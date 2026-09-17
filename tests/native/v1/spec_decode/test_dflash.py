# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RBLNDFlashProposer's draft-block geometry and cache write.

Every case here pins something a run measured rather than something the code
merely does, because each was a regression that only a full measurement run
surfaced:

  - the mask's query positions must name the block's real slots; naming the
    slots one block further on admitted the previous step's rejected draft K/V
  - the block is open within itself, not causal; closing it within itself
    reduced acceptance compared with leaving it open
  - one seq_idx and one block table for the whole drafter, because a second
    dynamic index on a partition is a compiler error that arrives as a segfault
  - the context write is contiguous on both sides, because a strided pair is
    staged through host memory and that staging buffer faults
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.v1.spec_decode.dflash as dflash_module
from tests.native.v1.spec_decode.utils import make_cad
from vllm_rbln.config import RBLNConfig
from vllm_rbln.v1.spec_decode.dflash import RBLNDFlashProposer

BLOCK_SIZE = 1024
WINDOW = 2048
NUM_SPEC = 7
QUERY_LEN = 1 + NUM_SPEC
MAX_SEQ = 8192


def _mask_self(sliding_window=WINDOW):
    """The attributes `_draft_block_mask` reads, and nothing else."""
    return SimpleNamespace(
        num_speculative_tokens=NUM_SPEC,
        sliding_window=sliding_window,
        block_size=BLOCK_SIZE,
    )


def _mask(seq_lens, sliding_window, num_reqs=None, max_seq_len=MAX_SEQ):
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    num_reqs = num_reqs if num_reqs is not None else len(seq_lens)
    return RBLNDFlashProposer._draft_block_mask(
        _mask_self(sliding_window),
        lens,
        num_reqs,
        num_reqs,
        max_seq_len,
        sliding_window,
    )


class TestDraftBlockMask:
    def test_shape_is_one_row_per_query_slot(self):
        for window in (None, WINDOW):
            mask = _mask([4000], window)
            assert tuple(mask.shape) == (1, 1, 1, QUERY_LEN, MAX_SEQ)

    @pytest.mark.parametrize("window", [None, WINDOW])
    def test_admits_nothing_past_the_block(self, window):
        """The regression that cost the most: a mask built from a length that
        counted the query block in sat eight slots further on, so every draft
        query admitted eight slots holding the previous step's rejected K/V."""
        seq_len = 4000
        mask = _mask([seq_len], window)[0, 0, 0]
        assert mask[:, seq_len + QUERY_LEN :].sum() == 0
        # ...and the block's own slots are all real keys, so they are admitted.
        assert mask[-1, seq_len : seq_len + QUERY_LEN].all()

    def test_block_is_open_within_itself(self):
        """Not causal: a block that only looked backwards is what the causal
        kernel family already gives, and this model would not need the
        mask-taking one at all."""
        seq_len = 4000
        mask = _mask([seq_len], WINDOW)[0, 0, 0]
        block = mask[:, seq_len : seq_len + QUERY_LEN]
        assert block.all(), "every query slot must see every other one"

    def test_sliding_row_sees_exactly_the_window(self):
        seq_len = 4000
        mask = _mask([seq_len], WINDOW)[0, 0, 0]
        context = mask[:, :seq_len]
        for row in range(QUERY_LEN):
            # The window is measured back from the row's own position, so the
            # rows nearest the block trade context slots for block slots.
            # The row's own slot is inside the block, so the context holds
            # one fewer than the window and slides forward with the row.
            assert int(context[row].sum()) == WINDOW - 1 - row
            first = int(context[row].nonzero()[0])
            assert first == seq_len - WINDOW + 1 + row

    def test_full_layer_sees_the_whole_context(self):
        seq_len = 4000
        mask = _mask([seq_len], None)[0, 0, 0]
        assert mask[:, : seq_len + QUERY_LEN].all()

    def test_rows_are_padded_with_zeros_not_dropped(self):
        mask = RBLNDFlashProposer._draft_block_mask(
            _mask_self(WINDOW),
            torch.tensor([4000, 3000], dtype=torch.int64),
            2,
            4,
            MAX_SEQ,
            WINDOW,
        )
        assert mask.shape[0] == 4
        assert mask[2:].sum() == 0


class TestContextWriteContiguity:
    """A strided copy pair is staged through host memory, and that staging
    buffer's recycled address is what faulted mid-run. Both sides have to be
    contiguous, which is only true one layer and head at a time."""

    NUM_KV_HEADS = 8
    HEAD_DIM = 128

    # block_axis 0 is the rbln_custom_ops layout, 1 the rbln_triton_ops one.
    # Neither side of the copy changes rank, so both reach the same conclusion.
    def _cache(self, block_axis):
        shape = [2, self.NUM_KV_HEADS, 1, BLOCK_SIZE, self.HEAD_DIM]
        shape.insert(block_axis, 4)
        return torch.zeros(shape, dtype=torch.bfloat16)

    @pytest.mark.parametrize("block_axis", [0, 1], ids=["blocks_first", "kv_first"])
    def test_all_heads_at_once_is_strided_on_both_sides(self, block_axis):
        cache = self._cache(block_axis)
        source = torch.zeros(6, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.bfloat16)
        assert not cache.select(block_axis, 1)[0, :, 0, 3:9, :].is_contiguous()
        assert not source[0:6].transpose(0, 1).is_contiguous()

    @pytest.mark.parametrize("block_axis", [0, 1], ids=["blocks_first", "kv_first"])
    def test_per_head_is_contiguous_on_both_sides(self, block_axis):
        cache = self._cache(block_axis)
        # Head-major, which is the layout the compiled projection now emits.
        source = torch.zeros(self.NUM_KV_HEADS, 6, self.HEAD_DIM, dtype=torch.bfloat16)
        one_block = cache.select(block_axis, 1)
        for head in range(self.NUM_KV_HEADS):
            assert one_block[0, head, 0, 3:9, :].is_contiguous()
            assert source[head, 0:6, :].is_contiguous()

    def test_a_write_run_never_leaves_its_block(self):
        """Runs are cut at block boundaries, which is what makes the
        destination a single contiguous span."""
        positions = torch.tensor([1020, 1021, 1022, 1023, 1024, 1025])
        blocks = (positions // BLOCK_SIZE).tolist()
        assert blocks == [0, 0, 0, 0, 1, 1]
        offsets = (positions % BLOCK_SIZE).tolist()
        assert offsets == [1020, 1021, 1022, 1023, 0, 1]


class TestSingleSequenceGuard:
    """`--max-num-seqs > 1` costs no errors and no output damage, only
    acceptance, which drops below the no-speculation baseline on the same
    prompts. The cause is not identified; the point here is that it fails
    loudly rather than quietly."""

    def test_one_sequence_is_allowed(self):
        RBLNDFlashProposer._require_single_sequence(SimpleNamespace(max_num_seqs=1))

    @pytest.mark.parametrize("max_num_seqs", [2, 4, 16])
    def test_a_wider_batch_is_refused(self, max_num_seqs):
        with pytest.raises(NotImplementedError, match="max-num-seqs 1"):
            RBLNDFlashProposer._require_single_sequence(
                SimpleNamespace(max_num_seqs=max_num_seqs)
            )


class TestSpanningBlockAllocation:
    SPANS = BLOCK_SIZE - 1  # a block starting here leaves its page; 0 does not

    def _run(self, table, ctx_lens):
        """Drive the real `_run_query_pass`; it raises once past the check."""
        n = len(ctx_lens)

        def reached(*args, **kwargs):
            raise RuntimeError("reached the draft pass")

        proposer = SimpleNamespace(
            arange_cpu=torch.arange(n + 1, dtype=torch.int32),
            dflash_causal=False,
            block_size=BLOCK_SIZE,
            positions=torch.zeros(n * QUERY_LEN, dtype=torch.int64),
            _dropped_rows=None,
            _build_draft_attn_metadata=reached,
        )
        cad = make_cad(
            [i * QUERY_LEN for i in range(n + 1)], [c + QUERY_LEN for c in ctx_lens]
        )
        cad.block_table_tensor = torch.tensor(table, dtype=torch.int32)
        RBLNDFlashProposer._run_query_pass(
            proposer,
            cad,
            n,
            QUERY_LEN,
            n * QUERY_LEN,
            torch.tensor(ctx_lens, dtype=torch.int32),
            torch.zeros(n, dtype=torch.int32),
        )
        return proposer

    @pytest.mark.parametrize(
        "table, ctx_lens",
        [
            ([[71, 0, 0, 0]], [0]),
            ([[71, 6, 0, 0]], [SPANS]),
        ],
        ids=["stays_on_its_page", "next_page_allocated"],
    )
    def test_a_step_proceeds_when_the_page_it_ends_on_is_allocated(
        self, table, ctx_lens
    ):
        with pytest.raises(RuntimeError, match="reached the draft pass"):
            self._run(table, ctx_lens)

    @pytest.mark.parametrize(
        "table, ctx_lens",
        [
            ([[71, 0, 0, 0]], [SPANS]),
            ([[71, 6]], [2 * BLOCK_SIZE - 1]),
            ([[71, 6, 0, 0], [80, 0, 0, 0]], [SPANS, SPANS]),
        ],
        ids=["unfilled", "past_the_table", "one_bad_row_of_two"],
    )
    def test_a_step_gives_up_when_it_is_not(self, table, ctx_lens):
        assert bool(self._run(table, ctx_lens)._dropped_rows.all())


class TestPlatformRefusals:
    """The three configurations DFlash cannot run on, all refused at
    construction and all before the base class does any work, so none of them
    reaches a device.

    Each fails silently otherwise: an eager context write goes through an
    attention op that exists only as a compiled kernel, and without device
    tensors the cache is allocated on `meta`, which accepts a host copy and
    discards it."""

    @staticmethod
    def _config(max_num_seqs=1, enforce_eager=False, compile_model=True):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            speculative_config=SimpleNamespace(enforce_eager=enforce_eager),
            additional_config=RBLNConfig(compile_model=compile_model),
        )

    def _construct(self):
        return RBLNDFlashProposer(self._config(), torch.device("cpu"))

    def test_a_wider_batch_is_refused_at_construction(self):
        with pytest.raises(NotImplementedError, match="max-num-seqs 1"):
            RBLNDFlashProposer(self._config(max_num_seqs=4), torch.device("cpu"))

    def test_eager_is_refused(self):
        with pytest.raises(NotImplementedError, match="cannot run eager"):
            RBLNDFlashProposer(self._config(enforce_eager=True), torch.device("cpu"))

    def test_compile_disabled_is_refused(self):
        with pytest.raises(NotImplementedError, match="cannot run eager"):
            RBLNDFlashProposer(self._config(compile_model=False), torch.device("cpu"))

    def test_host_visible_cache_is_required(self, monkeypatch):
        """Without device tensors the cache is on `meta` and the context write
        is dropped without an error."""
        monkeypatch.setattr(dflash_module, "USE_DEVICE_TENSOR", False)
        with pytest.raises(NotImplementedError, match="USE_DEVICE_TENSOR"):
            self._construct()


class TestDenseDrafterGuard:
    """A fused-MoE drafter is refused rather than run. The draft pass keeps only
    `num_tokens_across_dp` and drops the padded batch the ranks agreed on, so
    its expert dimension would not match its peers' and the group would hang --
    a silent stall, not an error. With MoE refused, a DP-idle rank may skip its
    draft unconditionally: the drafter runs no collective of its own."""

    @staticmethod
    def _model(*modules):
        return SimpleNamespace(modules=lambda: modules)

    def test_a_dense_drafter_is_allowed(self):
        RBLNDFlashProposer._require_dense_drafter(self._model(object(), object()))

    def test_a_moe_drafter_is_refused(self):
        moe = object.__new__(dflash_module.MoERunner)
        with pytest.raises(NotImplementedError, match="fused MoE"):
            RBLNDFlashProposer._require_dense_drafter(self._model(object(), moe))
