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

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.spec_decode.dflash import DFlashProposer

import vllm_rbln.v1.spec_decode.dflash as dflash_module
from tests.vllm.v1.spec_decode.utils import make_cad
from vllm_rbln.config import RBLNConfig
from vllm_rbln.v1.spec_decode.dflash import RBLNDFlashProposer
from vllm_rbln.v1.worker.dp_utils import (
    DPStatus,
    ShapeConfig,
    determine_batch_execution_and_padding,
)

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


class TestSchedulerCapacity:
    @pytest.mark.parametrize("max_num_seqs", [1, 2, 4, 16])
    def test_initialization_accepts_any_scheduler_capacity(
        self, monkeypatch, max_num_seqs
    ):
        """The configured capacity may exceed the number of active requests."""

        def initialize_base(proposer, **_kwargs):
            proposer.arange = torch.arange(NUM_SPEC + 1)
            proposer.dflash_causal = False

        monkeypatch.setattr(DFlashProposer, "__init__", initialize_base)
        monkeypatch.setattr(dflash_module, "USE_DEVICE_TENSOR", True)
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            additional_config=RBLNConfig(compile_model=True),
            speculative_config=SimpleNamespace(
                enforce_eager=False,
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(layer_types=[], sliding_window=None)
                ),
            ),
        )

        proposer = RBLNDFlashProposer(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

        assert proposer.runner is None
        assert proposer.arange_cpu.shape == (NUM_SPEC + 1,)


class _ReachedDraftPass(Exception):
    """Raised in place of the draft pass, so a step that does not give up is
    told apart from one that does. A named exception, not a bare RuntimeError:
    the caller catches it, and catching RuntimeError would swallow a real one."""


class TestSpanningBlockAllocation:
    SPANS = BLOCK_SIZE - 1  # a block starting here leaves its page; 0 does not

    def _run(self, table, ctx_lens):
        """Drive the real `_run_query_pass` and report whether it got past the
        allocation check, alongside the drafts it kept."""
        n = len(ctx_lens)

        def reached(*args, **kwargs):
            raise _ReachedDraftPass

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
        try:
            RBLNDFlashProposer._run_query_pass(
                proposer,
                cad,
                n,
                QUERY_LEN,
                n * QUERY_LEN,
                torch.tensor(ctx_lens, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
            )
            proposer.reached = False
        except _ReachedDraftPass:
            proposer.reached = True
        return proposer

    @pytest.mark.parametrize(
        "table, ctx_lens",
        [
            ([[71, 0, 0, 0]], [0]),
            ([[71, 6, 0, 0]], [SPANS]),
        ],
        ids=["stays_on_its_page", "next_page_allocated"],
    )
    def test_a_step_keeps_its_drafts_when_the_page_it_ends_on_is_allocated(
        self, table, ctx_lens
    ):
        proposer = self._run(table, ctx_lens)
        assert proposer.reached
        assert not bool(proposer._dropped_rows.any())

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
        proposer = self._run(table, ctx_lens)
        assert not proposer.reached
        assert bool(proposer._dropped_rows.all())


class TestPlatformRefusals:
    """The three configurations DFlash cannot run on, all refused at
    construction and all before the base class does any work, so none of them
    reaches a device.

    Each fails silently otherwise: an eager context write goes through an
    attention op that exists only as a compiled kernel, and without device
    tensors the cache is allocated on `meta`, which accepts a host copy and
    discards it."""

    @staticmethod
    def _config(enforce_eager=False, compile_model=True):
        return SimpleNamespace(
            speculative_config=SimpleNamespace(enforce_eager=enforce_eager),
            additional_config=RBLNConfig(compile_model=compile_model),
        )

    def _construct(self):
        return RBLNDFlashProposer(self._config(), torch.device("cpu"))

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
    """A DP-idle rank skips its draft, so a fused-MoE drafter would hang a busy
    rank's expert collective. Dense drafters run no collective of their own."""

    @staticmethod
    def _model(*modules):
        return SimpleNamespace(modules=lambda: modules)

    def test_a_dense_drafter_is_allowed(self):
        RBLNDFlashProposer._require_dense_drafter(self._model(object(), object()))

    def test_a_moe_drafter_is_refused(self):
        moe = object.__new__(dflash_module.MoERunner)
        with pytest.raises(NotImplementedError, match="fused MoE"):
            RBLNDFlashProposer._require_dense_drafter(self._model(object(), moe))


class TestQueryPassBatch:
    """The query pass runs at the decode bucket the target verifies at.

    The runner pads the target to its smallest fitting bucket. A drafter that
    stayed at the live request count ran a batch shape the target never did:
    one warmup never compiled, and the target/drafter batch asymmetry behind
    the acceptance collapse under padded verification. The full bucket ladder
    only hid it while every measured live count happened to be a bucket."""

    BUCKETS = [1, 2, 4]
    MAX_TOKENS = 4 * QUERY_LEN
    CONTEXT = 100  # well inside the first page, so no row crosses
    MASK_TOKEN = 151667
    STALE = 99  # what a previous step left in the buffers

    def _proposer(
        self,
        monkeypatch,
        num_reqs,
        *,
        dp_size=1,
        dp_status=None,
        max_tokens=None,
        specialized_moe_decode=False,
        dp_rank=0,
    ):
        max_tokens = self.MAX_TOKENS if max_tokens is None else max_tokens
        proposer = RBLNDFlashProposer.__new__(RBLNDFlashProposer)
        proposer.runner = SimpleNamespace(
            shape_config=ShapeConfig(
                decode_batch_buckets=self.BUCKETS,
                find_bucket=lambda n: next(b for b in self.BUCKETS if b >= n),
                max_num_tokens=max_tokens,
                specialized_moe_decode=specialized_moe_decode,
            ),
            dp_status=dp_status,
            kv_cache_bases=None,
            input_batch=SimpleNamespace(num_reqs=num_reqs),
        )
        proposer.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=dp_size)
        )
        proposer.dp_rank = dp_rank
        proposer.draft_has_moe = False
        proposer.num_speculative_tokens = NUM_SPEC
        proposer.block_size = BLOCK_SIZE
        proposer.dflash_causal = False
        proposer.max_num_tokens = max_tokens
        proposer.arange_cpu = torch.arange(num_reqs + 1, dtype=torch.int32)
        proposer.device = torch.device("cpu")
        proposer.parallel_drafting_token_id = self.MASK_TOKEN
        proposer.input_ids = torch.full((max_tokens,), self.STALE, dtype=torch.int32)
        proposer.positions = torch.full((max_tokens,), self.STALE, dtype=torch.int64)

        calls = SimpleNamespace(metadata=None, model=None, context=None)

        def build_metadata(cad, positions, num_reqs, num_reqs_padded):
            calls.metadata = (num_reqs, num_reqs_padded, cad.num_reqs)
            return {}

        def model(input_ids, positions, token_indices_to_sample):
            calls.model = (
                tuple(input_ids.shape),
                tuple(positions.shape),
                int(token_indices_to_sample.shape[0]),
            )
            # One argmax per mask position, numbered so a slice is checkable.
            return torch.arange(token_indices_to_sample.shape[0])

        @contextmanager
        def forward_context(per_layer, vllm_config, **kwargs):
            calls.context = kwargs
            yield

        proposer._build_draft_attn_metadata = build_metadata
        proposer.model_executable = model
        monkeypatch.setattr(dflash_module, "set_forward_context", forward_context)
        monkeypatch.setattr(
            dflash_module, "build_kv_cache_forward_context_kwargs", lambda bases: {}
        )
        return proposer, calls

    def _cad(self, num_reqs):
        return SimpleNamespace(
            seq_lens_cpu_upper_bound=MAX_SEQ,
            max_seq_len=self.CONTEXT,
            block_table_tensor=torch.ones(num_reqs, 4, dtype=torch.int32),
        )

    def _run(self, proposer, num_reqs):
        return proposer._run_query_pass(
            self._cad(num_reqs),
            num_reqs,
            QUERY_LEN,
            num_reqs * QUERY_LEN,
            torch.zeros(num_reqs, dtype=torch.int32),
            torch.full((num_reqs,), self.CONTEXT, dtype=torch.int32),
        )

    def test_a_live_count_between_buckets_runs_at_the_next_bucket(self, monkeypatch):
        proposer, calls = self._proposer(monkeypatch, num_reqs=3)
        drafts = self._run(proposer, 3)
        real, padded, described = calls.metadata
        assert (real, padded) == (3, 4)
        assert described == 3, "the metadata describes the real rows; the builder pads"
        assert calls.model == ((4, QUERY_LEN), (4, QUERY_LEN), 4 * NUM_SPEC)
        assert drafts.shape[0] == 4 * NUM_SPEC
        assert calls.context["num_tokens"] == 3 * QUERY_LEN
        # Off the DP path nothing pads the token dimension (`RBLNDPMetadata.make`).
        assert calls.context["num_padded_tokens"] is None
        assert calls.context["num_tokens_across_dp"] is None

    def test_a_bucket_sized_batch_is_not_padded(self, monkeypatch):
        proposer, calls = self._proposer(monkeypatch, num_reqs=2)
        self._run(proposer, 2)
        assert calls.metadata[:2] == (2, 2)
        assert calls.model[0] == (2, QUERY_LEN)
        assert (proposer.input_ids == self.STALE).all(), "nothing to define"

    def test_padded_rows_are_defined_not_stale(self, monkeypatch):
        proposer, _ = self._proposer(monkeypatch, num_reqs=3)
        self._run(proposer, 3)
        tail = slice(3 * QUERY_LEN, 4 * QUERY_LEN)
        assert (proposer.input_ids[tail] == self.MASK_TOKEN).all()
        assert (proposer.positions[tail] == 0).all()
        # The real rows belong to `_fill_first_pass_inputs` and are left alone.
        assert (proposer.input_ids[: 3 * QUERY_LEN] == self.STALE).all()
        assert (proposer.positions[: 3 * QUERY_LEN] == self.STALE).all()

    def test_propose_returns_one_row_per_real_request(self, monkeypatch):
        proposer, calls = self._proposer(monkeypatch, num_reqs=3)
        proposer.supports_mm_inputs = False
        proposer.hidden_size = 16
        proposer.model = SimpleNamespace(draft_id_to_target_id=None)
        proposer._fill_first_pass_inputs = lambda *args: (
            0,
            torch.zeros(3, dtype=torch.int32),
            torch.full((3,), self.CONTEXT, dtype=torch.int32),
        )
        proposer._write_context_kv = lambda *args: None

        drafts = proposer.propose(
            target_token_ids=torch.zeros(3, dtype=torch.int32),
            target_positions=torch.zeros(3, dtype=torch.int64),
            target_hidden_states=torch.zeros(3, 16),
            next_token_ids=torch.zeros(3, dtype=torch.int32),
            token_indices_to_sample=None,
            common_attn_metadata=self._cad(3),
        )

        assert calls.model[0] == (4, QUERY_LEN), "the graph ran at the bucket"
        # ...and the padded fourth row's drafts never reach the scheduler.
        assert torch.equal(drafts, torch.arange(3 * NUM_SPEC).view(3, NUM_SPEC))

    def test_dp_pads_the_token_dimension_to_the_bucket(self, monkeypatch):
        status = DPStatus(
            num_tokens=(3 * QUERY_LEN, 2 * QUERY_LEN),
            num_reqs=(3, 2),
            is_prefill=(False, False),
            is_idle=(False, False),
            num_tokens_across_dp=torch.tensor(
                [3 * QUERY_LEN, 2 * QUERY_LEN], dtype=torch.int32
            ),
        )
        proposer, calls = self._proposer(
            monkeypatch, num_reqs=3, dp_size=2, dp_status=status
        )
        self._run(proposer, 3)
        assert calls.metadata[:2] == (3, 4)
        assert calls.context["num_padded_tokens"] == 4 * QUERY_LEN
        assert calls.context["num_tokens_across_dp"].tolist() == [
            3 * QUERY_LEN,
            2 * QUERY_LEN,
        ]

    def test_a_bucket_wider_than_the_token_budget_is_refused(self, monkeypatch):
        """The live count fits the buffers, its bucket does not: refuse rather
        than fall back to the live shape, which is the mismatch itself."""
        proposer, _ = self._proposer(monkeypatch, num_reqs=3, max_tokens=3 * QUERY_LEN)
        with pytest.raises(AssertionError, match="decode bucket"):
            self._run(proposer, 3)

    @pytest.mark.parametrize(
        "tokens,reqs,prefill,idle,rank,specialized,expected_bucket",
        [
            ((8, 24), (1, 3), (False, False), (False, False), 0, True, 4),
            ((24, 8), (3, 1), (False, False), (False, False), 1, True, 4),
            ((8, 32), (1, 1), (False, True), (False, False), 0, True, 4),
            ((8, 2), (1, 2), (False, False), (False, False), 0, True, 4),
            ((32, 24), (1, 3), (True, False), (False, False), 0, True, 1),
            ((8, 24), (1, 3), (False, False), (False, False), 0, False, 1),
            ((8, 1), (1, 1), (False, False), (False, True), 0, True, 1),
        ],
        ids=[
            "busier-peer",
            "nonzero-rank",
            "prefilling-peer",
            "different-query-lengths",
            "local-prefill",
            "unspecialized",
            "idle-peer",
        ],
    )
    def test_dp_query_batch_matches_target(
        self,
        monkeypatch,
        tokens,
        reqs,
        prefill,
        idle,
        rank,
        specialized,
        expected_bucket,
    ):
        status = DPStatus(
            num_tokens=tokens,
            num_reqs=reqs,
            is_prefill=prefill,
            is_idle=idle,
            num_tokens_across_dp=torch.tensor(tokens, dtype=torch.int32),
        )
        num_reqs = reqs[rank]
        proposer, calls = self._proposer(
            monkeypatch,
            num_reqs,
            dp_size=2,
            dp_status=status,
            specialized_moe_decode=specialized,
            dp_rank=rank,
        )
        target_batch, _ = determine_batch_execution_and_padding(
            cfg=proposer.runner.shape_config,
            num_reqs=num_reqs,
            num_tokens=tokens[rank],
            is_prefill=prefill[rank],
            status=status,
        )
        assert target_batch.num_reqs_padded == expected_bucket

        self._run(proposer, num_reqs)

        assert calls.metadata[:2] == (num_reqs, expected_bucket)
        assert calls.model == (
            (expected_bucket, QUERY_LEN),
            (expected_bucket, QUERY_LEN),
            expected_bucket * NUM_SPEC,
        )
        assert calls.context["num_padded_tokens"] == expected_bucket * QUERY_LEN
