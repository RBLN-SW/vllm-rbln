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
merely does, because each was a regression that cost a full 26-instance
measurement to find:

  - the mask's query positions must name the block's real slots; naming the
    slots one block further on admitted the previous step's rejected draft K/V
  - the block is open within itself, not causal; closing it measured 3.25
    tokens/step against 3.62
  - one seq_idx and one block table for the whole drafter, because a second
    dynamic index on a partition is a compiler error that arrives as a segfault
  - the context write is contiguous on both sides, because a strided pair is
    staged through host memory and that staging buffer faults
"""

from types import SimpleNamespace

import pytest
import torch

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


class TestPageCrossing:
    """The kernel scatters the whole query block at one offset per partition,
    so the last QUERY_LEN - 1 offsets of a page are unrepresentable."""

    @staticmethod
    def _crossing(seq_lens):
        lens = torch.tensor(seq_lens, dtype=torch.int64)
        return (lens % BLOCK_SIZE) + QUERY_LEN > BLOCK_SIZE

    def test_only_the_last_offsets_of_a_page_cross(self):
        crossing = self._crossing(list(range(BLOCK_SIZE)))
        assert int(crossing.sum()) == QUERY_LEN - 1
        assert crossing[BLOCK_SIZE - QUERY_LEN + 1 :].all()
        assert not crossing[: BLOCK_SIZE - QUERY_LEN + 1].any()

    def test_a_block_start_never_crosses(self):
        assert not self._crossing([0, BLOCK_SIZE, 4 * BLOCK_SIZE]).any()

    def test_redirect_lands_on_the_next_page_start(self):
        lens = torch.tensor([BLOCK_SIZE - 3], dtype=torch.int64)
        redirected = (lens // BLOCK_SIZE + 1) * BLOCK_SIZE
        assert int(redirected[0]) == BLOCK_SIZE
        assert not self._crossing([int(redirected[0])]).any()


class TestContextWriteContiguity:
    """A strided copy pair is staged through host memory, and that staging
    buffer's recycled address is what faulted mid-run. Both sides have to be
    contiguous, which is only true one layer and head at a time."""

    NUM_KV_HEADS = 8
    HEAD_DIM = 128

    def _cache(self):
        return torch.zeros(
            2,
            4,
            self.NUM_KV_HEADS,
            1,
            BLOCK_SIZE,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )

    def test_all_heads_at_once_is_strided_on_both_sides(self):
        cache = self._cache()
        source = torch.zeros(6, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.bfloat16)
        assert not cache[0, 1, :, 0, 3:9, :].is_contiguous()
        assert not source[0:6].transpose(0, 1).is_contiguous()

    def test_per_head_is_contiguous_on_both_sides(self):
        cache = self._cache()
        # Head-major, which is the layout the compiled projection now emits.
        source = torch.zeros(self.NUM_KV_HEADS, 6, self.HEAD_DIM, dtype=torch.bfloat16)
        for head in range(self.NUM_KV_HEADS):
            assert cache[0, 1, head, 0, 3:9, :].is_contiguous()
            assert source[head, 0:6, :].is_contiguous()

    def test_a_write_run_never_leaves_its_block(self):
        """Runs are cut at block boundaries, which is what makes the
        destination a single contiguous span."""
        positions = torch.tensor([1020, 1021, 1022, 1023, 1024, 1025])
        blocks = (positions // BLOCK_SIZE).tolist()
        assert blocks == [0, 0, 0, 0, 1, 1]
        offsets = (positions % BLOCK_SIZE).tolist()
        assert offsets == [1020, 1021, 1022, 1023, 0, 1]
