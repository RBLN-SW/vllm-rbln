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

"""Tests for the padded EAGLE input-prep helpers (spec_decode/utils.py): pure
rejection-accounting math split out of EagleProposer, with expected values
pinned to upstream's tests/v1/spec_decode/test_eagle.py."""

import pytest
import torch
from vllm.platforms import current_platform

from vllm_rbln.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)

pytestmark = pytest.mark.maybe_use_device


def _t(data, dtype=torch.int64) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=current_platform.device_type)


class TestEaglePrepareNextTokenPadded:
    """1 + accepted-count and the next token to sample from: the last valid
    token, or the backup when nothing is valid or the request is discarded."""

    def test_selects_last_valid_token_and_counts(self):
        # Rows: two accepted, all five accepted, none accepted, three accepted.
        sampled = _t(
            [
                [0, 1, -1, -1, -1],
                [0, 1, 2, 3, 4],
                [-1, -1, -1, -1, -1],
                [0, 1, 2, -1, -1],
            ]
        )
        backup = _t([10, 20, 30, 40], dtype=torch.int32)
        discard = _t([False, False, False, False], dtype=torch.bool)

        next_ids, valid_count = eagle_prepare_next_token_padded(
            sampled, discard, backup, vocab_size=100
        )
        # Row 2 has no valid token -> backup 30; the rest take their last valid.
        assert next_ids.cpu().tolist() == [1, 4, 30, 2]
        assert valid_count.cpu().tolist() == [2, 5, 0, 3]

    def test_discard_mask_forces_backup_and_zero_count(self):
        # Same inputs, but request 3 is discarded: its (otherwise valid) result
        # is overridden by the backup token and its count zeroed.
        sampled = _t(
            [
                [0, 1, -1, -1, -1],
                [0, 1, 2, 3, 4],
                [-1, -1, -1, -1, -1],
                [0, 1, 2, -1, -1],
            ]
        )
        backup = _t([10, 20, 30, 40], dtype=torch.int32)
        discard = _t([False, False, False, True], dtype=torch.bool)

        next_ids, valid_count = eagle_prepare_next_token_padded(
            sampled, discard, backup, vocab_size=100
        )
        assert next_ids.cpu().tolist() == [1, 4, 30, 40]
        assert valid_count.cpu().tolist() == [2, 5, 0, 0]

    def test_token_at_or_above_vocab_size_is_invalid(self):
        # 150 >= vocab_size (100) is invalid, so it neither counts nor is
        # selected: the last valid token (3, at position 1) is returned.
        sampled = _t([[5, 3, 150]])
        backup = _t([99], dtype=torch.int32)
        discard = _t([False], dtype=torch.bool)

        next_ids, valid_count = eagle_prepare_next_token_padded(
            sampled, discard, backup, vocab_size=100
        )
        assert next_ids.cpu().tolist() == [3]
        assert valid_count.cpu().tolist() == [2]


class TestEaglePrepareInputsPadded:
    """Per-request rejected-token count and the token index to sample from
    (the last query position shifted back by the rejected count)."""

    def test_partial_full_and_heavy_rejection_mix(self):
        # 3 requests, 2 draft tokens each (spec len 2). valid = 1+accepted.
        cu_num_draft = _t([2, 4, 6])
        valid_count = _t([2, 3, 1])
        query_start_loc = _t([0, 3, 6, 9])

        token_indices, num_rejected = eagle_prepare_inputs_padded(
            cu_num_draft, valid_count, query_start_loc
        )
        # rejected = (2+1) - valid = [1, 0, 2]; index = qsl[1:]-1-rejected.
        assert num_rejected.cpu().tolist() == [1, 0, 2]
        assert token_indices.cpu().tolist() == [1, 5, 6]

    def test_request_without_draft_tokens_never_rejects(self):
        # Request 0 has 0 draft tokens (cu stays flat); its rejected count is
        # forced to 0 regardless of the valid count, unlike request 1.
        cu_num_draft = _t([0, 3])
        valid_count = _t([5, 2])
        query_start_loc = _t([0, 1, 4])

        token_indices, num_rejected = eagle_prepare_inputs_padded(
            cu_num_draft, valid_count, query_start_loc
        )
        # req0: no draft -> 0; req1: (3+1)-2 = 2.
        assert num_rejected.cpu().tolist() == [0, 2]
        assert token_indices.cpu().tolist() == [0, 1]

    def test_cumulative_counts_are_differenced_per_request(self):
        # cu = [3, 5, 10] must yield per-request drafts [3, 2, 5], not the raw
        # cumulative values, so rejection uses the right denominator.
        cu_num_draft = _t([3, 5, 10])
        valid_count = _t([2, 1, 4])
        query_start_loc = _t([0, 3, 5, 10])

        token_indices, num_rejected = eagle_prepare_inputs_padded(
            cu_num_draft, valid_count, query_start_loc
        )
        # rejected = [3+1-2, 2+1-1, 5+1-4] = [2, 2, 2].
        assert num_rejected.cpu().tolist() == [2, 2, 2]
        assert token_indices.cpu().tolist() == [0, 2, 7]
