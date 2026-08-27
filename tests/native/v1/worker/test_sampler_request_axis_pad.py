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

import pytest
import torch
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_rbln.v1.spec_decode.utils import eagle_prepare_inputs_padded
from vllm_rbln.v1.worker.rbln_model_runner import _pad_spec_decode_metadata

BUCKET = 4


def _metadata(num_draft_tokens: list[int]) -> SpecDecodeMetadata:
    """A shrunken batch: fewer requests than the bucket the sampler is compiled for."""
    num_sampled = [n + 1 for n in num_draft_tokens]
    cu_sampled = torch.tensor(num_sampled, dtype=torch.int32).cumsum(0).int()
    cu_draft = torch.tensor(num_draft_tokens, dtype=torch.int32).cumsum(0).int()
    total_draft = int(cu_draft[-1])
    return SpecDecodeMetadata(
        draft_token_ids=torch.zeros(total_draft, dtype=torch.int32),
        num_draft_tokens=list(num_draft_tokens),
        cu_num_draft_tokens=cu_draft,
        cu_num_sampled_tokens=cu_sampled,
        target_logits_indices=torch.arange(total_draft, dtype=torch.int32),
        bonus_logits_indices=(cu_sampled - 1).int(),
        logits_indices=torch.arange(int(cu_sampled[-1]), dtype=torch.int32),
    )


def test_pads_the_request_axis_to_the_bucket():
    md = _pad_spec_decode_metadata(_metadata([1, 1]), BUCKET)

    assert len(md.num_draft_tokens) == BUCKET
    assert md.cu_num_draft_tokens.shape[0] == BUCKET
    assert md.cu_num_sampled_tokens.shape[0] == BUCKET
    assert md.bonus_logits_indices.shape[0] == BUCKET


def test_padding_preserves_the_packed_layout_the_op_requires():
    original = _metadata([1, 1])
    md = _pad_spec_decode_metadata(original, BUCKET)

    # The op does `reshaped_target_probs[:N] = target_probs` with
    # N == sum(num_draft_tokens), and sizes its buffers by max_spec_len.
    assert sum(md.num_draft_tokens) == sum(original.num_draft_tokens)
    assert md.max_spec_len == original.max_spec_len
    assert torch.equal(md.target_logits_indices, original.target_logits_indices)
    assert torch.equal(md.logits_indices, original.logits_indices)


def test_padded_rows_reread_a_valid_row():
    original = _metadata([1, 1])
    md = _pad_spec_decode_metadata(original, BUCKET)

    gathered_rows = original.logits_indices.shape[0]
    assert int(md.bonus_logits_indices.max()) < gathered_rows
    assert md.num_draft_tokens[len(original.num_draft_tokens) :] == [0, 0]


def test_full_batch_is_returned_unchanged():
    original = _metadata([1] * BUCKET)

    assert _pad_spec_decode_metadata(original, BUCKET) is original


def test_eagle_input_prep_still_sees_an_unpadded_request_axis():
    """Guards the leak that padding inside _calc_spec_decode_metadata caused.

    eagle_prepare_inputs_padded differences cu_num_draft_tokens against tensors
    sized by the live request count, so it must not receive a padded metadata.
    """
    num_reqs = 2
    original = _metadata([1] * num_reqs)
    valid_sampled_tokens_count = torch.ones(num_reqs, dtype=torch.int32)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32) * 2

    token_indices, num_rejected = eagle_prepare_inputs_padded(
        original.cu_num_draft_tokens, valid_sampled_tokens_count, query_start_loc
    )
    assert token_indices.shape[0] == num_reqs
    assert num_rejected.shape[0] == num_reqs

    padded = _pad_spec_decode_metadata(original, BUCKET)
    with pytest.raises(RuntimeError, match="must match the size"):
        eagle_prepare_inputs_padded(
            padded.cu_num_draft_tokens, valid_sampled_tokens_count, query_start_loc
        )
