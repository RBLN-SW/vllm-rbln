# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer
from vllm_rbln.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)


# Verifies that discarded requests fall back to backup tokens
# and report zero valid samples.
def test_eagle_prepare_next_token_padded_uses_backup_for_discarded_requests():
    next_token_ids, valid_count = eagle_prepare_next_token_padded(
        sampled_token_ids=torch.tensor(
            [
                [10, 11, -1, -1],
                [3, 55, 60, -1],
                [-1, -1, -1, -1],
                [5, 6, 7, 8],
            ],
            dtype=torch.int32,
        ),
        discard_request_mask=torch.tensor([False, False, False, True]),
        backup_next_token_ids=torch.tensor([90, 91, 92, 93], dtype=torch.int32),
        vocab_size=50,
    )

    assert next_token_ids.dtype == torch.int32
    assert valid_count.dtype == torch.int32
    torch.testing.assert_close(
        next_token_ids,
        torch.tensor([11, 3, 92, 93], dtype=torch.int32),
    )
    torch.testing.assert_close(
        valid_count,
        torch.tensor([2, 1, 0, 0], dtype=torch.int32),
    )


# Verifies that padded input preparation returns both
# sample indices and rejected-token counts.
def test_eagle_prepare_inputs_padded_returns_indices_and_rejections():
    token_indices_to_sample, num_rejected_tokens_gpu = eagle_prepare_inputs_padded(
        cu_num_draft_tokens=torch.tensor([2, 5, 5, 8], dtype=torch.int32),
        valid_sampled_tokens_count=torch.tensor([3, 2, 0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 7, 8, 12], dtype=torch.int32),
    )

    assert token_indices_to_sample.dtype == torch.int32
    assert num_rejected_tokens_gpu.dtype == torch.int32
    torch.testing.assert_close(
        token_indices_to_sample,
        torch.tensor([2, 4, 7, 8], dtype=torch.int32),
    )
    torch.testing.assert_close(
        num_rejected_tokens_gpu,
        torch.tensor([0, 2, 0, 3], dtype=torch.int32),
    )


# Verifies that the proposer wrapper preserves the upstream
# three-value padded-input contract.
def test_prepare_inputs_padded_matches_upstream_tuple_contract():
    query_start_loc = torch.tensor([0, 3, 7, 8, 12], dtype=torch.int32)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        draft_token_ids=[[11, 12], [21, 22, 23], [], [31, 32, 33]],
        device=torch.device("cpu"),
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=torch.tensor([7, 7, 11, 9], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=query_start_loc_cpu[-1].item(),
        max_query_len=4,
        max_seq_len=seq_lens.max().item(),
        block_table_tensor=torch.zeros((4, 1), dtype=torch.int32),
        slot_mapping=torch.arange(query_start_loc_cpu[-1].item(), dtype=torch.int32),
        causal=True,
        dcp_local_seq_lens=None,
    )

    (
        spec_common_attn_metadata,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
    ) = RBLNEagleProposer.prepare_inputs_padded(
        object(),
        common_attn_metadata,
        spec_decode_metadata,
        torch.tensor([3, 2, 0, 1], dtype=torch.int32),
    )

    assert spec_common_attn_metadata.num_actual_tokens == 12
    assert spec_common_attn_metadata.max_query_len == 4
    assert spec_common_attn_metadata.max_seq_len == 13
    torch.testing.assert_close(
        token_indices_to_sample,
        torch.tensor([2, 4, 7, 8], dtype=torch.int32),
    )
    torch.testing.assert_close(
        num_rejected_tokens_gpu,
        torch.tensor([0, 2, 0, 3], dtype=torch.int32),
    )
