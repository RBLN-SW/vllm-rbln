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

from unittest.mock import MagicMock

import pytest
import torch
from vllm.forward_context import get_forward_context

from vllm_rbln.forward_context import set_forward_context


@pytest.fixture
def attn_metadata_mock():
    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionMetadata,
    )

    attn_metadata_mock = MagicMock(spec=RBLNFlashAttentionMetadata)
    attn_metadata_mock.num_actual_tokens = 16
    return attn_metadata_mock


def test_forward_context(vllm_config, attn_metadata_mock: MagicMock):
    with set_forward_context(
        attn_metadata_mock,
        vllm_config,
        num_tokens_across_dp=torch.tensor([0, 1]),
        num_padded_tokens=1,
    ):
        # assert dp_metadata class name is RBLNDPMetadata
        assert (
            get_forward_context().dp_metadata.__class__.__name__ == "RBLNDPMetadata"
        ), (
            f"Expected 'dp_metadata' class name is RBLNDPMetadata, \
                    got {get_forward_context().dp_metadata.__class__.__name__}"
        )


def test_num_tokens_and_reqs_across_dp_bit_pack(monkeypatch):
    """Round-trip the bit-packed (num_tokens, num_reqs, is_prefill, is_idle)
    all-reduce. Intercepts the inner all-gather so no real DP group is needed,
    and checks each field unpacks correctly -- including the new is_idle mask
    and the (prefill -> num_reqs None) behavior."""
    from vllm_rbln.forward_context import RBLNDPMetadata

    token_bits = 16

    def peer_encoded(num_tokens, num_reqs):
        return num_tokens | (num_reqs << token_bits)

    def make_fake_inner(peer_value):
        def fake_inner(encoded, dp_size, dp_rank):
            arr = [0, 0]
            arr[dp_rank] = encoded
            arr[1 - dp_rank] = peer_value
            return torch.tensor(arr, dtype=torch.int32)

        return fake_inner

    # Decode: this rank (rank0) = idle (num_tokens=1, num_reqs=1, is_idle);
    # peer (rank1) = busy (num_tokens=8, num_reqs=2).
    monkeypatch.setattr(
        RBLNDPMetadata,
        "num_tokens_across_dp",
        staticmethod(make_fake_inner(peer_encoded(8, 2))),
    )
    tokens, reqs, idle = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
        num_tokens=1,
        num_reqs=1,
        dp_size=2,
        dp_rank=0,
        is_prefill=False,
        is_idle=True,
    )
    assert tokens.tolist() == [1, 8]
    assert reqs is not None and reqs.tolist() == [1, 2]
    assert idle.tolist() == [1, 0]

    # Prefill on this rank -> num_reqs_across_dp is None; is_idle stays 0.
    monkeypatch.setattr(
        RBLNDPMetadata,
        "num_tokens_across_dp",
        staticmethod(make_fake_inner(peer_encoded(8, 2))),
    )
    tokens, reqs, idle = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
        num_tokens=5,
        num_reqs=1,
        dp_size=2,
        dp_rank=0,
        is_prefill=True,
        is_idle=False,
    )
    assert tokens.tolist() == [5, 8]
    assert reqs is None
    assert idle.tolist() == [0, 0]
