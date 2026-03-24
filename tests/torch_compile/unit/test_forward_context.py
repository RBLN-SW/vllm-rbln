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


@pytest.fixture
def attn_metadata_mock():
    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionMetadata,
    )

    attn_metadata_mock = MagicMock(spec=RBLNFlashAttentionMetadata)
    attn_metadata_mock.num_actual_tokens = 16
    return attn_metadata_mock


def test_forward_context(vllm_config, attn_metadata_mock: MagicMock):
    # forward_context
    from vllm.forward_context import get_forward_context, set_forward_context

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
