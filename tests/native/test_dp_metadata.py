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

# RBLNDPMetadata's DP token/req all-reduce. The collective is faked at the
# dist.all_reduce primitive, so the real bit-packing runs and the fake only sums
# in the other ranks' values.

from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.forward_context import RBLNDPMetadata


class TestMake:
    def test_non_dp_builds_single_slot(self):
        parallel_config = SimpleNamespace(data_parallel_size=1)
        meta = RBLNDPMetadata.make(parallel_config, num_tokens=8)
        assert meta.num_tokens_across_dp_cpu.cpu().tolist() == [8]
        assert meta.max_pads_across_dp is None

    @pytest.mark.parametrize(
        "extra",
        [
            {"num_tokens_across_dp": torch.tensor([8], dtype=torch.int32)},
            {"num_padded_tokens": 16},
        ],
    )
    def test_non_dp_rejects_dp_only_args(self, extra):
        parallel_config = SimpleNamespace(data_parallel_size=1)
        with pytest.raises(AssertionError):
            RBLNDPMetadata.make(parallel_config, num_tokens=8, **extra)

    def test_dp_uses_provided_counts_and_pad_buffer(self):
        parallel_config = SimpleNamespace(data_parallel_size=4)
        across = torch.tensor([8, 8, 8, 8], dtype=torch.int32)
        meta = RBLNDPMetadata.make(
            parallel_config,
            num_tokens=8,
            num_tokens_across_dp=across,
            num_padded_tokens=16,
        )
        assert meta.num_tokens_across_dp_cpu.cpu().tolist() == [8, 8, 8, 8]
        assert meta.max_pads_across_dp.shape == (16,)

    @pytest.mark.parametrize(
        "extra",
        [
            {"num_padded_tokens": 16},  # missing num_tokens_across_dp
            {"num_tokens_across_dp": torch.tensor([8, 8, 8, 8], dtype=torch.int32)},
        ],
    )
    def test_dp_requires_both_args(self, extra):
        parallel_config = SimpleNamespace(data_parallel_size=4)
        with pytest.raises(AssertionError):
            RBLNDPMetadata.make(parallel_config, num_tokens=8, **extra)
