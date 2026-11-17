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

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)

from vllm_rbln.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    RBLNUnquantizedFusedMoEMethod,
)


@pytest.mark.parametrize("limit", [None, 0.0, 10.0])
def test_clamp_support_is_checked_at_initialization(limit):
    moe = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        swiglu_limit=limit,
    )
    config = SimpleNamespace(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        if limit is None:
            method = UnquantizedFusedMoEMethod(moe)
            assert isinstance(method, RBLNUnquantizedFusedMoEMethod)
        else:
            with pytest.raises(NotImplementedError, match="unquantized.*swiglu_limit"):
                UnquantizedFusedMoEMethod(moe)
