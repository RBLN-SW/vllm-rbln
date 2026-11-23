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
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)

from vllm_rbln.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    RBLNUnquantizedFusedMoEMethod,
)

EXPERTS, HIDDEN, INTER, TOKENS = 2, 64, 64, 8


def _moe_config(activation, limit) -> FusedMoEConfig:
    parallel = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0,
        dp_rank=0, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="naive", enable_eplb=False,
    )  # fmt: skip
    return FusedMoEConfig(
        num_experts=EXPERTS,
        experts_per_token=1,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        num_local_experts=EXPERTS,
        num_logical_experts=EXPERTS,
        activation=activation,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.float32,
        swiglu_limit=limit,
    )


@pytest.mark.parametrize(
    ("activation", "act"),
    [
        (MoEActivation.SILU, torch.nn.functional.silu),
        (MoEActivation.GELU, torch.nn.functional.gelu),
    ],
)
@pytest.mark.parametrize("limit", [None, 0.0, 0.25])
def test_apply_clamps_the_glu_when_the_layer_has_a_limit(activation, act, limit):
    g = torch.Generator().manual_seed(0)
    layer = SimpleNamespace(
        w13_weight=torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=g) * 0.1,
        w2_weight=torch.randn(EXPERTS, HIDDEN, INTER, generator=g) * 0.1,
        activation=activation,
        expert_map=None,
        swiglu_alpha=None,
        swiglu_limit=limit,
        swiglu_beta=None,
    )
    x = torch.randn(TOKENS, HIDDEN, generator=g)
    routing = torch.softmax(torch.randn(EXPERTS, TOKENS, generator=g), dim=0)

    config = SimpleNamespace(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        method = RBLNUnquantizedFusedMoEMethod(_moe_config(activation, limit))
    out = method.apply(layer, x, routing)

    def reference(limit):
        expected = torch.zeros(TOKENS, HIDDEN)
        for e in range(EXPERTS):
            gate = x @ layer.w13_weight[e, :INTER].T
            up = x @ layer.w13_weight[e, INTER:].T
            if limit:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            expert = act(gate) * up @ layer.w2_weight[e].T
            expected += expert * routing[e, :, None]
        return expected

    torch.testing.assert_close(out, reference(limit), atol=1e-4, rtol=1e-4)
    if limit:
        assert not torch.allclose(out, reference(None), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("activation", "interleaved"),
    [(MoEActivation.SWIGLUOAI, True), (MoEActivation.SWIGLUOAI_UNINTERLEAVE, False)],
)
def test_apply_passes_the_whole_glu(activation, interleaved):
    """alpha, limit and beta reach the op, and SWIGLUOAI's interleaved w13 is split as
    such: swish(gate.clamp(max=L), alpha) * (up.clamp(+-L) + beta)."""
    alpha, limit, beta = 1.702, 7.0, 1.0
    g = torch.Generator().manual_seed(0)
    layer = SimpleNamespace(
        w13_weight=torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=g) * 0.1,
        w2_weight=torch.randn(EXPERTS, HIDDEN, INTER, generator=g) * 0.1,
        activation=activation,
        expert_map=None,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
        swiglu_beta=beta,
    )
    x = torch.randn(TOKENS, HIDDEN, generator=g)
    routing = torch.softmax(torch.randn(EXPERTS, TOKENS, generator=g), dim=0)

    config = SimpleNamespace(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        method = RBLNUnquantizedFusedMoEMethod(_moe_config(activation, limit))
    out = method.apply(layer, x, routing)

    expected = torch.zeros(TOKENS, HIDDEN)
    for e in range(EXPERTS):
        w13 = layer.w13_weight[e]
        if interleaved:
            gate_w, up_w = w13[::2], w13[1::2]
        else:
            gate_w, up_w = w13[:INTER], w13[INTER:]
        gate = (x @ gate_w.T).clamp(max=limit)
        up = (x @ up_w.T).clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(alpha * gate) * (up + beta)
        expected += (glu @ layer.w2_weight[e].T) * routing[e, :, None]
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
