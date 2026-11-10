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
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_rbln.config import RBLNConfig
from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

EXPERTS, HIDDEN, INTER, TOKENS, BLOCK = 2, 128, 128, 8, 128


def _method(layer, use_w8a8: bool = False) -> Fp8MoEMethod:
    config = SimpleNamespace(additional_config=RBLNConfig(use_w8a8=use_w8a8))
    with set_current_vllm_config(config):
        return Fp8MoEMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
            ),
            layer,
        )


def _layer(swiglu_limit, activation="silu"):
    g = torch.Generator().manual_seed(0)

    def fp8(*shape):
        return (torch.randn(*shape, generator=g) * 0.5).to(torch.float8_e4m3fn)

    def scale(*shape):
        return torch.rand(*shape, generator=g) * 0.2 + 0.1

    return SimpleNamespace(
        moe_config=SimpleNamespace(swiglu_limit=swiglu_limit),
        w13_weight=fp8(EXPERTS, 2 * INTER, HIDDEN),
        w2_weight=fp8(EXPERTS, HIDDEN, INTER),
        w13_weight_scale_inv=scale(EXPERTS, 2 * INTER // BLOCK, HIDDEN // BLOCK),
        w2_weight_scale_inv=scale(EXPERTS, HIDDEN // BLOCK, INTER // BLOCK),
        activation=SimpleNamespace(value=activation),
        expert_map=None,
        swiglu_limit=swiglu_limit,
    )


def _reference(layer, x, routing, limit):
    def deq(w, s):
        s = s.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
        return w.float() * s

    w13 = deq(layer.w13_weight, layer.w13_weight_scale_inv)
    w2 = deq(layer.w2_weight, layer.w2_weight_scale_inv)
    out = torch.zeros(TOKENS, HIDDEN)
    for e in range(EXPERTS):
        gate = x.float() @ w13[e, :INTER].T
        up = x.float() @ w13[e, INTER:].T
        if limit:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        out += (torch.nn.functional.silu(gate) * up @ w2[e].T) * routing[e, :, None]
    return out


@pytest.mark.parametrize("limit", [None, 0.0, 0.25])
def test_apply_clamps_the_swiglu_when_the_layer_has_a_limit(limit):
    layer = _layer(limit)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(TOKENS, HIDDEN, generator=g)
    routing = torch.softmax(torch.randn(EXPERTS, TOKENS, generator=g), dim=0)
    out = _method(layer).apply(layer, x, routing)
    expected = _reference(layer, x, routing, limit)
    torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
    if limit:  # the limit bites: the unclamped result differs
        assert not torch.allclose(out.float(), _reference(layer, x, routing, None))


@pytest.mark.parametrize("limit", [None, 0.0, 10.0])
def test_w8a8_checks_clamp_support_at_initialization(limit):
    if limit:
        with pytest.raises(NotImplementedError, match="W8A8.*SwiGLU clamp"):
            _method(_layer(limit), use_w8a8=True)
    else:
        assert _method(_layer(limit), use_w8a8=True).use_w8a8


def test_a_limit_needs_silu():
    layer = _layer(10.0, activation="gelu")
    with pytest.raises(NotImplementedError, match="silu"):
        _method(layer).apply(
            layer, torch.randn(TOKENS, HIDDEN), torch.ones(EXPERTS, TOKENS)
        )
