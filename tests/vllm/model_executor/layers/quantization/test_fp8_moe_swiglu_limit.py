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
"""Fp8MoEMethod.apply hands a RoutedExperts' swiglu_limit (GLM-5.3, DeepSeek-V4:
silu(min(gate, L)) * clamp(up, -L, L)) to the fp8 group-dequantize MoE op, run here
through its CPU body."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rebel")  # registers torch.ops.rbln_custom_ops on import
import rebel  # noqa: E402,F401

from vllm_rbln.model_executor.layers.quantization.fp8 import (  # noqa: E402
    Fp8MoEMethod,
)

EXPERTS, HIDDEN, INTER, TOKENS, BLOCK = 2, 128, 128, 8, 128


def _method(use_w8a8: bool = False) -> Fp8MoEMethod:
    method = Fp8MoEMethod.__new__(Fp8MoEMethod)
    method.use_w8a8 = use_w8a8
    method.weight_block_size = [BLOCK, BLOCK]
    return method


def _layer(swiglu_limit, activation="silu"):
    g = torch.Generator().manual_seed(0)

    def fp8(*shape):
        return (torch.randn(*shape, generator=g) * 0.5).to(torch.float8_e4m3fn)

    def scale(*shape):
        return torch.rand(*shape, generator=g) * 0.2 + 0.1

    return SimpleNamespace(
        w13_weight=fp8(EXPERTS, 2 * INTER, HIDDEN),
        w2_weight=fp8(EXPERTS, HIDDEN, INTER),
        w13_weight_scale_inv=scale(EXPERTS, 2 * INTER // BLOCK, HIDDEN // BLOCK),
        w2_weight_scale_inv=scale(EXPERTS, HIDDEN // BLOCK, INTER // BLOCK),
        activation=SimpleNamespace(value=activation),
        expert_map=None,
        swiglu_limit=swiglu_limit,
    )


def _reference(layer, x, routing, limit):
    """The experts written out: block-dequantize, then the clamped SwiGLU."""

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
    out = _method().apply(layer, x, routing)
    expected = _reference(layer, x, routing, limit)
    torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
    if limit:  # the limit bites: the unclamped result differs
        assert not torch.allclose(out.float(), _reference(layer, x, routing, None))


def test_w8a8_refuses_a_limit():
    with pytest.raises(NotImplementedError, match="W8A8"):
        _method(use_w8a8=True).apply(
            _layer(10.0), torch.randn(TOKENS, HIDDEN), torch.ones(EXPERTS, TOKENS)
        )


def test_a_limit_needs_silu():
    with pytest.raises(NotImplementedError, match="silu"):
        _method().apply(
            _layer(10.0, activation="gelu"),
            torch.randn(TOKENS, HIDDEN),
            torch.ones(EXPERTS, TOKENS),
        )
