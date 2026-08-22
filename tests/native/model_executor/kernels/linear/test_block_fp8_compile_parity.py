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

# Compile parity for the W8A16 block-FP8 kernel: CPU eager is the oracle, and the
# rbln-compiled graph (plus device eager on the device lane) must agree.
#
# Three shapes here are dictated by what the compiler accepts, not by taste:
#
#   - The fp8 weight is a module buffer, never a graph argument: a 1-byte dtype
#     cannot be a graph input at all. It is also what a real model holds.
#   - Every graph ends in the linear; a dequant with no matmul is rejected. The
#     dequant test therefore reads out through F.linear(I, W), which is W
#     transposed -- one output element per weight element, nothing to hide in.
#   - The weight spans more than one block per axis: a 128x128 weight, a 1x1
#     scale grid, fails to compile.

from typing import Any

import pytest
import torch
from vllm.platforms import current_platform

from vllm_rbln.compilation.compiler import compile as rbln_compile
from vllm_rbln.model_executor.kernels.linear.block_fp8 import (
    RBLNW8A16BlockFp8LinearKernel,
)

pytestmark = pytest.mark.use_device

_RTOL, _ATOL = 1e-2, 1e-2

# Production-sized fp8 block; toy blocks hit an rbln-compiler reshape edge case.
_BLOCK_N = _BLOCK_K = 128
# Non-square, so a transposed view would not silently have the right numel.
_OUT_FEATURES, _IN_FEATURES = 2 * _BLOCK_N, 4 * _BLOCK_K
_NUM_TOKENS = 8
# One scale per (block_n, block_k) tile of the weight.
_SCALE = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]

_ACT_DTYPES = [torch.float16, torch.bfloat16]
_ACT_DTYPE_IDS = ["fp16", "bf16"]

# Built once outside any compiled region so torch.compile sees weight_group_shape
# as a constant instead of tracing the (untraceable) object construction.
_KERNEL: Any = object.__new__(RBLNW8A16BlockFp8LinearKernel)
_KERNEL.weight_group_shape = (_BLOCK_N, _BLOCK_K)


def _agrees(actual, reference) -> bool:
    return torch.allclose(
        actual.cpu().float(), reference.float(), rtol=_RTOL, atol=_ATOL
    )


def _to_session_device(args):
    dev = current_platform.device_type
    return tuple(a.to(dev) if torch.is_tensor(a) else a for a in args)


def _assert_parity(build, *cpu_args):
    # Each step gets its own module: a weight-free compile relays the weight
    # buffer into the hardware layout in place -- that buffer is the weight pool,
    # not a copy -- so a compiled module no longer reads back as a plain tensor.
    dev = current_platform.device_type
    # 1. CPU eager oracle.
    ref = build()(*cpu_args)
    # 2. rbln-compiled, on whichever tensors the lane selects.
    dev_args = _to_session_device(cpu_args)
    assert _agrees(rbln_compile(build().to(dev), fullgraph=True)(*dev_args), ref)
    # 3. device eager, only when device tensors are on (--device-tensor 1).
    if dev == "rbln":
        assert _agrees(build().to(dev)(*dev_args), ref)


def _fp8_weight(out_features, in_features):
    return (
        torch.linspace(0.1, 2.0, out_features * in_features)
        .reshape(out_features, in_features)
        .to(torch.float8_e4m3fn)
    )


class _Dequant(torch.nn.Module):
    def __init__(self, weight, scale, act_dtype):
        super().__init__()
        self.act_dtype = act_dtype
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)

    def forward(self, readout):
        dequantized = _KERNEL._dequantize_block_fp8_weight(
            self.weight, self.scale, self.act_dtype
        )
        return torch.nn.functional.linear(readout, dequantized)


class _Apply(torch.nn.Module):
    def __init__(self, weight, scale):  # act dtype follows the activation
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)

    def forward(self, activation):
        return _KERNEL.apply_block_scaled_mm(activation, self.weight, None, self.scale)


@pytest.mark.parametrize("act_dtype", _ACT_DTYPES, ids=_ACT_DTYPE_IDS)
def test_dequantize_block_weight_matches_rbln_compiled(act_dtype):
    _assert_parity(
        lambda: _Dequant(
            _fp8_weight(_OUT_FEATURES, _IN_FEATURES),
            torch.tensor(_SCALE),
            act_dtype,
        ),
        torch.eye(_IN_FEATURES, dtype=act_dtype),
    )


@pytest.mark.parametrize("act_dtype", _ACT_DTYPES, ids=_ACT_DTYPE_IDS)
def test_apply_block_scaled_mm_matches_rbln_compiled(act_dtype):
    # Many tokens, the way a real decoder layer calls this.
    activation = (
        torch.linspace(0.5, 3.0, _NUM_TOKENS * _IN_FEATURES)
        .reshape(_NUM_TOKENS, _IN_FEATURES)
        .to(act_dtype)
    )
    _assert_parity(
        lambda: _Apply(_fp8_weight(_OUT_FEATURES, _IN_FEATURES), torch.tensor(_SCALE)),
        activation,
    )
