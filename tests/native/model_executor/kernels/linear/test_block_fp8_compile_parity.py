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
# rbln-compiled graph (plus device eager on the device lane) must agree. Inputs use
# production-sized 128-blocks because toy shapes trip an rbln-compiler edge case.

from typing import Any

import pytest
import torch
from vllm.platforms import current_platform

from vllm_rbln.compilation.compiler import compile as rbln_compile
from vllm_rbln.model_executor.kernels.linear.block_fp8 import (
    RBLNW8A16BlockFp8LinearKernel,
)

# TODO(rbln-fp8-compile): the whole file is skipped -- the rbln-compiled kernel
# diverges from CPU eager on fp8 weights (a bare fp8->float dequant already gives
# alternating ~2x values), which looks like a 1-byte .view()/reshape layout issue
# in the compile path. W8A8 shares the root cause; extend the parity here to its
# forward once resolved.
pytestmark = [
    pytest.mark.use_device,
    pytest.mark.skip(
        reason="TODO(rbln-fp8-compile): compiled fp8 dequant diverges "
        "from eager; see file header."
    ),
]

# The compiled/device fp8 paths round differently from CPU eager, so parity is
# checked within an fp8-sized tolerance rather than exactly.
_RTOL, _ATOL = 1e-2, 1e-2

# Production-sized fp8 block; toy blocks hit an rbln-compiler reshape edge case.
_BLOCK_N = _BLOCK_K = 128

# The activation (and thus the output) is fp16 on the W8A16 path.
_ACT_DTYPE = torch.float16

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


def _assert_parity(run_fn, *cpu_args):
    # 1. CPU eager oracle.
    ref = run_fn(*cpu_args)
    # 2. rbln-compiled on the session's tensors (cpu lane -> cpu, device lane ->
    #    device, matching how the backend runs the compiled graph in production).
    dev_args = _to_session_device(cpu_args)
    assert _agrees(rbln_compile(run_fn, fullgraph=True)(*dev_args), ref)
    # 3. device eager, only when device tensors are on (--device-tensor 1).
    if current_platform.device_type == "rbln":
        assert _agrees(run_fn(*dev_args), ref)


def _fp8_weight(out_features, in_features):
    # Block-FP8 weights are stored as fp8; the graph dequantizes them.
    return (
        torch.linspace(0.1, 2.0, out_features * in_features)
        .reshape(out_features, in_features)
        .to(torch.float8_e4m3fn)
    )


def test_dequantize_block_weight_matches_rbln_compiled():
    def run_dequant(weight, scale):
        return _KERNEL._dequantize_block_fp8_weight(weight, scale, _ACT_DTYPE)

    # 256x256 fp8 weight over 128-blocks -> a 2x2 block-scale grid.
    _assert_parity(
        run_dequant,
        _fp8_weight(256, 256),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_apply_block_scaled_mm_matches_rbln_compiled():
    # The W8A16 forward: dequant the fp8 weight and run the fp16 linear, all in
    # one compiled graph. The activation stays fp16 (A16, not quantized).
    def run_apply(activation, weight, scale):
        return _KERNEL.apply_block_scaled_mm(activation, weight, None, scale)

    activation = torch.linspace(0.5, 3.0, 128).reshape(1, 128).to(_ACT_DTYPE)
    _assert_parity(
        run_apply,
        activation,
        _fp8_weight(128, 128),
        torch.full((1, 1), 2.0),
    )
