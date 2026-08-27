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

# Compile parity for the unpacked WNA16 kernel: CPU eager is the oracle, and the
# rbln-compiled graph (plus device eager on the device lane) must agree. The
# kernel dequantizes an int8 weight and runs a linear; rebel_compiler is expected
# to recognise that pattern and map it to its quantized kernel, so a divergence
# here means the mapping changed the arithmetic.
#
# The int8 weight is a module buffer, never a graph argument: a 1-byte dtype
# cannot be a graph input at all. It is also what a real model holds.
#
# Channelwise is what RedHatAI/Qwen2.5-0.5B-quantized.w8a16 resolves to; grouped
# is the other branch through apply_weights, and a 64 and a 128 group reach it
# through different scale handling upstream.

import pytest
import torch
from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from vllm_rbln.compilation.compiler import compile as rbln_compile
from vllm_rbln.model_executor.kernels.linear.mixed_precision.unpacked_wna16 import (
    RBLNUnpackedwNa16LinearKernel,
)

pytestmark = pytest.mark.use_device

_RTOL, _ATOL = 1e-2, 1e-2

# Non-square, so a transposed view would not silently have the right numel.
_OUT_FEATURES, _IN_FEATURES = 128, 256
_NUM_TOKENS = 8

_ACT_DTYPES = [torch.float16, torch.bfloat16]
_ACT_DTYPE_IDS = ["fp16", "bf16"]

# The graph carries int8 for both, and only RBLN_QUANT_BITS says which.
_WEIGHT_TYPES = [scalar_types.uint8b128, scalar_types.uint4b8]
_WEIGHT_TYPE_IDS = ["int8", "int4"]

_GROUP_SIZES = [-1, 64, 128]
_GROUP_IDS = ["channelwise", "group64", "group128"]

# apply_weights always reshapes a grouped weight to 64-wide groups; a 128 group
# arrives as two of them, doubled by process_weights_after_loading.
_SCALE_GROUP = 64


# Each of the three on its own is fine, and the channelwise branch takes all
# three.
_GROUPED_BIAS_XFAIL = pytest.mark.xfail(
    reason="TODO(rbln-wna16-grouped-bias): grouped WNA16 with a bias does not "
    "compile for more than one token.",
)


@pytest.fixture(autouse=True)
def _quant_bits(request, monkeypatch):
    # Production sets this in process_weights_after_loading, which these tests
    # start downstream of.
    weight_type = request.node.callspec.params["weight_type"]
    monkeypatch.setenv("RBLN_QUANT_BITS", str(weight_type.mantissa))


def _agrees(actual, reference) -> bool:
    return torch.allclose(
        actual.cpu().float(), reference.float(), rtol=_RTOL, atol=_ATOL
    )


def _assert_parity(build, activation):
    # Each step gets its own module: a weight-free compile relays the weight
    # buffer into the hardware layout in place -- that buffer is the weight pool,
    # not a copy -- so a compiled module no longer reads back as a plain tensor.
    dev = current_platform.device_type
    # 1. CPU eager oracle.
    ref = build()(activation)
    # 2. rbln-compiled, on whichever tensors the lane selects.
    dev_activation = activation.to(dev)
    assert _agrees(rbln_compile(build().to(dev), fullgraph=True)(dev_activation), ref)
    # 3. device eager, only when device tensors are on (--device-tensor 1).
    if dev == "rbln":
        assert _agrees(build().to(dev)(dev_activation), ref)


def _quantized_weight():
    # -8..7 repeating: the whole int4 range, exact in every dtype here, so a byte
    # read out of place is a different number rather than a rounding difference.
    marker = (torch.arange(_IN_FEATURES) % 16) - 8
    return marker.unsqueeze(0).expand(_OUT_FEATURES, _IN_FEATURES).to(torch.int8)


def _scale(group_size, act_dtype):
    columns = 1 if group_size < 0 else _IN_FEATURES // _SCALE_GROUP
    return (
        ((torch.arange(_OUT_FEATURES * columns) % 4) + 1)
        .reshape(_OUT_FEATURES, columns)
        .to(act_dtype)
    )


class _Apply(torch.nn.Module):
    def __init__(self, group_size, act_dtype, weight_type, bias=None):
        super().__init__()
        self.kernel = RBLNUnpackedwNa16LinearKernel(
            MPLinearLayerConfig(
                full_weight_shape=(_IN_FEATURES, _OUT_FEATURES),
                partition_weight_shape=(_IN_FEATURES, _OUT_FEATURES),
                weight_type=weight_type,
                act_type=act_dtype,
                group_size=group_size,
                zero_points=False,
                has_g_idx=False,
            ),
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
        )
        self.register_buffer("weight_packed", _quantized_weight())
        self.register_buffer("weight_scale", _scale(group_size, act_dtype))
        self.register_buffer("bias", bias)

    def forward(self, activation):
        return self.kernel.apply_weights(self, activation, self.bias)


@pytest.mark.parametrize("weight_type", _WEIGHT_TYPES, ids=_WEIGHT_TYPE_IDS)
@pytest.mark.parametrize("act_dtype", _ACT_DTYPES, ids=_ACT_DTYPE_IDS)
@pytest.mark.parametrize("group_size", _GROUP_SIZES, ids=_GROUP_IDS)
def test_dequantized_weight_matches_rbln_compiled(group_size, act_dtype, weight_type):
    # The identity readout makes every output element one dequantized weight
    # element, so nothing accumulates over a misplaced byte.
    _assert_parity(
        lambda: _Apply(group_size, act_dtype, weight_type),
        torch.eye(_IN_FEATURES, dtype=act_dtype),
    )


@pytest.mark.parametrize("weight_type", _WEIGHT_TYPES, ids=_WEIGHT_TYPE_IDS)
@pytest.mark.parametrize("act_dtype", _ACT_DTYPES, ids=_ACT_DTYPE_IDS)
@pytest.mark.parametrize(
    "group_size",
    [
        -1,
        pytest.param(64, marks=_GROUPED_BIAS_XFAIL),
        pytest.param(128, marks=_GROUPED_BIAS_XFAIL),
    ],
    ids=_GROUP_IDS,
)
def test_apply_weights_matches_rbln_compiled(group_size, act_dtype, weight_type):
    # Many tokens and a bias, the way a real decoder layer calls this.
    activation = (
        torch.linspace(0.5, 3.0, _NUM_TOKENS * _IN_FEATURES)
        .reshape(_NUM_TOKENS, _IN_FEATURES)
        .to(act_dtype)
    )
    bias = (torch.arange(_OUT_FEATURES) / 8).to(act_dtype)
    _assert_parity(lambda: _Apply(group_size, act_dtype, weight_type, bias), activation)
