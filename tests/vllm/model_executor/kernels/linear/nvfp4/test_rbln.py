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

"""The RBLN NVFP4 linear reads the layer upstream's ModelOpt NVFP4 linear method
leaves (packed weight, linear-layout fp8 group scales, the global scale) as
upstream's own dequantization does. The op runs its CPU body here."""

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

from vllm_rbln.model_executor.kernels.linear.nvfp4.rbln import RBLNNvFp4LinearKernel

OUT_FEATURES, IN_FEATURES, GROUP = 96, 128, 16


def test_matches_upstream_dequantization():
    torch.manual_seed(0)
    layer = SimpleNamespace(
        weight=torch.randint(
            0, 256, (OUT_FEATURES, IN_FEATURES // 2), dtype=torch.uint8
        ),
        weight_scale=(torch.rand(OUT_FEATURES, IN_FEATURES // GROUP) * 4 + 0.25).to(
            torch.float8_e4m3fn
        ),
        weight_global_scale=torch.tensor(0.03, dtype=torch.float32),
    )
    x = torch.randn(2, 3, IN_FEATURES, dtype=torch.bfloat16)
    bias = torch.randn(OUT_FEATURES, dtype=torch.bfloat16)

    kernel = RBLNNvFp4LinearKernel.__new__(RBLNNvFp4LinearKernel)
    out = kernel.apply_weights(layer, x, bias)

    weight = dequantize_to_dtype(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        torch.float32,
        GROUP,
        swizzle=False,
    )
    expected = torch.nn.functional.linear(x.float(), weight, bias.float())
    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=2e-2)
