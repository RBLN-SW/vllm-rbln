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

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
)

from vllm_rbln.patches.mla import patched_get_and_maybe_dequant_weights


def test_mla_dequantises_compressed_tensors_block_fp8_without_running_apply():
    out_f, in_f = 256, 256
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(out_f, in_f, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.rand(out_f // 128, in_f // 128, generator=gen) + 0.5

    method = CompressedTensorsLinearMethod.__new__(CompressedTensorsLinearMethod)
    method.apply = None  # the generic path would call it with an identity
    layer = torch.nn.Module()
    layer.quant_method = method
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.weight_block_size = [128, 128]

    got = patched_get_and_maybe_dequant_weights(layer, torch.bfloat16)

    expected = weight.to(torch.float32) * scale.repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected.to(torch.bfloat16))
