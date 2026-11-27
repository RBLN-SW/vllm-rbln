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
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)

# group_quant_linear's name for the packed e2m1 weight: uint8 alone cannot tell
# it from INT4.
_PACKED_FP4_WEIGHT_DTYPE = "float4_e2m1fn"


class RBLNNvFp4LinearKernel(NvFp4LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        in_features = layer.weight.shape[1] * 2
        return torch.ops.rbln_custom_ops.group_quant_linear(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            bias,
            group_size=in_features // layer.weight_scale.shape[1],
            weight_dtype=_PACKED_FP4_WEIGHT_DTYPE,
        )
