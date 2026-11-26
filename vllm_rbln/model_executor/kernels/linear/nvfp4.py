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
import torch.nn.functional as F
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from torch.nn.parameter import Parameter
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)


class RBLNNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 linear as a dense matmul over the weight dequantised at load.

    There is no NVFP4 linear op on RBLN. Activations stay unquantized (W4A16)
    whatever the checkpoint's input scheme, as for the RBLN NVFP4 MoE.
    """

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        # Built inside the model's set_default_torch_dtype scope.
        self.dtype = torch.get_default_dtype()

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        out_features, packed_in = layer.weight.shape
        in_features = packed_in * 2
        weight = unpack_fp4_from_uint8(
            layer.weight.data, out_features, in_features, dtype=torch.float32
        )
        group_size = in_features // layer.weight_scale.shape[1]
        scale = layer.weight_scale.data.float().repeat_interleave(group_size, dim=1)
        # The scheme has already turned the global scale into a multiplier.
        weight = weight * scale * layer.weight_global_scale.data.float()
        layer.weight = Parameter(weight.to(self.dtype), requires_grad=False)
        del layer.weight_scale
        del layer.weight_global_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
