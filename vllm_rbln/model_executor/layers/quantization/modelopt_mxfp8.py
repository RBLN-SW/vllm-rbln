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
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

# MXFP8: fp8 (e4m3) weights with one e8m0 scale per 32 input elements.
MXFP8_GROUP_SIZE = 32
_E8M0_BIAS = 127


@register_weight_loader_v2_supported_method
class RBLNModelOptMxFp8LinearMethod(LinearMethodBase):
    """ModelOpt MXFP8 linear for RBLN.

    Upstream's MIXED_PRECISION config leaves an MXFP8 layer unquantized, which
    would load the e4m3 bytes into a bf16 parameter and drop the block scales.
    Here the layer loads the fp8 weight and its ``[N, K/32]`` e8m0 scale (the
    checkpoint's ``weight_scale_inv``, renamed by the model) and dequantizes
    them to one bf16 weight after loading.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if input_size_per_partition % MXFP8_GROUP_SIZE != 0:
            raise ValueError(
                f"MXFP8 needs the input size ({input_size_per_partition}) to be a "
                f"multiple of the group size {MXFP8_GROUP_SIZE}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_GROUP_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Dequantize to bf16 once, on the host: e8m0 -> 2^(e - 127) exactly (a
        # bf16 whose exponent field is e and mantissa zero IS that power of
        # two), times the fp8 weight per 32-wide group. The compiler's weight
        # layout pass takes the block-FP8 (128-granular) dequant pattern but
        # not a 32-wide group, so the layer runs as a plain bf16 linear.
        # TODO(perf): W8A16 for these linears needs a group-32 kernel path.
        weight = layer.weight.data
        device = weight.device
        out_features, in_features = weight.shape
        in_groups = in_features // MXFP8_GROUP_SIZE
        scale = (layer.weight_scale.data.to("cpu").to(torch.int16) << 7).view(
            torch.bfloat16
        )
        dequant = (
            weight.to("cpu").view(out_features, in_groups, MXFP8_GROUP_SIZE).to(torch.bfloat16)
            * scale[:, :, None]
        ).view(out_features, in_features)
        layer.weight = Parameter(dequant.to(device), requires_grad=False)
        # The scale is folded in; drop it so nothing (state dict, allocator)
        # keeps the uint8 copy around.
        del layer.weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)
