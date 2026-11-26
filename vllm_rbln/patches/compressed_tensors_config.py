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
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod,
)

from vllm_rbln.model_executor.layers.quantization.fp8 import (
    RBLNCompressedTensorsW8A8Fp8MoEMethod,
)
from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNCompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm_rbln.patches import add_registration


class RBLNCompressedTensorsConfig(CompressedTensorsConfig):
    """compressed-tensors config for RBLN."""

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            self._add_fused_moe_to_target_scheme_map()
            scheme = self.get_scheme_dict(layer, f"{prefix}.0.gate_proj")
            if scheme is not None and self._is_nvfp4_format(scheme.get("weights")):
                input_quant = scheme.get("input_activations")
                if input_quant is None or self._is_nvfp4_format(input_quant):
                    return RBLNCompressedTensorsW4A4Nvfp4MoEMethod(
                        layer.moe_config, use_a16=input_quant is None
                    )
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, CompressedTensorsW8A8Fp8MoEMethod) and method.block_quant:
            return RBLNCompressedTensorsW8A8Fp8MoEMethod(
                method.weight_quant, method.input_quant, layer.moe_config
            )
        return method


@add_registration(
    reason=(
        "Upstream's compressed-tensors fp8 and NVFP4 MoE methods pick among "
        "CUDA/ROCm kernels and have no RBLN backend, so block-fp8 and NVFP4 experts "
        "fail at load. Override the built-in compressed-tensors config so they run "
        "through the RBLN custom_moe op; every other scheme keeps upstream's method."
    )
)
def register_rbln_compressed_tensors_config() -> None:
    register_quantization_config("compressed-tensors")(RBLNCompressedTensorsConfig)
