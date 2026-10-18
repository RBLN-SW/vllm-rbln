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

from vllm_rbln.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8MoEMethod,
)
from vllm_rbln.patches import add_registration


class RBLNCompressedTensorsConfig(CompressedTensorsConfig):
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            self._add_fused_moe_to_target_scheme_map()
            schemes = [
                self.get_scheme_dict(layer, prefix + proj_name)
                for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
            ]
            scheme = schemes[0]
            if scheme is not None and all(s == scheme for s in schemes):
                weight_quant = scheme.get("weights")
                input_quant = scheme.get("input_activations")
                if (
                    weight_quant is not None
                    and weight_quant.num_bits == 8
                    and input_quant is None
                    and self._is_fp8_w8a16(weight_quant, input_quant)
                ):
                    return CompressedTensorsW8A16Fp8MoEMethod(
                        weight_quant, layer.moe_config
                    )
        return super().get_quant_method(layer, prefix)


@add_registration(
    reason=(
        "vLLM 0.31.0 has no compressed-tensors W8A16 FP8 MoE method. "
        "Register the RBLN config to select its weight-only FP8 MoE method "
        "while keeping upstream selection for other schemes and layers."
    ),
)
def register_rbln_compressed_tensors_config() -> None:
    register_quantization_config("compressed-tensors")(RBLNCompressedTensorsConfig)
