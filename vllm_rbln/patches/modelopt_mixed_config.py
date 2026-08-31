# Copyright 2025 Rebellions Inc. All rights reserved.
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
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)

from vllm_rbln.model_executor.layers.quantization.modelopt_fp8 import (
    RBLNModelOptFp8LinearMethod,
)
from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNModelOptNvFp4FusedMoE,
)
from vllm_rbln.patches import add_registration


class RBLNModelOptMixedPrecisionConfig(ModelOptMixedPrecisionConfig):
    """ModelOpt MIXED_PRECISION config for RBLN."""

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            quant_algo = self._resolve_quant_algo(prefix)
            if quant_algo == "NVFP4":
                return RBLNModelOptNvFp4FusedMoE(self.nvfp4_config, layer.moe_config)
            if quant_algo == "W4A16_NVFP4":
                return RBLNModelOptNvFp4FusedMoE(
                    self.w4a16_nvfp4_config, layer.moe_config
                )
        elif isinstance(layer, LinearBase) and not self.is_layer_excluded(prefix):
            if self._resolve_quant_algo(prefix) == "FP8":
                return RBLNModelOptFp8LinearMethod(self.fp8_config)
        return super().get_quant_method(layer, prefix)


@add_registration(
    reason=(
        "Override the built-in ModelOpt MIXED_PRECISION (modelopt_mixed) config "
        "so NVFP4 routed-expert MoE layers dequantise through the RBLN "
        "group-dequantise custom op instead of the upstream CUDA/Marlin kernels."
    )
)
def register_rbln_modelopt_mixed_config() -> None:
    register_quantization_config("modelopt_mixed")(RBLNModelOptMixedPrecisionConfig)
