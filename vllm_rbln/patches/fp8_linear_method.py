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
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

from vllm_rbln.model_executor.layers.quantization.activation_override import (
    activation_is_fp8,
    current_quantization_config,
    linear_w8a8_scope,
)
from vllm_rbln.patches import register_patch

_original_create_weights = Fp8LinearMethod.create_weights


@register_patch(
    target="vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.create_weights",
    reason=(
        "Pick the RBLN W8A8 or W8A16 block-fp8 kernel per layer from the "
        "--quantization-config activation override; upstream chooses the "
        "kernel without the layer name."
    ),
)
def rbln_fp8_linear_create_weights(
    self: Fp8LinearMethod, layer: torch.nn.Module, *args, **kwargs
) -> None:
    w8a8 = activation_is_fp8(
        current_quantization_config(),
        layer.prefix,
        "linear",
        self.quant_config.packed_modules_mapping,
    )
    with linear_w8a8_scope(w8a8):
        _original_create_weights(self, layer, *args, **kwargs)
