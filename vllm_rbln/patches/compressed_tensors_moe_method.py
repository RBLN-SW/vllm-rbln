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

# NOTE(RBLN): Importing the RBLN quantization.fp8 module here also registers the
# `rbln_custom_ops::custom_moe_glu_group_dequantize` custom op as an import side
# effect, so pulling in this patch module is enough to make the op available.
from vllm_rbln.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8MoEMethod,
)
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a16_fp8.CompressedTensorsW8A16Fp8MoEMethod",
    reason="",
)(CompressedTensorsW8A16Fp8MoEMethod)
