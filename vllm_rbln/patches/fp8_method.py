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
from vllm_rbln.model_executor.layers.quantization.fp8 import (
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod",
    reason=(
        "Replace upstream Fp8LinearMethod with the RBLN block-fp8 linear method. "
        "RBLN has no scaled-mm kernel, so create_weights must not call "
        "choose_scaled_mm_linear_kernel (which rejects out_features not divisible "
        "by block_n, e.g. DeepSeek's fused_qkv_a_proj=2112); apply dequantizes "
        "fp8 weights to bf16 and runs a bf16 GEMM."
    ),
)(Fp8LinearMethod)

register_patch(
    target="vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod",
    reason=(
        "Replace upstream Fp8MoEMethod with the RBLN block-fp8 MoE method. "
        "It dequantizes fp8 weights to bf16 and runs the "
        "custom_moe_glu_group_dequantize op with pre-computed routing weights "
        "supplied by RBLNFusedMoE (topk/softmax done outside the op)."
    ),
)(Fp8MoEMethod)
