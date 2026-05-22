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

from vllm_rbln import envs
from vllm_rbln.model_executor.layers.fused_moe.layer import (
    patched_fused_moe_forward,
    patched_unquantized_fused_moe_method_optimized,
    patched_unquantized_fused_moe_method_torch,
)
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.apply",
    reason=(
        "Route unquantized MoE execution through the optimized RBLN MoE custom op, "
        "including expert_map, token mask, top-k/post-norm parameters. "
        "(PR#202, PR#511)"
    ),
    owner_module=__name__,
    condition=lambda: envs.VLLM_RBLN_MOE_USE_OPT_KERNEL
    and envs.VLLM_RBLN_MOE_CUSTOM_KERNEL,
)(patched_unquantized_fused_moe_method_optimized)

register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.apply",
    reason=(
        "Provide an RBLN-compatible PyTorch fallback MoE implementation and force "
        "routing/tok-k computation to fp32 to avoid bf16 top-k correctness issues. "
        "(PR#192, PR#252)"
    ),
    owner_module=__name__,
    condition=lambda: not envs.VLLM_RBLN_MOE_USE_OPT_KERNEL
    and not envs.VLLM_RBLN_MOE_CUSTOM_KERNEL,
)(patched_unquantized_fused_moe_method_torch)

register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_oot",
    reason=(
        "Override FusedMoE.forward_oot to support RBLN MoE DP execution: "
        "multicast hidden states across DP ranks, compute router logits after "
        "multicast, call the patched quant_method.apply path, and combine "
        "DP outputs. (PR#145)"
    ),
    owner_module=__name__,
)(patched_fused_moe_forward)
