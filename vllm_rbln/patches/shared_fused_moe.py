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

from vllm_rbln.model_executor.layers.fused_moe.shared_fused_moe import (
    patched_shared_fused_moe_forward,
    patched_shared_fused_moe_init,
)
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.__init__",
    reason="To disable the upstream overlapped shared/fused MoE path on RBLN. (PR#293)",
    owner_module=__name__,
)(patched_shared_fused_moe_init)

register_patch(
    target="vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.forward",
    reason=(
        "Replace SharedFusedMoE's forward while perserving shared expert output "
        "handling and required TP all-reduce behavior. (PR#367)"
    ),
    owner_module=__name__,
)(patched_shared_fused_moe_forward)
