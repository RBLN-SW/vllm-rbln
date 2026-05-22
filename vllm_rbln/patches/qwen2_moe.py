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

from vllm_rbln.models.qwen2_moe import patched_qwen2_moe_forward
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.model_executor.models.qwen2_moe.Qwen2MoeSparseMoeBlock.forward",
    reason=(
        "To remove the unnecessary reshape and use the RBLN FusedMoE router-callable "
        "interface. (PR#367)"
    ),
    owner_module=__name__,
)(patched_qwen2_moe_forward)
