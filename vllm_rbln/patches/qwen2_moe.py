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
from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.qwen2_moe.Qwen2MoeSparseMoeBlock.forward",
    reason=(
        "To remove the unnecessary reshape and use the RBLN FusedMoE router-callable "
        "interface. (PR#367)"
    ),
)
def patched_qwen2_moe_forward(
    self: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor
) -> torch.Tensor:
    def router(h: torch.Tensor) -> torch.Tensor:
        return self.gate(h)[0]

    final_hidden_states = self.experts(hidden_states=hidden_states, router=router)
    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states + self.shared_expert(hidden_states)

    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states
