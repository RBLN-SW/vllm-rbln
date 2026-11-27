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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.afmoe import AfmoeMoE

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.afmoe.AfmoeMoE.forward",
    reason=(
        "Upstream unpacks a 2-D hidden shape and leaves route_scale (applied in "
        "its router), the shared expert and the TP all-reduce to the upstream "
        "MoERunner. RBLNMoERunner keeps [B, L, H], takes a router callable and "
        "does none of the three, so this forward does them."
    ),
)
def patched_afmoe_moe_forward(
    self: AfmoeMoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0]
    )
    final_hidden_states = final_hidden_states * self.route_scale
    if self.shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states.to(hidden_states.dtype)
