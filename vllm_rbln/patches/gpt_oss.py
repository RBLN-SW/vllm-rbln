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
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.models.gpt_oss import MLPBlock

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.gpt_oss.MLPBlock.forward",
    reason=(
        "Adapt GPT-OSS MLPBlock.forward to the RBLN FusedMoE interface: pass ",
        "the rounter callable instead of precomputed router logits so routing "
        "runs after RBLN DP multicast, then explicitly all-reduce TP outputs.",
    ),
)
def patched_gptoss_mlp_forward(
    self: MLPBlock, hidden_states: torch.Tensor
) -> torch.Tensor:
    final_hidden_states = self.experts(hidden_states=hidden_states, router=self.router)
    if get_tensor_model_parallel_world_size() > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states
