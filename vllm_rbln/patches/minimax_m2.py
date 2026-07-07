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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.minimax_m2 import MiniMaxM2MoE

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2MoE.forward",
    reason=(
        "Replace MiniMaxM2MoE.forward with an RBLN-friendly form. "
        "(1) Keep the 3D [B, L, H] hidden_states instead of unpacking a 2D "
        "shape (upstream's `num_tokens, hidden_dim = hidden_states.shape` "
        "raises on RBLN's 3D layout). "
        "(2) Pass `self.gate` as a router callable instead of precomputed "
        "router_logits, so routing runs after RBLN DP multicast. "
        "(3) Explicitly all-reduce TP outputs (upstream has no explicit "
        "all-reduce in this forward)."
    ),
)
def patched_minimax_m2_moe_forward(
    self: MiniMaxM2MoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    # router_logits: (num_tokens, n_experts)
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x.to(torch.float32))[0]
    )
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states
