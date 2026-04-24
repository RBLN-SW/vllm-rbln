# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock


def __qwen3_moe_forward_rsd(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # NOTE: hidden_states can have either 1D or 2D shape.
    # router_logits: (num_tokens, n_experts)
    shared_out, fused_out = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0]
    )
    final_hidden_states = (
        shared_out + fused_out if shared_out is not None else fused_out
    )

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states


Qwen3MoeSparseMoeBlock.forward = __qwen3_moe_forward_rsd
