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
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward",
    reason=(
        "Replace Qwen3MoeSparseMoeBlock.forward with an RBLN-friendly form. "
        "(1) Remove upstream's reshape operations. "
        "(2) Call `tensor_model_parallel_all_reduce` directly instead of "
        "`self.experts.maybe_all_reduce_tensor_model_parallel`."
    ),
)
def patched_qwen3_moe_forward(
    self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor
) -> torch.Tensor:
    assert hidden_states.dim() == 3  # [B, L, H]

    def router(h: torch.Tensor) -> torch.Tensor:
        return self.gate(h)[0]

    final_hidden_states = self.experts(hidden_states=hidden_states, router=router)
    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states + self.shared_expert(hidden_states)

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states
