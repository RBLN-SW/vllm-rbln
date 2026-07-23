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
from vllm.model_executor.models.deepseek_mtp import DeepSeekMultiTokenPredictorLayer

from vllm_rbln.patches import register_patch
from vllm_rbln.platform import USE_DEVICE_TENSOR


@register_patch(
    target=(
        "vllm.model_executor.models.deepseek_mtp."
        "DeepSeekMultiTokenPredictorLayer.forward"
    ),
    reason=(
        "Drop the position-0 embedding mask in the MTP drafter: "
        "the RBLN compiler rejects its position-conditional select with "
        "[UNEXPECTED_GRAPH] under device tensor."
    ),
    condition=lambda: USE_DEVICE_TENSOR,
)
def patched_deepseek_mtp_layer_forward(
    self: DeepSeekMultiTokenPredictorLayer,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_index: int = 0,
) -> torch.Tensor:
    assert inputs_embeds is not None
    # NOTE(RBLN): upstream masks inputs at position 0 here:
    #   inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    # Omitted on purpose -- see the patch `reason` above.
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = self.hnorm(previous_hidden_states)

    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
    )

    hidden_states, residual = self.mtp_block(
        positions=positions, hidden_states=hidden_states, residual=None
    )
    hidden_states = residual + hidden_states
    return hidden_states
