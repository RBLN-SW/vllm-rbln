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
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_rbln.custom_ops import custom_op, register_fake


@custom_op(
    "rbln_custom_ops::custom_moe_glu",
    mutates_args=(),
)
def custom_moe_glu(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    masked_routing_weight: torch.Tensor,
    hidden_act: str,
    expert_map: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Customized MoE GLU operation (optimized kernel version).

    Expected tensor shapes:
    - hidden_states: [batch * seq_len, hidden_size]
    - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
    - up_proj_weight: [num_experts, intermediate_size, hidden_size]
    - down_proj_weight: [num_experts, hidden_size, intermediate_size]
    - masked_routing_weight: [num_experts, batch * seq_len]
      (token dim may be padded to 64-align)
    - hidden_act: gate activation name ("silu"/"swish" or "gelu*")

    Returns:
        torch.Tensor: [batch * seq_len, hidden_size]
    """
    assert hidden_states.dtype == masked_routing_weight.dtype, (
        "hidden_states and masked_routing_weight must have the same dtype"
    )

    act = hidden_act.lower()
    if act in ("silu", "swish"):
        act_fn = torch.nn.functional.silu
    elif "gelu" in act:
        act_fn = torch.nn.functional.gelu
    else:
        raise ValueError(f"Unsupported hidden_act={hidden_act!r}")

    num_tokens = hidden_states.shape[0]
    out = torch.zeros_like(hidden_states)
    expert_cnt = gate_proj_weight.shape[0]
    # routing weight token dim may be padded to 64-align; slice to actual num_tokens
    routing_t = masked_routing_weight.transpose(0, 1)[:num_tokens, :]  # [num_tokens, E]
    for i in range(expert_cnt):
        gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
        up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
        mul = act_fn(gate) * up
        down = torch.nn.functional.linear(mul, down_proj_weight[i])
        out += down * routing_t[:, i : i + 1]
    return out


@register_fake("rbln_custom_ops::custom_moe_glu")
def custom_moe_glu_fake(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    masked_routing_weight: torch.Tensor,
    hidden_act: str,
    expert_map: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


class RBLNUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized MoE method for the RBLNMoERunner forward path.

    vLLM creates UnquantizedFusedMoEMethod directly when no quantization config
    is provided. Registering this OOT implementation preserves that selection
    path while routing execution through the RBLN MoE custom op.
    """

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # RBLNMoERunner computes router logits after RBLN DP multicast and then calls
        # quant_method.apply(layer, x, router_logits). Keep this router-logits interface
        # instead of upstream MoERunner's topk_weights/topk_ids interface.
        assert isinstance(w13 := layer.w13_weight, torch.Tensor)
        assert isinstance(w2 := layer.w2_weight, torch.Tensor)
        intermediate_size = w2.shape[-1]

        gate_proj_weight = w13[:, :intermediate_size, :]
        up_proj_weight = w13[:, intermediate_size:, :]
        down_proj_weight = w2

        orig_shape = x.shape
        num_tokens = orig_shape[:-1].numel()
        hidden_states = x.reshape(num_tokens, -1)
        masked_routing_weights = router_logits

        final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            masked_routing_weights,
            layer.activation.value,
            layer.expert_map,
            None,
            None,
            None,
        )
        return final_hidden_states.reshape(orig_shape)
