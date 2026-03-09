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
import torch.nn.functional as F
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# Define custom_moe_glu op based on environment variable
# VLLM_RBLN_MOE_USE_OPT_KERNEL: uses topk, post_norm, expert_map parameters
# VLLM_RBLN_MOE_CUSTOM_KERNEL: uses expert_select_count parameter
if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:

    @torch.library.custom_op(
        "rbln_custom_ops::custom_moe_glu",
        mutates_args=(),
    )
    def custom_moe_glu(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        topk: int,
        post_norm: bool,
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
        - masked_routing_weight: [batch * seq_len, num_experts]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight[:, i : i + 1]
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        topk: int,
        post_norm: bool,
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

else:

    @torch.library.custom_op(
        "rbln_custom_ops::custom_moe_glu",
        mutates_args=(),
    )
    def custom_moe_glu(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_select_count: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Customized MoE GLU operation (custom kernel version).

        Expected tensor shapes:
        - hidden_states: [batch * seq_len, hidden_size]
        - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
        - up_proj_weight: [num_experts, intermediate_size, hidden_size]
        - down_proj_weight: [num_experts, hidden_size, intermediate_size]
        - masked_routing_weight: [batch * seq_len, num_experts]
        - expert_select_count: [num_experts]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight[:, i : i + 1]
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_select_count: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


def unquantized_fused_moe_method_rbln(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
):
    # selected_experts
    w1 = layer.w13_weight
    w2 = layer.w2_weight

    orig_shape = x.shape  # noqa: F841
    hidden_size = x.shape[-1]
    num_tokens = x.shape[:-1].numel()  # noqa: F841
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = x.dtype
    top_k = layer.top_k

    hidden_states = x
    gating_output = router_logits
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights = topk_weights.to(torch.float)
    topk_weights, selected_experts = topk_weights.topk(top_k, dim=-1)
    if layer.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if layer.expert_map is not None:
        selected_experts = layer.expert_map[selected_experts]

    final_hidden_states = None

    # 1. build expert_mask & expert_weights
    # 2. FFN
    # topk_weights, expert_weights, expert_mask.shape = [b, seq, top_k]
    # NOTE - convert for loop scalar operation into tensor compare

    # [1,num_tokens,hidden_size]
    hidden_states = hidden_states.reshape(1, num_tokens, -1)
    # [num_experts,1,1,1]
    expert_idx_array = torch.arange(0, num_experts).reshape(num_experts, 1, 1, 1)
    # [1,1,num_tokens,topk]
    selected_experts_array = selected_experts.reshape(-1, 1, num_tokens, top_k)
    # [num_experts,1,num_tokens,topk]
    expert_mask_array = selected_experts_array == expert_idx_array
    # [num_experts,1,num_tokens,topk]
    topk_weights_array = topk_weights.reshape(-1, 1, num_tokens, top_k)
    # [num_experts,1,num_tokens,1]
    expert_weights_array = (topk_weights_array * expert_mask_array).sum(
        dim=-1, keepdim=True
    )
    # [1,num_tokens,1]
    temp_expert_weights = expert_weights_array[0]
    # NOTE - make explicit dependence between hidden_states and expert_weights
    # [1,num_tokens,hidden_size]
    # [1,num_tokens,1] <- broadcast add
    hidden_states = hidden_states + temp_expert_weights - temp_expert_weights
    # [num_experts,1,num_tokens,1] -> [num_experts,1,num_tokens,hidden_size]
    hidden_states = hidden_states.to(dtype)
    expert_weights_array = expert_weights_array.broadcast_to(
        (num_experts, 1, num_tokens, hidden_size)
    ).to(dtype)
    # solution1. make custom operation for expert loop
    # solution2. add dummy use of expert_weights_array
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_weights = expert_weights_array[expert_idx]
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[..., :intermediate_size])
        x = x[..., intermediate_size:] * gate
        x = F.linear(x, expert_w2)

        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    assert final_hidden_states is not None
    return final_hidden_states.reshape(orig_shape)


def get_tokens_mask(num_tokens: int, left=0.0, right=float("-inf")):
    num_tokens_across_dp = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
    num_tokens_across_dp = num_tokens_across_dp.unsqueeze(1)
    if num_tokens_across_dp.size(0) == 1:
        max_pad = num_tokens
    else:
        max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
    pos = torch.arange(max_pad, dtype=torch.int32).unsqueeze(0)  # [1, max_pad]
    tokens_mask = torch.where(
        pos < num_tokens_across_dp, left, right
    )  # [dp_size, max_pad]
    tokens_mask = tokens_mask.reshape(-1, 1)  # [dp_size * max_pad, 1]
    return tokens_mask


# based on custom fused moe expert kernel
def get_masked_routing_weights(router_logits, top_k, renormalize, expert_map):
    # routing_weights: (batch * sequence_length, n_experts)
    # selected_experts: (batch * sequence_length, top_k)
    if renormalize:
        router_logits = router_logits.to(torch.float)
        selected_weights, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
        selected_weights = torch.nn.functional.softmax(selected_weights, dim=1)
    else:
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1)
        routing_weights = routing_weights.to(torch.float)
        selected_weights, selected_experts = torch.topk(
            routing_weights, k=top_k, dim=-1
        )

    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(router_logits.shape[0], 1.0, 0.0)
        selected_weights = selected_weights * tokens_mask

    n_expert = router_logits.shape[1]
    if expert_map is not None:
        expert_map_within_bounds = torch.where(
            expert_map < 0, n_expert - 1, expert_map
        ).to(torch.int64)
        selected_experts = expert_map_within_bounds[selected_experts]

    # masked_routing_weights=selected_weights w/ non selected indicies zeros
    # selected_weights      = [..., top_k]
    # masked_routing_weights= [..., n_experts], selected_experts has only value
    masked_routing_weights = torch.zeros_like(router_logits, dtype=torch.float32)
    masked_routing_weights.scatter_(1, selected_experts, selected_weights)

    ## count selected tokens for each expert index from selected_experts
    zeros = torch.zeros(n_expert, dtype=torch.int32)

    if use_moe_tokens_mask:
        ones = torch.ones_like(selected_experts, dtype=torch.int32)
        tokens_mask = tokens_mask.to(torch.int32)
        ones = ones * tokens_mask
        ones = ones.view(-1)
    else:
        ones = torch.ones_like(selected_experts.view(-1), dtype=torch.int32)

    expert_select_count = torch.scatter_add(
        zeros, dim=0, index=selected_experts.view(-1), src=ones
    )

    return masked_routing_weights, expert_select_count


def unquantized_fused_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
):
    # selected_experts
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    # w13_weight- merged weight for gate_proj(w1_weight) and up_proj (w3_weight)
    # w2_weight - down_proj
    # gate_proj_weight - first half, layer.w13_weight[:intermediate_size]
    # up_proj_weight - second half, layer.w13_weight[intermediate_size:]
    # down_proj_weights = layer.w2_weight
    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    masked_routing_weights, expert_select_count = get_masked_routing_weights(
        router_logits, layer.top_k, layer.renormalize, layer.expert_map
    )
    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights,
        expert_select_count,
    )
    return final_hidden_states.reshape(orig_shape)


def unquantized_fused_optimize_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
):
    # selected_experts
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    # w13_weight- merged weight for gate_proj(w1_weight) and up_proj (w3_weight)
    # w2_weight - down_proj
    # gate_proj_weight - first half, layer.w13_weight[:intermediate_size]
    # up_proj_weight - second half, layer.w13_weight[intermediate_size:]
    # down_proj_weights = layer.w2_weight
    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    expert_map_const = None
    if layer.expert_map is not None:
        # Extract numpy array and create a fresh constant tensor
        expert_map_list = layer.expert_map.tolist()
        expert_map_const = torch.tensor(expert_map_list, dtype=torch.int32)

    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(num_tokens)
        router_logits = router_logits + tokens_mask

    # optimum-rbln/src/optimum/rbln/transformers/models/qwen3_moe/
    # qwen3_moe_architecture.py
    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        router_logits,
        layer.top_k,
        layer.renormalize,
        expert_map_const,
    )
    return final_hidden_states.reshape(orig_shape)


def fused_moe_forward_rbln(
    self: FusedMoE, hidden_states: torch.Tensor, router: torch.nn.Module
) -> torch.Tensor:
    assert self.quant_method is not None

    if self.dp_size > 1:
        org_hidden_shape = hidden_states.shape

        # input broadcast - all DPs broadcast hidden_states & router_logits
        # example) DP2, TP/EP2
        # dp_group = {{0, 2}, {1, 3}}
        # tp_group = {{0, 1}, {2, 3}}
        # 1. initially, each DP hidden_states = [1, 128, 1024]
        # 2. after multicast, all DPs hidden_states = [dp_size, 128, 1024]
        # - all DP ranks broadcast inputs to process group
        # 3. DP x TP/EP expert parallel
        # ex) 0, 1, 2, 3 has its own hidden_states = [dp_size, 128, 1024]
        # 4. dp_group all reduce - {0+2}, {1+3}, {0+2}, {1+3}
        # 5. select each DP rank output
        # 6. to_group all reduce - {0+2+1+3}, {0+2+1+3}, {0+2+1+3}, {0+2+1+3}
        hidden_states = self.naive_multicast(hidden_states)

    router_logits = router(hidden_states)

    # Matrix multiply.
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
    )

    if self.dp_size > 1:
        # output all_reduce == dp all_reduce + tp all_reduce
        all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
        hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
        final_hidden_states = all_hidden_states.reshape(hidden_shape_dp)

        max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
        num_tokens = org_hidden_shape[:-1].numel()  # noqa: F841
        start = self.dp_rank * max_pad
        end = start + num_tokens
        final_hidden_states = final_hidden_states[start:end]

        final_hidden_states = final_hidden_states.reshape(org_hidden_shape)

    return final_hidden_states


def fused_moe_naive_multicast_rbln(self: FusedMoE, x: torch.Tensor):
    # as-is : [num_tokens, hidden_size]
    # to-be : buffer = [data_parallel_size*batch, seq, hidden_size], broadcast
    #         hidden = [batch, seq, hidden_size]
    # x.shape = [1, seq, hidden_size]
    # assert len(x.shape) == 3

    x = x.reshape(1, -1, x.size(-1))
    max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
    num_tokens = x.size(1)
    num_repeat = max_pad // num_tokens
    # TODO: evaluate various padding approaches
    x = x.repeat(num_repeat, 1, 1)
    x = x.reshape(1, max_pad, -1)

    if not envs.VLLM_RBLN_DP_INPUT_ALL_GATHER:
        # each DP rank gather all inputs via torch.distributed.all_reduce
        # broadcast(value) == all_reduce(value for me or zeros for others)
        all_buffer = None
        zeros = x - x
        for rank in range(get_dp_group().world_size):
            rank_tensor = x if rank == self.dp_rank else zeros
            all_buffer = (
                torch.cat((all_buffer, rank_tensor), dim=0)
                if all_buffer is not None
                else rank_tensor
            )
        output = get_dp_group().all_reduce(all_buffer)
        return output
    else:
        # gather all inputs via torch.distributed.all_gather
        all_gather_buffer = get_dp_group().all_gather(x, dim=0)
        return all_gather_buffer


FusedMoE.forward_oot = fused_moe_forward_rbln

if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
    logger.info("[RBLN] fused moe, RBLN optimize moe custom kernel")
    UnquantizedFusedMoEMethod.forward_oot = unquantized_fused_optimize_moe_method_custom
elif envs.VLLM_RBLN_MOE_CUSTOM_KERNEL:
    logger.info("[RBLN] fused moe, RBLN moe custom kernel")
    UnquantizedFusedMoEMethod.forward_oot = unquantized_fused_moe_method_custom
else:
    logger.info("[RBLN] fused moe, pytorch native kernel")
    UnquantizedFusedMoEMethod.forward_oot = unquantized_fused_moe_method_rbln
FusedMoE.naive_multicast = fused_moe_naive_multicast_rbln
