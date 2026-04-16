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
from vllm_rbln.model_executor.layers.fused_moe.all2all import (  # noqa: F401 — registers custom ops on import
    CCL_ALL2ALL_GROUP_ID,
    ccl_combine_receive,  # noqa: F811
    ccl_combine_send,  # noqa: F811
    prepare_send_mask_matrix,
)

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
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Customized MoE GLU operation (optimized kernel version).

        Expected tensor shapes:
        - hidden_states: [batch * seq_len, hidden_size]
        - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
        - up_proj_weight: [num_experts, intermediate_size, hidden_size]
        - down_proj_weight: [num_experts, hidden_size, intermediate_size]
        - masked_routing_weight: [num_experts, batch * seq_len] (token dim may be padded to 64-align)

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        assert hidden_states.dtype == masked_routing_weight.dtype, "hidden_states and masked_routing_weight must have the same dtype"

        num_tokens = hidden_states.shape[0]
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        # routing weight token dim may be padded to 64-align; slice to actual num_tokens
        routing_t = masked_routing_weight.transpose(0, 1)[:num_tokens, :]  # [num_tokens, E]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * routing_t[:, i : i + 1]
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
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
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Customized MoE GLU operation (custom kernel version).

        Expected tensor shapes:
        - hidden_states: [batch * seq_len, hidden_size]
        - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
        - up_proj_weight: [num_experts, intermediate_size, hidden_size]
        - down_proj_weight: [num_experts, hidden_size, intermediate_size]
        - masked_routing_weight: [num_experts, batch * seq_len] (token dim may be padded to 64-align)
        - expert_select_count: [num_experts]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        assert hidden_states.dtype == masked_routing_weight.dtype, "hidden_states and masked_routing_weight must have the same dtype"

        num_tokens = hidden_states.shape[0]
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        # routing weight token dim may be padded to 64-align; slice to actual num_tokens
        routing_t = masked_routing_weight.transpose(0, 1)[:num_tokens, :]  # [num_tokens, E]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * routing_t[:, i : i + 1]
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
        dp_mask: torch.Tensor | None = None,
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


def get_tokens_mask(num_tokens: int, left=1.0, right=0.0):
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


def get_tokens_mask_DP(num_tokens: int, dp_rank: int, left=1.0, right=0.0):
    """Get token mask for a single DP rank. Returns [t, 1]."""
    num_tokens_across_dp = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
    # num_tokens_across_dp: [dp_size]
    my_num_tokens = num_tokens_across_dp[dp_rank]  # scalar tensor
    pos = torch.arange(num_tokens, dtype=torch.int32)  # [t]
    tokens_mask = torch.where(
        pos < my_num_tokens, left, right
    )  # [t]
    tokens_mask = tokens_mask.reshape(-1, 1)  # [t, 1]
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
    masked_routing_weights: torch.Tensor,
):
    # router_logits is now pre-computed masked_routing_weights
    # (topk + softmax already done in fused_moe_forward_rbln)
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    masked_routing_weights = masked_routing_weights.reshape(num_tokens, -1)

    # transpose to [num_experts, num_tokens] for custom_moe_glu
    masked_routing_weights_t = masked_routing_weights.transpose(0, 1)

    # compute expert_select_count from masked_routing_weights
    expert_select_count = (masked_routing_weights_t > 0).sum(dim=1).to(torch.int32)

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights_t,
        expert_select_count,
        None,
        None,
        None,
        None,
    )

    return final_hidden_states.reshape(orig_shape)


def unquantized_fused_optimize_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
):
    # router_logits is now pre-computed masked_routing_weights
    # (topk + softmax already done in fused_moe_forward_rbln)
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    masked_routing_weights = router_logits.reshape(num_tokens, -1)

    # transpose to [num_experts, num_tokens] for custom_moe_glu
    masked_routing_weights_t = masked_routing_weights.transpose(0, 1)

    expert_map_const = None
    if layer.expert_map is not None:
        expert_map_list = layer.expert_map.tolist()
        expert_map_const = torch.tensor(expert_map_list, dtype=torch.int32)

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights_t,
        expert_map_const,
        None,
        None,
        None,
        None,
    )
    return final_hidden_states.reshape(orig_shape)


def fused_moe_forward_rbln(
    self: FusedMoE, hidden_states: torch.Tensor, router: torch.nn.Module
) -> torch.Tensor:
    assert self.quant_method is not None

    if self.dp_size > 1:
        org_hidden_shape = hidden_states.shape
        R = self.dp_size
        num_tokens = org_hidden_shape[:-1].numel()
        t = num_tokens
        max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
        H_dim = hidden_states.shape[-1]

        # --- Step 1: Local routing on this rank's own tokens ---
        router_logits = router(hidden_states)
        router_logits_2d = router_logits.reshape(t, -1)  # [t, E]
        E = router_logits_2d.shape[-1]

        # Pad to max_pad so all DP ranks have the same tensor size for all_gather
        hidden_flat = hidden_states.reshape(t, -1)  # [t, H]
        if t < max_pad:
            router_logits_2d = F.pad(router_logits_2d, (0, 0, 0, max_pad - t), value=0.0)  # [max_pad, E]
            hidden_flat = F.pad(hidden_flat, (0, 0, 0, max_pad - t), value=0.0)  # [max_pad, H]

        # --- Step 2: all_gather router_logits across DP ranks ---
        # [max_pad, E] → [1, max_pad*E] → all_gather → [R, max_pad*E] → [R*max_pad, E]
        rl_flat = router_logits_2d.reshape(1, -1)  # [1, max_pad*E]
        all_rl_flat = get_dp_group().all_gather(rl_flat, dim=0)  # [R, max_pad*E]
        all_router_logits = all_rl_flat.reshape(R * max_pad, E)  # [R*max_pad, E]

        # --- Step 3: topk + softmax on all gathered tokens ---
        # [E, R*max_pad] for dim=0 topk (select top_k experts per token)
        all_router_logits_t = all_router_logits.transpose(0, 1)  # [E, R*max_pad]

        scoring_func = getattr(self, 'scoring_func', 'softmax')
        e_score_correction_bias = getattr(self, 'e_score_correction_bias', None)

        # deepseekv3 style: minimax m2.5
        if scoring_func == 'sigmoid':
            # scores_t = torch.sigmoid(all_router_logits_t)  # [E, R*max_pad]

            # sigmoid scoring with optional e_score_correction_bias
            scores = torch.sigmoid(all_router_logits)  # [R*max_pad, E]
            scores_t = scores.transpose(0, 1)  # [E, R*max_pad]
            scores_for_topk = scores_t
            if e_score_correction_bias is not None:
                scores_for_topk = scores_t + e_score_correction_bias.unsqueeze(1)  # [E, 1]
            _, selected_experts = torch.topk(scores_for_topk, k=self.top_k, dim=0)
            topk_weights = scores_t.gather(0, selected_experts)  # weights from original scores
            topk_weights = topk_weights / topk_weights.sum(dim=0, keepdim=True).clamp_min(1e-20)
        elif self.renormalize:
            # post_norm: topk first, then softmax on selected values
            topk_weights, selected_experts = torch.topk(
                all_router_logits_t, k=self.top_k, dim=0
            )
            topk_weights = F.softmax(topk_weights, dim=0)
        else:
            # pre_norm: softmax first, then topk
            routing_weights = F.softmax(all_router_logits_t, dim=0)
            topk_weights, selected_experts = torch.topk(
                routing_weights, k=self.top_k, dim=0
            )
        masked_routing_weights = torch.zeros_like(all_router_logits_t)  # [E, R*max_pad]
        masked_routing_weights.scatter_(0, selected_experts, topk_weights)

        # Apply token mask to zero out padded positions per DP rank
        use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
        if use_moe_tokens_mask:
            tokens_mask = get_tokens_mask(max_pad).transpose(1, 0)  # [1, R*max_pad]

            ## token dim padding (dim 0 right pad)
            T = masked_routing_weights.shape[1]
            pad_size = 0
            if T <= 8:
                pad_size = 64 - (T % 64)
                tokens_mask = F.pad(tokens_mask, (0, pad_size), value=0.0)
                masked_routing_weights = F.pad(masked_routing_weights, (0, pad_size), value=0.0)

            # [R*max_pad(+pad), E] * [R*max_pad(+pad), 1] (broadcast)
            masked_routing_weights = masked_routing_weights * tokens_mask
            masked_routing_weights_depadded = F.pad(masked_routing_weights, (0, -pad_size), value=0.0)

        # --- Pre-compute routing logit slices (used by dispatch and/or combine all2all) ---
        if envs.VLLM_RBLN_DISPATCH_ALL2ALL or envs.VLLM_RBLN_COMBINE_ALL2ALL:
            # all_routing_3d: [E, R, max_pad] for CCL send/receive kernels
            all_routing_3d = masked_routing_weights_depadded.reshape(E, R, max_pad)

            # Prepare per-rank router_logits slices
            e = E // R  # local experts per rank
            # send_rl: [E, max_pad] — this rank's routing for all experts
            send_rl = all_routing_3d[:, self.dp_rank, :]  # (E, max_pad)
            # recv_rl: [e, R*max_pad] — local experts' routing across all ranks
            e_start = self.dp_rank * e
            recv_rl = all_routing_3d[e_start : e_start + e].reshape(e, R * max_pad)  # (e, T)

        # --- Step 4: Dispatch tokens across DP ranks ---
        if envs.VLLM_RBLN_DISPATCH_ALL2ALL:
            # --- all2all dispatch path ---
            # hidden_flat: [max_pad, H] (already padded above)

            # ccl_dispatch_send
            send_buffer, send_sizes = (
                torch.ops.rbln_custom_ops.ccl_dispatch_send(
                    hidden_flat,
                    send_rl,
                    self.send_mask,
                    self.dp_rank,
                )
            )

            # ccl_all2all_x (naive P2P)
            recv_buffer = torch.ops.rbln_custom_ops.ccl_all2all_x_kernel(
                send_buffer,
                send_sizes,
                self.dp_size,
                CCL_ALL2ALL_GROUP_ID,
            )

            # ccl_dispatch_receive → unpacked: [R, max_pad, H]
            unpacked = torch.ops.rbln_custom_ops.ccl_dispatch_receive(
                recv_buffer,
                recv_rl,
                hidden_flat,
                self.dp_rank,
            )

            # unpacked: [R, max_pad, H] → flatten to [R*max_pad, H] for MoE
            gathered_hidden = unpacked.reshape(R * max_pad, H_dim)
        else:
            # --- all-gather dispatch path ---
            # Broadcast hidden_states to all DP ranks via all_gather
            # hidden_flat: [max_pad, H] → [1, max_pad, H] → all_gather → [R, max_pad, H]
            hidden_for_gather = hidden_flat.unsqueeze(0)  # [1, max_pad, H]
            all_hidden = get_dp_group().all_gather(hidden_for_gather, dim=0)  # [R, max_pad, H]
            gathered_hidden = all_hidden.reshape(R * max_pad, H_dim)

        # --- Step 5: MoE FFN computation ---
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=gathered_hidden,
            router_logits=masked_routing_weights, # [E*max_pad, T_padded]
        )

        # --- Step 6: Combine partial results and extract this rank's output ---
        if envs.VLLM_RBLN_COMBINE_ALL2ALL:
            # --- all2all combine path ---
            # MoE output: [R*max_pad, H] → [R, max_pad, H]
            combine_3d = final_hidden_states.reshape(R, max_pad, H_dim)

            # ccl_combine_send: pack expert outputs per destination rank
            # recv_rl (e, T): local experts' routing → sum-reduction for dest indices
            combine_send_buf, combine_send_sizes = (
                torch.ops.rbln_custom_ops.ccl_combine_send(
                    combine_3d,
                    recv_rl,
                    self.dp_rank,
                )
            )

            # ccl_all2all_x: exchange combine buffers (reuse same group ID)
            combine_recv_buf = torch.ops.rbln_custom_ops.ccl_all2all_x_kernel(
                combine_send_buf,
                combine_send_sizes,
                self.dp_size,
                CCL_ALL2ALL_GROUP_ID,
            )

            # ccl_combine_receive: unpack + sum-reduce → (max_pad, H)
            # send_rl (E, max_pad): this rank's full expert routing
            # self.send_mask: reused as expert_map (same matrix)
            final_hidden_states = torch.ops.rbln_custom_ops.ccl_combine_receive(
                combine_recv_buf,
                send_rl,
                self.send_mask,
                combine_3d[self.dp_rank],  # local rank's own contribution
                self.dp_rank,
            )
            # final_hidden_states: (max_pad, H)
        else:
            # --- reduce_scatter combine path ---
            # reduce_scatter: each rank receives only its own summed portion
            # (more efficient than all_reduce + slice since each rank only needs 1/R of data)
            hidden_shape_dp = (-1, 1, H_dim)
            all_hidden_states = final_hidden_states.reshape(hidden_shape_dp)
            assert all_hidden_states.shape[0] % self.dp_size == 0

            final_hidden_states = get_dp_group().reduce_scatter(all_hidden_states, dim=0)
            assert final_hidden_states.shape[0] == max_pad

        final_hidden_states = final_hidden_states[:t]
        final_hidden_states = final_hidden_states.reshape(org_hidden_shape)

        return final_hidden_states

    # --- DP == 1 path ---
    router_logits = router(hidden_states)

    # topk + softmax → masked_routing_weights (direct, no get_masked_routing_weights)
    orig_shape = hidden_states.shape
    num_tokens = orig_shape[:-1].numel()
    router_logits_2d = router_logits.reshape(num_tokens, -1)

    # transpose to [E, t] for dim=0 topk (matching detach_topk branch)
    router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]

    scoring_func = getattr(self, 'scoring_func', 'softmax')
    e_score_correction_bias = getattr(self, 'e_score_correction_bias', None)

    if scoring_func == 'sigmoid':
        # DeepSeek-V3 style: sigmoid scoring with optional e_score_correction_bias
        scores_t = torch.sigmoid(router_logits_t)  # [E, t]
        scores_for_topk = scores_t
        if e_score_correction_bias is not None:
            scores_for_topk = scores_t + e_score_correction_bias.unsqueeze(1)  # [E, 1]
        _, selected_experts = torch.topk(scores_for_topk, k=self.top_k, dim=0)
        topk_weights = scores_t.gather(0, selected_experts)  # weights from original scores
        topk_weights = topk_weights / topk_weights.sum(dim=0, keepdim=True).clamp_min(1e-20)
    elif self.renormalize:
        topk_weights, selected_experts = torch.topk(
            router_logits_t, k=self.top_k, dim=0
        )
        topk_weights = F.softmax(topk_weights, dim=0)
    else:
        routing_weights = F.softmax(router_logits_t, dim=0)
        topk_weights, selected_experts = torch.topk(
            routing_weights, k=self.top_k, dim=0
        )
    masked_routing_weights = torch.zeros_like(router_logits_t)  # [E, t]
    masked_routing_weights.scatter_(0, selected_experts, topk_weights)

    # Restore dtype to match hidden_states BEFORE tokens_mask multiply
    # (scatter_ promotes to float32, but LowerTopKRouting fuses
    #  scatter+cast(bf16) into contrib_topk_routing with bf16 output;
    #  multiply must happen after the cast so types match)

    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(num_tokens).transpose(1, 0)  # [1, t]

        ## token dim padding (dim 1 right pad)
        T = masked_routing_weights.shape[1]
        pad_size = 0
        if T <= 8:
            pad_size = 64 - (T % 64)
            tokens_mask = F.pad(tokens_mask, (0, pad_size), value=0.0)
            masked_routing_weights = F.pad(masked_routing_weights, (0, pad_size), value=0.0)

        # [t(+pad), E] * [t(+pad), 1] (broadcast)
        masked_routing_weights = masked_routing_weights * tokens_mask
        masked_routing_weights_depadded = F.pad(masked_routing_weights, (0, -pad_size), value=0.0)

    # pass as [t, E] to quant_method.apply (it will be reshaped inside)
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=masked_routing_weights,
    )

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


# ---------------------------------------------------------------------------
# Monkeypatch: FusedMoE.__init__ — register all2all masks when DP > 1
# ---------------------------------------------------------------------------
_original_fused_moe_init = FusedMoE.__init__


def _fused_moe_init_with_all2all(self, *args, **kwargs):
    _original_fused_moe_init(self, *args, **kwargs)
    use_dispatch_all2all = envs.VLLM_RBLN_DISPATCH_ALL2ALL
    use_combine_all2all = envs.VLLM_RBLN_COMBINE_ALL2ALL

    if self.dp_size > 1 and (use_dispatch_all2all or use_combine_all2all):
        R = self.dp_size
        E = self.global_num_experts
        # send_mask doubles as expert_map for combine_receive (same matrix)
        self.register_buffer(
            "send_mask",
            torch.tensor(
                prepare_send_mask_matrix(R, E),
                dtype=torch.float32,
            ),
        )
        logger.info(
            f"[RBLN] FusedMoE all2all masks registered "
            f"(dispatch={use_dispatch_all2all}, combine={use_combine_all2all}): "
            f"R={R}, E={E}, "
            f"send_mask={self.send_mask.shape}"
        )
    elif self.dp_size > 1:
        logger.info(
            f"[RBLN] FusedMoE using all-gather dispatch: "
            f"R={self.dp_size}, E={self.global_num_experts}"
        )


FusedMoE.__init__ = _fused_moe_init_with_all2all
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
