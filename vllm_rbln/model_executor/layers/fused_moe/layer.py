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

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# All2All mask generation helpers
# (recursive doubling all2all implementation)
# ---------------------------------------------------------------------------


def generate_H_matrix(R: int, my_rank: int) -> np.ndarray:
    """Hypercube send-stage mask (N, R) where N=log2(R)."""
    N = int(np.log2(R))
    H = np.zeros((N, R), dtype=int)
    for s in range(N):
        b = N - 1 - s
        for d in range(R):
            if ((d >> (b + 1)) == (my_rank >> (b + 1))) and (
                ((d >> b) & 1) != ((my_rank >> b) & 1)
            ):
                H[s, d] = 1
    return H


def generate_W_matrix(R: int, my_rank: int) -> np.ndarray:
    """Receive routing matrix (R, R)."""
    W = np.zeros((R, R), dtype=int)
    for i in range(R):
        if i == my_rank:
            W[i, my_rank] = 1
        else:
            diff = i ^ my_rank
            b = diff.bit_length() - 1
            for d in range(R):
                if (d >> b) == (my_rank >> b):
                    W[i, d] = 1
    return W


def generate_expert_mask(R: int, E: int) -> np.ndarray:
    """Expert ownership mask (R, E). Local expert index or -1."""
    mask = np.ones((R, E), dtype=int) * -1
    local_cnt = E // R
    for i in range(R):
        for j in range(local_cnt):
            mask[i, j + i * local_cnt] = j
    return mask


def prepare_send_mask_matrix(R: int, my_rank: int, E: int) -> np.ndarray:
    """(N, E) send mask for each hypercube stage."""
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    return np.matmul(generate_H_matrix(R, my_rank), expert_binary)


def prepare_recv_mask_matrix(R: int, my_rank: int, E: int) -> np.ndarray:
    """(R, E) recv mask."""
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    return np.matmul(generate_W_matrix(R, my_rank), expert_binary)


# ---------------------------------------------------------------------------
# Custom op: ccl_send_kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_send_kernel",
    mutates_args=(),
)
def ccl_send_kernel(
    hidden_states: Tensor,
    router_logits: Tensor,
    send_mask: Tensor,
    recv_mask: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = hidden_states.dtype
    send_mask = send_mask.to(dtype)
    recv_mask = recv_mask.to(dtype)

    t_dim = hidden_states.shape[0]
    t_padded = (t_dim + 63) // 64 * 64
    H_dim = hidden_states.shape[1]
    N_dim = send_mask.shape[0]
    R_dim = recv_mask.shape[0]

    local_router = router_logits[:, rank_id, :]
    send_buffer_logit = torch.matmul(send_mask, local_router)

    send_buffer = torch.zeros(N_dim, t_dim, H_dim, dtype=hidden_states.dtype)
    send_sizes = torch.zeros(N_dim, 64, dtype=torch.uint16)
    for s in range(N_dim):
        valid_idx = send_buffer_logit[s].nonzero(as_tuple=True)[0]
        send_buffer[s, : valid_idx.shape[0]] = hidden_states[valid_idx]
        send_sizes[s, 0] = valid_idx.shape[0]

    recv_buffer_logit = torch.einsum("re,ert->rt", recv_mask, router_logits)

    recv_indices = torch.full((R_dim, t_padded), 65535, dtype=torch.uint16)
    recv_sizes = torch.zeros(R_dim, 64, dtype=torch.uint16)
    for r in range(R_dim):
        valid_idx = recv_buffer_logit[r].nonzero(as_tuple=True)[0].to(torch.uint16)
        recv_indices[r, : valid_idx.shape[0]] = valid_idx
        recv_sizes[r, 0] = valid_idx.shape[0]

    return send_buffer, recv_indices, send_sizes, recv_sizes


@ccl_send_kernel.register_fake
def _ccl_send_kernel_fake(
    hidden_states: Tensor,
    router_logits: Tensor,
    send_mask: Tensor,
    recv_mask: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    t_dim = hidden_states.shape[0]
    t_padded = (t_dim + 63) // 64 * 64
    N_dim = send_mask.shape[0]
    R_dim = recv_mask.shape[0]
    H_dim = hidden_states.shape[1]
    return (
        torch.empty(N_dim, t_dim, H_dim, dtype=hidden_states.dtype),
        torch.empty(R_dim, t_padded, dtype=torch.uint16),
        torch.empty(N_dim, 64, dtype=torch.uint16),
        torch.empty(R_dim, 64, dtype=torch.uint16),
    )


# ---------------------------------------------------------------------------
# Custom op: ccl_all2all_kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_all2all_kernel",
    mutates_args=(),
)
def ccl_all2all_kernel(
    send_buffer: Tensor,
    send_sizes: Tensor,
    recv_sizes: Tensor,
    ccl_world_size: int,
    group_id: int,
) -> Tensor:
    # CPU stub — returns zeros of correct shape.
    # Real communication happens on device via CCL runtime.
    R = ccl_world_size
    t = send_buffer.shape[1]
    H = send_buffer.shape[2]
    return torch.zeros(R, t, H, dtype=send_buffer.dtype)


@ccl_all2all_kernel.register_fake
def _ccl_all2all_kernel_fake(
    send_buffer: Tensor,
    send_sizes: Tensor,
    recv_sizes: Tensor,
    ccl_world_size: int,
    group_id: int,
) -> Tensor:
    R_dim = recv_sizes.shape[0]
    t_dim = send_buffer.shape[1]
    H_dim = send_buffer.shape[2]
    return torch.empty(R_dim, t_dim, H_dim, dtype=send_buffer.dtype)


# ---------------------------------------------------------------------------
# Custom op: ccl_receive_kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_receive_kernel",
    mutates_args=(),
)
def ccl_receive_kernel(
    recv_buffer: Tensor,
    recv_indices: Tensor,
    recv_sizes: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]
    unpacked = torch.zeros(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)
    for r in range(R_dim):
        if r == rank_id:
            unpacked[r] = hidden_states
        else:
            num_valid = int(recv_sizes[r, 0])
            valid_idx = recv_indices[r, :num_valid].long()
            unpacked[r, valid_idx] = recv_buffer[r, :num_valid]
    return unpacked


@ccl_receive_kernel.register_fake
def _ccl_receive_kernel_fake(
    recv_buffer: Tensor,
    recv_indices: Tensor,
    recv_sizes: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]
    return torch.empty(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)


# ---------------------------------------------------------------------------
# CCL All2All group ID
# ---------------------------------------------------------------------------
CCL_ALL2ALL_GROUP_ID = 42


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
        - masked_routing_weight: [num_experts, batch * seq_len]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        assert hidden_states.dtype == masked_routing_weight.dtype, "hidden_states and masked_routing_weight must have the same dtype"

        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight.transpose(0, 1)[:, i : i + 1]
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
        - masked_routing_weight: [num_experts, batch * seq_len]
        - expert_select_count: [num_experts]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        assert hidden_states.dtype == masked_routing_weight.dtype, "hidden_states and masked_routing_weight must have the same dtype"

        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight.transpose(0, 1)[:, i : i + 1]
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

    # compute expert_select_count from masked_routing_weights
    expert_select_count = (masked_routing_weights_t > 0).sum(dim=1).to(torch.int32)

    tokens_mask = None
    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(num_tokens)

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
        tokens_mask,
    )

    print ("final_hidden_states dtype:", final_hidden_states.dtype)
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

    tokens_mask = None
    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(num_tokens)

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
        tokens_mask,
    )
    return final_hidden_states.reshape(orig_shape)


def fused_moe_forward_rbln(
    self: FusedMoE, hidden_states: torch.Tensor, router: torch.nn.Module
) -> torch.Tensor:
    assert self.quant_method is not None

    if self.dp_size > 1:
        org_hidden_shape = hidden_states.shape

        # --- Step 1: Local routing on this rank's own tokens ---
        router_logits = router(hidden_states)

        # --- Step 2: topk + softmax → masked_routing_weights ---
        # Direct softmax + topk + scatter (no tokens_mask / expert_map here)
        num_tokens = org_hidden_shape[:-1].numel()
        router_logits_2d = router_logits.reshape(num_tokens, -1)
        E = router_logits_2d.shape[-1]

        # transpose to [E, t] for dim=0 topk (matching detach_topk branch)
        router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]

        if self.renormalize:
            # post_norm: topk first, then softmax on selected values
            topk_weights, selected_experts = torch.topk(
                router_logits_t, k=self.top_k, dim=0
            )
            topk_weights = F.softmax(topk_weights, dim=0)
        else:
            # pre_norm: softmax first, then topk
            routing_weights = F.softmax(router_logits_t, dim=0)
            topk_weights, selected_experts = torch.topk(
                routing_weights, k=self.top_k, dim=0
            )
        masked_routing_weights = torch.zeros_like(router_logits_t)  # [E, t]
        masked_routing_weights.scatter_(0, selected_experts, topk_weights)
        # Restore dtype to match hidden_states (scatter_ may promote to float32)
        masked_routing_weights = masked_routing_weights.to(hidden_states.dtype)
        # masked_routing_weights: [E, t] (transposed)

        R = self.dp_size
        t = num_tokens

        # --- Step 3: all_gather routing weights across DP ranks ---
        # all_gather only supports dim=0, so gather as [1, E*t] → [R, E*t] → reshape to [E, R, t]
        mrw_flat = masked_routing_weights.reshape(1, -1)  # [1, E*t]
        all_routing_flat = get_dp_group().all_gather(mrw_flat, dim=0)  # [R, E*t]
        all_routing_3d = all_routing_flat.reshape(R, E, t).permute(1, 0, 2).contiguous()  # [E, R, t]

        # --- Step 4: CCL all2all dispatch ---
        # hidden_states for send: [t, H]
        hidden_flat = hidden_states.reshape(t, -1)

        # ccl_send
        send_buffer, recv_indices, send_sizes, recv_sizes = (
            torch.ops.rbln_custom_ops.ccl_send_kernel(
                hidden_flat,
                all_routing_3d,
                self.send_mask,
                self.recv_mask,
                self.dp_rank,
            )
        )

        # ccl_all2all
        recv_buffer = torch.ops.rbln_custom_ops.ccl_all2all_kernel(
            send_buffer,
            send_sizes,
            recv_sizes,
            self.dp_size,
            CCL_ALL2ALL_GROUP_ID,
        )

        # ccl_receive → unpacked: [R, t, H]
        unpacked = torch.ops.rbln_custom_ops.ccl_receive_kernel(
            recv_buffer,
            recv_indices,
            recv_sizes,
            hidden_flat,
            self.dp_rank,
        )

        # --- Step 5: MoE FFN on gathered tokens ---
        # unpacked: [R, t, H] → flatten to [R*t, H] for MoE
        H_dim = hidden_flat.shape[1]
        gathered_hidden = unpacked.reshape(R * t, H_dim)

        # all_routing for MoE: [E, R, t] → [E, R*t] → pass as [R*t, E]
        # (quant_method.apply receives [num_tokens, n_experts] shape;
        #  the method internally handles transpose to [E, t] if needed)
        all_routing_flat = all_routing_3d.reshape(E, R * t).transpose(0, 1)  # [R*t, E]

        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=gathered_hidden,
            router_logits=all_routing_flat,
        )

        # --- Step 6: Extract this rank's output ---
        # final_hidden_states: [R*t, H] → [R, t, H]
        final_hidden_states = final_hidden_states.reshape(R, t, H_dim)
        # Take only this rank's slice
        final_hidden_states = final_hidden_states[self.dp_rank]
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

    if self.renormalize:
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
    # Restore dtype to match hidden_states (scatter_ may promote to float32)
    masked_routing_weights = masked_routing_weights.to(hidden_states.dtype)

    # pass as [t, E] to quant_method.apply (it will be reshaped inside)
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=masked_routing_weights.transpose(0, 1).reshape(router_logits.shape),
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
    if self.dp_size > 1:
        R = self.dp_size
        rank_id = self.dp_rank
        E = self.global_num_experts
        self.register_buffer(
            "send_mask",
            torch.tensor(
                prepare_send_mask_matrix(R, rank_id, E), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "recv_mask",
            torch.tensor(
                prepare_recv_mask_matrix(R, rank_id, E), dtype=torch.float32
            ),
        )
        logger.info(
            "[RBLN] FusedMoE all2all masks registered: "
            f"R={R}, rank={rank_id}, E={E}, "
            f"send_mask={self.send_mask.shape}, recv_mask={self.recv_mask.shape}"
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
