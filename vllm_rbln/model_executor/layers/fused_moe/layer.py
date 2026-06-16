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


fused_moe_upstream__init__ = FusedMoE.__init__


def fused_moe_custom__init__(self, *args, **kwargs):
    fused_moe_upstream__init__(self, *args, **kwargs)

    self.expert_map_const = (
        self.expert_map.tolist() if self.expert_map is not None else None
    )


# Define custom_moe_glu op based on environment variable. Routing (topk +
# scoring + optional grouped-topk) is computed in PyTorch by
# fused_moe_forward_rbln; the ops below only apply pre-computed routing weights.
# VLLM_RBLN_MOE_USE_OPT_KERNEL: uses expert_map parameter
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
        n_group: int | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """
        Customized MoE GLU operation (optimized kernel version).

        Routing (topk + softmax/sigmoid, optional grouped-topk) is computed
        in PyTorch by ``fused_moe_forward_rbln``; this op only applies the
        pre-computed routing weights.

        Expected tensor shapes:
        - hidden_states: [batch * seq_len, hidden_size]
        - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
        - up_proj_weight: [num_experts, intermediate_size, hidden_size]
        - down_proj_weight: [num_experts, hidden_size, intermediate_size]
        - masked_routing_weight: [num_experts, batch * seq_len]
          (token dim may be padded to 64-align)

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        assert hidden_states.dtype == masked_routing_weight.dtype, (
            "hidden_states and masked_routing_weight must have the same dtype"
        )

        num_tokens = hidden_states.shape[0]
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        # routing weight token dim may be padded to 64-align; slice to num_tokens
        routing_t = masked_routing_weight.transpose(0, 1)[:num_tokens, :]  # [T, E]
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
        n_group: int | None = None,
        topk_group: int | None = None,
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

        Routing (topk + softmax/sigmoid, optional grouped-topk) is computed
        in PyTorch by ``fused_moe_forward_rbln``; this op only applies the
        pre-computed routing weights.

        Expected tensor shapes:
        - hidden_states: [batch * seq_len, hidden_size]
        - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
        - up_proj_weight: [num_experts, intermediate_size, hidden_size]
        - down_proj_weight: [num_experts, hidden_size, intermediate_size]
        - masked_routing_weight: [num_experts, batch * seq_len]
          (token dim may be padded to 64-align)
        - expert_select_count: [num_experts]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        assert hidden_states.dtype == masked_routing_weight.dtype, (
            "hidden_states and masked_routing_weight must have the same dtype"
        )

        num_tokens = hidden_states.shape[0]
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        # routing weight token dim may be padded to 64-align; slice to num_tokens
        routing_t = masked_routing_weight.transpose(0, 1)[:num_tokens, :]  # [T, E]
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


def _apply_grouped_topk_torch(
    router_logits_2d,  # [T, E] - raw logits (not transposed)
    top_k,
    num_expert_group,
    topk_group,
    scoring_func="softmax",
    renormalize=True,
    e_score_correction_bias=None,
):
    """Apply grouped topk routing in PyTorch.

    Groups experts, selects top groups per token, then applies topk routing
    within selected groups, and scatters results back to full expert space.

    Args:
        router_logits_2d: [T, E] raw router logits.
        top_k: Number of experts to select per token.
        num_expert_group: Number of expert groups (G).
        topk_group: Number of top groups to select per token.
        scoring_func: "softmax", "sigmoid", or None.
        renormalize: Whether to renormalize routing weights after topk.
        e_score_correction_bias: Optional [E] bias tensor (used with sigmoid).

    Returns:
        masked_routing_weights: [E, T] tensor with topk routing weights.
    """
    T, E = router_logits_2d.shape
    G = num_expert_group
    epg = E // G  # experts per group

    # Step 0: Apply scoring function on logits before grouping.
    # For sigmoid, group scores and final routing weights must be computed
    # from the sigmoid-activated scores (matching reference grouped_topk).
    if scoring_func == "sigmoid":
        router_logits_2d = router_logits_2d.sigmoid()

    # Step 1: Reshape to groups [T, G, E/G]
    grouped = router_logits_2d.reshape(T, G, epg)

    # Step 2: Score each group by sum of top-2 expert values
    group_top2_values, _ = torch.topk(grouped, 2, dim=2)  # [T, G, 2]
    group_scores = group_top2_values.sum(dim=2)  # [T, G]

    # Step 3: Select top topk_group groups per token
    _, selected_group_idx = torch.topk(group_scores, topk_group, dim=1)  # [T, tg]

    # Step 4: Gather selected groups [T, topk_group, epg]
    idx_expanded = selected_group_idx.unsqueeze(-1).expand(-1, -1, epg)
    gathered = torch.gather(grouped, 1, idx_expanded)  # [T, topk_group, epg]

    # Transpose to [topk_group, epg, T] then flatten to [topk_group*epg, T]
    gathered_t = gathered.permute(1, 2, 0).reshape(-1, T)  # [topk_group*epg, T]

    # Step 5: Apply topk routing on gathered experts
    if scoring_func == "sigmoid":
        scores_for_topk = gathered_t
        if e_score_correction_bias is not None:
            # Gather bias for selected groups per token
            bias_grouped = e_score_correction_bias.reshape(G, epg)  # [G, epg]
            bias_idx = selected_group_idx.unsqueeze(-1).expand(-1, -1, epg)
            gathered_bias = torch.gather(
                bias_grouped.unsqueeze(0).expand(T, -1, -1), 1, bias_idx
            )  # [T, topk_group, epg]
            gathered_bias_t = gathered_bias.permute(1, 2, 0).reshape(-1, T)
            scores_for_topk = gathered_t + gathered_bias_t
            _, selected_experts = torch.topk(scores_for_topk, k=top_k, dim=0)
            topk_weights = gathered_t.gather(0, selected_experts)
        else:
            topk_weights, selected_experts = torch.topk(
                scores_for_topk, k=top_k, dim=0
            )
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-20)
    elif scoring_func == "softmax":
        if renormalize:
            # post_norm: topk first (renormalize handles normalization)
            topk_weights, selected_experts = torch.topk(gathered_t, k=top_k, dim=0)
        else:
            # pre_norm: softmax first, then topk
            sw = F.softmax(gathered_t, dim=0)
            topk_weights, selected_experts = torch.topk(sw, k=top_k, dim=0)
    else:
        if renormalize:
            topk_weights, selected_experts = torch.topk(gathered_t, k=top_k, dim=0)
            topk_weights = F.softmax(topk_weights, dim=0)
        else:
            topk_weights, selected_experts = torch.topk(gathered_t, k=top_k, dim=0)

    # Create masked routing in gathered space [topk_group*epg, T]
    routed_flat = torch.zeros_like(gathered_t)
    routed_flat.scatter_(0, selected_experts, topk_weights)

    # Reshape back: [topk_group, epg, T] -> [T, topk_group, epg]
    routed_3d = routed_flat.reshape(topk_group, epg, T)
    routed_t = routed_3d.permute(2, 0, 1)  # [T, topk_group, epg]

    # Scatter back to full [T, G, epg]
    result = torch.zeros(T, G, epg, dtype=routed_t.dtype, device=routed_t.device)
    idx_expanded = selected_group_idx.unsqueeze(-1).expand(-1, -1, epg)
    result.scatter_(1, idx_expanded, routed_t)

    # Reshape to [T, E] then transpose to [E, T]
    return result.reshape(T, E).transpose(0, 1)  # [E, T]


def get_tokens_mask(num_tokens: int, left=1.0, right=0.0, device=None):
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
    if device is not None:
        tokens_mask = tokens_mask.to(device)
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
        tokens_mask = get_tokens_mask(
            router_logits.shape[0], 1.0, 0.0, device=selected_weights.device
        )
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
    # routing (topk + scoring + optional grouped-topk) is computed in PyTorch;
    # the custom op below only applies the resulting [E, T] weights.
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)

    # [num_experts, num_tokens(+pad)] masked routing weights
    masked_routing_weights_t = build_masked_routing_weights(layer, router_logits, x)

    # count selected tokens for each expert index
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
    # routing (topk + scoring + optional grouped-topk) is computed in PyTorch;
    # the custom op below only applies the resulting [E, T] weights.
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)

    # [num_experts, num_tokens(+pad)] masked routing weights
    masked_routing_weights_t = build_masked_routing_weights(layer, router_logits, x)

    expert_map_const = None
    if layer.expert_map is not None:
        assert getattr(layer, "expert_map_const", None) is not None
        # Keep tensor ops only: .tolist() + torch.tensor(list) graph-breaks under
        # PyTorch 2.10+ Dynamo when capture_scalar_outputs is false (pytorch#163807).
        expert_map_const = torch.tensor(layer.expert_map_const, dtype=torch.int32)

    # optimum-rbln/src/optimum/rbln/transformers/models/qwen3_moe/
    # qwen3_moe_architecture.py
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


def compute_masked_routing_weights(
    self: FusedMoE, router_logits: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Compute [E, T] masked routing weights in PyTorch.

    topk + scoring (softmax/sigmoid) and optional grouped-topk are applied
    here so the downstream custom op only applies the pre-computed weights.

    Returns:
        torch.Tensor: [num_experts, num_tokens] with non-selected entries zeroed.
    """
    router_logits_2d = router_logits.reshape(num_tokens, -1)  # [T, E]
    scoring_func = getattr(self, "scoring_func", "softmax")
    e_score_correction_bias = getattr(self, "e_score_correction_bias", None)
    use_grouped_topk = getattr(self, "use_grouped_topk", False)

    # Grouped-topk shares one code path across all scoring functions; the
    # scoring/renormalize handling lives inside _apply_grouped_topk_torch.
    if use_grouped_topk:
        return _apply_grouped_topk_torch(
            router_logits_2d,
            self.top_k,
            self.num_expert_group,
            self.topk_group,
            scoring_func=scoring_func,
            renormalize=self.renormalize,
            e_score_correction_bias=e_score_correction_bias,
        )  # [E, T]

    router_logits_t = router_logits_2d.transpose(0, 1)  # [E, T]

    if scoring_func == "sigmoid":
        # DeepSeek-V3 style: sigmoid scoring with optional correction bias
        scores_t = torch.sigmoid(router_logits_t)  # [E, T]
        scores_for_topk = scores_t
        if e_score_correction_bias is not None:
            scores_for_topk = scores_t + e_score_correction_bias.unsqueeze(1)
        _, selected_experts = torch.topk(scores_for_topk, k=self.top_k, dim=0)
        topk_weights = scores_t.gather(0, selected_experts)
        # clamp_min epsilon (1e-20, hard-coded) guards against division by
        # zero when all selected weights sum to 0. 1e-20 is below the
        # representable range of low-precision dtypes, so it is
        # saturated to the smallest representable positive value
        # instead of rounding to 0, preserving the div-by-zero guard.
        topk_weights = topk_weights / topk_weights.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-20)
    elif scoring_func == "softmax":
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
    else:
        # no scoring function: raw topk, optionally renormalized via softmax
        topk_weights, selected_experts = torch.topk(
            router_logits_t, k=self.top_k, dim=0
        )
        if self.renormalize:
            topk_weights = F.softmax(topk_weights, dim=0)

    masked_routing_weights = torch.zeros_like(router_logits_t)
    masked_routing_weights.scatter_(0, selected_experts, topk_weights)
    return masked_routing_weights  # [E, T]


def build_masked_routing_weights(
    layer: FusedMoE, router_logits: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Build [E, T(+pad)] masked routing weights for the routing-free custom ops.

    Computes topk + scoring (+ optional grouped-topk), applies the per-rank
    token mask with 64-align padding, and casts to ``hidden_states.dtype`` so
    the downstream custom ops (which assert matching dtypes) only need to apply
    the weights. Shared by the unquantized custom/optimize, fp8 and
    compressed-tensors MoE apply paths.
    """
    num_tokens = hidden_states.shape[:-1].numel()
    masked_routing_weights = compute_masked_routing_weights(
        layer, router_logits, num_tokens
    )  # [E, T]

    if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK:
        tokens_mask = get_tokens_mask(
            num_tokens, device=masked_routing_weights.device
        ).transpose(1, 0)  # [1, T]
        tokens_mask = tokens_mask.to(masked_routing_weights.dtype)

        # token dim 64-align padding (compiler requires >= 64 along token dim)
        token_dim = masked_routing_weights.shape[1]
        if token_dim <= 8:
            pad_size = 64 - (token_dim % 64)
            tokens_mask = F.pad(tokens_mask, (0, pad_size), value=0.0)
            masked_routing_weights = F.pad(
                masked_routing_weights, (0, pad_size), value=0.0
            )

        # [E, T(+pad)] * [1, T(+pad)] (broadcast)
        masked_routing_weights = masked_routing_weights * tokens_mask

    # custom MoE ops assert routing weights share hidden_states dtype
    return masked_routing_weights.to(hidden_states.dtype)


def fused_moe_forward_rbln(
    self: FusedMoE, hidden_states: torch.Tensor, router: torch.nn.Module
) -> torch.Tensor:
    assert self.quant_method is not None

    if self.moe_parallel_config.dp_size > 1:
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

    # NOTE: routing (topk + scoring + optional grouped-topk) is computed in
    # PyTorch inside each quant method's apply via build_masked_routing_weights
    # for the routing-free custom ops (unquantized custom/optimize, fp8,
    # compressed-tensors). Methods that still route inside the compiler op
    # (e.g. mxfp4) receive the raw router_logits unchanged.
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
    )

    if self.moe_parallel_config.dp_size > 1:
        # output all_reduce == dp all_reduce + tp all_reduce
        if envs.VLLM_RBLN_MOE_REDUCE_SCATTER:
            hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
            all_hidden_states = final_hidden_states.reshape(hidden_shape_dp)
            assert all_hidden_states.shape[0] % self.moe_parallel_config.dp_size == 0

            hidden_states = get_dp_group().reduce_scatter(all_hidden_states, dim=0)
            max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
            assert hidden_states.shape[0] == max_pad

            num_tokens = org_hidden_shape[:-1].numel()  # noqa: F841
            final_hidden_states = hidden_states[:num_tokens].contiguous()
        else:
            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
            final_hidden_states = all_hidden_states.reshape(hidden_shape_dp)

            max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
            num_tokens = org_hidden_shape[:-1].numel()  # noqa: F841
            start = self.moe_parallel_config.dp_rank * max_pad
            end = start + num_tokens
            final_hidden_states = final_hidden_states[start:end].contiguous()

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
            rank_tensor = x if rank == self.moe_parallel_config.dp_rank else zeros
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


FusedMoE.__init__ = fused_moe_custom__init__
FusedMoE.forward_oot = fused_moe_forward_rbln

if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
    logger.info("[RBLN] fused moe, RBLN optimize moe custom kernel")
    UnquantizedFusedMoEMethod.apply = unquantized_fused_optimize_moe_method_custom
elif envs.VLLM_RBLN_MOE_CUSTOM_KERNEL:
    logger.info("[RBLN] fused moe, RBLN moe custom kernel")
    UnquantizedFusedMoEMethod.apply = unquantized_fused_moe_method_custom
else:
    logger.info("[RBLN] fused moe, pytorch native kernel")
    UnquantizedFusedMoEMethod.apply = unquantized_fused_moe_method_rbln
FusedMoE.naive_multicast = fused_moe_naive_multicast_rbln
