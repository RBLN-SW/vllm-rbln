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
    ccl_combine_receive,  # noqa: F811
    ccl_combine_send,  # noqa: F811
    prepare_send_mask_matrix,
)

logger = init_logger(__name__)


fused_moe_upstream__init__ = FusedMoE.__init__


def fused_moe_custom__init__(self, *args, **kwargs):
    fused_moe_upstream__init__(self, *args, **kwargs)

    self.expert_map_const = (
        self.expert_map.tolist() if self.expert_map is not None else None
    )


# Define custom_moe_glu op. The body below is the PyTorch reference
# implementation; on device it is replaced by the RBLN custom kernel.
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
    hidden_act: str,
    expert_map: torch.Tensor | None = None,
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


@custom_moe_glu.register_fake
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

    hidden_states = x.reshape(num_tokens, -1)
    masked_routing_weights = router_logits

    expert_map_const = None
    if layer.expert_map is not None:
        assert getattr(layer, "expert_map_const", None) is not None
        # Keep tensor ops only: .tolist() + torch.tensor(list) graph-breaks under
        # PyTorch 2.10+ Dynamo when capture_scalar_outputs is false (pytorch#163807).
        expert_map_const = torch.tensor(layer.expert_map_const, dtype=torch.int32)

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights,
        layer.activation.value,
        expert_map_const,
        None,
        None,
        None,
    )
    return final_hidden_states.reshape(orig_shape)


def _apply_grouped_topk_torch(
    router_logits_2d,  # [T, E] - raw logits (not transposed)
    top_k,
    num_expert_group,
    topk_group,
    scoring_func=None,
    renormalize=True,
    e_score_correction_bias=None,
):
    """Apply grouped topk routing in PyTorch.

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

    # Step 1: Apply scoring function & reshape to groups [T, G, E/G]
    if scoring_func == "sigmoid":
        router_logits_2d = torch.sigmoid(router_logits_2d)
    grouped = router_logits_2d.reshape(T, G, epg)

    # Step 2: Score each group by sum of top-2 expert values
    group_top2_values, _ = torch.topk(grouped, 2, dim=2)  # [T, G, 2]
    group_scores = group_top2_values.sum(dim=2)  # [T, G]

    # Step 3: Select top topk_group groups per token
    _, selected_group_idx = torch.topk(
        group_scores, topk_group, dim=1
    )  # [T, topk_group]

    # Step 4: Gather selected groups [T, topk_group, epg]
    idx_expanded = selected_group_idx.unsqueeze(-1).expand(-1, -1, epg)
    gathered = torch.gather(grouped, 1, idx_expanded)  # [T, topk_group, epg]

    if e_score_correction_bias is not None:
        # --- with-bias branch ---
        # Scatter into [T, G, epg] filled with -inf, then flatten to [E, T] so
        # bias indices line up with expert IDs.
        minus_inf = torch.full(
            (T, G, epg),
            float("-inf"),
            dtype=gathered.dtype,
            device=gathered.device,
        )
        grouped_masked = minus_inf.scatter(1, idx_expanded, gathered)  # [T, G, epg]
        # [T, G, epg] -> [G, epg, T] -> [E, T]
        scores_t = grouped_masked.permute(1, 2, 0).reshape(E, T)

        scores_for_topk = scores_t + e_score_correction_bias.unsqueeze(1)
        _, selected_experts = torch.topk(scores_for_topk, k=top_k, dim=0)
        topk_weights = scores_t.gather(0, selected_experts)
        if scoring_func == "softmax":
            topk_weights = F.softmax(topk_weights, dim=0)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-20)

        result = torch.zeros_like(scores_t)  # [E, T]
        result.scatter_(0, selected_experts, topk_weights)
        return result  # [E, T]

    # --- no-bias branch ---
    # Flatten gathered groups to [topk_group*epg, T] and run topk routing there.
    gathered_flat = gathered.permute(1, 2, 0).reshape(-1, T)  # [topk_group*epg, T]

    if scoring_func == "softmax":
        if renormalize:
            # post_norm: topk first, then softmax on selected values
            topk_weights, selected_experts = torch.topk(gathered_flat, k=top_k, dim=0)
            topk_weights = F.softmax(topk_weights, dim=0)
        else:
            # pre_norm: softmax first, then topk
            sw = F.softmax(gathered_flat, dim=0)
            topk_weights, selected_experts = torch.topk(sw, k=top_k, dim=0)
    else:
        topk_weights, selected_experts = torch.topk(gathered_flat, k=top_k, dim=0)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-20)

    routed_flat = torch.zeros_like(gathered_flat)  # [topk_group*epg, T]
    routed_flat.scatter_(0, selected_experts, topk_weights)
    # [topk_group*epg, T] -> [topk_group, epg, T] -> [T, topk_group, epg]
    routed_t = routed_flat.reshape(topk_group, epg, T).permute(2, 0, 1)
    # Scatter back to full expert space: [T, G, epg]
    zeros = torch.zeros(T, G, epg, dtype=routed_t.dtype, device=routed_t.device)
    result_t = zeros.scatter(1, idx_expanded, routed_t)
    # [T, G, epg] -> [G, epg, T] -> [E, T]
    return result_t.permute(1, 2, 0).reshape(E, T)


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
        topk_weights = topk_weights / topk_weights.sum(dim=0, keepdim=True).clamp_min(
            1e-20
        )
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
        R = self.moe_parallel_config.dp_size
        num_tokens = org_hidden_shape[:-1].numel()
        t = num_tokens
        max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
        H_dim = hidden_states.shape[-1]

        # Pad hidden_states to max_pad so all DP ranks have the same tensor size
        hidden_flat = hidden_states.reshape(t, -1)  # [t, H]
        if t < max_pad:
            hidden_flat = F.pad(
                hidden_flat, (0, 0, 0, max_pad - t), value=0.0
            )  # [max_pad, H]

        if envs.VLLM_RBLN_DISPATCH_ALL2ALL:
            # --- Router DP path: local routing → all_gather logits ---
            router_logits = router(hidden_states)
            router_logits_2d = router_logits.reshape(t, -1)  # [t, E]
            E = router_logits_2d.shape[-1]

            if t < max_pad:
                router_logits_2d = F.pad(
                    router_logits_2d, (0, 0, 0, max_pad - t), value=0.0
                )  # [max_pad, E]

            # all_gather router_logits across DP ranks
            rl_flat = router_logits_2d.reshape(1, -1)
            all_rl_flat = get_dp_group().all_gather(rl_flat, dim=0)  # [R, max_pad*E]
            all_router_logits = all_rl_flat.reshape(R * max_pad, E)  # [R*max_pad, E]
        else:
            # --- origin/dev path: all-gather hidden first → router on full tokens ---
            # all-gather hidden_states across DP ranks (also serves as dispatch)
            hidden_for_gather = hidden_flat.unsqueeze(0)  # [1, max_pad, H]
            all_hidden = get_dp_group().all_gather(
                hidden_for_gather, dim=0
            )  # [R, max_pad, H]
            gathered_hidden = all_hidden.reshape(R * max_pad, H_dim)

            # Router on all gathered tokens (no Router DP)
            all_router_logits = router(gathered_hidden)  # [R*max_pad, E]
            all_router_logits = all_router_logits.reshape(R * max_pad, -1)
            E = all_router_logits.shape[-1]

        # --- topk + softmax on all gathered tokens ---
        # [E, R*max_pad] for dim=0 topk (select top_k experts per token)
        all_router_logits_t = all_router_logits.transpose(0, 1)  # [E, R*max_pad]

        scoring_func = getattr(self, "scoring_func", "softmax")
        e_score_correction_bias = getattr(self, "e_score_correction_bias", None)
        use_grouped_topk = getattr(self, "use_grouped_topk", False)

        if scoring_func == "sigmoid":
            if use_grouped_topk:
                masked_routing_weights = _apply_grouped_topk_torch(
                    all_router_logits,
                    self.top_k,
                    self.num_expert_group,
                    self.topk_group,
                    scoring_func=scoring_func,
                    renormalize=self.renormalize,
                    e_score_correction_bias=e_score_correction_bias,
                )  # [E, R*max_pad]
            else:
                # deepseekv3 style: minimax m2.5 (?)
                # sigmoid scoring with optional e_score_correction_bias
                scores = torch.sigmoid(all_router_logits)  # [R*max_pad, E]
                scores_t = scores.transpose(0, 1)  # [E, R*max_pad]
                scores_for_topk = scores_t
                if e_score_correction_bias is not None:
                    scores_for_topk = scores_t + e_score_correction_bias.unsqueeze(
                        1
                    )  # [E, 1]
                _, selected_experts = torch.topk(scores_for_topk, k=self.top_k, dim=0)
                topk_weights = scores_t.gather(
                    0, selected_experts
                )  # weights from original scores
                if self.renormalize:
                    topk_weights = topk_weights / topk_weights.sum(
                        dim=0, keepdim=True
                    ).clamp_min(1e-20)
                masked_routing_weights = torch.zeros_like(
                    all_router_logits_t
                )  # [E, R*max_pad]
                masked_routing_weights.scatter_(0, selected_experts, topk_weights)
        elif scoring_func == "softmax":
            if use_grouped_topk:
                masked_routing_weights = _apply_grouped_topk_torch(
                    all_router_logits,
                    self.top_k,
                    self.num_expert_group,
                    self.topk_group,
                    scoring_func=scoring_func,
                    renormalize=self.renormalize,
                    e_score_correction_bias=e_score_correction_bias,
                )  # [E, R*max_pad]
            else:
                if self.renormalize:
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
                masked_routing_weights = torch.zeros_like(
                    all_router_logits_t
                )  # [E, R*max_pad]
                masked_routing_weights.scatter_(0, selected_experts, topk_weights)
        else:
            if use_grouped_topk:
                masked_routing_weights = _apply_grouped_topk_torch(
                    all_router_logits,
                    self.top_k,
                    self.num_expert_group,
                    self.topk_group,
                    scoring_func=scoring_func,
                    renormalize=self.renormalize,
                    e_score_correction_bias=e_score_correction_bias,
                )  # [E, R*max_pad]
            else:
                topk_weights, selected_experts = torch.topk(
                    all_router_logits_t, k=self.top_k, dim=0
                )
                if self.renormalize:
                    topk_weights = topk_weights / topk_weights.sum(
                        dim=0, keepdim=True
                    ).clamp_min(1e-20)
                masked_routing_weights = torch.zeros_like(
                    all_router_logits_t
                )  # [E, R*max_pad]
                masked_routing_weights.scatter_(0, selected_experts, topk_weights)

        # Apply token mask to zero out padded positions per DP rank
        use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
        if use_moe_tokens_mask:
            tokens_mask = get_tokens_mask(
                max_pad, device=masked_routing_weights.device
            ).transpose(1, 0)  # [1, R*max_pad]
            masked_routing_weights = masked_routing_weights * tokens_mask

        # --- Pre-compute routing logit slices (used by all2all) ---
        if envs.VLLM_RBLN_DISPATCH_ALL2ALL or envs.VLLM_RBLN_COMBINE_ALL2ALL:
            # all_routing_3d: [E, R, max_pad] for CCL send/receive kernels
            all_routing_3d = masked_routing_weights.reshape(E, R, max_pad)

            # Prepare per-rank router_logits slices
            e = E // R  # local experts per rank
            # send_rl: [E, max_pad] — this rank's routing for all experts
            send_rl = all_routing_3d[
                :, self.moe_parallel_config.dp_rank, :
            ]  # (E, max_pad)
            # recv_rl: [e, R*max_pad] — local experts' routing across all ranks
            e_start = self.moe_parallel_config.dp_rank * e
            recv_rl = all_routing_3d[e_start : e_start + e].reshape(
                e, R * max_pad
            )  # (e, T)

        # --- Step 4: Dispatch tokens across DP ranks ---
        if envs.VLLM_RBLN_DISPATCH_ALL2ALL:
            # --- all2all dispatch path ---
            # hidden_flat: [max_pad, H] (already padded above)

            # ccl_dispatch_send
            send_buffer, send_sizes = torch.ops.rbln_custom_ops.ccl_dispatch_send(
                hidden_flat,
                send_rl,
                self.send_mask,
                self.moe_parallel_config.dp_rank,
            )

            # ccl_all2all_x (naive P2P)
            recv_buffer = torch.ops.rbln_custom_ops.ccl_all2all_x_kernel(
                send_buffer,
                send_sizes,
                self.moe_parallel_config.dp_size,
                get_dp_group().unique_name,
            )

            # ccl_dispatch_receive → unpacked: [R, max_pad, H]
            unpacked = torch.ops.rbln_custom_ops.ccl_dispatch_receive(
                recv_buffer,
                recv_rl,
                hidden_flat,
                self.moe_parallel_config.dp_rank,
            )

            # unpacked: [R, max_pad, H] → flatten to [R*max_pad, H] for MoE
            gathered_hidden = unpacked.reshape(R * max_pad, H_dim)
        # else: gathered_hidden was already computed via all-gather before router

        # --- Step 5: MoE FFN computation ---
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=gathered_hidden,
            router_logits=masked_routing_weights,  # [E*max_pad, T_padded]
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
                    self.moe_parallel_config.dp_rank,
                )
            )

            # ccl_all2all_x: all2all transfer of combined buffers
            combine_recv_buf = torch.ops.rbln_custom_ops.ccl_all2all_x_kernel(
                combine_send_buf,
                combine_send_sizes,
                self.moe_parallel_config.dp_size,
                get_dp_group().unique_name,
            )

            # ccl_combine_receive: unpack + sum-reduce → (max_pad, H)
            # send_rl (E, max_pad): this rank's full expert routing
            # self.send_mask: reused as expert_map (same matrix)
            final_hidden_states = torch.ops.rbln_custom_ops.ccl_combine_receive(
                combine_recv_buf,
                send_rl,
                self.send_mask,
                combine_3d[
                    self.moe_parallel_config.dp_rank
                ],  # local rank's own contribution
                self.moe_parallel_config.dp_rank,
            )
            # final_hidden_states: (max_pad, H)
        else:
            # --- reduce_scatter / all_reduce combine path ---
            if envs.VLLM_RBLN_MOE_REDUCE_SCATTER:
                # reduce_scatter: each rank receives only its own summed portion
                hidden_shape_dp = (-1, 1, H_dim)
                all_hidden_states = final_hidden_states.reshape(hidden_shape_dp)
                assert (
                    all_hidden_states.shape[0] % self.moe_parallel_config.dp_size == 0
                )

                final_hidden_states = get_dp_group().reduce_scatter(
                    all_hidden_states, dim=0
                )
                assert final_hidden_states.shape[0] == max_pad
            else:
                all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
                hidden_shape_dp = (-1, 1, H_dim)
                final_hidden_states = all_hidden_states.reshape(hidden_shape_dp)

                start = self.moe_parallel_config.dp_rank * max_pad
                end = start + t
                final_hidden_states = final_hidden_states[start:end]

        final_hidden_states = final_hidden_states[:t]
        final_hidden_states = final_hidden_states.reshape(org_hidden_shape)

        return final_hidden_states

    # --- DP == 1 path ---
    router_logits = router(hidden_states)

    # topk + softmax → masked_routing_weights
    orig_shape = hidden_states.shape
    num_tokens = orig_shape[:-1].numel()
    router_logits_2d = router_logits.reshape(num_tokens, -1)

    # transpose to [E, t] for dim=0 topk (matching detach_topk branch)
    scoring_func = getattr(self, "scoring_func", "softmax")
    e_score_correction_bias = getattr(self, "e_score_correction_bias", None)
    use_grouped_topk = getattr(self, "use_grouped_topk", False)

    if scoring_func == "sigmoid":
        if use_grouped_topk:
            masked_routing_weights = _apply_grouped_topk_torch(
                router_logits_2d,
                self.top_k,
                self.num_expert_group,
                self.topk_group,
                scoring_func=scoring_func,
                renormalize=self.renormalize,
                e_score_correction_bias=e_score_correction_bias,
            )  # [E, t]
        else:
            router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]
            # DeepSeek-V3 style: sigmoid scoring with optional e_score_correction_bias
            scores_t = torch.sigmoid(router_logits_t)  # [E, t]
            scores_for_topk = scores_t
            if e_score_correction_bias is not None:
                scores_for_topk = scores_t + e_score_correction_bias.unsqueeze(
                    1
                )  # [E, 1]
            _, selected_experts = torch.topk(scores_for_topk, k=self.top_k, dim=0)
            topk_weights = scores_t.gather(
                0, selected_experts
            )  # weights from original scores
            if self.renormalize:
                topk_weights = topk_weights / topk_weights.sum(
                    dim=0, keepdim=True
                ).clamp_min(1e-20)
            masked_routing_weights = torch.zeros_like(router_logits_t)  # [E, t]
            masked_routing_weights.scatter_(0, selected_experts, topk_weights)
    elif scoring_func == "softmax":
        if use_grouped_topk:
            masked_routing_weights = _apply_grouped_topk_torch(
                router_logits_2d,
                self.top_k,
                self.num_expert_group,
                self.topk_group,
                scoring_func=scoring_func,
                renormalize=self.renormalize,
                e_score_correction_bias=e_score_correction_bias,
            )  # [E, t]
        else:
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
    else:
        if use_grouped_topk:
            masked_routing_weights = _apply_grouped_topk_torch(
                router_logits_2d,
                self.top_k,
                self.num_expert_group,
                self.topk_group,
                scoring_func=scoring_func,
                renormalize=self.renormalize,
                e_score_correction_bias=e_score_correction_bias,
            )  # [E, t]
        else:
            router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]
            topk_weights, selected_experts = torch.topk(
                router_logits_t, k=self.top_k, dim=0
            )
            if self.renormalize:
                topk_weights = topk_weights / topk_weights.sum(
                    dim=0, keepdim=True
                ).clamp_min(1e-20)
            masked_routing_weights = torch.zeros_like(router_logits_t)  # [E, t]
            masked_routing_weights.scatter_(0, selected_experts, topk_weights)

    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(
            num_tokens, device=masked_routing_weights.device
        ).transpose(1, 0)  # [1, t]

        # [t, E] * [t, 1] (broadcast)
        masked_routing_weights = masked_routing_weights * tokens_mask

    # pass as [t, E] to quant_method.apply (it will be reshaped inside)
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=masked_routing_weights,
    )

    return final_hidden_states


# ---------------------------------------------------------------------------
# Monkeypatch: FusedMoE.__init__ — register all2all masks when DP > 1
# ---------------------------------------------------------------------------


def _fused_moe_init_with_all2all(self, *args, **kwargs):
    fused_moe_custom__init__(self, *args, **kwargs)
    use_dispatch_all2all = envs.VLLM_RBLN_DISPATCH_ALL2ALL
    use_combine_all2all = envs.VLLM_RBLN_COMBINE_ALL2ALL

    if self.moe_parallel_config.dp_size > 1 and (
        use_dispatch_all2all or use_combine_all2all
    ):
        R = self.moe_parallel_config.dp_size
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
            "[RBLN] FusedMoE all2all masks registered "
            "(dispatch=%s, combine=%s): "
            "R=%s, E=%s, "
            "send_mask=%s",
            use_dispatch_all2all,
            use_combine_all2all,
            R,
            E,
            self.send_mask.shape,
        )
    elif self.moe_parallel_config.dp_size > 1:
        logger.info(
            "[RBLN] FusedMoE using all-gather dispatch: R=%s, E=%s",
            self.moe_parallel_config.dp_size,
            self.global_num_experts,
        )


FusedMoE.__init__ = _fused_moe_init_with_all2all
FusedMoE.forward_oot = fused_moe_forward_rbln

logger.info("[RBLN] fused moe, RBLN moe custom kernel")
UnquantizedFusedMoEMethod.apply = unquantized_fused_moe_method_custom
