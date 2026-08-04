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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe import all2all
from vllm_rbln.model_executor.layers.fused_moe.utils import get_tokens_mask

logger = init_logger(__name__)


class RBLNMoERunner(MoERunner):
    """RBLN out-of-tree MoERunner.
    (See [#41184](https://github.com/vllm-project/vllm/pull/41184).)

    #41184 inverted FusedMoE/MoERunner: ``FusedMoE`` is now a factory function
    that builds a ``MoERunner`` (the nn.Module, a ``PluggableLayer``), and the
    expert weights + quant method live on ``self.routed_experts`` (a
    ``RoutedExperts``). RBLN subclasses ``MoERunner`` and, since it is a
    ``PluggableLayer``, the factory instantiates this class in its place.

    Differences from the upstream MoERunner:

    * ``forward(hidden_states, router)`` takes a **router callable** instead of
      precomputed ``router_logits``. RBLN computes router logits *after* the DP
      multicast so routing sees the multicast token layout; the patched model
      MoE blocks (qwen/deepseek/minimax/gpt-oss) pass ``router=`` accordingly.
    * Hidden states stay **3-D ``[B, L, H]``** throughout (RBLN convention); no
      2-D unpacking.
    * Routing (top-k + softmax/sigmoid, optional grouped-topk) is done inline and
      the resulting ``masked_routing_weights`` are passed to
      ``self._quant_method.apply(layer=self.routed_experts, x, router_logits=...)``
      **directly**, bypassing the modular/monolithic kernel plumbing.
    * When ``dp_size > 1``: tokens are padded to ``max_pads_across_dp`` and
      dispatched across DP ranks via all-gather or, when
      ``VLLM_RBLN_DISPATCH_ALL2ALL``/``COMBINE_ALL2ALL`` are set, via the RBLN
      CCL all2all kernels (``send_mask`` registered in ``__init__``); partial
      expert outputs are combined with reduce-scatter or the all2all-combine path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- #41184 remap: expose old FusedMoE attrs from the new locations ---
        # Alias keeps every downstream self.moe_parallel_config.{dp_size,dp_rank}
        # reference (in __init__ and forward) working unchanged.
        self.moe_parallel_config = self.moe_config.moe_parallel_config
        self.global_num_experts = self.routed_experts.global_num_experts

        # --- #41184 remap: top_k lives on moe_config ---
        self.top_k = self.moe_config.experts_per_token

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
                    all2all.prepare_send_mask_matrix(R, E),
                    dtype=torch.float32,
                ),
            )
            logger.debug(
                "[RBLN] MoERunner all2all masks registered "
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
            logger.debug(
                "[RBLN] MoERunner using all-gather dispatch: R=%s, E=%s",
                self.moe_parallel_config.dp_size,
                self.global_num_experts,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        assert self._quant_method is not None

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
                all_rl_flat = get_dp_group().all_gather(
                    rl_flat, dim=0
                )  # [R, max_pad*E]
                all_router_logits = all_rl_flat.reshape(
                    R * max_pad, E
                )  # [R*max_pad, E]
            else:
                # -- origin/dev path: all-gather hidden first → router on full tokens --
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

            scoring_func = getattr(self.router, "scoring_func", "softmax")
            renormalize = getattr(self.router, "renormalize", False)
            e_score_correction_bias = getattr(
                self.router, "e_score_correction_bias", None
            )
            num_expert_group = getattr(self.router, "num_expert_group", None)
            topk_group = getattr(self.router, "topk_group", None)
            use_grouped_topk = num_expert_group is not None

            if scoring_func == "sigmoid":
                if use_grouped_topk:
                    masked_routing_weights = _apply_grouped_topk_torch(
                        all_router_logits,
                        self.top_k,
                        num_expert_group,
                        topk_group,
                        scoring_func=scoring_func,
                        renormalize=renormalize,
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
                    _, selected_experts = torch.topk(
                        scores_for_topk, k=self.top_k, dim=0
                    )
                    topk_weights = scores_t.gather(
                        0, selected_experts
                    )  # weights from original scores
                    if renormalize:
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
                        num_expert_group,
                        topk_group,
                        scoring_func=scoring_func,
                        renormalize=renormalize,
                        e_score_correction_bias=e_score_correction_bias,
                    )  # [E, R*max_pad]
                else:
                    if renormalize:
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
                        num_expert_group,
                        topk_group,
                        scoring_func=scoring_func,
                        renormalize=renormalize,
                        e_score_correction_bias=e_score_correction_bias,
                    )  # [E, R*max_pad]
                else:
                    topk_weights, selected_experts = torch.topk(
                        all_router_logits_t, k=self.top_k, dim=0
                    )
                    if renormalize:
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
            final_hidden_states = self._quant_method.apply(
                layer=self.routed_experts,
                x=gathered_hidden,
                router_logits=masked_routing_weights,  # [E*max_pad, T_padded]
            )

            # --- Step 6: Combine partial results and extract this rank's output ---
            if envs.VLLM_RBLN_COMBINE_ALL2ALL:
                # --- all2all combine path ---
                # MoE output: [R*max_pad, H] → [R, max_pad, H]
                combine_3d = final_hidden_states.reshape(R, max_pad, H_dim)

                # ccl_combine_send: pack expert outputs per destination rank
                # recv_rl(e, T): local experts' routing → sum-reduction for dest indices
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
        scoring_func = getattr(self.router, "scoring_func", "softmax")
        renormalize = getattr(self.router, "renormalize", False)
        e_score_correction_bias = getattr(self.router, "e_score_correction_bias", None)
        num_expert_group = getattr(self.router, "num_expert_group", None)
        topk_group = getattr(self.router, "topk_group", None)
        use_grouped_topk = num_expert_group is not None

        if scoring_func == "sigmoid":
            if use_grouped_topk:
                masked_routing_weights = _apply_grouped_topk_torch(
                    router_logits_2d,
                    self.top_k,
                    num_expert_group,
                    topk_group,
                    scoring_func=scoring_func,
                    renormalize=renormalize,
                    e_score_correction_bias=e_score_correction_bias,
                )  # [E, t]
            else:
                router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]
                # DeepSeek-V3 style:
                # sigmoid scoring with optional e_score_correction_bias
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
                if renormalize:
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
                    num_expert_group,
                    topk_group,
                    scoring_func=scoring_func,
                    renormalize=renormalize,
                    e_score_correction_bias=e_score_correction_bias,
                )  # [E, t]
            else:
                router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]
                if renormalize:
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
                    num_expert_group,
                    topk_group,
                    scoring_func=scoring_func,
                    renormalize=renormalize,
                    e_score_correction_bias=e_score_correction_bias,
                )  # [E, t]
            else:
                router_logits_t = router_logits_2d.transpose(0, 1)  # [E, t]
                topk_weights, selected_experts = torch.topk(
                    router_logits_t, k=self.top_k, dim=0
                )
                if renormalize:
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
        final_hidden_states = self._quant_method.apply(
            layer=self.routed_experts,
            x=hidden_states,
            router_logits=masked_routing_weights,
        )

        return final_hidden_states


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
