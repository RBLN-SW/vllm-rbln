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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)


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
        dp_mask: torch.Tensor | None = None,
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
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)
elif envs.VLLM_RBLN_MOE_CUSTOM_KERNEL:
    # @torch.library.custom_op(
    #     "rbln_custom_ops::custom_moe_glu",
    #     mutates_args=(),
    # )
    # def custom_moe_glu(
    #     hidden_states: torch.Tensor,
    #     gate_proj_weight: torch.Tensor,
    #     up_proj_weight: torch.Tensor,
    #     down_proj_weight: torch.Tensor,
    #     masked_routing_weight: torch.Tensor,
    #     expert_select_count: torch.Tensor,
    #     gate_proj_bias: torch.Tensor | None = None,
    #     up_proj_bias: torch.Tensor | None = None,
    #     down_proj_bias: torch.Tensor | None = None,
    #     dp_mask: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     """
    #     Customized MoE GLU operation (custom kernel version).

    #     Expected tensor shapes:
    #     - hidden_states: [batch * seq_len, hidden_size]
    #     - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
    #     - up_proj_weight: [num_experts, intermediate_size, hidden_size]
    #     - down_proj_weight: [num_experts, hidden_size, intermediate_size]
    #     - masked_routing_weight: [batch * seq_len, num_experts]
    #     - expert_select_count: [num_experts]

    #     Returns:
    #         torch.Tensor: [batch * seq_len, hidden_size]
    #     """
    #     out = torch.zeros_like(hidden_states)
    #     expert_cnt = gate_proj_weight.shape[0]
    #     for i in range(expert_cnt):
    #         gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
    #         up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
    #         mul = torch.nn.functional.silu(gate) * up
    #         down = torch.nn.functional.linear(mul, down_proj_weight[i])
    #         out += down * masked_routing_weight[:, i : i + 1]
    #     return out

    # @custom_moe_glu.register_fake
    # def custom_moe_glu_fake(
    #     hidden_states: torch.Tensor,
    #     gate_proj_weight: torch.Tensor,
    #     up_proj_weight: torch.Tensor,
    #     down_proj_weight: torch.Tensor,
    #     masked_routing_weight: torch.Tensor,
    #     expert_select_count: torch.Tensor,
    #     gate_proj_bias: torch.Tensor | None = None,
    #     up_proj_bias: torch.Tensor | None = None,
    #     down_proj_bias: torch.Tensor | None = None,
    #     dp_mask: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     return torch.empty_like(hidden_states)
    raise NotImplementedError


def multicast(
    x: torch.Tensor,  # [num_tokens, hidden_size]
    dp_rank: int,
) -> torch.Tensor:
    num_tokens, _ = x.shape
    x = x.reshape(1, -1, x.size(-1))

    assert (dp_metadata := get_forward_context().dp_metadata) is not None
    max_pad = dp_metadata.max_pads_across_dp.shape[0]
    num_repeat = max_pad // num_tokens

    x = x.repeat(num_repeat, 1, 1)
    x = x.reshape(1, max_pad, -1)

    if not envs.VLLM_RBLN_DP_INPUT_ALL_GATHER:
        all_buffer = None
        zeros = x - x
        for rank in range(get_dp_group().world_size):
            rank_tensor = x if rank == dp_rank else zeros
            all_buffer = (
                torch.cat((all_buffer, rank_tensor), dim=0)
                if all_buffer is not None
                else rank_tensor
            )

        return get_dp_group().all_reduce(all_buffer)
    else:
        return get_dp_group().all_gather(x, dim=0)


def get_tokens_mask(num_tokens: int, left=1.0, right=0.0) -> torch.Tensor:
    assert (dp_metadata := get_forward_context().dp_metadata) is not None
    num_tokens_across_dp = dp_metadata.max_tokens_across_dp_cpu.unsqueeze(1)

    max_pad = (
        num_tokens
        if num_tokens_across_dp.shape[0] == 1
        else dp_metadata.max_pads_across_dp.shape[0]
    )
    pos = torch.arange(max_pad, dtype=torch.int32).unsqueeze(0)

    tokens_mask = torch.where(pos < num_tokens_across_dp, left, right)
    tokens_mask = tokens_mask.reshape(-1, 1)
    return tokens_mask


@register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.apply",
    reason=(
        "Route unquantized MoE execution through the optimized RBLN MoE custom op, "
        "including expert_map, token mask, top-k/post-norm parameters. "
        "(PR#202, PR#511)"
    ),
    condition=lambda: envs.VLLM_RBLN_MOE_USE_OPT_KERNEL
    and envs.VLLM_RBLN_MOE_CUSTOM_KERNEL,
)
def patched_unquantized_fused_moe_method_optimized(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(w13 := layer.w13_weight, torch.Tensor)
    assert isinstance(w2 := layer.w2_weight, torch.Tensor)
    intermediate_size = w2.shape[-1]

    gate_proj_weight = w13[:, :intermediate_size, :]
    up_proj_weight = w13[:, intermediate_size:, :]
    down_proj_weight = w2

    orig_shape = x.shape
    num_tokens = orig_shape[:-1].numel()
    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    tokens_mask = (
        get_tokens_mask(num_tokens) if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK else None
    )

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        router_logits,
        layer.top_k,
        layer.renormalize,
        layer.expert_map,
        None,
        None,
        None,
        tokens_mask,
    )
    return final_hidden_states.reshape(orig_shape)


@register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.apply",
    reason=(
        "Provide an RBLN-compatible PyTorch fallback MoE implementation and force "
        "routing/tok-k computation to fp32 to avoid bf16 top-k correctness issues. "
        "(PR#192, PR#252)"
    ),
    condition=lambda: not envs.VLLM_RBLN_MOE_USE_OPT_KERNEL
    and not envs.VLLM_RBLN_MOE_CUSTOM_KERNEL,
)
def patched_unquantized_fused_moe_method_torch(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(w13 := layer.w13_weight, torch.Tensor)
    assert isinstance(w2 := layer.w2_weight, torch.Tensor)
    num_experts, _, hidden_size = w13.shape
    intermediate_size = w2.shape[-1]
    top_k = layer.top_k
    dtype = x.dtype
    orig_shape = x.shape
    num_tokens = x.shape[:-1].numel()

    # Routing -> dense weights [E, 1, N, 1]
    topk_weights = router_logits.softmax(dim=-1, dtype=torch.float)
    topk_weights, topk_idx = topk_weights.topk(top_k, dim=-1)
    if layer.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if layer.expert_map is not None:
        topk_idx = layer.expert_map[topk_idx]
    expert_idx = torch.arange(num_experts).reshape(num_experts, 1, 1, 1)
    mask = topk_idx.reshape(1, 1, num_tokens, top_k) == expert_idx
    expert_weights = (
        (topk_weights.reshape(1, 1, num_tokens, top_k) * mask)
        .sum(dim=-1, keepdim=True)
        .to(dtype)
    )

    h = x.reshape(1, num_tokens, hidden_size)
    # Force dependency between hidden_states and routing.
    dep = expert_weights[0]
    h = h + dep - dep

    out = h.new_zeros(1, num_tokens, hidden_size)
    for e in range(num_experts):
        y = F.linear(h, w13[e])
        y = F.silu(y[..., :intermediate_size]) * y[..., intermediate_size:]
        y = F.linear(y, w2[e])
        out = out + y * expert_weights[e]

    return out.reshape(orig_shape)


@register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_oot",
    reason=(
        "Override FusedMoE.forward_oot to support RBLN MoE DP execution: "
        "multicast hidden states across DP ranks, compute router logits after "
        "multicast, call the patched quant_method.apply path, and combine "
        "DP outputs. (PR#145)"
    ),
)
def patched_fused_moe_forward(
    self: FusedMoE,
    hidden_states: torch.Tensor,
    router: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    assert self.quant_method is not None
    if self.moe_parallel_config.dp_size > 1:
        raise NotImplementedError

    router_logits = router(hidden_states)

    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
    )

    return final_hidden_states
