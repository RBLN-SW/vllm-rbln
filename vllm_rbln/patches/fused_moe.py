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
    scoring_func: str,
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
    scoring_func: str,
    topk: int,
    post_norm: bool,
    expert_map: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def multicast(
    x: torch.Tensor,
    dp_rank: int,
) -> torch.Tensor:
    """Gather every DP rank's hidden_states into ``[dp_size, max_pad, H]``.

    Args:
        x: Local hidden_states of any shape with last dim ``H``. ``max_pad``
            (from ``RBLNDPMetadata``) must be a multiple of the token count.
        dp_rank: Caller's rank within the DP group.

    Returns:
        Tensor of shape ``[dp_size, max_pad, H]``, identical on every rank.
        Padded positions carry duplicate tokens - mask them out before
        consuming (see ``get_tokens_mask``).
    """
    x = x.reshape(1, -1, x.size(-1))  # [1, num_tokens, H]
    _, num_tokens, _ = x.shape

    assert (dp_metadata := get_forward_context().dp_metadata) is not None
    max_pad = dp_metadata.max_pads_across_dp.shape[0]
    num_repeat = max_pad // num_tokens

    # TODO(RBLN): evaluate various padding approaches
    x = x.repeat(num_repeat, 1, 1)
    x = x.reshape(1, max_pad, -1)  # [1, max_pad, H]

    return get_dp_group().all_gather(x, dim=0)


def get_tokens_mask(num_tokens: int, left=1.0, right=0.0) -> torch.Tensor:
    """Real-vs-padding mask aligned with the DP multicast output layout.

    For every DP rank's slot in the multicast buffer, positions before
    that rank's actual token count get ``left`` and the reset get ``right``.
    Multiply this mask into routing weights (default ``(1.0, 0.0)``) or,
    with ``(0.0, float('-inf'))``, add it to router logits before softmax
    to suppress padded positions.

    In the DP=1 path no padding exists, so ``num_tokens`` is used as the
    pad length and the result is all ``left`` (effectively a no-op).

    Example:
        DP=2, ``max_pad=4``, rank 0 has 3 real tokens, rank 1 has 2.
        With defaults ``(left=1.0, right=0.0)``::

            rank 0 slot: [1.0, 1.0, 1.0, 0.0]
            rank 1 slot: [1.0, 1.0, 0.0, 0.0]

        Flattened return: ``[[1.0],[1.0],[1.0],[0.0],[1.0],[1.0],[0.0],[0.0]]``
        with shape ``[8, 1]``.

    Args:
        num_tokens: Used as ``max_pad`` only when DP=1 (where the metadata's
            ``max_pads_across_dp`` is ``None``); ignored otherwise
        left: Value for real-token positions.
        right: Value for padded positions.

    Returns:
        Tensor of shape ``[dp_size * max_pad, 1]``.
    """
    assert (dp_metadata := get_forward_context().dp_metadata) is not None
    num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.unsqueeze(1)

    max_pad = (
        num_tokens
        if num_tokens_across_dp.shape[0] == 1  # DP=1
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

    # Pre-score routing inputs at caller side; compiler custom op routing
    # expects already-scored values (no sigmoid applied inside the kernel).
    assert layer.scoring_func is not None, "scoring_func must be set"
    assert layer.scoring_func in {"softmax", "sigmoid"}
    if layer.scoring_func == "sigmoid":
        router_logits = torch.sigmoid(router_logits.to(torch.float32)).to(
            router_logits.dtype
        )

    tokens_mask = (
        get_tokens_mask(num_tokens) if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK else None
    )

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        router_logits,
        layer.scoring_func,
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
    target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward",
    reason=(
        "Override FusedMoE.forward to support RBLN MoE DP execution: "
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
    if self.moe_parallel_config.dp_size > 1:
        org_hidden_shape = hidden_states.shape

        # NOTE(RBLN): DP gather
        # Each rank holds only its own tokens, but the MoE experts are sharded
        # across DP ranks. Replicate every rank's hidden_states to every rank
        # so each can route the full batch through its local expert shard.
        #   [num_tokens, H] -> [dp_size, max_pad, H]
        # max_pad is agreed collectively in RBLNDPMetadata and is identical on every
        # rank, sh shapes match for the all_reduce below.
        hidden_states = multicast(hidden_states, self.moe_parallel_config.dp_rank)

    router_logits = router(hidden_states)

    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
    )

    if self.moe_parallel_config.dp_size > 1:
        # NOTE(RBLN): DP combine
        # Outputs are partial (each rank only ran its local expert shard over the full
        # batch). Sum across DP ranks, then keep only this rank's original token slice.

        # flatten to [dp_size * max_pad, 1, H]
        all_hidden_states = final_hidden_states.reshape(-1, 1, org_hidden_shape[-1])
        assert (all_hidden_states.shape[0] % self.moe_parallel_config.dp_size) == 0

        # sum-across-DP and scatter-by-dp_size in a single collective:
        # this rank receives only its own [max_pad, 1, H] slice of the sum.
        hidden_states = get_dp_group().reduce_scatter(all_hidden_states, dim=0)
        assert (
            hidden_states.shape[0]
            == get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
        )

        num_tokens = org_hidden_shape[:-1].numel()
        final_hidden_states = hidden_states[:num_tokens].reshape(org_hidden_shape)

    return final_hidden_states
