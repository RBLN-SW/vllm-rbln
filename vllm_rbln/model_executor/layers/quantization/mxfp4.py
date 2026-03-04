from typing import Callable, Optional, Union

import torch
import vllm.model_executor.layers.quantization.mxfp4 as upstream
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoEMethodBase)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe.layer import get_tokens_mask

logger = init_logger(__name__)


def _dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                      dtype: torch.dtype) -> torch.Tensor:
    """
    Args:
        blocks: uint8 [..., K // 2] containing packed FP4 values
        scales: uint8 [..., K // 32] containing E8M0 scales
        dtype: output dtype

    Returns:
        Dequantized tensor of shape [..., K]
    """
    # fmt: off
    FP4_VALUES = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    # fmt: on
    lut = torch.tensor(FP4_VALUES, dtype=dtype)

    # Convert E8M0 scales to exponents (subtract bias of 127)
    exponents = scales.to(torch.int32) - 127

    # Unpack FP4 nibbles
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    # Look up FP4 values
    val_lo = lut[idx_lo]
    val_hi = lut[idx_hi]

    # Interleave low and high nibbles
    *prefix_shape, B = blocks.shape
    out = torch.empty(*prefix_shape, B * 2, dtype=dtype)
    out[..., 0::2] = val_lo
    out[..., 1::2] = val_hi

    # Apply scales: each scale covers 32 elements
    # scales shape: [..., K // 32], out shape: [..., K]
    # Expand scales to match output shape
    exponents_expanded = exponents.unsqueeze(-1).expand(*exponents.shape, 32)
    exponents_expanded = exponents_expanded.reshape(*prefix_shape, -1)

    # ldexp: out * 2^exponents
    out = torch.ldexp(out, exponents_expanded[..., :out.shape[-1]])

    return out


def _swigluoai(gate: torch.Tensor, up: torch.Tensor, alpha: float,
               limit: float) -> torch.Tensor:
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


# kernel for gpt_oss, with built-in swigluoai activation
@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu_mxfp4",
    mutates_args=(),
)
def custom_moe_glu_mxfp4(
    hidden_states: torch.Tensor,
    gate_proj_blocks: torch.Tensor,
    gate_proj_scales: torch.Tensor,
    gate_proj_bias: torch.Tensor,
    up_proj_blocks: torch.Tensor,
    up_proj_scales: torch.Tensor,
    up_proj_bias: torch.Tensor,
    down_proj_blocks: torch.Tensor,
    down_proj_scales: torch.Tensor,
    down_proj_bias: torch.Tensor,
    router_logits: torch.Tensor,
    alpha: torch.Tensor,
    limit: torch.Tensor,
    k: int,
    post_norm: bool = True,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MoE GLU operation for GPT-OSS with mxfp4 quantization and swigluoai activation.

    Expected tensor shapes:
    - hidden_states: [num_tokens, hidden_size]
    - gate_proj_blocks: uint8 [num_experts, intermediate_size, hidden_size // 2]
    - gate_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_blocks: uint8 [num_experts, intermediate_size, hidden_size // 2]
    - up_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_blocks: uint8 [num_experts, hidden_size, intermediate_size // 2]
    - down_proj_scales: [num_experts, hidden_size, intermediate_size // 32]
    - down_proj_bias: [num_experts, hidden_size]
    - router_logits: [num_tokens, num_experts]
    - alpha: [], constant
    - limit: [], constant
    - expert_map: [num_experts],
      Mapping from global expert index to local expert index (in num_experts).
      Contains -1 for experts not assigned to the current rank.

    Returns:
        torch.Tensor: [num_tokens, hidden_size]
    """

    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.empty_like(hidden_states)

    # Reference torch native implementation

    num_tokens, hidden_size = hidden_states.shape
    num_local_experts = gate_proj_blocks.shape[0]
    # num_global_experts = router_logits.shape[1]
    dtype = hidden_states.dtype

    alpha_val = alpha.item()
    limit_val = limit.item()

    # Compute top-k routing
    # router_logits: [num_tokens, num_global_experts]
    top_k_values, top_k_indices = torch.topk(router_logits, k, dim=-1)

    # Apply softmax to get routing weights (only over selected experts)
    if post_norm:
        routing_weights = torch.softmax(top_k_values, dim=-1)
    else:
        # Pre-norm: softmax over all experts, then select top-k
        all_weights = torch.softmax(router_logits, dim=-1)
        routing_weights = torch.gather(all_weights,
                                       dim=-1,
                                       index=top_k_indices)

    # Initialize output
    output = torch.zeros(num_tokens, hidden_size, dtype=dtype)

    # Dequantize all expert weights once
    # gate_proj: [num_local_experts, intermediate_size, hidden_size]
    gate_proj_weights = _dequantize_mxfp4(gate_proj_blocks,
                                          gate_proj_scales,
                                          dtype=dtype)
    up_proj_weights = _dequantize_mxfp4(up_proj_blocks,
                                        up_proj_scales,
                                        dtype=dtype)
    down_proj_weights = _dequantize_mxfp4(down_proj_blocks,
                                          down_proj_scales,
                                          dtype=dtype)

    # Process each local expert
    for local_expert_idx in range(num_local_experts):
        # Determine which global expert this local expert corresponds to
        if expert_map is not None:
            # Find global expert index that maps to this local expert
            global_expert_idx = (expert_map == local_expert_idx).nonzero(
                as_tuple=True)[0]
            if len(global_expert_idx) == 0:
                continue
            global_expert_idx = global_expert_idx[0].item()
        else:
            global_expert_idx = local_expert_idx

        # Find tokens routed to this expert
        # top_k_indices: [num_tokens, k]
        expert_mask = (top_k_indices == global_expert_idx)  # [num_tokens, k]
        token_indices, k_indices = expert_mask.nonzero(as_tuple=True)

        if len(token_indices) == 0:
            continue

        # Get routing weights for these tokens
        weights = routing_weights[token_indices,
                                  k_indices]  # [num_selected_tokens]

        # Get hidden states for selected tokens
        selected_hidden = hidden_states[
            token_indices]  # [num_selected, hidden_size]

        # Get expert weights
        gate_w = gate_proj_weights[local_expert_idx]
        gate_b = gate_proj_bias[local_expert_idx]
        up_w = up_proj_weights[local_expert_idx]
        up_b = up_proj_bias[local_expert_idx]
        down_w = down_proj_weights[local_expert_idx]
        down_b = down_proj_bias[local_expert_idx]

        # Forward pass through expert MLP
        gate = selected_hidden @ gate_w.T + gate_b
        up = selected_hidden @ up_w.T + up_b
        activated = _swigluoai(gate, up, alpha_val, limit_val)
        expert_out = activated @ down_w.T + down_b  # [num_selected, hidden_size]

        # Apply routing weights and accumulate
        weighted_out = expert_out * weights.unsqueeze(-1)
        output.index_add_(0, token_indices, weighted_out.to(dtype))

    return output


@custom_moe_glu_mxfp4.register_fake
def custom_moe_glu_mxfp4_fake(
    hidden_states: torch.Tensor,
    gate_proj_blocks: torch.Tensor,
    gate_proj_scales: torch.Tensor,
    gate_proj_bias: torch.Tensor,
    up_proj_blocks: torch.Tensor,
    up_proj_scales: torch.Tensor,
    up_proj_bias: torch.Tensor,
    down_proj_blocks: torch.Tensor,
    down_proj_scales: torch.Tensor,
    down_proj_bias: torch.Tensor,
    router_logits: torch.Tensor,
    alpha: torch.Tensor,
    limit: torch.Tensor,
    k: int,
    post_norm: bool = True,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.moe = moe

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        # swigluoai constant value
        # gemm1_alpha = 1.702, gemm1_beta = 1.0, gemm1_clamp_limit = 7.0
        # gemm1_alpha = 1.702
        self.swiglu_alpha = torch.tensor(1.702, dtype=torch.float32)
        # gemm1_clamp_limit = 7.0
        self.swiglu_limit = torch.tensor(7.0, dtype=torch.float32)

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        assert isinstance(layer, FusedMoE)

        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = \
            intermediate_size_per_partition

        # NOTE: upstream rounds up intermediate_size_per_partition/hidden_size
        assert intermediate_size_per_partition % 64 == 0

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, FusedMoE)

        # w1
        layer.register_buffer("gate_proj_blocks",
                              layer.w13_weight.data[:, ::2])
        layer.register_buffer("gate_proj_scales",
                              layer.w13_weight_scale.data[:, ::2])
        layer.register_buffer("gate_proj_bias", layer.w13_bias.data[:, ::2])

        # w3
        layer.register_buffer("up_proj_blocks", layer.w13_weight.data[:, 1::2])
        layer.register_buffer("up_proj_scales",
                              layer.w13_weight_scale.data[:, 1::2])
        layer.register_buffer("up_proj_bias", layer.w13_bias.data[:, 1::2])

        # w2
        layer.register_buffer("down_proj_blocks", layer.w2_weight.data)
        layer.register_buffer("down_proj_scales", layer.w2_weight_scale.data)
        layer.register_buffer("down_proj_bias", layer.w2_bias.data)

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        # refer to custom_moe_glu
        orig_shape = x.shape  # noqa: F841
        num_tokens = orig_shape[:-1].numel()  # noqa: F841
        hidden_states = x.reshape(num_tokens, -1)
        router_logits = router_logits.reshape(num_tokens, -1)
        # x = x.view(-1, self.hidden_size)
        # router_logits = router_logits.view(-1, self.num_experts)
        # router_logits = router_logits.view(-1, self.moe.num_experts)

        if activation == "swigluoai":
            # TODO: use expert_map
            # FIXME(RBLN) - expert_map SHOULD be processed
            expert_map_const = None
            if expert_map is not None:
                # Extract numpy array and create a fresh constant tensor
                expert_map_list = expert_map.tolist()
                expert_map_const = torch.tensor(expert_map_list,
                                                dtype=torch.int32)

            use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
            if use_moe_tokens_mask:
                tokens_mask = get_tokens_mask(num_tokens)
                router_logits = router_logits + tokens_mask

            final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu_mxfp4(
                hidden_states,
                layer.gate_proj_blocks,
                layer.gate_proj_scales,
                layer.gate_proj_bias,
                layer.up_proj_blocks,
                layer.up_proj_scales,
                layer.up_proj_bias,
                layer.down_proj_blocks,
                layer.down_proj_scales,
                layer.down_proj_bias,
                router_logits,
                self.swiglu_alpha,
                self.swiglu_limit,
                top_k,
                renormalize,
                expert_map_const,
            )
        else:
            raise NotImplementedError(activation)

        return final_hidden_states.reshape(orig_shape)


# We do this because upstream uses Mxfp4MoEMethod for all non-xpu platforms
# and it doesn't expose interface for OOT kernels.
upstream.Mxfp4MoEMethod = Mxfp4MoEMethod
