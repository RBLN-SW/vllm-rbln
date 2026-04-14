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
import vllm.model_executor.layers.quantization.fp8 as upstream
from torch.nn.parameter import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    maybe_post_process_fp8_weight_block,
    process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    cutlass_fp8_supported,
    per_tensor_dequantize,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe.layer import get_tokens_mask

logger = init_logger(__name__)


class RBLNW8A16BlockFp8LinearOp:
    """
    This class executes a RBLN Blocked FP8 linear layer
    """

    def __init__(
        self,
        weight_group_shape: GroupShape,
        act_quant_group_shape: GroupShape,
    ):
        self.weight_group_shape = weight_group_shape
        self.act_quant_group_shape = act_quant_group_shape
        logger.info(
            "RBLN W8A16 block fp8 weight group shape = %s", self.weight_group_shape
        )
        logger.info(
            "RBLN W8A16 block fp8 act quant group shape = %s",
            self.act_quant_group_shape,
        )
        assert self.act_quant_group_shape == GroupShape(1, 128)

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # FIXME(RBLN) - REBEL evt0 DOES NOT support w8a8
        # current implementation is based on w8a16 (imported from optimum minimax2)
        # TODO(RBLN) - REBEL evt1 SHOULD support w8a8 fp8 linear operation
        # further implementation will be basedon on w8a8
        return self._w8a16_block_fp8_matmul(
            input,
            weight,
            weight_scale,
            list(self.weight_group_shape),
            input_scale,
            bias,
        )

    def _w8a16_block_fp8_matmul(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: list[int],
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _input_dtype = input.dtype
        out_features, in_features = weight.shape
        bs0, bs1 = int(block_size[0]), int(block_size[1])
        out_blocks = out_features // bs0
        in_blocks = in_features // bs1

        weight = weight.view(out_blocks, bs0, in_blocks, bs1).to(_input_dtype)
        weight_scale = weight_scale.view(out_blocks, in_blocks).to(_input_dtype)
        scaled_weight = (weight * weight_scale[:, None, :, None]).reshape(
            out_features, in_features
        )
        output = torch.nn.functional.linear(input, scaled_weight, bias)
        return output


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None
        self.use_marlin = False
        self.use_deep_gemm = False

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        if self.weight_block_size:
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
        else:
            self.act_q_group_shape = GroupShape.PER_TENSOR

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            self.w8a8_block_fp8_linear = RBLNW8A16BlockFp8LinearOp(
                weight_group_shape=GroupShape(*self.weight_block_size),
                act_quant_group_shape=self.act_q_group_shape,
            )
        else:
            # Use per-token quantization for better perf if dynamic and cutlass
            if self.act_q_static:
                activation_quant_key = kFp8StaticTensorSym
            elif cutlass_fp8_supported():
                activation_quant_key = kFp8DynamicTokenSym
            else:
                activation_quant_key = kFp8DynamicTensorSym

            self.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=activation_quant_key,
                weight_quant_key=kFp8StaticTensorSym,
                out_dtype=torch.get_default_dtype(),
                module_name=self.__class__.__name__,
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        if self.quant_config.is_checkpoint_fp8_serialized:
            weight = create_fp8_weight_parameter(
                output_size_per_partition, input_size_per_partition, weight_loader
            )
        else:
            # For non-serialized checkpoints, use original dtype
            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                scale = create_fp8_scale_parameter(
                    PerTensorScaleParameter,
                    output_partition_sizes,
                    input_size_per_partition,
                    None,
                    weight_loader,
                )
                if not hasattr(scale, "scale_type"):
                    set_weight_attrs(scale, {"scale_type": "weight_scale"})
                layer.register_parameter("weight_scale", scale)
            else:
                assert not self.act_q_static
                assert self.weight_block_size is not None
                scale = create_fp8_scale_parameter(
                    BlockQuantScaleParameter,
                    output_partition_sizes,
                    input_size_per_partition,
                    self.weight_block_size,
                    weight_loader,
                )
                if not hasattr(scale, "scale_type"):
                    set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.act_q_static:
                scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
                if not hasattr(scale, "scale_type"):
                    set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # size_k_first = True
        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static
            # size_k_first = False

            weight, weight_scale = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale_inv
            )
            # Delete the weight_scale_inv parameter to avoid confusion
            # with the weight_scale parameter
            del layer.weight_scale_inv

        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = upstream.ops.scaled_fp8_quant(
                layer.weight, scale=None
            )
            weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            if not self.use_marlin:
                weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                    weight,
                    weight_scale,
                    layer.logical_widths,
                    getattr(layer, "input_scale", None),
                )
                if self.act_q_static:
                    assert input_scale is not None
                    input_scale = input_scale.max()
            weight = weight.t()

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = (
            Parameter(input_scale, requires_grad=False)
            if input_scale is not None
            else None
        )

        if self.block_quant:
            maybe_post_process_fp8_weight_block(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # use BF16 dequant
        # fp8 -> dequantize bf16 -> bf16 torch.nn.functional.linear
        if True:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.w8a8_block_fp8_linear.apply(
                    input=x,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    input_scale=layer.input_scale,
                    bias=bias,
                )
            else:
                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        if self.block_quant:
            assert self.weight_block_size is not None

            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_swiglu_group_dequantize",
    mutates_args=(),
)
def custom_moe_swiglu_group_dequantize(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    gate_proj_scale: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_scale: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_scale: torch.Tensor,
    router_logits: torch.Tensor,
    group_size: torch.Tensor,
    topk: int,
    e_score_correction_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Customized MoE SwiGLU operation.

    Expected tensor shapes:
    - hidden_states: [batch*seq_len, hidden_size]
    - gate_proj_weight: [num_experts, hidden_size, intermediate_size]
    - gate_proj_scale: [num_experts, intermediate_size, hidden_size // 128]
    - up_proj_weight: [num_experts, hidden_size, intermediate_size]
    - up_proj_scale: [num_experts, intermediate_size, hidden_size // 128]
    - down_proj_weight: [num_experts, intermediate_size, hidden_size]
    - down_proj_scale: [num_experts, hidden_size, intermediate_size // 128]
    - router_logits: [batch*seq_len, num_experts]
    - group_size: group size for weight scale
    - topk: top k experts to select
    - e_score_correction_bias:
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_bias: [num_experts, hidden_size]

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """

    def _dequantize_blockwise_weight(
        weight: torch.Tensor,
        scale: torch.Tensor,
        in_block_size: int,
        out_block_size: int | None = None,
    ) -> torch.Tensor:
        # `weight` is [num_experts, out_features, in_features].
        # `scale` is [num_experts, out_blocks, in_blocks].
        out_features = weight.shape[1]
        in_features = weight.shape[2]
        out_blocks = scale.shape[1]
        out_block_size = out_block_size or (
            (out_features + out_blocks - 1) // out_blocks
        )

        expanded = scale.repeat_interleave(out_block_size, dim=1).repeat_interleave(
            in_block_size, dim=2
        )
        expanded = expanded[:, :out_features, :in_features]
        return weight.to(hidden_states.dtype) * expanded.to(hidden_states.dtype)

    in_block_size = int(group_size.item())
    gate_out_block = (
        gate_proj_weight.shape[1] + gate_proj_scale.shape[1] - 1
    ) // gate_proj_scale.shape[1]
    down_out_block = (
        down_proj_weight.shape[1] + down_proj_scale.shape[1] - 1
    ) // down_proj_scale.shape[1]

    gate_proj_weight_dq = _dequantize_blockwise_weight(
        gate_proj_weight, gate_proj_scale, in_block_size, gate_out_block
    )
    up_proj_weight_dq = _dequantize_blockwise_weight(
        up_proj_weight, up_proj_scale, in_block_size, gate_out_block
    )
    down_proj_weight_dq = _dequantize_blockwise_weight(
        down_proj_weight, down_proj_scale, in_block_size, down_out_block
    )

    routing_weights = router_logits.float()
    scores_for_choice = routing_weights
    if e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + e_score_correction_bias

    _, topk_ids = torch.topk(scores_for_choice, topk, dim=-1, sorted=False)
    topk_weights = routing_weights.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    topk_weights = topk_weights.to(hidden_states.dtype)

    if dp_mask is not None:
        topk_weights = topk_weights * dp_mask.to(topk_weights.dtype)

    num_experts = gate_proj_weight_dq.shape[0]
    if expert_map is not None:
        safe_expert_map = torch.where(expert_map < 0, num_experts - 1, expert_map).to(
            topk_ids.dtype
        )
        topk_ids = safe_expert_map[topk_ids]

    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(
        topk_ids, num_classes=num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        topk_slot, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue

        current_state = hidden_states[token_idx]
        gate = torch.nn.functional.linear(
            current_state,
            gate_proj_weight_dq[expert_idx],
            gate_proj_bias[expert_idx] if gate_proj_bias is not None else None,
        )
        up = torch.nn.functional.linear(
            current_state,
            up_proj_weight_dq[expert_idx],
            up_proj_bias[expert_idx] if up_proj_bias is not None else None,
        )
        swiglu = torch.nn.functional.silu(gate) * up
        down = torch.nn.functional.linear(
            swiglu,
            down_proj_weight_dq[expert_idx],
            down_proj_bias[expert_idx] if down_proj_bias is not None else None,
        )
        current_hidden_states = down * topk_weights[token_idx, topk_slot, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


@custom_moe_swiglu_group_dequantize.register_fake
def custom_moe_swiglu_group_dequantize_fake(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    gate_proj_scale: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_scale: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_scale: torch.Tensor,
    router_logits: torch.Tensor,
    group_size: torch.Tensor,
    topk: int,
    e_score_correction_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.moe = layer.moe_config
        self.quant_config = quant_config

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        self.rocm_aiter_moe_enabled = False

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, FusedMoE)
        # Lazy import to avoid importing triton too early.

        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            w13_weight = layer.w13_weight.data
            w13_weight_scale_inv = layer.w13_weight_scale_inv.data
            w2_weight = layer.w2_weight
            w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(
                w13_weight_scale_inv, requires_grad=False
            )
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(
                w2_weight_scale_inv, requires_grad=False
            )
        # If checkpoint is fp16, quantize in place.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = upstream.current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.local_num_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    upstream.ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    upstream.ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = upstream.rocm_aiter_ops.shuffle_weights(
                    layer.w13_weight, layer.w2_weight
                )

                layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)
        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    logger.warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer."
                    )
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    layer.w13_weight[expert_id][start : start + shard_size, :], _ = (
                        upstream.ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id]
                        )
                    )
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # NOTE(RBLN): fp8 MoE implementation uses custom op
        orig_shape = x.shape  # noqa: F841
        num_tokens = orig_shape[:-1].numel()  # noqa: F841
        hidden_states = x.reshape(num_tokens, -1)
        router_logits = router_logits.reshape(num_tokens, -1)
        router_logits = torch.sigmoid(router_logits)

        intermediate_size = layer.w2_weight.shape[-1]

        # w13_weight: merged gate(up) weights, w2_weight: down weights
        gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
        up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
        down_proj_weight = layer.w2_weight

        down_proj_weight_scale = layer.w2_weight_scale_inv

        # w13_weight_scale_inv: [E, 2 * scale_intermediate, scale_hidden]
        scale_intermediate_size = layer.w13_weight_scale_inv.shape[1] // 2
        gate_proj_weight_scale = layer.w13_weight_scale_inv[
            :, :scale_intermediate_size, :
        ]
        up_proj_weight_scale = layer.w13_weight_scale_inv[
            :, scale_intermediate_size:, :
        ]

        e_score_correction_bias = kwargs.get("e_score_correction_bias")
        if e_score_correction_bias is None:
            e_score_correction_bias = getattr(layer, "e_score_correction_bias", None)

        expert_map_const = None
        if layer.expert_map is not None:
            expert_map_const = layer.expert_map
            if expert_map_const.dtype != torch.int32:
                expert_map_const = expert_map_const.to(dtype=torch.int32)
            expert_map_const = expert_map_const.detach().clone()

        tokens_mask = None
        use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
        if use_moe_tokens_mask:
            tokens_mask = get_tokens_mask(num_tokens)

        final_hidden_states = (
            torch.ops.rbln_custom_ops.custom_moe_swiglu_group_dequantize(
                hidden_states,
                gate_proj_weight,
                gate_proj_weight_scale,
                up_proj_weight,
                up_proj_weight_scale,
                down_proj_weight,
                down_proj_weight_scale,
                router_logits,
                torch.tensor(self.weight_block_size[1], dtype=torch.int32),
                layer.top_k,
                e_score_correction_bias,
                None,  # gate_proj_bias
                None,  # up_proj_bias
                None,  # down_proj_bias
                expert_map_const,
                tokens_mask,
            )
        )

        return final_hidden_states.reshape(orig_shape)


# fp8 linear method - w8a8, custom fp8 linear operation
upstream.Fp8LinearMethod = Fp8LinearMethod
# fp8 moe method - based on fused_moe/layer.py, custom_moe_glu
upstream.Fp8MoEMethod = Fp8MoEMethod
