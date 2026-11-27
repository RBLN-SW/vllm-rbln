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

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn.parameter import Parameter
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    modular_kernel,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    per_tensor_dequantize,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.moe = layer.moe_config
        self.quant_config = quant_config

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None

        from vllm_rbln.config import get_rbln_config

        self.use_w8a8 = get_rbln_config().use_w8a8

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
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
        assert isinstance(layer, RoutedExperts)
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
            fp8_dtype = current_platform.fp8_dtype()
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
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

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
                        ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    )
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

    def select_gemm_impl(
        self,
        prepare_finalize: modular_kernel.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> modular_kernel.FusedMoEExpertsModular:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # NOTE(RBLN): fp8 MoE implementation uses custom op
        # router_logits is now pre-computed masked_routing_weights
        # (topk + softmax already done in fused_moe_forward_rbln)
        return _block_fp8_moe(
            layer,
            x,
            router_logits,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            int(self.weight_block_size[1]),
            self.use_w8a8,
        )


class RBLNCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    """Block-fp8 compressed-tensors MoE on the RBLN fp8 custom_moe path.

    Upstream's weights and checkpoint names are kept; its fp8 kernel selection,
    which has no RBLN backend, is skipped. W8A16 or W8A8 follows use_w8a8, as
    for Fp8MoEMethod.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
    ) -> None:
        FusedMoEMethodBase.__init__(self, moe)
        assert weight_quant.strategy == QuantizationStrategy.BLOCK
        assert input_quant.dynamic
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.weight_block_size = weight_quant.block_structure
        self.block_quant = True
        self.static_input_scales = False

        from vllm_rbln.config import get_rbln_config

        self.use_w8a8 = get_rbln_config().use_w8a8

    @property
    def is_monolithic(self) -> bool:
        # RBLNMoERunner calls apply() directly; no modular kernel to build.
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            setattr(
                layer, name, Parameter(getattr(layer, name).data, requires_grad=False)
            )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        return _block_fp8_moe(
            layer,
            x,
            router_logits,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            int(self.weight_block_size[1]),
            self.use_w8a8,
        )

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError("RBLNMoERunner calls apply(), not apply_monolithic()")


def _block_fp8_moe(
    layer: RoutedExperts,
    x: torch.Tensor,
    masked_routing_weights: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_k: int,
    use_w8a8: bool,
) -> torch.Tensor:
    """custom_moe over block-fp8 experts; w13 stacks gate then up."""
    orig_shape = x.shape
    num_tokens = orig_shape[:-1].numel()
    hidden_states = x.reshape(num_tokens, -1)
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    # w13_scale: [E, 2 * scale_intermediate, scale_hidden]
    scale_intermediate_size = w13_scale.shape[1] // 2
    gate_proj_weight_scale = w13_scale[:, :scale_intermediate_size, :]
    up_proj_weight_scale = w13_scale[:, scale_intermediate_size:, :]
    swiglu_limit = layer.swiglu_limit or None

    if use_w8a8:
        fp8_dtype = torch.float8_e4m3fn
        finfo = torch.finfo(fp8_dtype)
        compute_dtype = hidden_states.dtype
        hs_shape = hidden_states.shape
        s_g = hidden_states.reshape(-1, block_k).to(torch.float32)
        amax = s_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        scale = amax / finfo.max
        hs_q = (s_g / scale).clamp(finfo.min, finfo.max).to(fp8_dtype).reshape(hs_shape)
        hs_scale = scale.reshape(hs_shape[0], hs_shape[1] // block_k)
        final_hidden_states = torch.ops.rbln_custom_ops.custom_moe(
            hs_q,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            masked_routing_weights,
            layer.activation.value,
            gate_proj_scale=gate_proj_weight_scale,
            up_proj_scale=up_proj_weight_scale,
            down_proj_scale=w2_scale,
            group_size=block_k,
            hidden_states_scale=hs_scale,
            out_dtype=compute_dtype,
            expert_map=layer.expert_map,
            swiglu_limit=swiglu_limit,
        )
    else:
        final_hidden_states = torch.ops.rbln_custom_ops.custom_moe(
            hidden_states,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            masked_routing_weights,
            layer.activation.value,
            gate_proj_scale=gate_proj_weight_scale,
            up_proj_scale=up_proj_weight_scale,
            down_proj_scale=w2_scale,
            group_size=block_k,
            expert_map=layer.expert_map,
            swiglu_limit=swiglu_limit,
        )

    return final_hidden_states.reshape(orig_shape)
