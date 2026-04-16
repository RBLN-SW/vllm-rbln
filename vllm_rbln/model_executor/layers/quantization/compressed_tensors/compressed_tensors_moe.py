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
import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe as upstream
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn.parameter import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.utils import set_weight_attrs

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe.layer import get_tokens_mask

logger = init_logger(__name__)


class CompressedTensorsW8A16Fp8MoEMethod(upstream.CompressedTensorsMoEMethod):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        assert self.strategy in (
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.BLOCK,
        ), (
            f"CompressedTensorsW8A16Fp8MoEMethod only supports strategies "
            f"CHANNEL, TENSOR, BLOCK, got {self.strategy}"
        )
        
        if self.strategy == QuantizationStrategy.TENSOR:
            raise NotImplementedError("Tensor strategy is not supported yet")
        self.weight_block_size = (
            weight_quant.block_structure
            if self.strategy == QuantizationStrategy.BLOCK
            else None
        )

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
        layer.weight_block_size = self.weight_block_size

        params_dtype = torch.float8_e4m3fn
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        tp_size = get_tensor_model_parallel_world_size()

        if self.strategy == QuantizationStrategy.BLOCK:
            block_n, block_k = self.weight_block_size[0], self.weight_block_size[1]
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
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

        if self.strategy == QuantizationStrategy.BLOCK:
            block_n, block_k = self.weight_block_size[0], self.weight_block_size[1]
            w13_scale_shape = (
                num_experts,
                w13_num_shards
                * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
            )
            w2_scale_shape = (
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
            )
            scale_quant_method = FusedMoeWeightScaleSupported.BLOCK.value
        elif self.strategy == QuantizationStrategy.CHANNEL:
            w13_scale_shape = (
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                1,
            )
            w2_scale_shape = (num_experts, hidden_size, 1)
            scale_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        else:  # TENSOR
            w13_scale_shape = (num_experts,)
            w2_scale_shape = (num_experts,)
            scale_quant_method = FusedMoeWeightScaleSupported.TENSOR.value

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(w13_scale_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update({"quant_method": scale_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(w2_scale_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

        if getattr(layer, "_expert_map", None) is not None:
            layer._expert_map_list = layer._expert_map.data.to(dtype=torch.int32).tolist()

    @property
    def is_monolithic(self) -> bool:
        return False

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        num_tokens = orig_shape[:-1].numel()
        hidden_states = x.reshape(num_tokens, -1)
        router_logits = router_logits.reshape(num_tokens, -1)

        intermediate_size = layer.w2_weight.shape[-1]

        gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
        up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
        down_proj_weight = layer.w2_weight

        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale

        if self.strategy == QuantizationStrategy.CHANNEL:
            scale_intermediate_size = w13_scale.shape[1] // 2
            gate_proj_weight_scale = w13_scale[:, :scale_intermediate_size, :]
            up_proj_weight_scale = w13_scale[:, scale_intermediate_size:, :]
            down_proj_weight_scale = w2_scale
            # When channel strategy, group_size is set to 0
            group_size = 0
        else:  # BLOCK
            scale_intermediate_size = w13_scale.shape[1] // 2
            gate_proj_weight_scale = w13_scale[:, :scale_intermediate_size, :]
            up_proj_weight_scale = w13_scale[:, scale_intermediate_size:, :]
            down_proj_weight_scale = w2_scale
            group_size = self.weight_block_size[1]

        e_score_correction_bias = getattr(layer, "e_score_correction_bias", None)

        expert_map_const = None
        if layer.expert_map is not None:
            expert_map_const = torch.tensor(layer._expert_map_list, dtype=torch.int32)

        tokens_mask = None
        if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK:
            tokens_mask = get_tokens_mask(num_tokens)

        if layer.use_grouped_topk:
            n_group = layer.num_expert_group
            topk_group = layer.topk_group
        else:
            n_group = None
            topk_group = None

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
                torch.tensor(group_size, dtype=torch.int32),
                layer.top_k,
                e_score_correction_bias,
                None,  # gate_proj_bias
                None,  # up_proj_bias
                None,  # down_proj_bias
                expert_map_const,
                tokens_mask,
                n_group,
                topk_group,
            )
        )

        return final_hidden_states.reshape(orig_shape)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    @property
    def supports_eplb(self) -> bool:
        return True


upstream.CompressedTensorsW8A16Fp8MoEMethod = CompressedTensorsW8A16Fp8MoEMethod
