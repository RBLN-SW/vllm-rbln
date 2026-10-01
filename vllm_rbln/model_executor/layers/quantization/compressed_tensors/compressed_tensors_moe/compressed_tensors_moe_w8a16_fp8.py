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
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn.parameter import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs


class CompressedTensorsW8A16Fp8MoEMethod(CompressedTensorsMoEMethod):
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
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # NOTE: hidden_size and intermediate_size_per_partition are read-only
        # @property on FusedMoE in vLLM 0.22 (no setter); the local args are
        # used directly below. Only assign the settable attributes here.
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

        w13_scale_shape: tuple[int, ...]
        w2_scale_shape: tuple[int, ...]
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

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    @property
    def is_monolithic(self) -> bool:
        return False

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Routing (topk + scoring + optional grouped-topk) is computed upstream
        # in fused_moe_forward_rbln; router_logits already holds the pre-computed
        # [E, T] masked routing weights (matches dev-0.22.0 Fp8MoEMethod.apply).
        orig_shape = x.shape
        num_tokens = orig_shape[:-1].numel()
        hidden_states = x.reshape(num_tokens, -1)
        masked_routing_weights = router_logits

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
            group_size = 0
        else:  # BLOCK
            scale_intermediate_size = w13_scale.shape[1] // 2
            gate_proj_weight_scale = w13_scale[:, :scale_intermediate_size, :]
            up_proj_weight_scale = w13_scale[:, scale_intermediate_size:, :]
            down_proj_weight_scale = w2_scale
            group_size = self.weight_block_size[1]

        final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu_group_dequantize(
            hidden_states,
            gate_proj_weight,
            gate_proj_weight_scale,
            up_proj_weight,
            up_proj_weight_scale,
            down_proj_weight,
            down_proj_weight_scale,
            masked_routing_weights,
            torch.tensor(group_size, dtype=torch.int32),
            layer.activation.value,
            None,  # gate_proj_bias
            None,  # up_proj_bias
            None,  # down_proj_bias
            layer.expert_map,
        )

        return final_hidden_states.reshape(orig_shape)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    @property
    def supports_eplb(self) -> bool:
        return True
