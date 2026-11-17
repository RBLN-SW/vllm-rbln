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
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptMxFp8FusedMoE,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

# MXFP8: fp8 (e4m3) weights with one e8m0 scale per 32 input elements.
MXFP8_GROUP_SIZE = 32


@register_weight_loader_v2_supported_method
class RBLNModelOptMxFp8LinearMethod(LinearMethodBase):
    """ModelOpt MXFP8 linear for RBLN."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if input_size_per_partition % MXFP8_GROUP_SIZE != 0:
            raise ValueError(
                f"MXFP8 needs the input size ({input_size_per_partition}) to be a "
                f"multiple of the group size {MXFP8_GROUP_SIZE}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_GROUP_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        # bf16 shares E8M0's 8-bit exponent, so every scale converts exactly.
        layer.weight_scale = Parameter(
            layer.weight_scale.data.view(torch.float8_e8m0fnu).to(torch.bfloat16),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_features, in_features = layer.weight.shape
        weight = (
            layer.weight.view(
                out_features, in_features // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE
            ).to(x.dtype)
            * layer.weight_scale.to(x.dtype)[:, :, None]
        ).view(out_features, in_features)
        return torch.nn.functional.linear(x, weight, bias)


class RBLNModelOptMxFp8FusedMoE(ModelOptMxFp8FusedMoE):
    """MXFP8 MoE that runs the RBLN group-dequantise custom op.

    The e4m3 weights and raw uint8 e8m0 scales go to the op as loaded; the
    uint8 scale is what routes it to the packed MoE kernel, which dequantises
    per 32-element group on the device (W8A16).
    """

    def __init__(self, quant_config: ModelOptMxFp8Config, moe: FusedMoEConfig) -> None:
        # Do not call ModelOptMxFp8FusedMoE.__init__(): it selects an upstream
        # CUDA/ROCm MXFP8 backend.
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_block_size = [1, MXFP8_GROUP_SIZE]
        self.quant_config = quant_config
        self.mxfp8_backend = None
        self.experts_cls = None

    @property
    def is_monolithic(self) -> bool:
        # Prevent vLLM from trying to initialize modular-kernel plumbing.
        # RBLNMoERunner.forward calls apply() directly.
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        self._check_weight_dtypes(layer)
        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        orig_shape = x.shape
        num_tokens = orig_shape[:-1].numel()
        hidden_states = x.reshape(num_tokens, -1)
        masked_routing_weights = router_logits

        # w13 stacks gate (w1) then up (w3) along the output dim.
        intermediate_size = layer.w13_weight.shape[1] // 2
        gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
        up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
        gate_proj_scale = layer.w13_weight_scale[:, :intermediate_size, :]
        up_proj_scale = layer.w13_weight_scale[:, intermediate_size:, :]

        out = torch.ops.rbln_custom_ops.custom_moe_glu_group_dequantize(
            hidden_states,
            gate_proj_weight,
            gate_proj_scale,
            up_proj_weight,
            up_proj_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            masked_routing_weights,
            torch.tensor(MXFP8_GROUP_SIZE, dtype=torch.int32),
            layer.activation.value,
            None,  # gate_proj_bias
            None,  # up_proj_bias
            None,  # down_proj_bias
            layer.expert_map,
            swiglu_alpha=layer.swiglu_alpha,
            swiglu_limit=layer.swiglu_limit,
            swiglu_beta=layer.swiglu_beta,
        )
        return out.reshape(orig_shape)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError


class RBLNModelOptMxFp8Config(ModelOptMxFp8Config):
    """ModelOpt MXFP8 config for RBLN: routed experts take the RBLN MoE method."""

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts) and not self.is_layer_excluded(prefix):
            return RBLNModelOptMxFp8FusedMoE(self, layer.moe_config)
        return super().get_quant_method(layer, prefix)
