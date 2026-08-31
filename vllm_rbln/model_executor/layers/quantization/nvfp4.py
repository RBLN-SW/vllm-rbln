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

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)

_PACKED_FP4_WEIGHT_DTYPE = "float4_e2m1fn"


class RBLNModelOptNvFp4FusedMoE(ModelOptNvFp4FusedMoE):
    """NVFP4 MoE that dequantises through the RBLN group-dequantise custom op."""

    def __init__(self, quant_config: ModelOptNvFp4Config, moe: FusedMoEConfig) -> None:
        FusedMoEMethodBase.__init__(self, moe)
        self.quant_config = quant_config
        self.moe = moe
        self.use_a16 = quant_config.quant_method == "W4A16_NVFP4"
        self.use_global_sf = False

    @property
    def is_monolithic(self) -> bool:
        # RBLNMoERunner.forward calls apply() directly;
        return True

    def maybe_make_prepare_finalize(self, routing_tables=None):
        raise RuntimeError

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        layer.w13_weight_scale_2 = Parameter(
            layer.w13_weight_scale_2.data, requires_grad=False
        )
        layer.w2_weight_scale_2 = Parameter(
            layer.w2_weight_scale_2.data, requires_grad=False
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
            torch.tensor(self.quant_config.group_size, dtype=torch.int32),
            layer.activation.value,
            None,  # gate_proj_bias
            None,  # up_proj_bias
            None,  # down_proj_bias
            layer.expert_map,
            gate_proj_scale_2=layer.w13_weight_scale_2[:, 0].contiguous(),
            up_proj_scale_2=layer.w13_weight_scale_2[:, 1].contiguous(),
            down_proj_scale_2=layer.w2_weight_scale_2,
            weight_dtype=_PACKED_FP4_WEIGHT_DTYPE,
        )
        return out.reshape(orig_shape)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError
