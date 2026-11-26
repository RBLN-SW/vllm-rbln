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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
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
        self.nvfp4_backend = None
        self.experts_cls = None
        self.use_global_sf = False

        group_size = quant_config.group_size
        if group_size not in (16, 32):
            raise ValueError(
                f"RBLN NVFP4 MoE requires {group_size=} in (16, 32); the packed "
                f"FP4 kernel takes no other group size"
            )

    @property
    def is_monolithic(self) -> bool:
        # Prevent vLLM from trying to initialize modular-kernel plumbing.
        # RBLNMoERunner.forward calls apply() directly.
        return True

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
        return _nvfp4_moe(layer, x, router_logits, self.quant_config.group_size)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError


class RBLNCompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsW4A4Nvfp4MoEMethod):
    """compressed-tensors NVFP4 MoE through the RBLN group-dequantise custom op.

    Upstream's weights and checkpoint names are kept. The activations stay
    unquantized (W4A16) whatever the checkpoint's input scheme, as for
    RBLNModelOptNvFp4FusedMoE.
    """

    def __init__(self, moe: FusedMoEConfig, use_a16: bool) -> None:
        FusedMoEMethodBase.__init__(self, moe)
        self.group_size = 16
        self.use_a16 = use_a16
        self.nvfp4_backend = None
        self.experts_cls = None
        self.use_global_sf = False

    @property
    def is_monolithic(self) -> bool:
        # RBLNMoERunner.forward calls apply() directly.
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # compressed-tensors stores the second-level scale as its reciprocal.
        for w in ("w13", "w2"):
            packed = getattr(layer, f"{w}_weight_packed").data
            global_scale = getattr(layer, f"{w}_weight_global_scale").data
            scale = getattr(layer, f"{w}_weight_scale").data
            for name in ("weight_packed", "weight_global_scale", "input_global_scale"):
                delattr(layer, f"{w}_{name}")
            setattr(layer, f"{w}_weight", Parameter(packed, requires_grad=False))
            setattr(layer, f"{w}_weight_scale", Parameter(scale, requires_grad=False))
            setattr(
                layer,
                f"{w}_weight_scale_2",
                Parameter(1.0 / global_scale, requires_grad=False),
            )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        return _nvfp4_moe(layer, x, router_logits, self.group_size)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError("RBLNMoERunner calls apply(), not apply_monolithic()")


def _nvfp4_moe(
    layer: RoutedExperts,
    x: torch.Tensor,
    masked_routing_weights: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """custom_moe over packed-fp4 experts; w13 stacks gate (w1) then up (w3)."""
    orig_shape = x.shape
    num_tokens = orig_shape[:-1].numel()
    hidden_states = x.reshape(num_tokens, -1)

    intermediate_size = layer.w13_weight.shape[1] // 2
    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    gate_proj_scale = layer.w13_weight_scale[:, :intermediate_size, :]
    up_proj_scale = layer.w13_weight_scale[:, intermediate_size:, :]

    out = torch.ops.rbln_custom_ops.custom_moe(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        layer.w2_weight,
        masked_routing_weights,
        layer.activation.value,
        gate_proj_scale=gate_proj_scale,
        up_proj_scale=up_proj_scale,
        down_proj_scale=layer.w2_weight_scale,
        group_size=group_size,
        expert_map=layer.expert_map,
        gate_proj_scale_2=layer.w13_weight_scale_2[:, 0].contiguous(),
        up_proj_scale_2=layer.w13_weight_scale_2[:, 1].contiguous(),
        down_proj_scale_2=layer.w2_weight_scale_2,
        weight_dtype=_PACKED_FP4_WEIGHT_DTYPE,
        swiglu_alpha=layer.swiglu_alpha,
        swiglu_limit=layer.swiglu_limit,
        swiglu_beta=layer.swiglu_beta,
    )
    return out.reshape(orig_shape)
