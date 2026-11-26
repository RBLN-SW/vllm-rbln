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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)

_PACKED_FP4_WEIGHT_DTYPE = "float4_e2m1fn"


def apply_rbln_nvfp4_moe(
    layer: RoutedExperts,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Run NVFP4 experts through rbln_custom_ops.custom_moe.

    The layer holds packed FP4 weights (w13_weight, w2_weight), their e4m3
    block scales (w*_weight_scale) and the per-expert fp32 second-level
    scales (w13_weight_scale_2 [E, 2] for gate/up, w2_weight_scale_2 [E]),
    with the ModelOpt meaning: dequantised = fp4 * block_scale * scale_2.
    """
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
        return apply_rbln_nvfp4_moe(
            layer, x, router_logits, self.quant_config.group_size
        )

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError


class RBLNCompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsW4A4Nvfp4MoEMethod):
    """compressed-tensors NVFP4 MoE on the RBLN group-dequantise custom op.

    Loads under upstream's names, then hands custom_moe what the ModelOpt
    method does: the packed FP4 values and e4m3 block scales are the same,
    and compressed-tensors stores scale_2 as its reciprocal (global_scale).
    Like the ModelOpt method, it dequantises weights only, so the input
    global scales go unused.
    """

    def __init__(
        self,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        use_a16: bool = False,
    ) -> None:
        # Upstream __init__ picks a CUDA NVFP4 backend and raises without one.
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.group_size = 16
        self.use_a16 = use_a16
        self.nvfp4_backend = None
        self.experts_cls = None
        self.use_global_sf = False

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        def param(t: torch.Tensor) -> Parameter:
            return Parameter(t.data, requires_grad=False)

        layer.w13_weight = param(layer.w13_weight_packed)
        layer.w2_weight = param(layer.w2_weight_packed)
        del layer.w13_weight_packed
        del layer.w2_weight_packed
        layer.w13_weight_scale = param(layer.w13_weight_scale)
        layer.w2_weight_scale = param(layer.w2_weight_scale)
        # custom_moe takes gate and up scales apart, so unlike upstream the
        # two w13 halves keep their own.
        layer.w13_weight_scale_2 = param(1.0 / layer.w13_weight_global_scale)
        layer.w2_weight_scale_2 = param(1.0 / layer.w2_weight_global_scale)
        del layer.w13_weight_global_scale
        del layer.w2_weight_global_scale

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        return apply_rbln_nvfp4_moe(layer, x, router_logits, self.group_size)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError
