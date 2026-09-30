# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 quantization for RBLN: block-FP8 linears and MXFP4 routed experts.

Both keep the checkpoint's packed weights and fold its ue8m0 (e8m0fnu) scales, once at load,
into bf16 scales expanded along the output channels:

* FP8 linear: weight [N, K] e4m3 with a [N / 128, K / 128] block scale -> a [N, K / 128] bf16
  scale (each block row repeated over its 128 output channels), dequantized as
  [N, K / 128, 128] * [N, K / 128, 1] (W8A16, the compiler's group-128 fp8 dense).
* MXFP4 experts: w13 / w2 [E, N, K / 2] packed e2m1 with a [E, N, K / 32] scale (already per
  output channel) -> bf16, through ``custom_moe_glu_group_dequantize`` (group 32). V4's
  ``silu(gate.clamp(max=L)) * up.clamp(+-L)`` is the op's SwiGLU-OAI with alpha 1, beta 0.

An e8m0 byte e is 2^(e - 127) exactly; a bf16 whose exponent field is e and mantissa zero is that
power of two, so the fold is a shift, lossless.
"""

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

_PACKED_FP4_WEIGHT_DTYPE = "float4_e2m1fn"
MXFP4_GROUP_SIZE = 32


def e8m0_to_bf16(scale: torch.Tensor) -> torch.Tensor:
    """e8m0 bytes (uint8 or float8_e8m0fnu) -> the same powers of two in bf16."""
    raw = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    device = raw.device
    return (raw.to("cpu").to(torch.int16) << 7).view(torch.bfloat16).to(device)


class RBLNDeepseekV4BlockFp8LinearMethod(Fp8LinearMethod):
    """Block-FP8 (128 x 128, ue8m0) linear as W8A16 with a folded per-output-row scale."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        out_features = weight.shape[0]
        block_n = int(self.weight_block_size[0])
        # Folded on the host: the weights may already sit on the RBLN device, where an eager
        # op (repeat_interleave, a strided slice) is not runnable at compile time.
        device = layer.weight_scale_inv.device
        scale = layer.weight_scale_inv.data.to("cpu")
        if scale.dtype == torch.float8_e8m0fnu or scale.dtype == torch.uint8:
            scale = e8m0_to_bf16(scale)
        else:
            scale = scale.to(torch.bfloat16)
        scale = scale.repeat_interleave(block_n, dim=0)[:out_features].contiguous().to(device)
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        del layer.weight_scale_inv

    def dequantized_weight(self, layer: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
        out_features, in_features = layer.weight.shape
        block_k = int(self.weight_block_size[1])
        return (
            layer.weight.view(out_features, in_features // block_k, block_k).to(dtype)
            * layer.weight_scale.to(dtype)[:, :, None]
        ).view(out_features, in_features)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dequantized_weight(layer, x.dtype), bias)


class RBLNDeepseekV4Mxfp4MoEMethod(Mxfp4MoEMethod):
    """MXFP4 routed experts through the RBLN group-dequantise MoE op."""

    def __init__(self, moe: FusedMoEConfig, swiglu_limit: float) -> None:
        # Skip Mxfp4MoEMethod.__init__: it selects a CUDA / Triton backend.
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_dtype = "mxfp4"
        self.swiglu_limit = float(swiglu_limit)

    @property
    def is_monolithic(self) -> bool:
        # RBLNMoERunner.forward calls apply() directly.
        return True

    @property
    def skip_forward_padding(self) -> bool:
        return False

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, act_dtype,
                            moe_parallel_config):
        return FusedMoEMethodBase.maybe_roundup_sizes(
            self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            e8m0_to_bf16(layer.w13_weight_scale.data), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            e8m0_to_bf16(layer.w2_weight_scale.data), requires_grad=False
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        orig_shape = x.shape
        hidden_states = x.reshape(orig_shape[:-1].numel(), -1)
        # router_logits is the runner's [E, T] masked routing weights.
        intermediate_size = layer.w13_weight.shape[1] // 2
        out = torch.ops.rbln_custom_ops.custom_moe_glu_group_dequantize(
            hidden_states,
            layer.w13_weight[:, :intermediate_size, :],
            layer.w13_weight_scale[:, :intermediate_size, :],
            layer.w13_weight[:, intermediate_size:, :],
            layer.w13_weight_scale[:, intermediate_size:, :],
            layer.w2_weight,
            layer.w2_weight_scale,
            router_logits,
            torch.tensor(MXFP4_GROUP_SIZE, dtype=torch.int32),
            "swigluoai",
            None,  # gate_proj_bias
            None,  # up_proj_bias
            None,  # down_proj_bias
            layer.expert_map,
            weight_dtype=_PACKED_FP4_WEIGHT_DTYPE,
            swiglu_alpha=1.0,
            swiglu_limit=self.swiglu_limit,
            swiglu_beta=0.0,
        )
        return out.reshape(orig_shape)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError


class RBLNDeepseekV4FP8Config(DeepseekV4FP8Config):
    """``deepseek_v4_fp8`` for RBLN: the two methods above, nothing CUDA."""

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        skipped = is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        )
        if isinstance(layer, RoutedExperts):
            if self.expert_dtype != "fp4":
                raise NotImplementedError(
                    "RBLN DeepSeek-V4 runs the MXFP4-expert checkpoints only; got "
                    f"expert_dtype={self.expert_dtype!r}"
                )
            if skipped:
                raise NotImplementedError(f"unquantized routed experts at {prefix}")
            from vllm.config import get_current_vllm_config

            hf_config = get_current_vllm_config().model_config.hf_text_config
            return RBLNDeepseekV4Mxfp4MoEMethod(layer.moe_config, hf_config.swiglu_limit)
        if isinstance(layer, LinearBase):
            if skipped:
                return UnquantizedLinearMethod()
            return RBLNDeepseekV4BlockFp8LinearMethod(self)
        return super().get_quant_method(layer, prefix)
