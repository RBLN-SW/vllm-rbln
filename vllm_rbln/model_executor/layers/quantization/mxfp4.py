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
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import (
    GptOssMxfp4Config,
    GptOssMxfp4MoEMethod,
    Mxfp4MoeBackend,
)

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe.utils import get_tokens_mask

logger = init_logger(__name__)


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
    scoring_func: str,
    alpha: torch.Tensor,
    limit: torch.Tensor,
    k: int,
    post_norm: bool = True,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
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
    - scoring_func: str
    - alpha: [], constant
    - limit: [], constant
    - expert_map: [num_experts],
      Mapping from global expert index to local expert index (in num_experts).
      Contains -1 for experts not assigned to the current rank.

    Returns:
        torch.Tensor: [num_tokens, hidden_size]
    """
    return torch.empty_like(hidden_states)


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
    scoring_func: str,
    alpha: torch.Tensor,
    limit: torch.Tensor,
    k: int,
    post_norm: bool = True,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


class RBLNGptOssMxfp4Config(GptOssMxfp4Config):
    """GPT-OSS MXFP4 quantization config for RBLN.

    This keeps upstream GPT-OSS MXFP4 quantization detection and non-MoE
    fallbacks, but selects the RBLN MoE quantization method for routed experts
    so GPT-OSS MXFP4 MoE layers execute through the RBLN custom op.
    """

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return RBLNGptOssMxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)


class RBLNGptOssMxfp4MoEMethod(GptOssMxfp4MoEMethod):
    def __init__(self, moe: FusedMoEConfig) -> None:
        # Do not call GptOssMxfp4MoEMethod.__init__().
        # It selects upstream CUDA/CPU/XPU MXFP4 backends.
        FusedMoEMethodBase.__init__(self, moe)
        self.mxfp4_backend = Mxfp4MoeBackend.NONE
        self.moe_kernel = None

        self.swiglu_alpha = torch.tensor(1.702, dtype=torch.float32)
        self.swiglu_limit = torch.tensor(7.0, dtype=torch.float32)

    @property
    def is_monolithic(self) -> bool:
        # Prevent vLLM from trying to initialize modular-kernel plumbing.
        # RBLN FusedMoE.forward calls apply() directly.
        return True

    @property
    def skip_forward_padding(self) -> bool:
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )

        if hidden_size % 32 != 0:
            raise ValueError(f"RBLN GPT-OSS MXFP4 requires {hidden_size=} % 32 == 0")
        if intermediate_size_per_partition % 64 != 0:
            raise ValueError(
                "RBLN GPT-OSS MXFp4 requires "
                f"{intermediate_size_per_partition=} % 64 == 0"
            )
        return hidden_size, intermediate_size_per_partition

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        self._set_buffer(layer, "gate_proj_blocks", layer.w13_weight.data[:, ::2])
        self._set_buffer(layer, "gate_proj_scales", layer.w13_weight_scale.data[:, ::2])
        self._set_buffer(layer, "up_proj_blocks", layer.w13_weight.data[:, 1::2])
        self._set_buffer(layer, "up_proj_scales", layer.w13_weight_scale.data[:, 1::2])

        if not hasattr(layer, "w13_bias") or not hasattr(layer, "w2_bias"):
            raise NotImplementedError("RBLN GPT-OSS MXFP4 requires MoE bias tensors")

        self._set_buffer(layer, "gate_proj_bias", layer.w13_bias.data[:, ::2])
        self._set_buffer(layer, "up_proj_bias", layer.w13_bias.data[:, 1::2])
        self._set_buffer(layer, "down_proj_blocks", layer.w2_weight.data)
        self._set_buffer(layer, "down_proj_scales", layer.w2_weight_scale.data)
        self._set_buffer(layer, "down_proj_bias", layer.w2_bias.data)

    @staticmethod
    def _set_buffer(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
        if name in layer._buffers:
            layer._buffers[name] = value
        else:
            layer.register_buffer(name, value)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if layer.activation != MoEActivation.SWIGLUOAI:
            raise NotImplementedError(layer.activation)

        orig_shape = x.shape
        num_tokens = orig_shape[:-1].numel()
        hidden_states = x.reshape(num_tokens, -1)
        router_logits = router_logits.reshape(num_tokens, -1)

        # Pre-score routing inputs at caller side; compiler custom op routing
        # expects already-scored values (no sigmoid applied inside the kernel).
        assert layer.scoring_func is not None, "FusedMoE.scoring_func must be set"
        assert layer.scoring_func in {"softmax", "sigmoid"}
        if layer.scoring_func == "sigmoid":
            router_logits = torch.sigmoid(router_logits.to(torch.float32)).to(
                router_logits.dtype
            )

        tokens_mask = (
            get_tokens_mask(num_tokens) if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK else None
        )

        out = torch.ops.rbln_custom_ops.custom_moe_glu_mxfp4(
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
            layer.scoring_func,
            self.swiglu_alpha,
            self.swiglu_limit,
            layer.top_k,
            layer.renormalize,
            layer.expert_map,
            tokens_mask,
        )
        return out.reshape(orig_shape)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise RuntimeError
