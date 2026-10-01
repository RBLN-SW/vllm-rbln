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
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.utils import replace_parameter

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class RBLNW8A16Fp8LinearKernel(FP8ScaledMMLinearKernel):
    """Weight-only FP8 linear with per-channel or per-tensor weight scales.

    Selected by CompressedTensorsW8A16Fp8 for checkpoints whose config_groups
    declare `strategy: channel` (or `tensor`) with no input_activations, e.g.
    a.x-k1. The activation stays in its original dtype, so there is nothing to
    quantize at runtime: dequantize the weight and run a plain linear, which
    rebel_compiler detects and maps to the actual kernel.
    """

    @classmethod
    def is_supported(cls, _: int | None = None) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        group_shape = config.weight_quant_key.scale.group_shape
        if not (group_shape.is_per_channel() or group_shape.is_per_tensor()):
            return False, (
                "RBLN W8A16 FP8 linear kernel requires per-channel or "
                f"per-tensor weight scales, got {group_shape}."
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # CompressedTensorsW8A16Fp8 canonicalizes the weight to (K, N) for
        # scaled_mm kernels, leaving a non-contiguous transposed view. F.linear
        # wants (N, K), so transpose it back here: the runtime graph then holds
        # no transpose, and the weight stays contiguous for weight-free compile.
        weight_name = self.layer_param_names[0]
        weight = getattr(layer, weight_name)
        replace_parameter(layer, weight_name, weight.t().contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(RBLN): the base class quantizes the activation to fp8 before
        # calling apply_scaled_mm. This is a weight-only scheme, so bypass that
        # path entirely and dequantize the weight instead.
        weight, weight_scale, _, _ = self._get_layer_params(layer)
        out_dtype = self.config.out_dtype
        if out_dtype is None:
            out_dtype = x.dtype

        # weight: (N, K) fp8, weight_scale: (N, 1) fp32. Per-tensor scales are
        # expanded to channels by the scheme, so both strategies broadcast the
        # same way.
        weight = weight.to(x.dtype) * weight_scale.reshape(-1, 1).to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias).to(out_dtype)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "RBLNW8A16Fp8LinearKernel dequantizes the weight in "
            "apply_weights; apply_scaled_mm is unused."
        )
