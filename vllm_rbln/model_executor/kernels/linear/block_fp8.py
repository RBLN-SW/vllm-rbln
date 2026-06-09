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
from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class RBLNW8A16BlockFp8LinearKernel(Fp8BlockScaledMMLinearKernel):
    apply_input_quant = False

    @classmethod
    def is_supported(cls, _: int | None = None) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        ok, reason = super().can_implement(config)
        if not ok:
            return ok, reason

        weight_group_shape = config.weight_quant_key.scale.group_shape
        block_n = int(weight_group_shape.row)
        block_k = int(weight_group_shape.col)

        if block_n <= 0 or block_k <= 0:
            return False, (
                "RBLN block FP8 linear kernel requires positive block size, "
                f"got ({block_n}, {block_k})."
            )

        out_features, in_features = config.weight_shape
        if out_features % block_n != 0:
            return False, (
                "RBLN block FP8 linear kernel requires output features to be divisible "
                f"by block_n. got {out_features=}, {block_n=}"
            )

        if in_features % block_k != 0:
            return False, (
                "RBLN block FP8 linear kernel requires input features to be divisible "
                f"by block_k. got {in_features=}, {block_k=}"
            )

        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        del As

        weight = self._dequantize_block_fp8_weight(
            weight=B,
            weight_scale=Bs,
            dtype=A.dtype,
        )
        return torch.nn.functional.linear(A, weight)

    def _dequantize_block_fp8_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        block_n, block_k = [int(v) for v in self.weight_group_shape]
        out_features, in_features = weight.shape

        out_blocks = out_features // block_n
        in_blocks = in_features // block_k

        weight = weight.view(out_blocks, block_n, in_blocks, block_k).to(dtype)
        weight_scale = weight_scale.view(out_blocks, in_blocks).to(dtype)

        return (weight * weight_scale[:, None, :, None]).reshape(
            out_features, in_features
        )
