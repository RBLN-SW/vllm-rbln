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
    def check_shape(
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

        _, in_features = config.weight_shape

        if in_features % block_k != 0:
            return False, (
                "RBLN block FP8 linear kernel requires input features to be divisible "
                f"by block_k. got {in_features=}, {block_k=}"
            )

        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        ok, reason = cls.check_shape(config)
        if not ok:
            return ok, reason

        from vllm_rbln import envs

        if not envs.VLLM_RBLN_USE_W8A16:
            return False, "RBLN W8A16 block fp8 kernel applies only on W8A16 devices."
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

        in_blocks = in_features // block_k

        weight_scale = weight_scale.repeat_interleave(block_n, dim=0)[:out_features, :]
        weight = weight.view(out_features, in_blocks, block_k).to(dtype)

        return (weight * weight_scale[:, :, None].to(dtype)).view(
            out_features, in_features
        )


class RBLNW8A8BlockFp8LinearKernel(RBLNW8A16BlockFp8LinearKernel):
    """W8A8 block FP8 linear for RBLN."""

    FP8_DTYPE = torch.float8_e4m3fn

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        ok, reason = cls.check_shape(config)
        if not ok:
            return ok, reason

        from vllm_rbln import envs

        if envs.VLLM_RBLN_USE_W8A16:
            return False, "RBLN W8A8 block fp8 kernel applies only on W8A8 devices."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        del As
        _block_n, block_k = [int(v) for v in self.weight_group_shape]

        a_3d = A.reshape(1, *A.shape)
        x_q, x_scale = self._per_token_group_quant_fp8(a_3d, block_k)
        x_deq = x_q.to(A.dtype) * x_scale.repeat_interleave(block_k, dim=-1).to(A.dtype)

        weight = self._dequantize_block_fp8_weight(
            weight=B,
            weight_scale=Bs,
            dtype=A.dtype,
        )
        return torch.nn.functional.linear(x_deq.reshape(-1, x_deq.shape[-1]), weight)

    @classmethod
    def _per_token_group_quant_fp8(
        cls,
        x: torch.Tensor,
        group_size: int,
        eps: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        finfo = torch.finfo(cls.FP8_DTYPE)
        orig_shape = x.shape
        x_g = x.reshape(-1, group_size).to(torch.float32)
        amax = x_g.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = amax / finfo.max
        x_q = (x_g / scale).clamp(finfo.min, finfo.max)
        x_q = x_q.reshape(orig_shape)
        scale = scale.reshape(*orig_shape[:-1], orig_shape[-1] // group_size)
        return x_q, scale
