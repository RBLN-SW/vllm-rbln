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

"""A pre-norm block boundary on the residual ops the compiler cuts across chiplets.

A pre-norm layer adds a block's output to the residual and normalizes the sum for
the next block in one call. On prefill steps that call runs on two ops:
rbln_custom_ops::residual_write leaves the new residual cut by tokens, and a read
gathers the next block's input. Decode steps do not cut: at decode batch sizes the
cut's scatter and gather cost more than the norm work it splits across the chiplets.

- RMSNorm: the "rbln" implementation of the fused_add_rms_norm IR op, read with
  residual_read; decode steps keep the native op.
- GemmaRMSNorm (weight stored zero-centered, scaled by 1 + w): RBLNGemmaRMSNorm,
  read with zero_centered_residual_read, which takes w as stored.
"""

import torch
from torch import Tensor
from vllm import ir
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_rbln.utils import is_prefill_step

# The architectures whose block boundaries run on the residual ops.
RESIDUAL_CUT_ARCHITECTURES = frozenset(
    {
        "AXK2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    }
)


def cuts_residual(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    return model_config is not None and bool(
        RESIDUAL_CUT_ARCHITECTURES.intersection(model_config.architectures)
    )


def _supports_args(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    # The ops take the [B, T, h] rows the model runner hands every layer.
    return weight is not None and variance_size is None and x.dim() == 3


@ir.ops.fused_add_rms_norm.register_impl("rbln", supports_args=_supports_args)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if not is_prefill_step():
        return ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
            x, x_residual, weight, epsilon, variance_size
        )
    residual = torch.ops.rbln_custom_ops.residual_write(x_residual, x)
    return torch.ops.rbln_custom_ops.residual_read(residual, weight, epsilon), residual


class RBLNGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm whose block-boundary call runs on the residual ops.

    Upstream hands the IR op the computed 1 + w: a constant the decode graph cannot
    share once prefill feeds it to a kernel, and one the runtime cannot load by the
    weight's name. So both steps read with zero_centered_residual_read(w); the
    compiler cuts it on prefill chunks only, and decode adds the residual as the
    native op does.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps)
        self.cuts_residual = cuts_residual(get_current_vllm_config())

    def forward(
        self, x: Tensor, residual: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        if residual is None or not self.cuts_residual or x.dim() != 3:
            return super().forward(x, residual)
        ops = torch.ops.rbln_custom_ops
        if is_prefill_step():
            residual = ops.residual_write(residual, x)
        else:
            residual = x + residual
        x = ops.zero_centered_residual_read(
            residual, self.weight, self.variance_epsilon
        )
        return x, residual
