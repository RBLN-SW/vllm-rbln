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

"""fused_add_rms_norm on the residual ops the compiler cuts across the chiplets.

A pre-norm layer adds a block's output to the residual and normalizes the sum for
the next block in one fused_add_rms_norm. The "rbln" implementation splits that
boundary in two: rbln_custom_ops::residual_write leaves the new residual cut by rows
(tokens in prefill, batch rows in decode), and residual_read gathers the next
block's input. The compiler decides whether a step is cut; an uncut step runs both
whole.
"""

import torch
from torch import Tensor
from vllm import ir

# The architectures whose fused_add_rms_norm puts "rbln" first.
RESIDUAL_CUT_ARCHITECTURES = frozenset(
    {"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"}
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
    residual = torch.ops.rbln_custom_ops.residual_write(x_residual, x)
    return torch.ops.rbln_custom_ops.residual_read(residual, weight, epsilon), residual
