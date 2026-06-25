# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.model_executor.layers.rotary_embedding import mrope as vllm_mrope


def apply_interleaved_rope_oot(
    x: torch.Tensor,
    mrope_section: list[int],
) -> torch.Tensor:
    idx = torch.arange(x.shape[-1], device=x.device)
    h_mask = ((idx % 3) == 1) & (idx < mrope_section[1] * 3)
    w_mask = ((idx % 3) == 2) & (idx < mrope_section[2] * 3)
    # Avoid vLLM's in-place x[..., 1::3] / x[..., 2::3] slice writes. Those
    # lower to aten::copy_ with a strided slice pattern unsupported by RBLN.
    h = h_mask.to(dtype=x.dtype)
    w = w_mask.to(dtype=x.dtype)
    t = 1 - h - w
    return x[0] * t + x[1] * h + x[2] * w


vllm_mrope.apply_interleaved_rope = apply_interleaved_rope_oot
