# Copyright 2025 Rebellions Inc. All rights reserved.

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
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)


def deepseek_scaling_rope_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-native implementation equivalent to forward()."""
    assert key is not None
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    query_rot = query[..., : self.rotary_dim]
    key_rot = key[..., : self.rotary_dim]
    if self.rotary_dim < self.head_size:
        query_pass = query[..., self.rotary_dim :]
        key_pass = key[..., self.rotary_dim :]

    cos_sin = cos_sin_cache[
        torch.add(positions, offsets) if offsets is not None else positions
    ]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if self.is_neox_style:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if self.rotary_dim < self.head_size:
        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)
    else:
        query = query_rot
        key = key_rot
    return query, key


DeepseekScalingRotaryEmbedding.forward = deepseek_scaling_rope_forward
