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

import logging

log = logging.getLogger("torch._dynamo")


def deepseek_scaling_rope_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-native implementation equivalent to forward()."""
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()

    query_rot = query[..., : self.rotary_dim]
    key_rot = key[..., : self.rotary_dim]
    if self.rotary_dim < self.head_size:
        query_pass = query[..., self.rotary_dim :]
        key_pass = key[..., self.rotary_dim :]

    self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if self.is_neox_style:
        # NOTE(woosuk): Here we assume that the positions tensor has the
        # shape [batch_size, seq_len].
        cos = cos.repeat(1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2)
    else:
        cos = torch.stack([cos, cos], dim=-1).reshape(cos_sin.shape).unsqueeze(-2)
        sin = torch.stack([sin, sin], dim=-1).reshape(cos_sin.shape).unsqueeze(-2)

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


DeepseekScalingRotaryEmbedding.forward_oot = deepseek_scaling_rope_forward
