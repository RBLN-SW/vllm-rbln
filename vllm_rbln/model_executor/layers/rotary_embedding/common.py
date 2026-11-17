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
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox


def register_rotate_half_caches(self: RotaryEmbeddingBase) -> None:
    # For best compatibility with rbln, we use the rotate_half-style RoPE.
    cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
    if self.is_neox_style:
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
    else:
        cos = torch.stack([cos, cos], dim=-1).reshape(cos.shape[0], -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(sin.shape[0], -1)
    self.register_buffer("cos_cache", cos, persistent=False)
    self.register_buffer("sin_cache", sin, persistent=False)


def apply_rotate_half(
    self: RotaryEmbeddingBase,
    batch_size: int,
    seq_len: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NOTE(RBLN): For best compatibility with rbln,
    # tensors are reshaped/transposed as follows:
    # - cos, sin: (batch_size, 1, seq_len, rotary_dim * 2)
    # - query, key: (batch_size, num_heads, seq_len, head_size)
    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
    cos = cos.to(query.dtype)
    sin = sin.to(query.dtype)

    query_shape = query.shape
    query = query.view(batch_size, seq_len, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_rot = query_rot.transpose(1, 2)
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    query_rot = query_rot.transpose(1, 2)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        query = query_rot.reshape(query_shape)
    else:
        query_pass = query[..., self.rotary_dim :]
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(batch_size, seq_len, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_rot = key_rot.transpose(1, 2)
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin
    key_rot = key_rot.transpose(1, 2)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        key = key_rot.reshape(key_shape)
    else:
        key_pass = key[..., self.rotary_dim :]
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key
