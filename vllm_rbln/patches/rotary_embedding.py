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
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from vllm_rbln.patches import register_patch

original_rope_init = RotaryEmbedding.__init__
original_mrope_init = MRotaryEmbedding.__init__


def _register_rotate_half_caches(self: RotaryEmbeddingBase) -> None:
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


def _apply_rotate_half(
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


@register_patch(
    target="vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.__init__",
    reason=(
        "The RBLN path needs rotate-half style RoPE caches because that "
        "cache layout is more compatible with RBLN execution than the "
        "upstream layout."
    ),
)
def patched_rope_init(
    self: RotaryEmbedding,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    init_cache: bool = True,
):
    original_rope_init(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        init_cache,
    )
    _register_rotate_half_caches(self)


@register_patch(
    target="vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_oot",
    reason=(
        "The RBLN path needs a dedicated RoPE execution path that consumes "
        "the rotate-half cache layout prepared at initialization and "
        "applies rotary embeddings with RBLN-friendly tensor layouts."
    ),
)
def patched_rope_forward_oot(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: [batch_size, seq_len]
        query: [batch_size, seq_len, num_heads * head_size]
        key: [batch_size, seq_len, num_kv_heads * head_size]
    """
    if offsets is not None:
        positions = positions + offsets

    batch_size, seq_len = positions.shape[0], positions.shape[1]
    positions_flat = positions.flatten()
    cos = self.cos_cache.index_select(0, positions_flat).view(
        batch_size, 1, seq_len, -1
    )
    sin = self.sin_cache.index_select(0, positions_flat).view(
        batch_size, 1, seq_len, -1
    )
    return _apply_rotate_half(self, batch_size, seq_len, cos, sin, query, key)


@register_patch(
    target="vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding.__init__",
    reason=(
        "MRotaryEmbedding is a sibling of RotaryEmbedding, so the RotaryEmbedding "
        "patch never builds the rotate-half caches its RBLN forward reads. The "
        "per-channel axis table replaces upstream's strided section writes, which "
        "the compiler cannot lower."
    ),
)
def patched_mrope_init(self: MRotaryEmbedding, *args, **kwargs) -> None:
    original_mrope_init(self, *args, **kwargs)
    _register_rotate_half_caches(self)
    if not self.mrope_section:
        return
    # Which of the T/H/W position axes each rotary channel takes: interleaved
    # [THWTHW...TT] (upstream apply_interleaved_rope) or chunked [TTT..HHH..WWW].
    half = self.rotary_dim // 2
    if self.mrope_interleaved:
        channel = torch.arange(half)
        axis = torch.zeros(half, dtype=torch.long)
        for index in (1, 2):
            selected = (channel % 3 == index) & (
                channel < self.mrope_section[index] * 3
            )
            axis = torch.where(selected, torch.full_like(axis, index), axis)
    else:
        axis = torch.repeat_interleave(
            torch.arange(3), torch.tensor(self.mrope_section)
        )
    # The rotate-half caches lay the half out twice (neox) or pairwise (gptj).
    axis = axis.repeat(2) if self.is_neox_style else axis.repeat_interleave(2)
    self.register_buffer("mrope_axis_table", axis, persistent=False)


@register_patch(
    target="vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding.forward_oot",
    reason=(
        "MRotaryEmbedding has no RBLN forward, and its native one reads a 2-D "
        "positions tensor as [3, N]; the RBLN layout is [batch, seq], with "
        "[3, batch, seq] for multimodal positions."
    ),
)
def patched_mrope_forward_oot(
    self: MRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: [batch_size, seq_len] for text, where every axis shares the
            position, or [3, batch_size, seq_len] holding the T/H/W positions
        query: [batch_size, seq_len, num_heads * head_size] or
            [batch_size * seq_len, num_heads * head_size]
        key: the same layout with num_kv_heads
    """
    if offsets is not None:
        positions = positions + offsets

    if positions.ndim == 2:
        batch_size, seq_len = positions.shape
        positions_flat = positions.flatten()
        cos = self.cos_cache.index_select(0, positions_flat).view(
            batch_size, 1, seq_len, -1
        )
        sin = self.sin_cache.index_select(0, positions_flat).view(
            batch_size, 1, seq_len, -1
        )
        return _apply_rotate_half(self, batch_size, seq_len, cos, sin, query, key)

    assert positions.ndim == 3 and positions.shape[0] == 3, positions.shape
    assert self.mrope_section, "multimodal positions need mrope_section"
    batch_size, seq_len = positions.shape[1], positions.shape[2]
    positions_flat = positions.flatten()
    cos_axes = self.cos_cache.index_select(0, positions_flat).view(
        3, batch_size, 1, seq_len, -1
    )
    sin_axes = self.sin_cache.index_select(0, positions_flat).view(
        3, batch_size, 1, seq_len, -1
    )
    table = self.mrope_axis_table
    cos = torch.where(table == 1, cos_axes[1], cos_axes[0])
    cos = torch.where(table == 2, cos_axes[2], cos)
    sin = torch.where(table == 1, sin_axes[1], sin_axes[0])
    sin = torch.where(table == 2, sin_axes[2], sin)
    return _apply_rotate_half(self, batch_size, seq_len, cos, sin, query, key)
