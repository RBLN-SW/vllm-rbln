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
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox

from vllm_rbln.patches import register_patch

original_rope_init = RotaryEmbedding.__init__


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
    # NOTE(RBLN): For best compatibility with rbln,
    # tensors are reshaped/transposed as follows:
    # - cos, sin: (batch_size, 1, seq_len, rotary_dim * 2)
    # - query, key: (batch_size, num_heads, seq_len, head_size)

    if offsets is not None:
        positions = positions + offsets

    batch_size, seq_len = positions.shape[0], positions.shape[1]
    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj

    positions_flat = positions.flatten()
    cos = (
        self.cos_cache.index_select(0, positions_flat)
        .view(batch_size, 1, seq_len, -1)
        .to(query.dtype)
    )
    sin = (
        self.sin_cache.index_select(0, positions_flat)
        .view(batch_size, 1, seq_len, -1)
        .to(query.dtype)
    )

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
    target="vllm.model_executor.layers.rotary_embedding.mrope.apply_interleaved_rope",
    reason=(
        "Avoid advanced indexing/split patterns in upstream M-RoPE that lower "
        "to unsupported RBLN strided_slice operations."
    ),
)
def patched_apply_interleaved_rope_oot(
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
