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

from vllm_rbln.model_executor.layers.rotary_embedding.common import (
    apply_rotate_half,
    register_rotate_half_caches,
)
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
    # fp32 ops on RBLN may run in reduced precision; compute the RoPE cache on CPU.
    target_device = torch.get_default_device()
    with torch.device("cpu"):
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
        register_rotate_half_caches(self)
    self.to(target_device)


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
    key: torch.Tensor | None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    return apply_rotate_half(self, batch_size, seq_len, cos, sin, query, key)
