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
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from vllm_rbln.patches.rotary_embedding import (
    _apply_rotate_half,
    _register_rotate_half_caches,
)


class RBLNMRotaryEmbedding(MRotaryEmbedding):
    """M-RoPE for RBLN: the rotate-half caches of the RBLN RoPE forward, and a
    per-channel axis table in place of upstream's strided section writes, which
    the compiler cannot lower."""

    def __init__(self, *args, **kwargs) -> None:
        # fp32 ops on RBLN may run in reduced precision; compute the RoPE cache on CPU.
        target_device = torch.get_default_device()
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
            _register_rotate_half_caches(self)
            if self.mrope_section:
                # Which of the T/H/W position axes each rotary channel takes:
                # interleaved [THWTHW...TT] (upstream apply_interleaved_rope)
                # or chunked [TTT..HHH..WWW].
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
                # The rotate-half caches lay the half out twice (neox) or
                # pairwise (gptj).
                axis = (
                    axis.repeat(2) if self.is_neox_style else axis.repeat_interleave(2)
                )
                self.register_buffer("mrope_axis_table", axis, persistent=False)
        self.to(target_device)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream's native forward reads 2-D positions as [3, N]; the RBLN
        layout is [batch, seq].

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
