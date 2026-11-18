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
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    DeepseekV4ScalingRotaryEmbedding,
)


class RBLNDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """DeepSeek scaling RoPE for RBLN."""

    def __init__(self, *args, **kwargs) -> None:
        # fp32 ops on RBLN may run in reduced precision; compute the RoPE cache on CPU.
        target_device = torch.get_default_device()
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        self.to(target_device)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
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


class RBLNDeepseekV4ScalingRotaryEmbedding(DeepseekV4ScalingRotaryEmbedding):
    def __init__(self, *args, **kwargs) -> None:
        target_device = torch.get_default_device()
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        self.to(target_device)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
        head_major: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """positions [B, L]; query / key [B, L, ..., D], or [B, H, L, D]
        with head_major=True."""
        if offsets is not None:
            positions = positions + offsets
        batch, seq_len = positions.shape[0], positions.shape[1]
        cos_sin = self.cos_sin_cache.index_select(0, positions.flatten()).to(
            query.dtype
        )
        cos, sin = cos_sin.view(batch, seq_len, -1).chunk(2, dim=-1)
        if self.is_neox_style:
            cos, sin = torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
        else:
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        if inverse:
            sin = -sin
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        out: list[torch.Tensor | None] = []
        for x in (query, key):
            if x is None:
                out.append(None)
                continue
            headless = x.dim() == 3
            if headless:
                x = x.unsqueeze(2)
            c, s = cos, sin
            if head_major:
                c, s = c.unsqueeze(1), s.unsqueeze(1)
            else:
                for _ in range(x.dim() - 3):
                    c, s = c.unsqueeze(-2), s.unsqueeze(-2)
            nope, rope = x.split([x.shape[-1] - self.rotary_dim, self.rotary_dim], -1)
            rot = (rope * c + rotate_fn(rope) * s).to(x.dtype)
            x = torch.cat((nope, rot), dim=-1)
            out.append(x.squeeze(2) if headless else x)
        return out[0], out[1]
