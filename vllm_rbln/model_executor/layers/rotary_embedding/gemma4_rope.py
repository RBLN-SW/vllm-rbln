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

from typing import Any

import torch
import vllm.model_executor.layers.rotary_embedding as rope_module
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


class Gemma4RotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        self.rope_angles = rotary_dim // 2
        self.nope_angles = (head_size // 2) - self.rope_angles

        # Gemma4 rotates every head dimension, but non-rotated slots are
        # zero-padded in inv_freq so they become identity rotations.
        super().__init__(
            head_size,
            head_size,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        freq_exponents = (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float)
            / self.head_size
        )
        inv_freq = 1.0 / (base**freq_exponents)

        if self.nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq, torch.zeros(self.nope_angles, dtype=torch.float)]
            )

        return inv_freq

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", rope_angles={self.rope_angles}, nope_angles={self.nope_angles}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


rope_original_get_rope = rope_module.get_rope


def gemma4_get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    rope_parameters = rope_parameters or {}
    scaling_type = rope_parameters.get("rope_type", "default")
    if scaling_type != "proportional":
        return rope_original_get_rope(
            head_size=head_size,
            max_position=max_position,
            is_neox_style=is_neox_style,
            rope_parameters=rope_parameters,
            dtype=dtype,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

    if dtype is None:
        dtype = torch.get_default_dtype()

    rope_parameters_tuple = {
        k: tuple(v) if isinstance(v, list) else v for k, v in rope_parameters.items()
    }
    rope_parameters_args = tuple(rope_parameters_tuple.items())

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    base = rope_parameters.get("rope_theta", 10000)
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
        raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")
    rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in rope_module._ROPE_DICT:
        return rope_module._ROPE_DICT[key]

    rotary_emb = Gemma4RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
    )
    rope_module._ROPE_DICT[key] = rotary_emb
    return rotary_emb


if not getattr(rope_module.get_rope, "_vllm_rbln_gemma4_patched", False):
    gemma4_get_rope._vllm_rbln_gemma4_patched = True
    rope_module.get_rope = gemma4_get_rope
