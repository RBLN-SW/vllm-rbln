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
"""Qwen-VL prefill helpers.

Holds the two prefill concerns kept out of the main model class to keep it
readable:

- ``iter_modalities`` / ``MODALITIES``: the per-modality kwarg/config layout,
  shared by every prefill path (full, MRoPE-only, partial).
- ``QwenVLPartialPrefixMixin``: rebuilding the uncached tail on a partial
  prefix-cache hit. Cohesive and orthogonal to the normal prefill path.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterator

import torch

from .base import ModelInputForRBLN


@dataclass(frozen=True)
class ModalitySpec:
    """Per-modality kwarg keys and the config attribute for its placeholder id."""

    name: str  # "image" | "video"
    grid_key: str  # grid_thw kwarg key
    pixel_key: str  # pixel-values kwarg key
    embeds_key: str  # cached-embeds kwarg key / input "type" marker
    token_attr: str  # config attribute holding the placeholder token id


MODALITIES: tuple[ModalitySpec, ModalitySpec] = (
    ModalitySpec("image", "image_grid_thw", "pixel_values", "image_embeds",
                 "image_token_id"),
    ModalitySpec("video", "video_grid_thw", "pixel_values_videos",
                 "video_embeds", "video_token_id"),
)


def iter_modalities(
    image_input: Any, video_input: Any
) -> Iterator[tuple[ModalitySpec, Any]]:
    """Pair each modality spec with its parsed input (image first, video next)."""
    yield MODALITIES[0], image_input
    yield MODALITIES[1], video_input


class QwenVLPartialPrefixMixin:
    """Build the uncached tail of a partial prefix-cache hit.

    On a partial hit the boundary may fall inside an image, so the tail is built
    manually rather than via ``_preprocess_prefill`` (whose ``get_rope_index``
    chokes on the tail's orphaned image-pad tokens). The cached prefix KV is
    reused separately via ``copy_cached_kv_blocks``.
    """

    # Provided by the host model class; declared for type-checking only.
    if TYPE_CHECKING:
        model: Any

        def _parse_and_validate_image_input(self, **kwargs: Any) -> Any | None: ...
        def _parse_and_validate_video_input(self, **kwargs: Any) -> Any | None: ...
        def _compute_mrope_position(
            self, input_ids, attention_mask, image_input, video_input
        ) -> dict: ...
        def _mm_feature_counts(
            self, grid_thw: torch.Tensor, merge_size: int
        ) -> torch.Tensor: ...

    def _build_partial_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Build the uncached tail's embeds/positions (and Qwen3-VL deepstack)."""
        inputs_embeds, visual_pos_mask, deepstack_embeds = (
            self._build_partial_inputs_embeds(model_input)
        )
        position_embed, rope_deltas = self._recompute_full_mrope_position(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=inputs_embeds,
            position_embed=position_embed,
            visual_pos_mask=visual_pos_mask,
            deepstack_embeds=deepstack_embeds,
        )

    def _scatter_tail_mm(
        self, model_input: ModelInputForRBLN
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor | None],
        dict[str, list[torch.Tensor] | None],
    ]:
        """Scatter each kept item's uncached-tail features into the tail embeds.

        Returns ``(inputs_embeds, masks, sides)``: per-modality tail placeholder
        masks and encoder side outputs (deepstack for Qwen3-VL, else ``None``).
        """
        partial = model_input.partial_prefix
        assert partial is not None
        tail_ids = model_input.input_tokens
        inputs_embeds = self.model.embed_tokens(tail_ids).to(
            self.model.rbln_config.dtype
        )

        mm_kwargs = model_input.multi_modal_kwargs or {}
        tail_starts = partial.mm_embed_tail_starts or {}
        image_input = self._parse_and_validate_image_input(**mm_kwargs)
        video_input = self._parse_and_validate_video_input(**mm_kwargs)

        config = self.model.config
        merge = config.vision_config.spatial_merge_size
        masks: dict[str, torch.Tensor | None] = {s.name: None for s in MODALITIES}
        sides: dict[str, list[torch.Tensor] | None] = {s.name: None for s in MODALITIES}
        for spec, mm_input in iter_modalities(image_input, video_input):
            if mm_input is None:
                continue
            tail_feats, side = self._encode_and_slice_mm(
                pixel_values=mm_input[spec.pixel_key],
                grid_thw=mm_input[spec.grid_key],
                tail_starts=tail_starts.get(spec.name, []),
                merge=merge,
            )
            mask = tail_ids == getattr(config, spec.token_attr)
            inputs_embeds[mask] = tail_feats.to(inputs_embeds.dtype)
            masks[spec.name] = mask
            sides[spec.name] = side
        return inputs_embeds, masks, sides

    def _build_partial_inputs_embeds(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Tail ``(inputs_embeds, visual_pos_mask, deepstack_embeds)``. Base has
        no extras (``None``); Qwen3-VL overrides to pack deepstack.
        """
        inputs_embeds, _masks, _sides = self._scatter_tail_mm(model_input)
        return inputs_embeds, None, None

    @staticmethod
    def _slice_to_tail(
        feats: torch.Tensor, counts: list[int], tail_starts: list[int]
    ) -> torch.Tensor:
        """Concatenate each item's features cut to its tail: item ``i`` keeps
        ``feats[item_i][tail_starts[i]:]`` (``feats`` = items concatenated)."""
        parts = []
        offset = 0
        for count, tail_start in zip(counts, tail_starts):
            parts.append(feats[offset : offset + count][tail_start:])
            offset += count
        return torch.cat(parts, dim=0)

    def _encode_and_slice_mm(
        self,
        *,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        tail_starts: list[int],
        merge: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Encode whole items, slice each to its tail. Returns ``(tail_feats,
        tail_deepstack)``; base has no deepstack (``None``), Qwen3-VL overrides.
        """
        feats = self.model.visual(pixel_values, grid_thw=grid_thw)
        counts = self._mm_feature_counts(grid_thw, merge).tolist()
        assert len(counts) == len(tail_starts), (
            f"kept-item count mismatch: {len(counts)} grids vs "
            f"{len(tail_starts)} tail starts"
        )
        return self._slice_to_tail(feats, counts, tail_starts), None

    def _recompute_full_mrope_position(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tail ``position_embed`` for a partial hit: MRoPE needs the whole
        layout, so compute positions over the full prompt (encoder skipped) and
        slice to the tail. Also returns full-sequence ``rope_deltas`` for decode.
        """
        partial = model_input.partial_prefix
        assert partial is not None
        full_input_ids = partial.full_input_tokens
        num_cached = partial.num_cached_tokens

        image_input = None
        video_input = None
        if partial.mrope_mm_kwargs:
            image_input = self._parse_and_validate_image_input(
                **partial.mrope_mm_kwargs
            )
            video_input = self._parse_and_validate_video_input(
                **partial.mrope_mm_kwargs
            )

        attention_mask = torch.ones_like(full_input_ids)
        params = self._compute_mrope_position(
            full_input_ids, attention_mask, image_input, video_input
        )
        # position_embed: [2, batch, 1, N, head_dim]; slice the sequence (dim=-2)
        # to the uncached tail.
        position_embed = params["position_embed"][..., num_cached:, :]
        return position_embed, params["rope_deltas"]
