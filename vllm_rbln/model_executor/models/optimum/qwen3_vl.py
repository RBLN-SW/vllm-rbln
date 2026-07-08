# Copyright 2026 Rebellions Inc. All rights reserved.

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
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLVideoPixelInputs,
)

from .base import ModelInputForRBLN
from .qwen_vl import RBLNOptimumQwen2_5_VLForConditionalGeneration

logger = init_logger(__name__)


class RBLNOptimumQwen3VLForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Qwen3-VL reuses Qwen2.5-VL classes with the same implementation.
    However, since Qwen3-VL does not require second_per_grid_ts,
    certain methods are overridden to exclude it from the model inputs.
    """

    def _build_prefill_params(self, preprocess_outputs: tuple) -> dict:
        # deepstack_embeds
        # [1, 3, num_patches, embedding_dim] -> [3, num_patches, embedding_dim]
        deepstack_embeds = preprocess_outputs[4]
        return {
            "inputs_embeds": preprocess_outputs[0],
            "position_embed": preprocess_outputs[1],
            "rope_deltas": preprocess_outputs[2],
            "visual_pos_mask": preprocess_outputs[3],
            "deepstack_embeds": deepstack_embeds.squeeze(0)
            if deepstack_embeds is not None
            else None,
        }

    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Qwen3-VL doesn't need additional arguments"""
        pass

    def _process_image_input(self, image_input) -> dict:
        result = {}
        if image_input is not None and image_input.get("type") == "pixel_values":
            image_embeds, deepstack = self.model.visual(
                image_input["pixel_values"], grid_thw=image_input["image_grid_thw"]
            )
            result["image_embeds"] = image_embeds
            result["deepstack_image_embeds"] = deepstack
            result["image_grid_thw"] = image_input["image_grid_thw"]
        return result

    def _process_video_input(self, video_input) -> dict:
        result = {}
        if video_input is not None and video_input.get("type") == "pixel_values_videos":
            video_embeds, deepstack = self.model.visual(
                video_input["pixel_values_videos"],
                grid_thw=video_input["video_grid_thw"],
            )
            result["video_embeds"] = video_embeds
            result["deepstack_video_embeds"] = deepstack
            result["video_grid_thw"] = video_input["video_grid_thw"]
            second_per_grid_ts = video_input.get("second_per_grid_ts", None)
            if second_per_grid_ts is not None:
                result["second_per_grid_ts"] = second_per_grid_ts
        return result

    def _encode_and_slice_mm(
        self,
        *,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        tail_starts: list[int],
        merge: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Slice image + per-layer deepstack features to the uncached tail."""
        feats, deepstack = self.model.visual(pixel_values, grid_thw=grid_thw)
        counts = self._mm_feature_counts(grid_thw, merge).tolist()
        assert len(counts) == len(tail_starts), (
            f"kept-item count mismatch: {len(counts)} grids vs "
            f"{len(tail_starts)} tail starts"
        )
        tail_feats = self._slice_to_tail(feats, counts, tail_starts)
        tail_deepstack = [
            self._slice_to_tail(layer, counts, tail_starts) for layer in deepstack
        ]
        return tail_feats, tail_deepstack

    def _extract_cached_deepstack(
        self,
        image_caches: list[dict],
        video_caches: list[dict],
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Concatenate cached per-layer deepstack features across items."""

        def _concat(caches: list[dict], key: str):
            present = [c for c in caches if c.get(key) is not None]
            if not present:
                return None
            num_layers = len(present[0][key])
            return [
                torch.cat([c[key][layer].to(self.dtype) for c in present], dim=0)
                for layer in range(num_layers)
            ]

        return (
            _concat(image_caches, "deepstack_image_embeds"),
            _concat(video_caches, "deepstack_video_embeds"),
        )

    def _pack_partial_deepstack(
        self,
        masks: dict[str, torch.Tensor | None],
        deepstacks: dict[str, list[torch.Tensor] | None],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pack tail-sliced deepstack + visual mask via optimum's
        ``_prepare_deepstack`` (batch dim squeezed to match the full path)."""
        if deepstacks["image"] is None and deepstacks["video"] is None:
            return None, None
        visual_pos_mask, deepstack_visual = self.model._prepare_deepstack(
            masks["image"],
            masks["video"],
            deepstacks["image"],
            deepstacks["video"],
        )
        deepstack_embeds = (
            # [1, num_layers, seq, hidden] -> [num_layers, seq, hidden]
            deepstack_visual.squeeze(0) if deepstack_visual is not None else None
        )
        return visual_pos_mask, deepstack_embeds

    def _build_partial_inputs_embeds(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Tail ``inputs_embeds`` + packed deepstack/visual mask (partial hit)."""
        inputs_embeds, masks, deepstacks = self._scatter_tail_mm(model_input)
        visual_pos_mask, deepstack_embeds = self._pack_partial_deepstack(
            masks, deepstacks
        )
        return inputs_embeds, visual_pos_mask, deepstack_embeds

    def build_prefill_inputs_from_cache(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list[dict],
        **kwargs,
    ) -> dict:
        """EC consumer: forward the cached deepstack features, then reuse the
        common base path."""
        image_caches = [c for c in cached_mm_outputs if "image_embeds" in c]
        video_caches = [c for c in cached_mm_outputs if "video_embeds" in c]
        deepstack_image_embeds, deepstack_video_embeds = self._extract_cached_deepstack(
            image_caches, video_caches
        )
        extra: dict = {}
        if deepstack_image_embeds is not None:
            extra["deepstack_image_embeds"] = deepstack_image_embeds
        if deepstack_video_embeds is not None:
            extra["deepstack_video_embeds"] = deepstack_video_embeds
        return super().build_prefill_inputs_from_cache(
            input_ids, cached_mm_outputs, **extra, **kwargs
        )

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts=torch.Tensor | None,
    ):
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )


class RBLNOptimumQwen3VLMoeForConditionalGeneration(
    RBLNOptimumQwen3VLForConditionalGeneration
):
    """
    Qwen3-VL MoE model shares the same input structure as Qwen3-VL,
    so it inherits from RBLNOptimumQwen3VLForConditionalGeneration without changes.
    """

    pass
