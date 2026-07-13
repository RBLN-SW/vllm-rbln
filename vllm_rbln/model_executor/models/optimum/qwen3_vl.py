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
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLVideoPixelInputs,
)

from .base import ModelInputForRBLN
from .qwen_vl import (
    MODALITIES,
    ModalitySpec,
    RBLNOptimumQwen2_5_VLForConditionalGeneration,
    TailFeatureProvider,
)

logger = init_logger(__name__)


@dataclass
class DeepstackTailProvider(TailFeatureProvider):
    """``TailFeatureProvider`` extended with Qwen3-VL deepstack side outputs.
    ``features_for`` fills ``deepstack_by_modality`` in place as it runs, so the
    shared ``_scatter_tail_mm`` places the tail features while the deepstack is
    collected here for later packing.
    """

    deepstack_by_modality: dict[str, list[torch.Tensor] | None] = field(
        default_factory=dict
    )


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

    def _encode_tail_feats_and_deepstack(
        self,
        *,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        tail_starts: list[int],
        merge: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the vision encoder once and slice both the image features and the
        per-layer deepstack side outputs to the uncached tail."""
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

    def _make_encoder_tail_provider(self, mm_kwargs: dict) -> DeepstackTailProvider:
        """Encoder tail provider that also captures each modality's tail deepstack
        into ``deepstack_by_modality`` (one ``visual()`` call per item)."""
        image_input = self._parse_and_validate_image_input(**mm_kwargs)
        video_input = self._parse_and_validate_video_input(**mm_kwargs)
        by_name = {
            MODALITIES[0].name: image_input,
            MODALITIES[1].name: video_input,
        }
        deepstack_by_modality: dict[str, list[torch.Tensor] | None] = {
            s.name: None for s in MODALITIES
        }

        def features_for(
            spec: ModalitySpec, tail_starts: list[int], merge: int
        ) -> torch.Tensor | None:
            mm_input = by_name[spec.name]
            if mm_input is None:
                return None
            tail_feats, tail_deepstack = self._encode_tail_feats_and_deepstack(
                pixel_values=mm_input[spec.pixel_key],
                grid_thw=mm_input[spec.grid_key],
                tail_starts=tail_starts,
                merge=merge,
            )
            deepstack_by_modality[spec.name] = tail_deepstack
            return tail_feats

        return DeepstackTailProvider(
            features_for=features_for,
            deepstack_by_modality=deepstack_by_modality,
        )

    def _make_cache_tail_provider(
        self, cached_mm_outputs: list[dict]
    ) -> DeepstackTailProvider:
        """Cache tail provider that also captures the cached per-modality deepstack
        into ``deepstack_by_modality``."""
        deepstack_by_modality: dict[str, list[torch.Tensor] | None] = {
            s.name: None for s in MODALITIES
        }

        def features_for(
            spec: ModalitySpec, tail_starts: list[int], merge: int
        ) -> torch.Tensor | None:
            caches = [c for c in cached_mm_outputs if spec.embeds_key in c]
            if not caches:
                return None
            feats = torch.cat(
                [c[spec.embeds_key].to(self.dtype) for c in caches], dim=0
            )
            grid_thw = torch.cat(
                [c[spec.grid_key].to(torch.int64) for c in caches], dim=0
            )
            counts = self._mm_feature_counts(grid_thw, merge).tolist()
            assert len(counts) == len(tail_starts), (
                f"kept-item count mismatch: {len(counts)} grids vs "
                f"{len(tail_starts)} tail starts"
            )
            deepstack_by_modality[spec.name] = self._slice_cached_side_to_tail(
                caches, spec, counts, tail_starts
            )
            return self._slice_to_tail(feats, counts, tail_starts)

        return DeepstackTailProvider(
            features_for=features_for,
            deepstack_by_modality=deepstack_by_modality,
        )

    def _build_partial_tail(
        self, model_input: ModelInputForRBLN, provider: DeepstackTailProvider
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Tail ``inputs_embeds`` + packed deepstack/visual mask (partial hit).

        ``_scatter_tail_mm`` places the tail features and, as a side effect, fills
        ``provider.deepstack_by_modality`` with each present modality's tail
        deepstack. Placeholder masks are recomputed here (cheap) to pack it.
        """
        inputs_embeds = self._scatter_tail_mm(model_input, provider)
        deepstack_by_modality = provider.deepstack_by_modality
        tail_ids = model_input.input_tokens
        config = self.model.config
        masks: dict[str, torch.Tensor | None] = {
            spec.name: (tail_ids == getattr(config, spec.token_attr))
            if deepstack_by_modality.get(spec.name) is not None
            else None
            for spec in MODALITIES
        }
        visual_pos_mask, deepstack_embeds = self._pack_partial_deepstack(
            masks, deepstack_by_modality
        )
        return inputs_embeds, visual_pos_mask, deepstack_embeds

    def _build_full_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Full prefill; also carries Qwen3-VL visual_pos_mask + deepstack."""
        input_ids = model_input.input_tokens
        image_input = None
        video_input = None
        if model_input.multi_modal_kwargs:
            image_input = self._parse_and_validate_image_input(
                **model_input.multi_modal_kwargs
            )
            video_input = self._parse_and_validate_video_input(
                **model_input.multi_modal_kwargs
            )
        self._assert_mm_grid_tokens_match(input_ids, image_input, video_input)
        attention_mask = torch.ones_like(input_ids)
        prefill_params = self.preprocess_prefill(
            input_ids, attention_mask, image_input, video_input
        )
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=prefill_params["inputs_embeds"],
            position_embed=position_embed,
            visual_pos_mask=prefill_params.get("visual_pos_mask"),
            deepstack_embeds=prefill_params.get("deepstack_embeds"),
        )

    def _build_partial_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Partial prefill tail; also carries Qwen3-VL visual_pos_mask + deepstack."""
        mm_kwargs = model_input.multi_modal_kwargs or {}
        provider = self._make_encoder_tail_provider(mm_kwargs)
        inputs_embeds, visual_pos_mask, deepstack_embeds = self._build_partial_tail(
            model_input, provider
        )
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=inputs_embeds,
            position_embed=position_embed,
            visual_pos_mask=visual_pos_mask,
            deepstack_embeds=deepstack_embeds,
        )

    def _build_partial_prefill_inputs_from_cache(
        self,
        model_input: ModelInputForRBLN,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None,
        running_requests_ids: list[str] | None,
        mrope_position_deltas: dict[str, float] | None,
    ) -> dict:
        """EC-consumer partial prefill; also carries visual_pos_mask + deepstack."""
        provider = self._make_cache_tail_provider(cached_mm_outputs)
        inputs_embeds, visual_pos_mask, deepstack_embeds = self._build_partial_tail(
            model_input, provider
        )
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        if running_requests_ids and mrope_position_deltas is not None:
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

        params = {
            "inputs_embeds": inputs_embeds,
            "position_embed": position_embed,
            "cache_position": cache_position,
        }
        if visual_pos_mask is not None:
            params["visual_pos_mask"] = visual_pos_mask
        if deepstack_embeds is not None:
            params["deepstack_embeds"] = deepstack_embeds
        return params

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        """Prefill forward that feeds visual_pos_mask + deepstack to the prefill
        decoder; decode is unchanged (delegated to the base)."""
        if not model_input.is_prompt:
            return super().forward(model_input, **kwargs)

        input_ids = model_input.input_tokens
        request_nums = input_ids.shape[0]
        assert len(model_input.running_requests_ids) == request_nums, (
            f"The number of running requests is "
            f"{len(model_input.running_requests_ids)}, "
            f"but the shape of input_ids is {input_ids.shape}"
        )
        decoder_kwargs = self.preprocess_for_decoder(
            True, model_input.block_tables, input_ids, model_input.input_positions
        )
        prefill_kwargs = {
            "inputs_embeds": model_input.inputs_embeds,
            "position_embed": model_input.position_embed,
            "block_tables": decoder_kwargs.pop("block_tables"),
            "cache_position": decoder_kwargs.pop("cache_position"),
        }
        if model_input.visual_pos_mask is not None:
            prefill_kwargs["visual_pos_mask"] = model_input.visual_pos_mask
        if model_input.deepstack_embeds is not None:
            prefill_kwargs["deepstack_embeds"] = model_input.deepstack_embeds
        return self.model.prefill_decoder(**prefill_kwargs).logits

    def _slice_cached_side_to_tail(self, caches, spec, counts, starts):
        """Tail-slice the cached per-layer deepstack for one modality (EC)."""
        key = f"deepstack_{spec.name}_embeds"
        present = [c for c in caches if c.get(key) is not None]
        if not present:
            return None
        num_layers = len(present[0][key])
        layers = [
            torch.cat([c[key][layer].to(self.dtype) for c in present], dim=0)
            for layer in range(num_layers)
        ]
        return [self._slice_to_tail(layer, counts, starts) for layer in layers]

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
        second_per_grid_ts: torch.Tensor | None = None,
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
