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
from dataclasses import replace
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLVideoPixelInputs,
)

from .base import ModelInputForRBLN
from .qwen2_vl import MODALITIES, RBLNOptimumQwen2_5_VLForConditionalGeneration

logger = init_logger(__name__)


class RBLNOptimumQwen3VLForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Qwen3-VL reuses Qwen2.5-VL classes with the same implementation.
    However, since Qwen3-VL does not require second_per_grid_ts,
    certain methods are overridden to exclude it from the model inputs.

    Qwen3-VL also emits per-layer *deepstack* side outputs from the vision
    encoder. They flow alongside the base multimodal features (the ``mm`` dict
    produced by ``embed_multimodal`` / ``_cache_to_mm``), are tail-sliced by
    ``_build_partial_mm_embeds``, and packed for the prefill decoder by
    ``_pack_deepstack_from_mm``.
    """

    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Qwen3-VL doesn't need additional arguments"""
        pass

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

    def _build_full_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Whole-prompt prefill + Qwen3-VL deepstack: the partial path without
        the tail slice.
        """
        mm = self.embed_multimodal(**(model_input.multi_modal_kwargs or {}))
        inputs_embeds = self.embed_input_ids(model_input.input_tokens, mm)
        visual_pos_mask, deepstack_embeds = self._pack_deepstack_from_mm(
            model_input.input_tokens, mm
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

    def _build_partial_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Uncached-tail prefill + Qwen3-VL deepstack. Same flow as the base
        ``RBLNOptimumQwenVLForConditionalGeneration`` plus the deepstack pack.
        """
        assert model_input.partial_prefix is not None
        mm = self.embed_multimodal(**(model_input.multi_modal_kwargs or {}))
        mm = self._build_partial_mm_embeds(model_input.partial_prefix, mm)
        inputs_embeds = self.embed_input_ids(model_input.input_tokens, mm)
        visual_pos_mask, deepstack_embeds = self._pack_deepstack_from_mm(
            model_input.input_tokens, mm
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

    def build_prefill_inputs_from_cache(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None = None,
        running_requests_ids: list[str] | None = None,
        mrope_position_deltas: dict[str, float] | None = None,
        model_input: ModelInputForRBLN | None = None,
    ) -> dict:
        """EC consumer + Qwen3-VL deepstack. Same flow as the base; the
        whole-prompt features (incl. cached deepstack) come from
        ``_cache_to_mm``. Partial hits additionally tail-slice.
        """
        assert model_input is not None
        if model_input.partial_prefix is not None:
            return self._build_partial_prefill_inputs_from_cache(
                model_input,
                cached_mm_outputs,
                cache_position=cache_position,
                running_requests_ids=running_requests_ids,
                mrope_position_deltas=mrope_position_deltas,
            )

        mm = self._cache_to_mm(cached_mm_outputs)
        inputs_embeds = self.embed_input_ids(input_ids, mm)
        visual_pos_mask, deepstack_embeds = self._pack_deepstack_from_mm(input_ids, mm)
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

    def _build_partial_prefill_inputs_from_cache(
        self,
        model_input: ModelInputForRBLN,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None,
        running_requests_ids: list[str] | None,
        mrope_position_deltas: dict[str, float] | None,
    ) -> dict:
        """EC-consumer partial prefill + Qwen3-VL deepstack. Same flow as the
        base version; the tail features come from ``_cache_to_mm``.
        """
        assert model_input.partial_prefix is not None
        mm = self._cache_to_mm(cached_mm_outputs)
        mm = self._build_partial_mm_embeds(model_input.partial_prefix, mm)
        inputs_embeds = self.embed_input_ids(model_input.input_tokens, mm)
        visual_pos_mask, deepstack_embeds = self._pack_deepstack_from_mm(
            model_input.input_tokens, mm
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

    def _cache_to_mm(self, cached_mm_outputs: list[dict]) -> dict:
        """Also carry the producer's cached per-layer deepstack."""
        mm = super()._cache_to_mm(cached_mm_outputs)
        image_caches = [c for c in cached_mm_outputs if "image_embeds" in c]
        video_caches = [c for c in cached_mm_outputs if "video_embeds" in c]
        deepstack_image_embeds, deepstack_video_embeds = self._extract_cached_deepstack(
            image_caches, video_caches
        )
        if deepstack_image_embeds is not None:
            mm["deepstack_image_embeds"] = deepstack_image_embeds
        if deepstack_video_embeds is not None:
            mm["deepstack_video_embeds"] = deepstack_video_embeds
        return mm

    def _build_partial_mm_embeds(
        self, partial_prefix: Any, multimodal_embeddings: Any
    ) -> dict:
        """Also tail-slice the per-layer deepstack alongside the base features."""
        sliced = super()._build_partial_mm_embeds(partial_prefix, multimodal_embeddings)
        if not sliced:
            return sliced
        mm = multimodal_embeddings
        tail_starts = partial_prefix.mm_embed_tail_starts or {}
        merge = self.model.config.vision_config.spatial_merge_size
        for spec in MODALITIES:
            ds_key = f"deepstack_{spec.name}_embeds"
            layers = mm.get(ds_key)
            if layers is None:
                continue
            counts = self._mm_feature_counts(mm[spec.grid_key], merge).tolist()
            starts = tail_starts.get(spec.name, [])
            sliced[ds_key] = [
                self._slice_to_tail(layer, counts, starts) for layer in layers
            ]
        return sliced

    def _pack_deepstack_from_mm(
        self, input_ids: torch.Tensor, mm: dict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build each modality's placeholder mask + tail deepstack from the
        (already tail-sliced) ``mm`` and pack them for the prefill decoder."""
        config = self.model.config
        masks: dict[str, torch.Tensor | None] = {}
        deepstacks: dict[str, list[torch.Tensor] | None] = {}
        for spec in MODALITIES:
            layers = mm.get(f"deepstack_{spec.name}_embeds")
            deepstacks[spec.name] = layers
            masks[spec.name] = (
                input_ids == getattr(config, spec.token_attr)
                if layers is not None
                else None
            )
        return self._pack_partial_deepstack(masks, deepstacks)

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


class RBLNOptimumQwen3VLMoeForConditionalGeneration(
    RBLNOptimumQwen3VLForConditionalGeneration
):
    """
    Qwen3-VL MoE model shares the same input structure as Qwen3-VL,
    so it inherits from RBLNOptimumQwen3VLForConditionalGeneration without changes.
    """

    pass


class RBLNOptimumQwen3_5ForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Vision-language Qwen3.5 for RBLN.

    Qwen3.5 is "Qwen3-VL WITHOUT deepstack": a Qwen3-VL-style vision tower + mRoPE, but
    a HYBRID text backbone (GatedDeltaNet ``linear_attention`` layers + gated
    ``full_attention``). Its vision encoder returns ONLY the merged image embeds (a
    single tensor, no deepstack tuple), so it inherits the non-deepstack multimodal /
    prefill path from Qwen2.5-VL rather than the Qwen3-VL wrapper — whose
    ``_process_image_input`` unpacks a ``(image_embeds, deepstack)`` 2-tuple and whose
    prefill build packs deepstack, both of which break on Qwen3.5's single-tensor
    visual output. The only thing Qwen3.5 changes vs Qwen2.5-VL is that it has no
    ``second_per_grid_ts`` and names its placeholder ``image_token_id``.

    The ``linear_attention`` layers' ``conv_state``/``recurrent_state`` caches and the
    0/1 control masks (``conv_state_mask``/``recurrent_state_mask``/``valid_mask``) are
    handled entirely inside optimum-rbln's ``RBLNQwen3_5RuntimeModel``; this wrapper
    passes only the standard prefill/decode kwargs. ``full_attention`` layers keep the
    on-device paged KV cache.
    """

    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        # Qwen3.5 (like Qwen3-VL) has no ``second_per_grid_ts``; its ``get_rope_index``
        # separates videos by timestamps instead.
        pass

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None = None,
    ):
        # Qwen2.5-VL's version raises when ``second_per_grid_ts`` is None; Qwen3.5 never
        # has it, so build the carrier without requiring it (mirrors Qwen3-VL).
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _image_token_id(self) -> int:
        # Qwen3.5's HF config names the placeholder ``image_token_id`` (top-level), not
        # ``image_token_index`` as the mixin default assumes. Used by the prefill path's
        # embed_input_ids() image-token scatter, so this must be correct.
        return self.model.config.image_token_id

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        """Qwen3.5 prefill must feed ``batch_idx`` (the base Qwen-VL forward does not).

        The GatedDeltaNet ``linear_attention`` conv/recurrent state is a max-batch
        static DRAM cache; ``batch_idx`` selects which slot this prefilled request
        writes so decode reads the same slot. optimum-rbln declares ``batch_idx`` as a
        REQUIRED prefill graph input (``get_input_info``), so omitting it raises
        ``KeyError: 'batch_idx'`` in ``RBLNQwen3_5RuntimeModel._run``.

        NOTE: this assumes the request maps to cache slot 0, which holds for
        ``max_num_seqs == 1``. Multi-request continuous batching needs a per-request
        slot assignment threaded from the model runner (TODO).
        """
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables
        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            prefill_kwargs = {
                "inputs_embeds": model_input.inputs_embeds,
                "position_embed": model_input.position_embed,
                "block_tables": block_tables,
                "cache_position": cache_position,
                # linear-attention cache slot (single-slot assumption; see NOTE).
                "batch_idx": 0,
            }
            logits = self.model.prefill_decoder(**prefill_kwargs).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.decoder = self.model.decoders[padded_batch_size]
            input_ids = kwargs.pop("input_ids")
            inputs_embeds = self.model.embed_tokens(input_ids).to(self.dtype)
            logits = self.model.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=model_input.position_embed,
                block_tables=block_tables,
            ).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits
