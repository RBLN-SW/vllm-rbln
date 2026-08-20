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
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLImageEmbeddingInputs,
    Qwen2VLImagePixelInputs,
    Qwen2VLVideoEmbeddingInputs,
    Qwen2VLVideoPixelInputs,
)

from vllm_rbln.utils.optimum.bucket import select_bucket_size

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModalitySpec:
    """Per-modality kwarg keys and the config attribute for its placeholder id."""

    name: str  # "image" | "video"
    grid_key: str  # grid_thw kwarg key
    pixel_key: str  # pixel-values kwarg key
    embeds_key: str  # cached-embeds kwarg key / input "type" marker
    token_attr: str  # config attribute holding the placeholder token id


MODALITIES: tuple[ModalitySpec, ModalitySpec] = (
    ModalitySpec(
        "image", "image_grid_thw", "pixel_values", "image_embeds", "image_token_id"
    ),
    ModalitySpec(
        "video",
        "video_grid_thw",
        "pixel_values_videos",
        "video_embeds",
        "video_token_id",
    ),
)


class RBLNOptimumQwenVLForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
    RBLNOptimumDecoderMixin,
    ABC,
):
    """
    Unified class for both Qwen2-VL and Qwen2.5-VL models.
    Automatically detects model type based on the model configuration.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        if self._is_ec_producer_only():
            return
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def get_prefill_decoder(self):
        return self.model.prefill_decoder

    def get_language_model(self):
        return self.model

    @abstractmethod
    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """
        Add model-specific arguments to preprocessing args.

        Args:
            preprocess_args: Dictionary of preprocessing arguments to modify
            video_input: Video input data
        """
        pass

    @abstractmethod
    def _create_image_pixel_inputs(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> Any:
        """Create image pixel inputs based on model type"""
        pass

    @abstractmethod
    def _create_image_embedding_inputs(
        self, image_embeds: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> Any:
        """Create image embedding inputs based on model type"""
        pass

    @abstractmethod
    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None,
    ) -> Any:
        """Create video pixel inputs based on model type"""
        pass

    @abstractmethod
    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw) -> Any:
        """Create video embedding inputs based on model type"""
        pass

    def _parse_and_validate_image_input(self, **kwargs: Any) -> Any | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return self._create_image_pixel_inputs(
                pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )

        if image_embeds is not None:
            return self._create_image_embedding_inputs(
                image_embeds=image_embeds, image_grid_thw=image_grid_thw
            )

        # fallback return if both are None
        return None

    def _parse_and_validate_video_input(self, **kwargs: object) -> Any | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return self._create_video_pixel_inputs(
                pixel_values_videos, video_grid_thw, second_per_grid_ts
            )

        if video_embeds is not None:
            return self._create_video_embedding_inputs(video_embeds, video_grid_thw)

        # fallback return if both are None
        return None

    def _process_image_input(self, image_input) -> dict:
        result = {}
        if image_input is not None and image_input.get("type") == "pixel_values":
            result["image_embeds"] = self.model.visual(
                image_input["pixel_values"], grid_thw=image_input["image_grid_thw"]
            )
            result["image_grid_thw"] = image_input["image_grid_thw"]
        return result

    def _process_video_input(self, video_input) -> dict:
        result = {}
        if video_input is not None and video_input.get("type") == "pixel_values_videos":
            result["video_embeds"] = self.model.visual(
                video_input["pixel_values_videos"],
                grid_thw=video_input["video_grid_thw"],
            )
            result["video_grid_thw"] = video_input["video_grid_thw"]
            second_per_grid_ts = video_input.get("second_per_grid_ts", None)
            if second_per_grid_ts is not None:
                result["second_per_grid_ts"] = second_per_grid_ts
        return result

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | dict:
        """Encode the vision inputs into the whole-prompt cacheable unit (per
        modality: features + grid, plus Qwen3-VL deepstack).

        The single non-EC encode entry point: the EC producer
        (``_run_encoder_and_save``, which caches the result) and both non-EC
        prefill builders call it, then scatter (full) or tail-slice + scatter
        (partial). EC-consumer prefill sources the same representation from the
        cache via ``_cache_to_mm`` instead.
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)
        if image_input is None and video_input is None:
            return []

        # Merge the per-modality encoder outputs into a single cacheable dict
        # (consumed on the decode side by build_prefill_inputs_from_cache()).
        result = {}
        result.update(self._process_image_input(image_input))
        result.update(self._process_video_input(video_input))
        return result

    def _build_full_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Whole-prompt prefill: the partial path without the tail slice
        (encode the full prompt, then scatter). MRoPE positions come from the
        shared ``_build_prefill_position_embed`` (num_cached == 0 keeps the
        whole prompt).
        """
        mm = self.embed_multimodal(**(model_input.multi_modal_kwargs or {}))
        inputs_embeds = self.embed_input_ids(model_input.input_tokens, mm)
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=inputs_embeds,
            position_embed=position_embed,
        )

    def _build_partial_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Uncached-tail prefill. Mirrors the base
        ``RBLNOptimumMultimodalMixin`` flow (encode → tail-slice → scatter),
        plus the Qwen-VL MRoPE positions.
        """
        assert model_input.partial_prefix is not None
        mm = self.embed_multimodal(**(model_input.multi_modal_kwargs or {}))
        mm = self._build_partial_mm_embeds(model_input.partial_prefix, mm)
        inputs_embeds = self.embed_input_ids(model_input.input_tokens, mm)
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=inputs_embeds,
            position_embed=position_embed,
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
        """Build prefill_decoder kwargs from cached encoder outputs (EC
        consumer). Same flow as the non-EC prefill; the whole-prompt features
        come from the encoder cache (``_cache_to_mm``) instead of the vision
        encoder. A partial prefix-cache hit additionally tail-slices via
        ``_build_partial_prefill_inputs_from_cache``.
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
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        if running_requests_ids and mrope_position_deltas is not None:
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()
        return {
            "inputs_embeds": inputs_embeds,
            "position_embed": position_embed,
            "cache_position": cache_position,
        }

    def _build_partial_prefill_inputs_from_cache(
        self,
        model_input: ModelInputForRBLN,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None,
        running_requests_ids: list[str] | None,
        mrope_position_deltas: dict[str, float] | None,
    ) -> dict:
        """EC-consumer partial prefill. Same flow as the non-EC
        ``_build_partial_prefill_forward_inputs``; the tail features come from
        the encoder cache (``_cache_to_mm``) instead of the vision encoder.
        """
        assert model_input.partial_prefix is not None
        mm = self._cache_to_mm(cached_mm_outputs)
        mm = self._build_partial_mm_embeds(model_input.partial_prefix, mm)
        inputs_embeds = self.embed_input_ids(model_input.input_tokens, mm)
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        if running_requests_ids and mrope_position_deltas is not None:
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

        return {
            "inputs_embeds": inputs_embeds,
            "position_embed": position_embed,
            "cache_position": cache_position,
        }

    def _cache_to_mm(self, cached_mm_outputs: list[dict]) -> dict:
        """Merge the producer's per-item cached encoder outputs into the same
        whole-prompt representation ``embed_multimodal`` produces (per-modality
        features concatenated across items). Qwen3-VL overrides to also carry
        the cached deepstack.
        """
        mm: dict = {}
        for spec in MODALITIES:
            caches = [c for c in cached_mm_outputs if spec.embeds_key in c]
            if not caches:
                continue
            mm[spec.embeds_key] = torch.cat([c[spec.embeds_key] for c in caches], dim=0)
            mm[spec.grid_key] = torch.cat(
                [c[spec.grid_key].to(torch.int64) for c in caches], dim=0
            )
        return mm

    def _build_partial_mm_embeds(
        self, partial_prefix: Any, multimodal_embeddings: Any
    ) -> dict:
        """Slice each modality's whole-prompt features down to the uncached
        tail (the encoder already ran over full items). Qwen3-VL overrides to
        also slice its deepstack side outputs.
        """
        if not isinstance(multimodal_embeddings, dict) or not multimodal_embeddings:
            return {}
        mm = multimodal_embeddings
        tail_starts = partial_prefix.mm_embed_tail_starts or {}
        merge = self.model.config.vision_config.spatial_merge_size
        sliced: dict = {}
        for spec in MODALITIES:
            if spec.embeds_key not in mm:
                continue
            counts = self._mm_feature_counts(mm[spec.grid_key], merge).tolist()
            starts = tail_starts.get(spec.name, [])
            assert len(counts) == len(starts), (
                f"kept-item count mismatch: {len(counts)} grids vs "
                f"{len(starts)} tail starts"
            )
            sliced[spec.embeds_key] = self._slice_to_tail(
                mm[spec.embeds_key], counts, starts
            )
        return sliced

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed the text tokens, then scatter each modality's features over
        its placeholder positions.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        if not isinstance(multimodal_embeddings, dict) or not multimodal_embeddings:
            return inputs_embeds
        config = self.model.config
        for spec in MODALITIES:
            embeds = multimodal_embeddings.get(spec.embeds_key)
            if embeds is None:
                continue
            mask = input_ids == getattr(config, spec.token_attr)
            inputs_embeds[mask] = embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def _build_prefill_position_embed(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MRoPE ``(position_embed, rope_deltas)`` for prefill, unified across
        full and partial prefix-cache hits.

        Each multimodal item shifts every later token's position, so MRoPE
        positions depend on the whole prompt layout and cannot be computed from
        the uncached tail alone. They are therefore always computed over the
        full prompt with the encoder skipped (grids only), then sliced to the
        uncached window ``[num_cached:]``:

        - full prefill: ``num_cached == 0``, so the whole prompt is kept;
        - partial hit: only the uncached tail is kept.

        ``rope_deltas`` is over the full sequence (used for decode positions).
        """
        partial = model_input.partial_prefix
        if partial is not None:
            full_input_ids = partial.full_input_tokens
            num_cached = partial.num_cached_tokens
            mm_kwargs = partial.mrope_mm_kwargs
        else:
            full_input_ids = model_input.input_tokens
            num_cached = 0
            mm_kwargs = model_input.multi_modal_kwargs

        image_input = None
        video_input = None
        if mm_kwargs:
            image_input = self._parse_and_validate_image_input(**mm_kwargs)
            video_input = self._parse_and_validate_video_input(**mm_kwargs)

        attention_mask = torch.ones_like(full_input_ids)
        params = self._compute_mrope_position(
            full_input_ids, attention_mask, image_input, video_input
        )
        # position_embed: [2, batch, 1, N, head_dim]; slice the sequence (dim=-2)
        # to the uncached window (whole prompt when num_cached == 0).
        position_embed = params["position_embed"][..., num_cached:, :]
        return position_embed, params["rope_deltas"]

    def _compute_mrope_position(
        self, input_ids, attention_mask, image_input, video_input
    ) -> dict:
        """MRoPE positions only: run ``get_rope_index`` with grids but no
        ``pixel_values`` (encoder skipped). Returns ``{position_embed,
        rope_deltas}``.
        """
        preprocess_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        for spec, mm_input in zip(MODALITIES, (image_input, video_input)):
            preprocess_args[spec.pixel_key] = None
            preprocess_args[spec.grid_key] = (
                mm_input[spec.grid_key] if mm_input is not None else None
            )
        # second_per_grid_ts (video, Qwen2.5-VL) feeds get_rope_index too.
        self._add_model_specific_args(preprocess_args, video_input)

        outputs = self.model._preprocess_prefill(**preprocess_args)
        # outputs[1]/[2] = position_embed/rope_deltas across all variants; the
        # rest of the tuple's arity differs, so don't unpack it.
        return {"position_embed": outputs[1], "rope_deltas": outputs[2]}

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

    @staticmethod
    def _mm_feature_counts(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
        """Per-item feature/placeholder count from ``grid_thw`` (patches merged
        ``merge_size x merge_size``, 1:1 token<->feature)."""
        return grid_thw.prod(dim=-1) // (merge_size**2)

    def compute_decode_position_embed(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> torch.Tensor:
        """Decode-step MRoPE: advance each request's position from its stored
        delta (``cache_position + mrope_position_delta``) and return the padded
        position embeddings (cos/sin). Mirrors upstream vLLM's
        ``get_next_input_positions_tensor``.
        """
        cache_position = model_input.input_positions
        running_requests_ids = model_input.running_requests_ids
        # int32 mirrors the cache_position dtype the prior decode path used
        # (cast in preprocess_for_decoder before computing position embeds).
        cache_position = cache_position.to(torch.int32)
        padded_batch_size = self.decoder_batch_size
        if self.use_multiple_decoder:
            padded_batch_size = select_bucket_size(
                len(running_requests_ids), self.decoder_batch_sizes
            )

        position_embeds = []
        for b_id, request_id in enumerate(running_requests_ids):
            delta = cache_position[b_id] + mrope_position_deltas[request_id]
            position_ids = torch.arange(1).view(1, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_embed = self.model._get_position_embeddings(
                torch.zeros(1, dtype=self.dtype), position_ids
            )
            position_embeds.append(position_embed)

        for _ in range(padded_batch_size - len(running_requests_ids)):
            position_embeds.append(torch.zeros_like(position_embeds[0]))

        return torch.cat(position_embeds, dim=1)

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        # FIXME This should be removed in the future
        # by moving the padding logic into model runner.
        assert len(model_input.running_requests_ids) == request_nums, (
            f"The number of running requests is "
            f"{len(model_input.running_requests_ids)}, "
            f"but the shape of input_ids is {input_ids.shape}"
        )

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
            }
            logits = self.model.prefill_decoder(**prefill_kwargs).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.decoder = self.model.decoders[padded_batch_size]
            input_ids = kwargs.pop("input_ids")
            inputs_embeds = self.model.embed_tokens(input_ids)
            logits = self.model.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=model_input.position_embed,
                block_tables=block_tables,
            ).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits


class RBLNOptimumQwen2_5_VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Add second_per_grid_ts for Qwen2.5-VL"""
        if video_input is not None:
            second_per_grid_ts = video_input.get("second_per_grid_ts", None)
            if second_per_grid_ts is not None:
                preprocess_args["second_per_grid_ts"] = second_per_grid_ts

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return Qwen2_5_VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return Qwen2_5_VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts=torch.Tensor | None,
    ):
        if second_per_grid_ts is None:
            raise ValueError(
                "second_per_grid_ts is required for Qwen2.5-VL video inputs."
            )
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2_5_VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
        )


class RBLNOptimumQwen2VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Qwen2-VL doesn't need additional arguments"""
        pass

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return Qwen2VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return Qwen2VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None,
    ):
        # NOTE Qwen2-VL doesn't use second_per_grid_ts
        return Qwen2VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
        )
