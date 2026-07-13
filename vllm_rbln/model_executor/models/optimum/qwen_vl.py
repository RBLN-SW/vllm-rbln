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
from collections.abc import Callable, Iterator
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


def iter_modalities(
    image_input: Any, video_input: Any
) -> Iterator[tuple[ModalitySpec, Any]]:
    """Pair each modality spec with its parsed input (image first, video next)."""
    yield MODALITIES[0], image_input
    yield MODALITIES[1], video_input


# Returns one modality's uncached-tail features for a partial prefix-cache hit,
# given the modality spec, its per-item uncached-tail start offsets, and the
# spatial merge size (``None`` when the modality is absent). The two providers
# below (``_make_encoder_tail_provider`` / ``_make_cache_tail_provider``) differ
# only in where the features come from — the vision encoder vs the encoder cache.
TailFeatureFn = Callable[[ModalitySpec, list[int], int], torch.Tensor | None]


@dataclass
class TailFeatureProvider:
    """Supplies each modality's uncached-tail features for a partial
    prefix-cache hit. ``features_for(spec, tail_starts, merge)`` returns that
    modality's tail features (or ``None`` when absent).

    Models with encoder side outputs (Qwen3-VL deepstack) extend this with a
    subclass that also accumulates those outputs; base Qwen2/2.5-VL needs only
    ``features_for``.
    """

    features_for: TailFeatureFn


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

    def _build_prefill_params(self, preprocess_outputs: tuple) -> dict:
        return {
            "inputs_embeds": preprocess_outputs[0],
            "position_embed": preprocess_outputs[1],
            "rope_deltas": preprocess_outputs[2],
        }

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

    def preprocess_prefill(
        self,
        input_ids,
        attention_mask,
        image_input,
        video_input,
    ) -> dict:
        """Build ``_preprocess_prefill`` kwargs and run it. Cached ``*_embeds``
        skip the encoder.
        """
        preprocess_args = self._build_preprocess_args(
            input_ids, attention_mask, image_input, video_input
        )
        preprocess_outputs = self.model._preprocess_prefill(**preprocess_args)
        return self._build_prefill_params(preprocess_outputs)

    def _build_preprocess_args(
        self,
        input_ids,
        attention_mask,
        image_input,
        video_input,
    ) -> dict:
        """Assemble the per-modality kwargs passed to ``_preprocess_prefill``.

        Qwen3-VL reuses this and adds its cached deepstack kwargs on the EC
        full-prefill path (see ``build_prefill_inputs_from_cache``).
        """
        preprocess_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        for spec, mm_input in iter_modalities(image_input, video_input):
            if mm_input is None:
                preprocess_args[spec.pixel_key] = None
                preprocess_args[spec.grid_key] = None
                continue
            preprocess_args[spec.grid_key] = mm_input[spec.grid_key]
            if mm_input.get("type") == spec.embeds_key:
                logger.info(
                    "Prefill: using cached %s embeddings (encoder skipped)",
                    spec.name,
                )
                preprocess_args[spec.embeds_key] = mm_input[spec.embeds_key]
                preprocess_args[spec.pixel_key] = None
            else:
                logger.info("Prefill: running visual encoder (%s)", spec.pixel_key)
                preprocess_args[spec.pixel_key] = mm_input[spec.pixel_key]

        self._add_model_specific_args(preprocess_args, video_input)
        return preprocess_args

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
        for spec, mm_input in iter_modalities(image_input, video_input):
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

    @staticmethod
    def _mm_feature_counts(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
        """Per-item feature/placeholder count from ``grid_thw`` (patches merged
        ``merge_size x merge_size``, 1:1 token<->feature)."""
        return grid_thw.prod(dim=-1) // (merge_size**2)

    def _assert_mm_grid_tokens_match(self, input_ids, image_input, video_input) -> None:
        """Assert each item's grid-derived feature count matches its placeholder
        count in ``input_ids``. Skipped if token ids / merge size are missing.
        """
        config = getattr(self.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        merge_size = getattr(vision_config, "spatial_merge_size", None)
        if merge_size is None:
            return

        for spec, mm_input in iter_modalities(image_input, video_input):
            token_id = getattr(config, spec.token_attr, None)
            if mm_input is None or token_id is None:
                continue
            grid_thw = mm_input.get(spec.grid_key)
            if grid_thw is None:
                continue
            num_embed_tokens = int(self._mm_feature_counts(grid_thw, merge_size).sum())
            num_placeholders = int((input_ids == token_id).sum())
            self._assert_mm_tokens_match(num_placeholders, num_embed_tokens)

    def _build_full_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
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

        # Positions come from the shared _build_prefill_position_embed so the
        # full and partial paths compute MRoPE identically (over the full
        # prompt, then slice [num_cached:]; num_cached == 0 here keeps the whole
        # prompt). preprocess_prefill also computed positions internally; that
        # extra get_rope_index pass is index-only (encoder already ran) and its
        # position output is discarded here.
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=prefill_params["inputs_embeds"],
            position_embed=position_embed,
        )

    def _build_partial_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Build the uncached tail's embeds/positions."""
        mm_kwargs = model_input.multi_modal_kwargs or {}
        provider = self._make_encoder_tail_provider(mm_kwargs)
        inputs_embeds = self._scatter_tail_mm(model_input, provider)
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(
            model_input,
            inputs_embeds=inputs_embeds,
            position_embed=position_embed,
        )

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

    # --- Partial prefix-cache-hit tail feature providers -------------------
    # A ``TailFeatureProvider`` yields, per modality, the uncached-tail features
    # for a partial prefix-cache hit. Its two factories capture the ONLY
    # difference between the non-EC and EC-consumer partial paths: where
    # full-item features come from — the vision encoder vs the producer's
    # encoder cache. ``_scatter_tail_mm`` is otherwise identical for both.
    # Qwen3-VL overrides the factories to return a ``DeepstackTailProvider``
    # that also accumulates its deepstack side outputs; base Qwen2/2.5-VL has
    # no side outputs.

    def _make_encoder_tail_provider(self, mm_kwargs: dict) -> TailFeatureProvider:
        """Tail provider that runs the vision encoder (non-EC path)."""
        image_input = self._parse_and_validate_image_input(**mm_kwargs)
        video_input = self._parse_and_validate_video_input(**mm_kwargs)
        by_name = {
            MODALITIES[0].name: image_input,
            MODALITIES[1].name: video_input,
        }

        def features_for(
            spec: ModalitySpec, tail_starts: list[int], merge: int
        ) -> torch.Tensor | None:
            mm_input = by_name[spec.name]
            if mm_input is None:
                return None
            return self._encode_tail_feats(
                pixel_values=mm_input[spec.pixel_key],
                grid_thw=mm_input[spec.grid_key],
                tail_starts=tail_starts,
                merge=merge,
            )

        return TailFeatureProvider(features_for=features_for)

    def _make_cache_tail_provider(
        self, cached_mm_outputs: list[dict]
    ) -> TailFeatureProvider:
        """Tail provider that reads the producer's encoder cache (EC path)."""

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
            return self._slice_to_tail(feats, counts, tail_starts)

        return TailFeatureProvider(features_for=features_for)

    def _scatter_tail_mm(
        self, model_input: ModelInputForRBLN, provider: TailFeatureProvider
    ) -> torch.Tensor:
        """Scatter each kept item's uncached-tail features into the tail embeds,
        pulling full-item features from ``provider`` (vision encoder or encoder
        cache), and return the tail ``inputs_embeds``.
        """
        partial = model_input.partial_prefix
        assert partial is not None
        tail_ids = model_input.input_tokens
        inputs_embeds = self.model.embed_tokens(tail_ids).to(
            self.model.rbln_config.dtype
        )
        tail_starts = partial.mm_embed_tail_starts or {}

        config = self.model.config
        merge = config.vision_config.spatial_merge_size
        for spec in MODALITIES:
            tail_feats = provider.features_for(
                spec, tail_starts.get(spec.name, []), merge
            )
            if tail_feats is None:
                continue
            mask = tail_ids == getattr(config, spec.token_attr)
            inputs_embeds[mask] = tail_feats.to(inputs_embeds.dtype)
        return inputs_embeds

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

    def _encode_tail_feats(
        self,
        *,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        tail_starts: list[int],
        merge: int,
    ) -> torch.Tensor:
        """Run the vision encoder on whole items, then slice each to its
        uncached tail. Returns the tail features. (Qwen3-VL uses its own
        ``_encode_tail_feats_and_deepstack`` to also produce deepstack.)
        """
        feats = self.model.visual(pixel_values, grid_thw=grid_thw)
        counts = self._mm_feature_counts(grid_thw, merge).tolist()
        assert len(counts) == len(tail_starts), (
            f"kept-item count mismatch: {len(counts)} grids vs "
            f"{len(tail_starts)} tail starts"
        )
        return self._slice_to_tail(feats, counts, tail_starts)

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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | dict:
        """Encoder-cache (EC) producer step: encode the vision inputs into the
        cacheable unit that the consumer later merges back.

        Used ONLY on the EC producer path (``_run_encoder_and_save``), which is
        gated to Qwen3-VL (``ec_enabled_model``). The normal prefill path does
        not call this: it builds inputs via ``preprocess_prefill`` (full) or
        ``_scatter_tail_mm`` (partial hit). So for Qwen2/2.5-VL, or Qwen3-VL
        without EC, this method is unreachable.
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
        """
        Build prefill_decoder kwargs from cached encoder outputs (EC consumer).

        On a partial prefix-cache hit (``model_input.partial_prefix`` set) the
        cached embeds are tail-sliced and MRoPE is recomputed over the full
        prompt, mirroring the non-EC partial prefill path. Otherwise the whole
        prompt is rebuilt from the cached embeds.
        """
        if model_input is not None and model_input.partial_prefix is not None:
            return self._build_partial_prefill_inputs_from_cache(
                model_input,
                cached_mm_outputs,
                cache_position=cache_position,
                running_requests_ids=running_requests_ids,
                mrope_position_deltas=mrope_position_deltas,
            )

        image_input, video_input = self._cache_to_embedding_inputs(cached_mm_outputs)
        attention_mask = torch.ones_like(input_ids)
        prefill_params = self.preprocess_prefill(
            input_ids, attention_mask, image_input, video_input
        )
        self._record_cache_rope_deltas(
            prefill_params, running_requests_ids, mrope_position_deltas
        )
        return prefill_params

    def _cache_to_embedding_inputs(
        self, cached_mm_outputs: list[dict]
    ) -> tuple[Any | None, Any | None]:
        """Rebuild the whole-prompt image/video embedding inputs from the
        producer's cached encoder outputs (EC full-prefill path)."""
        model_dtype = self.dtype
        image_caches = [c for c in cached_mm_outputs if "image_embeds" in c]
        video_caches = [c for c in cached_mm_outputs if "video_embeds" in c]

        image_input = None
        video_input = None

        if image_caches:
            image_embeds = torch.cat(
                [c["image_embeds"].to(model_dtype) for c in image_caches], dim=0
            )
            image_grid_thw = torch.cat(
                [c["image_grid_thw"].to(torch.int64) for c in image_caches], dim=0
            )
            image_input = self._create_image_embedding_inputs(
                image_embeds=image_embeds, image_grid_thw=image_grid_thw
            )

        if video_caches:
            video_embeds = torch.cat(
                [c["video_embeds"].to(model_dtype) for c in video_caches], dim=0
            )
            video_grid_thw = torch.cat(
                [c["video_grid_thw"].to(torch.int64) for c in video_caches], dim=0
            )
            video_input = self._create_video_embedding_inputs(
                video_embeds=video_embeds, video_grid_thw=video_grid_thw
            )
            # Qwen2.5-VL: second_per_grid_ts is per-video metadata; carry the
            # first feature's value as a best-effort for mixed batches.
            if "second_per_grid_ts" in video_caches[0]:
                video_input["second_per_grid_ts"] = video_caches[0][
                    "second_per_grid_ts"
                ]

        return image_input, video_input

    @staticmethod
    def _record_cache_rope_deltas(
        prefill_params: dict,
        running_requests_ids: list[str] | None,
        mrope_position_deltas: dict[str, float] | None,
    ) -> None:
        """Pop ``rope_deltas`` off the prefill params and record it for decode."""
        rope_deltas = prefill_params.pop("rope_deltas", None)
        if (
            rope_deltas is not None
            and running_requests_ids
            and mrope_position_deltas is not None
        ):
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

    def _build_partial_prefill_inputs_from_cache(
        self,
        model_input: ModelInputForRBLN,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None,
        running_requests_ids: list[str] | None,
        mrope_position_deltas: dict[str, float] | None,
    ) -> dict:
        """EC-consumer prefill for a partial prefix-cache hit. Mirrors the
        non-EC ``_build_partial_prefill_forward_inputs``: build the uncached
        tail's embeds from the cached features and recompute MRoPE positions
        over the full prompt, sliced to the tail.
        """
        provider = self._make_cache_tail_provider(cached_mm_outputs)
        inputs_embeds = self._scatter_tail_mm(model_input, provider)
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        if running_requests_ids and mrope_position_deltas is not None:
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

        return {
            "inputs_embeds": inputs_embeds,
            "position_embed": position_embed,
            "cache_position": cache_position,
        }


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
