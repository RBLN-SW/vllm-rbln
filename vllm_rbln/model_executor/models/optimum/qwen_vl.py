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
from dataclasses import replace
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


class RBLNOptimumQwenVLForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumMultimodalMixin, RBLNOptimumDecoderMixin, ABC
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
        **extra_preprocess_args,
    ) -> dict:
        """Build ``_preprocess_prefill`` kwargs and run it.

        Cached ``*_embeds`` skip the encoder; ``extra_preprocess_args`` pass
        through to the model (e.g. Qwen3-VL deepstack embeds).
        """

        preprocess_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Per modality: (parsed input, grid key, pixel key, embeds key, label).
        for mm_input, grid_key, pixel_key, embeds_key, label in (
            (image_input, "image_grid_thw", "pixel_values", "image_embeds", "image"),
            (
                video_input,
                "video_grid_thw",
                "pixel_values_videos",
                "video_embeds",
                "video",
            ),
        ):
            if mm_input is None:
                preprocess_args[pixel_key] = None
                preprocess_args[grid_key] = None
                continue
            preprocess_args[grid_key] = mm_input[grid_key]
            if mm_input.get("type") == embeds_key:
                logger.info(
                    "Prefill: using cached %s embeddings (encoder skipped)", label
                )
                preprocess_args[embeds_key] = mm_input[embeds_key]
                preprocess_args[pixel_key] = None
            else:
                logger.info("Prefill: running visual encoder (%s)", pixel_key)
                preprocess_args[pixel_key] = mm_input[pixel_key]

        # Add model-specific parameters
        self._add_model_specific_args(preprocess_args, video_input)

        # Model-specific pass-through args (e.g. Qwen3-VL deepstack embeds).
        preprocess_args.update(extra_preprocess_args)

        # Call the actual preprocessing
        preprocess_outputs = self.model._preprocess_prefill(**preprocess_args)
        prefill_params = self._build_prefill_params(preprocess_outputs)
        return prefill_params

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
        for mm_input, grid_key, pixel_key in (
            (image_input, "image_grid_thw", "pixel_values"),
            (video_input, "video_grid_thw", "pixel_values_videos"),
        ):
            preprocess_args[pixel_key] = None
            preprocess_args[grid_key] = (
                mm_input[grid_key] if mm_input is not None else None
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

        for mm_input, grid_key, token_id in (
            (image_input, "image_grid_thw", getattr(config, "image_token_id", None)),
            (video_input, "video_grid_thw", getattr(config, "video_token_id", None)),
        ):
            if mm_input is None or token_id is None:
                continue
            grid_thw = mm_input.get(grid_key)
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

        mrope_position_deltas[model_input.running_requests_ids[0]] = prefill_params[
            "rope_deltas"
        ].item()
        return replace(
            model_input,
            inputs_embeds=prefill_params["inputs_embeds"],
            position_embed=prefill_params["position_embed"],
            visual_pos_mask=prefill_params.get("visual_pos_mask"),
            deepstack_embeds=prefill_params.get("deepstack_embeds"),
        )

    def _build_partial_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Prefill for a partial prefix-cache hit (boundary may end inside an
        image). The cached prefix KV is reused via ``copy_cached_kv_blocks``;
        here we build the uncached tail's embeds/positions (and Qwen3-VL
        deepstack). Done manually, not via ``_preprocess_prefill``, whose
        ``get_rope_index`` chokes on the tail's orphaned image-pad tokens.
        """
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
        # (modality, parsed input, pixel key, grid key, placeholder token id)
        modalities = (
            (
                "image",
                image_input,
                "pixel_values",
                "image_grid_thw",
                config.image_token_id,
            ),
            (
                "video",
                video_input,
                "pixel_values_videos",
                "video_grid_thw",
                config.video_token_id,
            ),
        )
        masks: dict[str, torch.Tensor | None] = {"image": None, "video": None}
        sides: dict[str, list[torch.Tensor] | None] = {"image": None, "video": None}
        for modality, mm_input, pixel_key, grid_key, token_id in modalities:
            if mm_input is None:
                continue
            tail_feats, side = self._encode_and_slice_mm(
                pixel_values=mm_input[pixel_key],
                grid_thw=mm_input[grid_key],
                tail_starts=tail_starts.get(modality, []),
                merge=merge,
            )
            mask = tail_ids == token_id
            inputs_embeds[mask] = tail_feats.to(inputs_embeds.dtype)
            masks[modality] = mask
            sides[modality] = side
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
                # Required for prefix-cache reuse: tells the prefill where to
                # write KV (offset = num cached tokens) and to attend to the
                # copied cached KV before it. Without it the prefill writes from
                # 0 and ignores the reused KV. Harmless when nothing is cached
                # (cache_position starts at 0).
                "cache_position": cache_position,
            }
            # Qwen3-VL / Qwen3-VL-Moe feed visual_pos_mask + deepstack_embeds to
            # the prefill decoder; Qwen2/2.5-VL leave these None and skip them.
            if model_input.visual_pos_mask is not None:
                prefill_kwargs["visual_pos_mask"] = model_input.visual_pos_mask
            if model_input.deepstack_embeds is not None:
                prefill_kwargs["deepstack_embeds"] = model_input.deepstack_embeds
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

    def get_language_model(self):
        return self.model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | dict:
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
        **extra_preprocess_args,
    ) -> dict:
        """
        Build prefill_decoder kwargs from cached encoder outputs (EC consumer).

        ``extra_preprocess_args`` are passed through to ``preprocess_prefill``
        (e.g. Qwen3-VL forwards its cached deepstack features there).
        """
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

        attention_mask = torch.ones_like(input_ids)
        prefill_params = self.preprocess_prefill(
            input_ids,
            attention_mask,
            image_input,
            video_input,
            **extra_preprocess_args,
        )

        rope_deltas = prefill_params.pop("rope_deltas", None)
        if (
            rope_deltas is not None
            and running_requests_ids
            and mrope_position_deltas is not None
        ):
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

        return prefill_params


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
