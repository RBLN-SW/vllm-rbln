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

    Prefill inputs are built with the shared multimodal interface — the vision
    encoder runs in ``embed_multimodal`` and its outputs are scattered onto the
    placeholder positions in ``embed_input_ids``. MRoPE position embeddings
    (which the generic path does not need) are then produced by
    ``_compute_mrope``. This mirrors optimum-rbln's ``_preprocess_prefill``,
    which fuses those steps into one call, but keeps them separated so Qwen
    shares the same entry points as the other multimodal models.
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

    # ----- multimodal token ids (Qwen names them *_token_id on the config) -----

    def _image_token_id(self) -> int:
        return self.model.config.image_token_id

    def _video_token_id(self) -> int:
        return self.model.config.video_token_id

    def _vision_start_token_id(self) -> int:
        return self.model.config.vision_start_token_id

    # ----- encoder outputs (visual) --------------------------------------------

    def _process_image_input(self, image_input) -> dict:
        result = {}
        if image_input is not None and image_input.get("type") == "pixel_values":
            visual_out = self.model.visual(
                image_input["pixel_values"], grid_thw=image_input["image_grid_thw"]
            )
            # Qwen3-VL.visual returns (image_embeds, deepstack_features),
            # Qwen2/2.5-VL.visual returns a single tensor.
            if isinstance(visual_out, tuple):
                result["image_embeds"] = visual_out[0]
                if len(visual_out) > 1:
                    result["deepstack_image_embeds"] = visual_out[1]
            else:
                result["image_embeds"] = visual_out
            result["image_grid_thw"] = image_input["image_grid_thw"]
        return result

    def _process_video_input(self, video_input) -> dict:
        result = {}
        if video_input is not None and video_input.get("type") == "pixel_values_videos":
            visual_out = self.model.visual(
                video_input["pixel_values_videos"],
                grid_thw=video_input["video_grid_thw"],
            )
            if isinstance(visual_out, tuple):
                result["video_embeds"] = visual_out[0]
                if len(visual_out) > 1:
                    result["deepstack_video_embeds"] = visual_out[1]
            else:
                result["video_embeds"] = visual_out
            result["video_grid_thw"] = video_input["video_grid_thw"]
            second_per_grid_ts = video_input.get("second_per_grid_ts", None)
            if second_per_grid_ts is not None:
                result["second_per_grid_ts"] = second_per_grid_ts
        return result

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

    # ----- input embeddings (text lookup + multimodal scatter) -----------------

    def _scatter_mm(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter ``embeds`` onto the ``token_id`` placeholder positions.

        Mirrors the per-modality masked_scatter in optimum-rbln's
        ``_preprocess_prefill``, including the token/feature count check.
        """
        mask = input_ids == token_id
        n_tokens = int(mask.sum().item())
        n_features = embeds.shape[0]
        if n_tokens != n_features:
            raise ValueError(
                f"Multimodal features and tokens do not match: "
                f"tokens: {n_tokens}, features: {n_features}"
            )
        embeds = embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(mask_expanded, embeds)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | dict | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Text embedding lookup, then scatter the encoder outputs over the
        # placeholder positions. ``multimodal_embeddings`` is the dict produced
        # by embed_multimodal() (image/video embeds + grids). Image and video
        # are scattered in two independent passes so the result is invariant to
        # their interleaving order within a request.
        inputs_embeds = self.model.embed_tokens(input_ids).to(self.dtype)
        if not multimodal_embeddings:
            return inputs_embeds

        mm = multimodal_embeddings
        image_embeds = mm.get("image_embeds")
        if image_embeds is not None:
            inputs_embeds = self._scatter_mm(
                inputs_embeds, input_ids, self._image_token_id(), image_embeds
            )
        video_embeds = mm.get("video_embeds")
        if video_embeds is not None:
            inputs_embeds = self._scatter_mm(
                inputs_embeds, input_ids, self._video_token_id(), video_embeds
            )
        return inputs_embeds

    # ----- MRoPE position embeddings -------------------------------------------

    def _video_grid_slice(
        self, video_grid_thw: torch.Tensor | None, video_idx: int, video_nums: int
    ) -> tuple[torch.Tensor | None, int]:
        """Select this request's rows of ``video_grid_thw`` and advance the
        cursor. Default: one grid row per video (Qwen2/2.5-VL)."""
        if video_grid_thw is None:
            return None, video_idx
        return video_grid_thw[
            video_idx : video_idx + video_nums
        ], video_idx + video_nums

    def _rope_index_extra(
        self, mm: dict, video_idx: int, video_nums: int
    ) -> dict[str, Any]:
        """Extra keyword args for ``_get_rope_index_func`` (model-specific).
        Default: none. Qwen2.5-VL adds ``second_per_grid_ts``."""
        return {}

    def _compute_mrope(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm: dict | None,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request MRoPE: derive position ids from the vision layout, then
        build the padded (cos/sin) position embeddings. Reimplements the loop in
        optimum-rbln's ``_preprocess_prefill`` on top of the model's
        ``_get_rope_index_func`` / ``_get_position_embeddings``. Returns
        ``(position_embed, rope_deltas)``.
        """
        mm = mm or {}
        image_grid_thw = mm.get("image_grid_thw")
        video_grid_thw = mm.get("video_grid_thw")

        batch_size = input_ids.shape[0]
        max_inputs_len = input_ids.shape[1]
        text_config = self.model.config.text_config
        head_dim = (
            getattr(text_config, "head_dim", None)
            or text_config.hidden_size // text_config.num_attention_heads
        )
        all_position_embeds = torch.zeros(
            2, batch_size, 1, max_inputs_len, head_dim, dtype=self.dtype
        )
        all_rope_deltas = []

        image_token_id = self._image_token_id()
        video_token_id = self._video_token_id()
        vision_start_token_id = self._vision_start_token_id()
        image_idx, video_idx = 0, 0

        for b_idx in range(batch_size):
            input_id = input_ids[b_idx : b_idx + 1][:, attention_mask[b_idx].bool()]
            vision_start_indices = torch.argwhere(
                input_id == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_id[0][vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum())
            video_nums = int((vision_tokens == video_token_id).sum())

            # mm_token_type_ids (0=text, 1=image, 2=video), derived from input_id.
            mm_token_type_ids = torch.zeros_like(input_id, dtype=torch.int)
            mm_token_type_ids[input_id == image_token_id] = 1
            mm_token_type_ids[input_id == video_token_id] = 2

            image_grid_slice = (
                image_grid_thw[image_idx : image_idx + image_nums]
                if image_grid_thw is not None
                else None
            )
            video_idx_before = video_idx
            video_grid_slice, video_idx = self._video_grid_slice(
                video_grid_thw, video_idx, video_nums
            )

            position_ids, rope_deltas = self.model._get_rope_index_func(
                input_id,
                mm_token_type_ids,
                image_grid_slice,
                video_grid_slice,
                **self._rope_index_extra(mm, video_idx_before, video_nums),
            )
            image_idx += image_nums

            position_embed = self.model._get_position_embeddings(
                inputs_embeds, position_ids
            )
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(
                dim=-2, index=mask_indices, source=position_embed
            )
            all_rope_deltas.append(rope_deltas)

        return all_position_embeds, torch.stack(all_rope_deltas)

    def _extra_prefill_params(self, input_ids: torch.Tensor, mm: dict) -> dict:
        """Extra prefill_decoder kwargs beyond inputs_embeds/position_embed.
        Default: none. Qwen3-VL adds deepstack (visual_pos_mask, deepstack_embeds).
        """
        return {}

    # ----- input validation / typed inputs -------------------------------------

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

    # ----- prefill entry points ------------------------------------------------

    def build_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
        """Prefill: run the visual encoder, merge embeddings, and compute MRoPE,
        returning ``(inputs_embeds, position_embed, rope_delta)``. The runner
        owns and stores the per-request rope delta; this hook only returns it.
        """
        input_ids = model_input.input_tokens
        mm = self.embed_multimodal(**(model_input.multi_modal_kwargs or {})) or None

        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = self.embed_input_ids(input_ids, mm)
        position_embed, rope_deltas = self._compute_mrope(
            input_ids, attention_mask, mm, inputs_embeds
        )
        return inputs_embeds, position_embed, rope_deltas.item()

    def _concat_deepstack(self, caches: list[dict], key: str):
        present = [c for c in caches if c.get(key) is not None]
        if not present:
            return None
        num_layers = len(present[0][key])
        return [
            torch.cat([c[key][layer].to(self.dtype) for c in present], dim=0)
            for layer in range(num_layers)
        ]

    def _build_mm_from_cache(self, cached_mm_outputs: list[dict]) -> dict:
        """Reassemble the multimodal dict from cached encoder outputs (EC
        consumer). Same shape as embed_multimodal() so it feeds embed_input_ids()
        and _compute_mrope() unchanged.
        """
        image_caches = [c for c in cached_mm_outputs if "image_embeds" in c]
        video_caches = [c for c in cached_mm_outputs if "video_embeds" in c]

        mm: dict = {}
        if image_caches:
            mm["image_embeds"] = torch.cat(
                [c["image_embeds"].to(self.dtype) for c in image_caches], dim=0
            )
            mm["image_grid_thw"] = torch.cat(
                [c["image_grid_thw"].to(torch.int64) for c in image_caches], dim=0
            )
            deepstack = self._concat_deepstack(image_caches, "deepstack_image_embeds")
            if deepstack is not None:
                mm["deepstack_image_embeds"] = deepstack

        if video_caches:
            mm["video_embeds"] = torch.cat(
                [c["video_embeds"].to(self.dtype) for c in video_caches], dim=0
            )
            mm["video_grid_thw"] = torch.cat(
                [c["video_grid_thw"].to(torch.int64) for c in video_caches], dim=0
            )
            # Qwen2.5-VL: second_per_grid_ts is per-video metadata; carry the
            # first feature's value as a best-effort for mixed batches.
            if "second_per_grid_ts" in video_caches[0]:
                mm["second_per_grid_ts"] = video_caches[0]["second_per_grid_ts"]
            deepstack = self._concat_deepstack(video_caches, "deepstack_video_embeds")
            if deepstack is not None:
                mm["deepstack_video_embeds"] = deepstack

        return mm

    def build_prefill_inputs_from_cache(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None = None,
        running_requests_ids: list[str] | None = None,
        mrope_position_deltas: dict[str, float] | None = None,
    ) -> dict:
        """Build prefill_decoder kwargs from cached encoder outputs (EC consumer).
        The visual encoder was already run on the producer side, so this merges
        the cached embeddings and computes MRoPE without re-encoding.
        """
        mm = self._build_mm_from_cache(cached_mm_outputs)

        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = self.embed_input_ids(input_ids, mm or None)
        position_embed, rope_deltas = self._compute_mrope(
            input_ids, attention_mask, mm, inputs_embeds
        )

        if running_requests_ids and mrope_position_deltas is not None:
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

        prefill_params = {
            "inputs_embeds": inputs_embeds,
            "position_embed": position_embed,
        }
        prefill_params.update(self._extra_prefill_params(input_ids, mm))
        return prefill_params

    # ----- decode --------------------------------------------------------------

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
            logits = self.model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                position_embed=model_input.position_embed,
                block_tables=block_tables,
            ).logits
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

    def get_language_model(self):
        return self.model


class RBLNOptimumQwen2_5_VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    def _rope_index_extra(
        self, mm: dict, video_idx: int, video_nums: int
    ) -> dict[str, Any]:
        """Qwen2.5-VL passes per-video temporal spacing to get_rope_index."""
        second_per_grid_ts = mm.get("second_per_grid_ts")
        if second_per_grid_ts is None:
            return {}
        return {
            "second_per_grid_ts": second_per_grid_ts[video_idx : video_idx + video_nums]
        }

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


class RBLNOptimumQwen3VLForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Qwen3-VL reuses Qwen2.5-VL classes with the same implementation.
    However, since Qwen3-VL does not require second_per_grid_ts,
    certain methods are overridden to exclude it from the model inputs.
    Qwen3-VL also carries deepstack visual features into the decoder.
    """

    def _rope_index_extra(
        self, mm: dict, video_idx: int, video_nums: int
    ) -> dict[str, Any]:
        """Qwen3-VL doesn't use second_per_grid_ts."""
        return {}

    def _video_grid_slice(
        self, video_grid_thw: torch.Tensor | None, video_idx: int, video_nums: int
    ) -> tuple[torch.Tensor | None, int]:
        # Qwen3-VL indexes video_grid_thw by temporal chunks: each video spans
        # ``grid[row, 0]`` (T) rows, so consume rows until ``video_nums`` videos
        # are covered.
        if video_grid_thw is None:
            return None, video_idx
        start_row = video_idx
        consumed_video_chunks = 0
        while (
            video_idx < video_grid_thw.shape[0] and consumed_video_chunks < video_nums
        ):
            consumed_video_chunks += int(video_grid_thw[video_idx, 0].item())
            video_idx += 1
        return video_grid_thw[start_row:video_idx], video_idx

    def _extra_prefill_params(self, input_ids: torch.Tensor, mm: dict) -> dict:
        # Deepstack features are scattered by the decoder over the same visual
        # positions as the main embeddings, so pass the placeholder mask and the
        # per-layer deepstack embeds. Shapes mirror optimum-rbln's
        # ``_preprocess_prefill`` output ([1, 3, ...] -> [3, ...]).
        image_mask = (
            input_ids == self._image_token_id()
            if mm.get("image_embeds") is not None
            else None
        )
        video_mask = (
            input_ids == self._video_token_id()
            if mm.get("video_embeds") is not None
            else None
        )
        visual_pos_mask, deepstack_visual_embeds = self.model._prepare_deepstack(
            image_mask,
            video_mask,
            mm.get("deepstack_image_embeds"),
            mm.get("deepstack_video_embeds"),
        )
        return {
            "visual_pos_mask": visual_pos_mask,
            "deepstack_embeds": deepstack_visual_embeds.squeeze(0)
            if deepstack_visual_embeds is not None
            else None,
        }

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
