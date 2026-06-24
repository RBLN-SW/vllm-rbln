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
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoPixelInputs,
)

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)
from .optimum_attention import (
    AttentionManager,
    InnerAttentionEntry,
    InnerAttentionStrategy,
    InnerR1,
    InnerR2,
)

logger = init_logger(__name__)


class RBLNOptimumExaone4_5_ForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumMultimodalMixin, RBLNOptimumDecoderMixin
):
    # EXAONE-4.5 reuses the Qwen2.5-VL multimodal placeholders.
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
        self.strategy = InnerAttentionStrategy()
        self.attention_manager: AttentionManager[
            InnerAttentionStrategy, InnerAttentionEntry, InnerR1, InnerR2
        ] = AttentionManager(self.strategy)
        self.is_hybrid = getattr(self.model.rbln_config, "cache_impl", None) == "hybrid"

    def preprocess_prefill(self, input_ids, attention_mask, image_input, video_input):
        """
        Common preprocessing logic for prefill inputs.
        Calls model-specific parameter preparation method.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            image_input: Image input data
            video_input: Video input data

        Returns:
            Prefill input embeddings tensor.
        """

        # Prepare base arguments common to all models
        preprocess_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_input["pixel_values"]
            if image_input is not None
            else None,
            "image_grid_thw": image_input["image_grid_thw"]
            if image_input is not None
            else None,
            "pixel_values_videos": video_input["pixel_values_videos"]
            if video_input is not None
            else None,
            "video_grid_thw": video_input["video_grid_thw"]
            if video_input is not None
            else None,
        }

        # Call the actual preprocessing
        return self.model._preprocess_prefill(**preprocess_args)

    def get_language_model(self):
        return self.model

    def _image_token_id(self) -> int:
        # EXAONE-4.5's HF config names the placeholder `image_token_id`
        # (not `image_token_index` as the mixin default assumes).
        return self.model.config.image_token_id

    def _process_image_input(self, image_input) -> dict:
        result = {}
        if image_input is not None and image_input.get("type") == "pixel_values":
            image_embeds = self.model.visual(
                image_input["pixel_values"], grid_thw=image_input["image_grid_thw"]
            )
            result["image_embeds"] = image_embeds
            result["image_grid_thw"] = image_input["image_grid_thw"]
        return result

    def _process_video_input(self, video_input) -> dict:
        result = {}
        if video_input is not None and video_input.get("type") == "pixel_values_videos":
            video_embeds = self.model.visual(
                video_input["pixel_values_videos"],
                grid_thw=video_input["video_grid_thw"],
            )
            result["video_embeds"] = video_embeds
            result["video_grid_thw"] = video_input["video_grid_thw"]
        return result

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | dict:
        """Vision-only encode entry point (SupportsMultiModal / EC producer).

        Like Qwen-VL, the cacheable unit is a dict of image/video embeds plus
        their grid_thw rather than a flat list of per-item embeddings.
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)
        if image_input is None and video_input is None:
            return []

        result = {}
        result.update(self._process_image_input(image_input))
        result.update(self._process_video_input(video_input))
        return result

    def build_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list,
        *,
        cache_position: torch.Tensor | None = None,
        running_requests_ids: list[str] | None = None,
    ) -> dict:
        # NOTE: this guard is currently unreachable — init_model() only enables
        # the EC path for "RBLNQwen3VLForConditionalGeneration", so EXAONE-4.5
        # never enters here today. It documents the contract for when EC is
        # extended: the sliding-window/hybrid-cache prefill needs the
        # attention_manager state that build_prefill_inputs does not yet provide.
        raise NotImplementedError(
            "EC disaggregation is not implemented for EXAONE-4.5."
        )

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
        second_per_grid_ts: torch.Tensor | None = None,
    ):
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _create_video_embedding_inputs(
        self, video_embeds, video_grid_thw, second_per_grid_ts=None
    ):
        return Qwen2_5_VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        is_prompt = model_input.is_prompt

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids = self.attention_manager.get(
            is_prompt,
            self.decoder_batch_size,
            running_requests_ids,
            finished_requests_ids,
        )

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )

        padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            image_input = None
            video_input = None
            if model_input.multi_modal_kwargs:
                image_input = self._parse_and_validate_image_input(
                    **model_input.multi_modal_kwargs
                )
                video_input = self._parse_and_validate_video_input(
                    **model_input.multi_modal_kwargs
                )
            if image_input is None and video_input is None:
                inputs_embeds = None

            attention_mask = torch.ones_like(input_ids)
            inputs_embeds = self.preprocess_prefill(
                input_ids, attention_mask, image_input, video_input
            )
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx], dtype=torch.int16)
            logits = self.model.prefill_decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                local_block_tables=local_block_table_id,
                block_tables=block_tables if self.is_hybrid else None,
            ).logits
            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=prefill_batch_idx,
            )
        else:
            self.model.decoder = self.model.decoders[padded_batch_size]
            inputs_embeds = self.model.embed_tokens(input_ids).to(
                self.model.rbln_config.dtype
            )
            local_block_table_id, cache_position = self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
            )
            logits = self.model.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                local_block_tables=local_block_table_id,
                block_tables=block_tables if self.is_hybrid else None,
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
        # Parsed only to match the qwen2_5_vl schema; not consumed by the
        # forward path (optimum-rbln does not take second_per_grid_ts).
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return self._create_video_pixel_inputs(
                pixel_values_videos, video_grid_thw, second_per_grid_ts
            )

        if video_embeds is not None:
            return self._create_video_embedding_inputs(
                video_embeds, video_grid_thw, second_per_grid_ts
            )

        # fallback return if both are None
        return None
