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
from .qwen2_vl import split_by_grid_thw

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
        )
        self.is_hybrid = getattr(self.model.rbln_config, "cache_impl", None) == "hybrid"

    def get_language_model(self):
        return self.model

    def _image_token_id(self) -> int:
        # EXAONE-4.5's HF config names the placeholder `image_token_id`
        # (not `image_token_index` as the mixin default assumes).
        return self.model.config.image_token_id

    def _embed_text_tokens(
        self, input_ids: torch.Tensor, is_multimodal: torch.Tensor
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids).to(self.dtype)

    def _process_image_input(self, image_input) -> list[torch.Tensor]:
        if image_input is None or image_input.get("type") != "pixel_values":
            return []
        grid_thw = image_input["image_grid_thw"]
        embeds = self.model.visual(image_input["pixel_values"], grid_thw=grid_thw)
        return split_by_grid_thw(embeds, grid_thw)

    def _process_video_input(self, video_input) -> list[torch.Tensor]:
        if video_input is None or video_input.get("type") != "pixel_values_videos":
            return []
        grid_thw = video_input["video_grid_thw"]
        embeds = self.model.visual(
            video_input["pixel_values_videos"], grid_thw=grid_thw
        )
        return split_by_grid_thw(embeds, grid_thw)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)
        return [
            *self._process_image_input(image_input),
            *self._process_video_input(video_input),
        ]

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
        cache_slot_ids = model_input.cache_slot_ids
        assert cache_slot_ids is not None
        block_tables = model_input.block_tables if self.is_hybrid else None

        if model_input.is_prompt:
            return self.model.prefill_decoder(
                input_ids=model_input.input_tokens,
                inputs_embeds=model_input.inputs_embeds,
                cache_position=model_input.input_positions,
                local_block_tables=cache_slot_ids,
                block_tables=block_tables,
            ).logits

        self.model.decoder = self.model.decoders[model_input.padded_batch_size]
        logits = self.model.decoder(
            input_ids=model_input.input_tokens,
            inputs_embeds=self.model.embed_tokens(model_input.input_tokens),
            cache_position=model_input.input_positions,
            local_block_tables=cache_slot_ids,
            block_tables=block_tables,
        ).logits
        return logits[: len(model_input.running_requests_ids)]

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
