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
from vllm.multimodal.inputs import MultiModalFeatureSpec

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)


def split_by_grid_thw(
    embeds: torch.Tensor, grid_thw: torch.Tensor
) -> list[torch.Tensor]:
    """Cut the encoder's concatenated output back into per-item tensors.

    Item i has `t * h * w / k**2` tokens; `k**2` is read off the output so this
    also works on an EC producer, which has no text config.
    """
    patches = grid_thw.prod(dim=-1)
    merge_unit = int(patches.sum()) // embeds.shape[0]
    return list(embeds.split((patches // merge_unit).tolist()))


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

    supports_mrope = True

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
        )

    def get_prefill_decoder(self):
        return self.model.prefill_decoder

    def get_language_model(self):
        return self.model

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
        """One 2D tensor per item, in kwargs order; the runner caches each by
        mm_hash."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)
        return [
            *self._process_image_input(image_input),
            *self._process_video_input(video_input),
        ]

    def _image_token_id(self) -> int:
        return self.model.config.image_token_id

    def _embed_text_tokens(
        self, input_ids: torch.Tensor, is_multimodal: torch.Tensor
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def build_prefill_forward_inputs(
        self, model_input: ModelInputForRBLN
    ) -> ModelInputForRBLN:
        model_input = super().build_prefill_forward_inputs(model_input)
        return replace(model_input, position_embed=self._position_embed(model_input))

    def build_decode_forward_inputs(
        self, model_input: ModelInputForRBLN
    ) -> ModelInputForRBLN:
        return replace(model_input, position_embed=self._position_embed(model_input))

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Whole-prompt MRoPE positions [3, N] and the decode delta, from HF's
        get_rope_index that optimum-rbln exposes on the model."""
        input_ids = torch.tensor([input_tokens])
        config = self.model.config
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == config.image_token_id] = 1
        mm_token_type_ids[input_ids == config.video_token_id] = 2
        features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        images = [f for f in features if f.modality == "image"]
        videos = [f for f in features if f.modality == "video"]
        position_ids, rope_deltas = self.model._get_rope_index_func(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=self._grid_thw(images, "image_grid_thw"),
            video_grid_thw=self._grid_thw(videos, "video_grid_thw"),
            **self._video_rope_kwargs(videos),
        )
        return position_ids[:, 0], int(rope_deltas)

    @staticmethod
    def _grid_thw(
        features: list[MultiModalFeatureSpec], key: str
    ) -> torch.Tensor | None:
        if not features:
            return None
        return torch.stack([f.data[key].data for f in features]).to(torch.int64)

    def _video_rope_kwargs(
        self, video_features: list[MultiModalFeatureSpec]
    ) -> dict[str, torch.Tensor]:
        """Extra get_rope_index kwargs a variant needs for videos; none here."""
        return {}

    def _position_embed(self, model_input: ModelInputForRBLN) -> torch.Tensor:
        """cos/sin for the runner's MRoPE positions, [2, padded_batch_size, 1,
        seq_len, head_dim], zero in the padding rows."""
        assert model_input.mrope_positions is not None
        embed = self.model._get_position_embeddings(
            torch.zeros(1, dtype=self.dtype), model_input.mrope_positions
        )
        rows: torch.Tensor | slice = (
            slice(0, len(model_input.running_requests_ids))
            if model_input.batch_rows is None
            else model_input.batch_rows
        )
        out = torch.zeros_like(embed)
        out[:, rows] = embed[:, rows]
        return out

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        if model_input.is_prompt:
            return self.model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                position_embed=model_input.position_embed,
                block_tables=model_input.block_tables,
                cache_position=model_input.input_positions,
            ).logits

        self.model.decoder = self.model.decoders[model_input.padded_batch_size]
        logits = self.model.decoder(
            inputs_embeds=self.model.embed_tokens(model_input.input_tokens),
            cache_position=model_input.input_positions,
            position_embed=model_input.position_embed,
            block_tables=model_input.block_tables,
        ).logits
        return logits[: len(model_input.running_requests_ids)]


class RBLNOptimumQwen2_5_VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    def _video_rope_kwargs(
        self, video_features: list[MultiModalFeatureSpec]
    ) -> dict[str, torch.Tensor]:
        if not video_features:
            return {}
        return {
            "second_per_grid_ts": torch.tensor(
                [float(f.data["second_per_grid_ts"].data) for f in video_features]
            )
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
