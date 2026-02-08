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
from typing import Any, Dict, Optional

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs, Qwen2_5_VLVideoPixelInputs)
from vllm.model_executor.models.qwen2_vl import (Qwen2VLImageEmbeddingInputs,
                                                 Qwen2VLImagePixelInputs,
                                                 Qwen2VLVideoEmbeddingInputs,
                                                 Qwen2VLVideoPixelInputs)

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumQwenVLForConditionalGeneration(RBLNOptimumModelBase,
                                                RBLNOptimumDecoderMixin,
                                                SupportsMultiModal, ABC):
    """
    Unified class for both Qwen2-VL and Qwen2.5-VL models.
    Automatically detects model type based on the model configuration.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )
        self.rope_deltas: Dict = dict()

    def preprocess_prefill(self, input_ids, attention_mask, image_input,
                           video_input):
        """
        Common preprocessing logic for prefill inputs.
        Calls model-specific parameter preparation method.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            image_input: Image input data
            video_input: Video input data
            
        Returns:
            Tuple of (inputs_embeds, position_embed, rope_deltas)
        """

        # Prepare base arguments common to all models
        preprocess_args = {
            "input_ids":
            input_ids,
            "attention_mask":
            attention_mask,
            "pixel_values":
            image_input["pixel_values"] if image_input is not None else None,
            "image_grid_thw":
            image_input["image_grid_thw"] if image_input is not None else None,
            "pixel_values_videos":
            video_input["pixel_values_videos"]
            if video_input is not None else None,
            "video_grid_thw":
            video_input["video_grid_thw"] if video_input is not None else None,
        }

        # Add model-specific parameters
        self._add_model_specific_args(preprocess_args, video_input)

        # Call the actual preprocessing
        return self.model._preprocess_prefill(**preprocess_args)

    @abstractmethod
    def _add_model_specific_args(self, preprocess_args: dict,
                                 video_input: Any):
        """
        Add model-specific arguments to preprocessing args.
        
        Args:
            preprocess_args: Dictionary of preprocessing arguments to modify
            video_input: Video input data
        """
        pass

    @abstractmethod
    def _create_image_pixel_inputs(self, pixel_values: torch.Tensor,
                                   image_grid_thw: torch.Tensor) -> Any:
        """Create image pixel inputs based on model type"""
        pass

    @abstractmethod
    def _create_image_embedding_inputs(self, image_embeds: torch.Tensor,
                                       image_grid_thw: torch.Tensor) -> Any:
        """Create image embedding inputs based on model type"""
        pass

    @abstractmethod
    def _create_video_pixel_inputs(
            self, pixel_values_videos: torch.Tensor,
            video_grid_thw: torch.Tensor,
            second_per_grid_ts: Optional[torch.Tensor]) -> Any:
        """Create video pixel inputs based on model type"""
        pass

    @abstractmethod
    def _create_video_embedding_inputs(self, video_embeds,
                                       video_grid_thw) -> Any:
        """Create video embedding inputs based on model type"""
        pass

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids

        is_prompt = model_input.is_prompt

        if is_prompt:
            image_input = None
            video_input = None
            if model_input.multi_modal_kwargs:
                image_input = self._parse_and_validate_image_input(
                    **model_input.multi_modal_kwargs)
                video_input = self._parse_and_validate_video_input(
                    **model_input.multi_modal_kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None

            cur_request_id = running_requests_ids[0]
            attention_mask = torch.ones_like(input_ids)

            (inputs_embeds, position_embed,
             rope_deltas) = self.preprocess_prefill(input_ids, attention_mask,
                                                    image_input, video_input)

            if finished_requests_ids:
                for request_id in finished_requests_ids:
                    self.rope_deltas.pop(request_id)
            self.rope_deltas[cur_request_id] = rope_deltas.item()

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, cache_position)
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            logits = self.model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=position_embed,
                block_tables=block_tables,
            ).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.decoder = self.model.decoders[padded_batch_size]
            input_ids = kwargs.pop("input_ids")

            inputs_embeds, position_embed = self._preprocess_embeds(
                input_ids, cache_position, running_requests_ids,
                padded_batch_size)
            logits = self.model.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=position_embed,
                block_tables=block_tables,
            ).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def _preprocess_embeds(
        self,
        input_ids: torch.LongTensor,
        cache_position: torch.LongTensor,
        running_requests_ids: list[str],
        padded_batch_size: int,
    ):
        if padded_batch_size != cache_position.shape[0]:
            raise RuntimeError(
                f"Cache position size mismatch: got {cache_position.shape[0]},",
                " expected {padded_batch_size}.")

        inputs_embeds = self.model.embed_tokens(input_ids)
        position_embeds = []
        for b_id, request_id in enumerate(running_requests_ids):
            delta = cache_position[b_id] + self.rope_deltas[request_id]
            position_ids = torch.arange(1).view(1, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_embed = self.model._get_position_embeddings(
                torch.zeros(1, dtype=self.dtype), position_ids)
            position_embeds.append(position_embed)

        for _ in range(padded_batch_size - len(running_requests_ids)):
            position_embed = torch.zeros_like(position_embeds[0])
            position_embeds.append(position_embed)

        position_embeds = torch.cat(position_embeds, dim=1)

        return inputs_embeds, position_embeds

    def _parse_and_validate_image_input(self, **kwargs: Any) -> Optional[Any]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return self._create_image_pixel_inputs(
                pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            return self._create_image_embedding_inputs(
                image_embeds=image_embeds, image_grid_thw=image_grid_thw)

        # fallback return if both are None
        return None

    def _parse_and_validate_video_input(self,
                                        **kwargs: object) -> Optional[Any]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return self._create_video_pixel_inputs(pixel_values_videos,
                                                   video_grid_thw,
                                                   second_per_grid_ts)

        if video_embeds is not None:
            return self._create_video_embedding_inputs(video_embeds,
                                                       video_grid_thw)

        # fallback return if both are None
        return None


class RBLNOptimumQwen2_5_VLForConditionalGeneration(
        RBLNOptimumQwenVLForConditionalGeneration):

    def _add_model_specific_args(self, preprocess_args: dict,
                                 video_input: Any):
        """Add second_per_grid_ts for Qwen2.5-VL"""
        if video_input is not None:
            preprocess_args["second_per_grid_ts"] = video_input[
                "second_per_grid_ts"]

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                          pixel_values=pixel_values,
                                          image_grid_thw=image_grid_thw)

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return Qwen2_5_VLImageEmbeddingInputs(type="image_embeds",
                                              image_embeds=image_embeds,
                                              image_grid_thw=image_grid_thw)

    def _create_video_pixel_inputs(self,
                                   pixel_values_videos: torch.Tensor,
                                   video_grid_thw: torch.Tensor,
                                   second_per_grid_ts=Optional[torch.Tensor]):
        if second_per_grid_ts is None:
            raise ValueError(
                "second_per_grid_ts is required for Qwen2.5-VL video inputs.")
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts)

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2_5_VLVideoEmbeddingInputs(type="video_embeds",
                                              video_embeds=video_embeds,
                                              video_grid_thw=video_grid_thw)


class RBLNOptimumQwen2VLForConditionalGeneration(
        RBLNOptimumQwenVLForConditionalGeneration):

    def _add_model_specific_args(self, preprocess_args: dict,
                                 video_input: Any):
        """Qwen2-VL doesn't need additional arguments"""
        pass

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return Qwen2VLImagePixelInputs(type="pixel_values",
                                       pixel_values=pixel_values,
                                       image_grid_thw=image_grid_thw)

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return Qwen2VLImageEmbeddingInputs(type="image_embeds",
                                           image_embeds=image_embeds,
                                           image_grid_thw=image_grid_thw)

    def _create_video_pixel_inputs(self, pixel_values_videos: torch.Tensor,
                                   video_grid_thw: torch.Tensor,
                                   second_per_grid_ts: Optional[torch.Tensor]):
        # NOTE Qwen2-VL doesn't use second_per_grid_ts
        return Qwen2VLVideoPixelInputs(type="pixel_values_videos",
                                       pixel_values_videos=pixel_values_videos,
                                       video_grid_thw=video_grid_thw)

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2VLVideoEmbeddingInputs(type="video_embeds",
                                           video_embeds=video_embeds,
                                           video_grid_thw=video_grid_thw)
