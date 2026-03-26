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
from abc import ABC
from typing import Any

import torch
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config
from transformers.models.exaone4_5.processing_exaone4_5 import Exaone4_5_Processor
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLVideoEmbeddingInputs,
    Qwen2VLVideoPixelInputs,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import (
    AttentionManager,
    InnerAttentionEntry,
    InnerAttentionStrategy,
    InnerR1,
    InnerR2,
)

logger = init_logger(__name__)


class EXAONE4_5ImageEmbeddingInputs(Qwen2_5_VLImageEmbeddingInputs):
    pass


class EXAONE4_5ImagePixelInputs(Qwen2_5_VLImagePixelInputs):
    pass


# NOTE: EXAONE4_5 does not require second_per_grid_ts.
class EXAONE4_5VideoPixelInputs(Qwen2VLVideoPixelInputs):
    pass


class EXAONE4_5VideoEmbeddingInputs(Qwen2VLVideoEmbeddingInputs):
    pass


class EXAONE4_5_DummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    pass


class EXAONE4_5ProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Exaone4_5_Config)

    def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Processor:
        return self.ctx.get_hf_processor(
            Exaone4_5_Processor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class EXAONE4_5MultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    def apply(self, *args, **kwargs):
        hf_processor_mm_kwargs = kwargs.pop("hf_processor_mm_kwargs", {})
        do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames", False)
        if do_sample_frames:
            raise NotImplementedError(
                "`do_sample_frames=True` is not supported yet. "
                "Please set `do_sample_frames` to False."
            )
        hf_processor_mm_kwargs.pop("fps", None)
        kwargs["hf_processor_mm_kwargs"] = hf_processor_mm_kwargs
        return super().apply(*args, **kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    EXAONE4_5MultiModalProcessor,
    info=EXAONE4_5ProcessingInfo,
    dummy_inputs=EXAONE4_5_DummyInputsBuilder,
)
class RBLNOptimumExaone4_5_ForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumDecoderMixin, SupportsMultiModal, ABC
):
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
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

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return EXAONE4_5ImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return EXAONE4_5ImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ):
        return EXAONE4_5VideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return EXAONE4_5VideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
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

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return self._create_video_pixel_inputs(pixel_values_videos, video_grid_thw)

        if video_embeds is not None:
            return self._create_video_embedding_inputs(video_embeds, video_grid_thw)

        # fallback return if both are None
        return None
