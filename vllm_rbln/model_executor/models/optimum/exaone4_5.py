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
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLVideoPixelInputs

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


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
        self.is_hybrid = getattr(self.model.rbln_config, "cache_impl", None) == "hybrid"
        self._local_table_by_request_id: dict[str, int] = {}

    def _release_finished_slots(self, finished_request_ids: list[str]) -> None:
        for request_id in finished_request_ids:
            self._local_table_by_request_id.pop(request_id, None)

    def _allocate_slot_for_request(self, request_id: str) -> int:
        slot = self._local_table_by_request_id.get(request_id)
        if slot is not None:
            return slot

        used = set(self._local_table_by_request_id.values())
        for candidate in range(self.decoder_batch_size):
            if candidate not in used:
                self._local_table_by_request_id[request_id] = candidate
                return candidate
        raise RuntimeError("No available local attention table slots.")

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

        # Add model-specific parameters
        self._add_model_specific_args(preprocess_args, video_input)

        # Call the actual preprocessing
        return self.model._preprocess_prefill(**preprocess_args)

    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Add video kwargs only when available."""
        if video_input is not None and "second_per_grid_ts" in video_input:
            preprocess_args["second_per_grid_ts"] = video_input["second_per_grid_ts"]

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
        # Exaone4_5 path does not require second_per_grid_ts.
        return Qwen2VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2_5_VLVideoEmbeddingInputs(
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
        self._release_finished_slots(finished_requests_ids)

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

            attention_mask = torch.ones_like(input_ids)

            inputs_embeds = self.preprocess_prefill(
                input_ids, attention_mask, image_input, video_input
            )

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            prefill_kwargs = {
                "inputs_embeds": inputs_embeds,
                "cache_position": cache_position,
                "block_tables": block_tables,
            }
            if self.is_hybrid:
                assert len(running_requests_ids) == 1
                prefill_batch_idx = self._allocate_slot_for_request(
                    running_requests_ids[0]
                )
                prefill_kwargs["local_block_tables"] = torch.tensor(
                    [prefill_batch_idx], dtype=torch.int16, device=block_tables.device
                )

            logits = self.model.prefill_decoder(**prefill_kwargs).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.decoder = self.model.decoders[padded_batch_size]
            input_ids = kwargs.pop("input_ids")
            inputs_embeds = self.model.embed_tokens(input_ids).to(
                self.model.rbln_config.dtype
            )
            decoder_kwargs = {
                "inputs_embeds": inputs_embeds,
                "cache_position": cache_position,
                "block_tables": block_tables,
            }
            if self.is_hybrid:
                local_ids = [
                    self._allocate_slot_for_request(request_id)
                    for request_id in running_requests_ids
                ]
                used_ids = set(local_ids)
                pad_value = next(
                    (i for i in range(padded_batch_size) if i not in used_ids), 0
                )
                local_block_tables = torch.full(
                    (padded_batch_size, 1),
                    pad_value,
                    dtype=torch.int16,
                    device=block_tables.device,
                )
                local_block_tables[: len(local_ids), 0] = torch.tensor(
                    local_ids, dtype=torch.int16, device=block_tables.device
                )
                decoder_kwargs["local_block_tables"] = local_block_tables

            logits = self.model.decoder(**decoder_kwargs).logits
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
