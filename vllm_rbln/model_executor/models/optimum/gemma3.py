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
from typing import Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma3_mm import (Gemma3DummyInputsBuilder,
                                                  Gemma3ImageInputs,
                                                  Gemma3ImagePixelInputs,
                                                  Gemma3MultiModalProcessor,
                                                  Gemma3ProcessingInfo)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import (
    VllmModelForTextGeneration)
from vllm.model_executor.models.utils import flatten_bn
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import (HybridAttentionImageManager,
                                HybridAttentionImageStrategy)

logger = init_logger(__name__)

IMG_PAD_TOKEN_ID = -1
PAD_TOKEN_ID = 0


class RBLNGemma3MultiModalProcessor(Gemma3MultiModalProcessor):

    def _pad_for_gemma3(self, prompt_ids: list[int]):
        token_type_ids = (torch.tensor(prompt_ids) ==
                          self.info.get_hf_processor().image_token_id)

        image_prefill_chunk_size = self.info.get_hf_processor(
        ).image_seq_length
        # Find image start positions
        image_starts = [
            s for s in torch.where(token_type_ids)[0]
            if torch.all(token_type_ids[s:s + image_prefill_chunk_size])
        ]
        padded_seq_len = 0
        for image_start in image_starts:
            pad_needed = (
                image_prefill_chunk_size -
                (image_start + padded_seq_len) % image_prefill_chunk_size)
            padded_seq_len += pad_needed
        # To pad the image token
        # not using pad_token_id
        # NOTE: Left padding for Gemma3
        prompt_ids = [IMG_PAD_TOKEN_ID] * padded_seq_len + prompt_ids
        return prompt_ids

    def apply(self, *args, **kwargs):
        # NOTE: Check if padding works correctly
        output = super().apply(*args, **kwargs)
        prompt_ids = self._pad_for_gemma3(output["prompt_token_ids"])

        output["prompt_token_ids"] = prompt_ids

        return output


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma3MultiModalProcessor,
    info=Gemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class RBLNOptimumGemma3ForConditionalGeneration(
        RBLNOptimumModelBase,
        RBLNOptimumDecoderMixin,
        VllmModelForTextGeneration,
        SupportsMultiModal,
):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        # NOTE:
        # model_config.vocab_size != tokenizer.vocab_size in Gemma3
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config.language_model,
                "use_multiple_decoder",
                False,
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.
            decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )
        self.strategy = HybridAttentionImageStrategy(IMG_PAD_TOKEN_ID)
        self.attention_manager: HybridAttentionImageManager \
            = HybridAttentionImageManager(self.strategy)

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions
        block_tables = model_input.block_tables

        is_prompt = model_input.is_prompt

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids, padded_cache_lengths, attention_masks = \
        self.attention_manager.get(
                is_prompt,
                self.decoder_batch_size,
                running_requests_ids,
                finished_requests_ids,
                input_ids=input_ids,
            )
        # When processing input_ids, we initially pad them with IMG_PAD_TOKEN_ID
        # instead of PAD_TOKEN_ID.
        #
        # IMG_PAD_TOKEN_ID does not have an embedding. It is used only to mark
        # image boundaries and to prevent multiple images from being placed
        # within the same attention window. It is used in generating
        # attention mask during the prefill phase.
        #
        # PAD_TOKEN_ID, on the other hand, has a valid embedding and is used for
        # standard padding.
        #
        # After the attention mask is generated, IMG_PAD_TOKEN_ID is replaced
        # with PAD_TOKEN_ID.
        input_ids = self.postprocess_input_ids(input_ids)

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, position_ids)

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            inputs_embeds = None
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx],
                                                dtype=torch.int16)
            # token_type_ids model_input != token_type_ids of gemma3
            # https://github.com/huggingface/transformers/blob/d0c9c66d1c09df3cd70bf036e813d88337b20d4c/src/transformers/models/gemma3/processing_gemma3.py#L143
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids ==
                           self.model.config.image_token_index] = 1

            pixel_values = self.get_pixel_values(model_input)
            inputs_embeds = self.model._preprocess_prefill(
                input_ids, inputs_embeds, pixel_values)
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            assert attention_masks is not None
            attention_mask = attention_masks[0]
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                attention_mask=attention_mask,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=token_type_ids,
            )
            logits = output.logits
            updated_attention_mask = output.attention_mask
            updated_padded_cache_length = output.padded_cache_lengths

            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=sliding_window_table_ids[0],
                pad_len=updated_padded_cache_length,
                attention_mask=updated_attention_mask,
            )
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.language_model.decoder = (
                self.model.language_model.decoders[padded_batch_size])
            (
                local_block_table_id,
                cache_position,
                position_ids,
                attention_mask,
            ) = self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
                pad_lens=padded_cache_lengths,
                attention_masks=attention_masks,
            )

            attention_mask = self.attention_manager.update(
                running_requests_ids,
                attention_mask,
                cache_position,
            )

            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_pixel_values(self, model_input: ModelInputForRBLN):
        image_input = None

        if model_input.multi_modal_kwargs:
            image_input = self._parse_and_validate_image_input(
                **model_input.multi_modal_kwargs)
            if image_input is not None:
                assert image_input["type"] == "pixel_values"
                pixel_values = image_input["pixel_values"]
        else:
            pixel_values = None

        return pixel_values

    def _parse_and_validate_image_input(
            self, **kwargs: Any) -> Optional[Gemma3ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        num_crops = kwargs.pop("num_crops", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        assert image_embeds is None, "Gemma3 does not support image_embeds."
        if pixel_values is None:
            return None

        if not isinstance(pixel_values, torch.Tensor | list):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if not isinstance(num_crops, torch.Tensor | list):
            raise ValueError("Incorrect type of num_crops. "
                             f"Got type: {type(num_crops)}")

        image_size = config.vision_config.image_size

        return Gemma3ImagePixelInputs(
            pixel_values=flatten_bn(pixel_values, concat=True),
            num_patches=flatten_bn(num_crops, concat=True) + 1,
            resolve_bindings={
                "h": image_size,
                "w": image_size
            })

    def postprocess_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_image_pad_token = (input_ids == IMG_PAD_TOKEN_ID)
        input_ids[is_image_pad_token] = PAD_TOKEN_ID
        return input_ids
