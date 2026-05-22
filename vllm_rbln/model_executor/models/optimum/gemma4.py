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
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ImageInputs,
    Gemma4ImagePixelInputs,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    Gemma4AudioInputs,
    Gemma4VideoInputs,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal, MultiModalEmbeddings
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import HybridAttentionImageManager, HybridAttentionImageStrategy
logger = init_logger(__name__)

PAD_TOKEN_ID = 0


class RBLNGemma4MultiModalProcessor(Gemma4MultiModalProcessor):
    def _pad_for_gemma4(self, prompt_ids: list[int]):
        token_type_ids = (
            torch.tensor(prompt_ids) == self.info.get_hf_processor().image_token_id
        )

        image_prefill_chunk_size = self.info.get_hf_processor().image_seq_length
        # Find image start positions
        image_starts = [
            s
            for s in torch.where(token_type_ids)[0]
            if torch.all(token_type_ids[s : s + image_prefill_chunk_size])
        ]
        padded_seq_len = 0
        for image_start in image_starts:
            pad_needed = (
                image_prefill_chunk_size
                - (image_start + padded_seq_len) % image_prefill_chunk_size
            )
            padded_seq_len += pad_needed
        # Left padding for Gemma3 image boundary alignment
        prompt_ids = [PAD_TOKEN_ID] * padded_seq_len + prompt_ids
        return prompt_ids

    def apply(self, *args, **kwargs):
        # NOTE: Check if padding works correctly
        output = super().apply(*args, **kwargs)
        prompt_ids = self._pad_for_gemma4(output["prompt_token_ids"])

        output["prompt_token_ids"] = prompt_ids

        return output


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class RBLNOptimumGemma4ForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumDecoderMixin,
    VllmModelForTextGeneration,
    SupportsMultiModal,
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        # NOTE:
        # model_config.vocab_size != tokenizer.vocab_size in Gemma3
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config.language_model,
                "use_multiple_decoder",
                False,
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )
        self.strategy = HybridAttentionImageStrategy(PAD_TOKEN_ID)
        self.attention_manager: HybridAttentionImageManager = (
            HybridAttentionImageManager(self.strategy)
        )
        # config = vllm_config.model_config.hf_config
        with self._mark_language_model(vllm_config):
            self.language_model = self.model.language_model
        with self._mark_tower_model(vllm_config, "image"):
            from transformers import AutoModelForImageTextToText
            hf_model_id = "google/gemma-4-31B-it"
            hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_id).to(dtype=torch.bfloat16).eval()
            self.vision_tower = hf_model.model.vision_tower
            self.embed_vision = hf_model.model.embed_vision
            # self.vision_tower = self.model.vision_tower

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions
        block_tables = model_input.block_tables

        is_prompt = model_input.is_prompt

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids, padded_cache_lengths, attention_masks = (
            self.attention_manager.get(
                is_prompt,
                self.decoder_batch_size,
                running_requests_ids,
                finished_requests_ids,
                input_ids=input_ids,
            )
        )
        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, position_ids
        )

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
            local_block_table_id = torch.tensor([prefill_batch_idx], dtype=torch.int16)
            # FIXME It is disappeared in transformers 5.5.4
            # token_type_ids model_input != token_type_ids of gemma3
            # https://github.com/huggingface/transformers/blob/d0c9c66d1c09df3cd70bf036e813d88337b20d4c/src/transformers/models/gemma3/processing_gemma3.py#L143
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids == self.model.config.image_token_id] = 1

            pixel_values = self.get_pixel_values(model_input)
            inputs_embeds = self.model._preprocess_prefill(
                input_ids, inputs_embeds, pixel_values
            )
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            assert attention_masks is not None
            attention_mask = attention_masks[0]
            print("@@@ token_type_ids", token_type_ids)
            print("@@@@ prefill_decoder", inputs_embeds)
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                # attention_mask=attention_mask,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=token_type_ids,
            )
            logits = output.logits
            # updated_attention_mask = output.attention_mask
            updated_padded_cache_length = output.padded_cache_lengths
            print("@@ attention_mask", attention_mask)
            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=sliding_window_table_ids[0],
                pad_len=updated_padded_cache_length,
                attention_mask=attention_mask.unsqueeze(0), # unused
            )
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.language_model.decoder = self.model.language_model.decoders[
                padded_batch_size
            ]
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

            # attention_mask = self.attention_manager.update(
            #     running_requests_ids,
            #     attention_mask,
            #     cache_position,
            # )

            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                # attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    # def embed_input_ids(
    #     self,
    #     input_ids: torch.Tensor,
    #     multimodal_embeddings: MultiModalEmbeddings | None = None,
    #     *,
    #     is_multimodal: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     # FIXME embed_input_ids in super class?
    #     # print("@@@ input_ids", input_ids)
    #     # # This is to satisfy the type checker for each overload
    #     # if multimodal_embeddings is None or is_multimodal is None:
    #     #     print("@@@ multimodal_embeddings", multimodal_embeddings)
    #     #     inputs_embeds = self.model._preprocess_prefill(
    #     #         input_ids, None, multimodal_embeddings
    #     #     )
    #     #     print("@@@ inputs_embeds", inputs_embeds)
    #     #     return inputs_embeds
    #     print("@@ multimodal_embeddings", multimodal_embeddings)
    #     inputs_embeds = self.model._preprocess_prefill(
    #         input_ids, None, multimodal_embeddings
    #     )
    #     print("@@ inputs_embeds", inputs_embeds)
    #     return inputs_embeds


    # ------------------------------------------------------------------ #
    # Image processing
    # ------------------------------------------------------------------ #

    def _process_image_input(
        self,
        image_input: Gemma4ImageInputs,
    ):
        vision_outputs = self.vision_tower(
            pixel_values=image_input["pixel_values"],
            pixel_position_id=image_input["pixel_position_id"],
        )
        last_hidden_state = vision_outputs.last_hidden_state
        multimodal_embeddings = self.embed_vision(inputs_embeds=last_hidden_state)

        return multimodal_embeddings

    # ------------------------------------------------------------------ #
    # MultiModalEmbeddings interface
    # ------------------------------------------------------------------ #

    # def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
    #     mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    #     multimodal_embeddings: list[torch.Tensor] = []

    #     for modality, multimodal_input in mm_input_by_modality.items():
    #         if multimodal_input is None:
    #             continue
    #         if modality == "image":
    #             multimodal_embeddings.extend(
    #                 self._process_image_input(multimodal_input)
    #             )
    #         else:
    #             raise NotImplementedError("modality: video, audio")
            # if modality == "image":
            #     multimodal_embeddings.extend(
            #         self._process_image_input(multimodal_input)
            #     )
            # elif modality == "video":
            #     multimodal_embeddings.extend(
            #         self._process_video_input(multimodal_input)
            #     )
            # elif modality == "audio":
            #     multimodal_embeddings.extend(
            #         self._process_audio_input(multimodal_input)
            #     )
        # print("@@ multimodal_embeddings", multimodal_embeddings)
        # return multimodal_embeddings


    def get_pixel_values(self, model_input: ModelInputForRBLN):
        image_input = None

        if model_input.multi_modal_kwargs:
            multimodal_inputs = self._parse_and_validate_multimodal_inputs(
                **model_input.multi_modal_kwargs
            )
            if multimodal_inputs is not None:
                # FIXME it's different with upstream vllm 0.19.1
                print("@@ image_input", image_input)
                # assert image_input["type"] == "pixel_values"
                image_input = multimodal_inputs["image"]
                pixel_values = image_input["pixel_values"]
        else:
            pixel_values = None

        return pixel_values

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, Gemma4ImageInputs | Gemma4AudioInputs | Gemma4VideoInputs | None]:
        mm_input_by_modality = {}
        for input_key in list(kwargs):
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if (
                input_key == "input_features_padded"
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        print("@@@ mm_input_by_modality", mm_input_by_modality)
        return mm_input_by_modality

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma4ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma4 does not support image_embeds."
        if pixel_values is None:
            return None
        return Gemma4ImagePixelInputs(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )