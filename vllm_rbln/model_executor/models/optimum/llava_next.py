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
from typing import Any, Union

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.llava_next import (
    LlavaNextImageInputs,
    LlavaNextImagePixelInputs,
)

from .base import ModelInputForRBLN, version_error
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)


class RBLNOptimumLlavaNextForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumDecoderMixin, RBLNOptimumMultimodalMixin
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

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
                self.model.rbln_config.language_model, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def _forward(
        self,
        is_prefill: bool,
        block_tables: torch.Tensor,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: Union[
            list[torch.Tensor], torch.Tensor
        ] = None,  # vllm keyword argument
        **kwargs,
    ):
        if is_prefill:
            if self.model.language_model.prefill_decoder is None:
                raise version_error

            logits = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits
        else:
            if self.model.language_model.decoder is None:
                raise version_error

            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits

        return logits

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        is_prompt = model_input.is_prompt

        request_nums = input_ids.shape[0]

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )
        input_ids = kwargs.pop("input_ids")
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")
        if not is_prompt:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.language_model.decoder = self.model.language_model.decoders[
                padded_batch_size
            ]

        inputs_embeds = None
        if is_prompt:
            multimodal_embeddings = self.embed_multimodal(
                **(model_input.multi_modal_kwargs or {})
            )
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings or None,
            )

        logits = self._forward(
            is_prefill=is_prompt,
            block_tables=block_tables,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_language_model(self):
        return self.model.language_model

    def _process_image_input(
        self, image_input: LlavaNextImageInputs
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        image_sizes = image_input["image_sizes"]
        config = self.model.config

        # Vision tower + multi-modal projector, compiled by optimum-rbln.
        # Returns a tuple of per-image features split by num_patches.
        image_features = self.model.get_image_features(
            pixel_values,
            image_sizes,
            vision_feature_layer=config.vision_feature_layer,
            vision_feature_select_strategy=config.vision_feature_select_strategy,
        )
        # spatial_unpad merge + image_newline insertion. Returns the flattened
        # features for all images plus the per-image token counts.
        image_features, feature_lens = self.model.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=config.vision_feature_select_strategy,
            image_newline=self.model.image_newline,
        )
        return list(torch.split(image_features, feature_lens.tolist()))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Mirrors optimum-rbln's _preprocess_prefill: image tokens are in-vocab
        # for LLaVA-NeXT, so no PAD masking is needed; just scatter the packed
        # image features over the image-token positions.
        config = self.model.config
        if is_multimodal is None:
            is_multimodal = input_ids == config.image_token_index

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        mm_embeds = torch.cat(list(multimodal_embeddings)).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        scatter_mask = is_multimodal.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(scatter_mask, mm_embeds)

    def _parse_and_validate_image_input(
        self, **kwargs: Any
    ) -> LlavaNextImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            expected_h = expected_w = config.vision_config.image_size
            return LlavaNextImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                resolve_bindings={
                    "h": expected_h,
                    "w": expected_w,
                },
            )

        if image_embeds is not None:
            raise NotImplementedError(
                "Image embeds are not supported in this version for RBLN"
            )

        raise AssertionError("This line should be unreachable.")
