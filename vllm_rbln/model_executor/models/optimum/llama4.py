# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.mllama4 import Llama4ImagePatchInputs

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)


class RBLNOptimumLlama4ForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumMultimodalMixin, RBLNOptimumDecoderMixin
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        raise ValueError("Only image modality is supported")

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config=vllm_config)
        language_config = self.model.rbln_config.language_model
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=language_config.use_multiple_decoder,
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=language_config.decoder_batch_sizes,
        )

    def get_language_model(self):
        return self.model.language_model

    def _image_token_id(self) -> int:
        return self.model.config.image_token_id

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Llama4ImagePatchInputs | None:
        if kwargs.get("image_embeds") is not None:
            raise NotImplementedError("Llama 4 image_embeds inputs are not supported")
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        return Llama4ImagePatchInputs(
            pixel_values=pixel_values,
            patches_per_image=kwargs.pop("patches_per_image"),
            aspect_ratios=kwargs.pop("aspect_ratios"),
        )

    def _process_image_input(
        self, image_input: Llama4ImagePatchInputs
    ) -> list[torch.Tensor]:
        features = self.model.get_image_features(
            image_input["pixel_values"]
        ).last_hidden_state
        return [
            image.flatten(0, 1)
            for image in features.split(image_input["patches_per_image"].tolist())
        ]

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        language_model = self.model.language_model
        if model_input.is_prompt:
            return language_model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                cache_position=model_input.input_positions,
                block_tables=model_input.block_tables,
            ).logits

        language_model.decoder = language_model.decoders[model_input.padded_batch_size]
        # Llama 4 compiles both prefill and decode with inputs_embeds.
        return language_model.decoder(
            inputs_embeds=self.model.get_input_embeddings()(model_input.input_tokens),
            cache_position=model_input.input_positions,
            block_tables=model_input.block_tables,
        ).logits
