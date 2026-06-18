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
from vllm.model_executor.models.paligemma import (
    PaliGemmaImageEmbeddingInputs,
    PaliGemmaImageInputs,
    PaliGemmaImagePixelInputs,
)

from optimum.rbln.configuration_utils import RBLNModelConfig
from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN

from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

PAD_TOKEN_ID = 0


class RBLNOptimumPaliGemmaForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumDecoderMixin, RBLNOptimumMultimodalMixin
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

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

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )

        if is_prompt:
            block_tables = kwargs.pop("block_tables")
            cache_position = kwargs.pop("cache_position")

            # inputs_embeds are computed at the runner level (embed_multimodal
            # + embed_input_ids); see RBLNOptimumModelRunner._build_forward_inputs.
            inputs_embeds = model_input.inputs_embeds
            logits = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.language_model.decoder = self.model.language_model.decoders[
                padded_batch_size
            ]
            # NOTE(eunji.lee): attention_mask, position_ids are required
            # to paligemma in optimum-rbln.
            # They depends on the version of gemma in paligemma.
            attention_mask, position_ids = self.generate_params_for_gemma(
                padded_batch_size,
                self.model.rbln_config.language_model,
                kwargs["cache_position"],
            )
            logits = self.model.language_model.decoder(
                attention_mask=attention_mask, position_ids=position_ids, **kwargs
            ).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_language_model(self):
        return self.model.language_model

    def _process_image_input(
        self, image_input: PaliGemmaImageInputs
    ) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])

        # Vision tower + multi-modal projector, compiled by optimum-rbln.
        # Returns (num_images, num_image_tokens, hidden_size).
        image_features = self.model.get_image_features(image_input["data"])
        return list(image_features)

    def _embed_text_tokens(
        self, input_ids: torch.Tensor, is_multimodal: torch.Tensor
    ) -> torch.Tensor:
        # PaliGemma's image token can be OOV; PAD-mask those positions before
        # the text embedding lookup (mirrors optimum-rbln's _preprocess_prefill).
        config = self.model.config
        if config.image_token_index >= config.text_config.vocab_size:
            input_ids = input_ids.masked_fill(is_multimodal, PAD_TOKEN_ID)
        return self.model.get_input_embeddings()(input_ids)

    def _parse_and_validate_image_input(
        self, **kwargs: Any
    ) -> PaliGemmaImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            h = w = config.vision_config.image_size
            return PaliGemmaImagePixelInputs(
                type="pixel_values",
                data=pixel_values,
                resolve_bindings={"h": h, "w": w},
            )

        if image_embeds is not None:
            return PaliGemmaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def generate_params_for_gemma(
        self,
        padded_batch_size: int,
        rbln_model_config: RBLNModelConfig,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate attention mask and position ids for gemma.
        """
        max_seq_len = rbln_model_config.max_seq_len
        seq_range = torch.arange(max_seq_len).unsqueeze(0)  # (1, max_seq_len,)
        attention_mask = (seq_range <= cache_position).to(rbln_model_config.torch_dtype)
        position_ids = cache_position.clone()
        return attention_mask, position_ids
