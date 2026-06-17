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
from vllm.model_executor.models.blip2 import (
    Blip2ImageEmbeddingInputs,
    Blip2ImageInputs,
    Blip2ImagePixelInputs,
)

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)


class RBLNOptimumBlip2ForConditionalGeneration(
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

        is_prompt = model_input.is_prompt

        request_nums = input_ids.shape[0]

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )

        if is_prompt:
            block_tables = kwargs.pop("block_tables")
            cache_position = kwargs.pop("cache_position")

            # inputs_embeds are computed at the runner level (embed_multimodal
            # + embed_input_ids); see RBLNOptimumModelRunner._maybe_embed_inputs.
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

            logits = self.model.language_model.decoder(**kwargs).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_language_model(self):
        return self.model.language_model

    def _process_image_input(self, image_input: Blip2ImageInputs) -> list[torch.Tensor]:
        # FIXME new API in optimum-rbln
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])

        # Replicates the vision-encode portion of optimum-rbln's
        # _preprocess_prefill: vision_model -> Q-Former (query_tokens) ->
        # language_projection. optimum-rbln does not expose a standalone
        # get_image_features for BLIP-2.
        model = self.model
        pixel_values = image_input["data"]

        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs[0]
        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        # (num_images, num_query_tokens, text_hidden_size)
        language_model_inputs = model.language_projection(query_output)
        return list(language_model_inputs)

    def _parse_and_validate_image_input(self, **kwargs: Any) -> Blip2ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            expected_h = expected_w = config.vision_config.image_size
            return Blip2ImagePixelInputs(
                type="pixel_values",
                data=pixel_values,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        if image_embeds is not None:
            return Blip2ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")
