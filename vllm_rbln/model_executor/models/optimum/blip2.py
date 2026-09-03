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
    RBLNOptimumModelBase, RBLNOptimumMultimodalMixin, RBLNOptimumDecoderMixin
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
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        if model_input.is_prompt:
            return self.model.language_model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                cache_position=model_input.input_positions,
                block_tables=model_input.block_tables,
            ).logits

        self.model.language_model.decoder = self.model.language_model.decoders[
            model_input.padded_batch_size
        ]
        logits = self.model.language_model.decoder(
            input_ids=model_input.input_tokens,
            cache_position=model_input.input_positions,
            block_tables=model_input.block_tables,
        ).logits
        return logits[: len(model_input.running_requests_ids)]

    def get_language_model(self):
        return self.model.language_model

    def _process_image_input(self, image_input: Blip2ImageInputs) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])

        # Vision model + Q-Former + language projection, compiled by
        # optimum-rbln.
        pixel_values = image_input["data"]
        # (num_images, num_query_tokens, text_hidden_size)
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        return list(image_features)

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
