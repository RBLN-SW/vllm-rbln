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
from vllm.model_executor.models.interfaces import SupportsMultiModal

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumBlip2ForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumDecoderMixin, SupportsMultiModal
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

        image_input = None
        pixel_values = None

        padded_batch_size = self.decoder_batch_size
        request_nums = input_ids.shape[0]

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )

        if is_prompt:
            if model_input.multi_modal_kwargs:
                image_input = self._parse_and_validate_image_input(
                    **model_input.multi_modal_kwargs
                )
                if image_input is not None:
                    assert image_input["type"] == "pixel_values"
                    pixel_values = image_input["data"]

            block_tables = kwargs.pop("block_tables")
            input_ids = kwargs.pop("input_ids")
            cache_position = kwargs.pop("cache_position")

            inputs_embeds = self.model._preprocess_prefill(
                pixel_values=pixel_values,
                input_ids=input_ids,
            )
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
