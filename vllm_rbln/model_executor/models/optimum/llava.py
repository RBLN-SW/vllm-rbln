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
from typing import List, Optional, Union

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llava import (LlavaImageInputs,
                                              LlavaImagePixelInputs,
                                              PixtralHFImagePixelInputs)
from vllm.model_executor.models.utils import flatten_bn

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumLlavaForConditionalGeneration(RBLNOptimumModelBase,
                                               RBLNOptimumDecoderMixin,
                                               SupportsMultiModal):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config.language_model,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.
            decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def _forward(
        self,
        is_prefill: bool,
        block_tables: torch.Tensor,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        cache_position: Union[List[torch.Tensor],
                              torch.Tensor] = None,  # vllm keyword argument
        **kwargs,
    ):
        if is_prefill:
            # NOTE inputs_embeds will be generated inside _preprocess_prefill
            inputs_embeds = self.model._preprocess_prefill(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )

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

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        is_prompt = model_input.is_prompt

        request_nums = input_ids.shape[0]
        if model_input.multi_modal_kwargs:
            image_input = self._parse_and_validate_image_input(
                **model_input.multi_modal_kwargs)
            if image_input is not None:
                if image_input["type"] == "pixel_values":
                    pixel_values = image_input["pixel_values"]
                    image_sizes = None
                elif image_input["type"] == "pixel_values_pixtral":
                    pixel_values = image_input["pixel_values"]
                    image_sizes = torch.tensor(
                        pixel_values.shape[-2:]).unsqueeze(0)
        else:
            pixel_values = None
            image_sizes = None

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, cache_position)
        input_ids = kwargs.pop("input_ids")
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")
        if not is_prompt:
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.language_model.decoder = \
                self.model.language_model.decoders[padded_batch_size]

        logits = self._forward(
            is_prefill=is_prompt,
            block_tables=block_tables,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            cache_position=cache_position,
        )

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, torch.Tensor | list):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if config.vision_config.model_type == "pixtral":
                return PixtralHFImagePixelInputs(
                    type="pixel_values_pixtral",
                    pixel_values=flatten_bn(pixel_values),
                )

            expected_h = expected_w = config.vision_config.image_size
            return LlavaImagePixelInputs(
                type="pixel_values",
                pixel_values=flatten_bn(pixel_values, concat=True),
                resolve_bindings={
                    "h": expected_h,
                    "w": expected_w
                },
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor | list):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            raise NotImplementedError(
                "Image embeds are not supported in this version for RBLN")
        raise AssertionError("This line should be unreachable.")
