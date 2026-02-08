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
from vllm.model_executor.models.idefics3 import (Idefics3ImageEmbeddingInputs,
                                                 Idefics3ImagePixelInputs,
                                                 ImageInputs)
from vllm.model_executor.models.interfaces import SupportsMultiModal

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumIdefics3ForConditionalGeneration(RBLNOptimumModelBase,
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
            use_multiple_decoder=getattr(self.model.rbln_config.text_model,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.text_model.
            decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, cache_position)

        if is_prompt:
            if model_input.multi_modal_kwargs:
                image_input = self._parse_and_validate_image_input(
                    **model_input.multi_modal_kwargs)

            # Only when image input is given
            if image_input is not None:
                pixel_values = image_input["pixel_values"].unsqueeze(0)
                pixel_attention_mask = image_input[
                    "pixel_attention_mask"].unsqueeze(0)
            else:
                pixel_values = None
                pixel_attention_mask = None

            block_tables = kwargs.pop("block_tables")
            input_ids = kwargs.pop("input_ids")
            cache_position = kwargs.pop("cache_position")

            inputs_embeds = self.model._preprocess_prefill(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
            )
            logits = self.model.text_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.text_model.decoder = self.model.text_model.decoders[
                padded_batch_size]
            logits = self.model.text_model.decoder(**kwargs).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.model.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: Any) -> Optional[ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor | list):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return Idefics3ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        if pixel_values is not None:
            if not isinstance(pixel_values, torch.Tensor | list):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            pixel_attention_mask = kwargs.pop("pixel_attention_mask")
            if not isinstance(pixel_attention_mask, torch.Tensor | list):
                raise ValueError("Incorrect type of pixel_attention_mask. "
                                 f"Got type: {type(pixel_attention_mask)}")

            num_patches = kwargs.pop("num_patches")
            if not isinstance(num_patches, torch.Tensor | list):
                raise ValueError("Incorrect type of num_patches. "
                                 f"Got type: {type(num_patches)}")

            expected_h = expected_w = config.vision_config.image_size
            return Idefics3ImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
                resolve_bindings={
                    "h": expected_h,
                    "w": expected_w
                },
            )

        raise AssertionError("This line should be unreachable.")
