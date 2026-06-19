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
from vllm.model_executor.models.idefics3 import (
    Idefics3ImageEmbeddingInputs,
    Idefics3ImagePixelInputs,
    ImageInputs,
)

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)


class RBLNOptimumIdefics3ForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumMultimodalMixin, RBLNOptimumDecoderMixin
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
                self.model.rbln_config.text_model, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.text_model.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        if is_prompt:
            kwargs = self.preprocess_for_prefill(
                block_tables, input_ids, cache_position
            )
            block_tables = kwargs.pop("block_tables")
            cache_position = kwargs.pop("cache_position")

            inputs_embeds = model_input.inputs_embeds
            logits = self.model.text_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits
        else:
            kwargs = self.preprocess_for_decode(block_tables, input_ids, cache_position)
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.text_model.decoder = self.model.text_model.decoders[
                padded_batch_size
            ]
            logits = self.model.text_model.decoder(**kwargs).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_language_model(self):
        return self.model.text_model

    def _process_image_input(self, image_input: ImageInputs) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])

        # Vision model + pixel-shuffle connector, compiled by optimum-rbln.
        # get_image_features expects a leading batch dim
        # (batch_size, num_images, num_channels, height, width).
        pixel_values = image_input["pixel_values"].unsqueeze(0)
        pixel_attention_mask = image_input["pixel_attention_mask"].unsqueeze(0)
        image_features = self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )
        return list(image_features)

    def _image_token_id(self) -> int:
        # Idefics3Config exposes only `image_token_id` (no `image_token_index`
        # attribute_map alias, unlike PaliGemma/Gemma3/LLaVA), so the mixin
        # default would raise AttributeError here.
        return self.model.config.image_token_id

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
                    f"You supplied {tuple(d.shape)}."
                )

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(self, **kwargs: Any) -> ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return Idefics3ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        if pixel_values is not None:
            pixel_attention_mask = kwargs.pop("pixel_attention_mask")
            num_patches = kwargs.pop("num_patches")

            expected_h = expected_w = config.vision_config.image_size
            return Idefics3ImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        raise AssertionError("This line should be unreachable.")
