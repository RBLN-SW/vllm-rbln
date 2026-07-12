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
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4_mm import (
    Gemma4AudioInputs,
    Gemma4ImageInputs,
    Gemma4ImagePixelInputs,
    Gemma4VideoInputs,
)

from .gemma3 import (
    RBLNOptimumGemma3ForConditionalGeneration,
)

logger = init_logger(__name__)


class RBLNOptimumGemma4ForConditionalGeneration(
    RBLNOptimumGemma3ForConditionalGeneration
):
    def _image_token_id(self) -> int:
        # Gemma4Config names the image placeholder `image_token_id`
        # (Gemma3Config uses `image_token_index`, the base-class default).
        return self.model.config.image_token_id

    def _process_image_input(
        self, image_input: Gemma4ImageInputs
    ) -> list[torch.Tensor]:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]

        # get_image_features returns one flat [total_soft_tokens, hidden] tensor
        # (padding stripped by the pooler mask), so split it back per image as
        # the base embed path and partial prefix-cache slicing expect.
        # Upstream vLLM splits the same way:
        # https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/model_executor/models/gemma4_mm.py#L1348
        image_embeds = self.model.get_image_features(
            pixel_values=pixel_values, pixel_position_ids=pixel_position_ids
        )

        # The pooler averages each k**2 patch block into one soft token, so
        # soft_tokens(i) = non_padding_patches(i) // k**2 (padding is (-1, -1)).
        k2 = self.model.config.vision_config.pooling_kernel_size**2
        valid_patches = (~(pixel_position_ids == -1).all(dim=-1)).sum(dim=-1)
        valid_lens = (valid_patches // k2).tolist()
        return list(torch.split(image_embeds, valid_lens))

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
            # NOTE: Other multimodal types (audio, video) will be implemented later.

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
