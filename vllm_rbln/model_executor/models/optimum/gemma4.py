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
    Gemma4DummyInputsBuilder,
    Gemma4ImageInputs,
    Gemma4ImagePixelInputs,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    Gemma4VideoInputs,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .gemma3 import (
    RBLNChunkedPrefillPadMixin,
    RBLNOptimumGemma3ForConditionalGeneration,
)

logger = init_logger(__name__)


class RBLNGemma4MultiModalProcessor(
    RBLNChunkedPrefillPadMixin, Gemma4MultiModalProcessor
):
    # Reuses the shared alloc_len-based left-pad. Gemma4 differs from gemma3 only
    # in image bucketing: a list of buckets (variable soft-token count per image),
    # from which the planner picks the smallest bucket that fits each image run —
    # exactly mirroring optimum-rbln's RBLNGemma4RuntimeModel._resolve_image_chunk.
    # The old per-image pre/post-pad approach is obsolete: tight-packing plus
    # dead-tail masking inside optimum-rbln now enforce the no-image/text-mixing
    # rule, so only the total pad count (not its placement) matters.
    def _image_buckets(self) -> list[int]:
        cfg = self._rbln_cfg()
        sizes = cfg.get("image_prefill_chunk_sizes")
        if sizes is None:
            # Some configs persist the scalar form instead of the list.
            size = cfg.get("image_prefill_chunk_size")
            return [size] if size is not None else []
        return list(sizes) if isinstance(sizes, (list, tuple)) else [sizes]


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
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

        image_embeds = self.model.get_image_features(
            pixel_values=pixel_values, pixel_position_ids=pixel_position_ids
        )

        return image_embeds

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
