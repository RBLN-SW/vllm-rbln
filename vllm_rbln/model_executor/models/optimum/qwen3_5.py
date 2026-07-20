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
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLVideoPixelInputs

from .qwen_vl import RBLNOptimumQwen2_5_VLForConditionalGeneration


class RBLNOptimumQwen3_5ForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Vision-language Qwen3.5 for RBLN.

    Qwen3.5 is essentially "Qwen3-VL without deepstack": a Qwen3-VL-style vision
    encoder feeds a HYBRID text backbone (GatedDeltaNet ``linear_attention`` layers
    + gated ``full_attention`` layers). Because there is no deepstack, optimum-rbln's
    ``_preprocess_prefill`` returns the base 3-tuple
    ``(inputs_embeds, position_embed, rope_deltas)``, so this class inherits the base
    ``_build_prefill_params`` from Qwen2.5-VL rather than the 5-tuple (deepstack +
    visual_pos_mask) override in ``RBLNOptimumQwen3VLForConditionalGeneration``.

    The ``linear_attention`` layers' ``conv_state``/``recurrent_state`` caches and the
    0/1 control masks (``conv_state_mask``/``recurrent_state_mask``/``valid_mask``) are
    handled entirely inside optimum-rbln's ``RBLNQwen3_5RuntimeModel``; this wrapper
    passes only the standard prefill/decode kwargs, exactly like the other Qwen-VL
    wrappers. ``full_attention`` layers keep the on-device paged KV cache.
    """

    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        # Qwen3.5 (like Qwen3-VL) has no ``second_per_grid_ts``; its ``get_rope_index``
        # separates videos by timestamps instead, so nothing extra is added here.
        pass

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts=None,
    ):
        # Mirrors Qwen3-VL: build the Qwen2.5-VL video-pixel carrier WITHOUT requiring
        # ``second_per_grid_ts`` (Qwen2.5-VL's own override raises when it is None).
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _image_token_id(self) -> int:
        # Qwen3.5's HF config names the placeholder ``image_token_id`` (top-level),
        # not ``image_token_index`` as the mixin default assumes. Only reached on the
        # EC-producer path (not enabled for Qwen3.5 today), but kept correct.
        return self.model.config.image_token_id
