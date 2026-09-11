# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import replace
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec

from .base import ModelInputForRBLN
from .qwen2_vl import RBLNOptimumQwen2_5_VLForConditionalGeneration, split_by_grid_thw

logger = init_logger(__name__)


class RBLNOptimumQwen3VLForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Qwen3-VL reuses Qwen2.5-VL classes with the same implementation.
    However, since Qwen3-VL does not require second_per_grid_ts,
    certain methods are overridden to exclude it from the model inputs.

    Qwen3-VL also emits per-layer deepstack features. `embed_multimodal` packs
    them after each item's embeddings along the hidden axis, so the cache and
    the window cut carry them unchanged; `embed_input_ids` strips them and
    `_pack_deepstack` lays them out for the graph.
    """

    def _video_rope_kwargs(
        self, video_features: list[MultiModalFeatureSpec]
    ) -> dict[str, torch.Tensor]:
        # Qwen3-VL's get_rope_index takes no second_per_grid_ts.
        return {}

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None = None,
    ):
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _process_image_input(self, image_input) -> list[torch.Tensor]:
        if image_input is None or image_input.get("type") != "pixel_values":
            return []
        grid_thw = image_input["image_grid_thw"]
        embeds, deepstack = self.model.visual(
            image_input["pixel_values"], grid_thw=grid_thw
        )
        return split_by_grid_thw(torch.cat([embeds, *deepstack], dim=-1), grid_thw)

    def _process_video_input(self, video_input) -> list[torch.Tensor]:
        if video_input is None or video_input.get("type") != "pixel_values_videos":
            return []
        grid_thw = video_input["video_grid_thw"]
        embeds, deepstack = self.model.visual(
            video_input["pixel_values_videos"], grid_thw=grid_thw
        )
        return split_by_grid_thw(torch.cat([embeds, *deepstack], dim=-1), grid_thw)

    def _num_deepstack_layers(self) -> int:
        return len(self.model.config.vision_config.deepstack_visual_indexes)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter each item's leading hidden block; the deepstack blocks go
        through _pack_deepstack."""
        if multimodal_embeddings:
            hidden = multimodal_embeddings[0].shape[-1] // (
                1 + self._num_deepstack_layers()
            )
            multimodal_embeddings = [e[:, :hidden] for e in multimodal_embeddings]
        return super().embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def build_prefill_forward_inputs(
        self, model_input: ModelInputForRBLN
    ) -> ModelInputForRBLN:
        model_input = super().build_prefill_forward_inputs(model_input)
        visual_pos_mask, deepstack_embeds = self._pack_deepstack(model_input)
        return replace(
            model_input,
            visual_pos_mask=visual_pos_mask,
            deepstack_embeds=deepstack_embeds,
        )

    def _pack_deepstack(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The graph's deepstack inputs: the [1, seq] multimodal mask and a
        [num_layers, seq, hidden] tensor with each item's layers at its
        positions."""
        mm_embeds, mask = model_input.mm_embeds, model_input.is_mm_embed
        if not mm_embeds:
            return None, None
        assert mask is not None
        num_layers = self._num_deepstack_layers()
        packed = torch.cat(mm_embeds)
        hidden = packed.shape[-1] // (1 + num_layers)
        # [N, L * hidden] -> [L, N, hidden]
        deepstack = packed[:, hidden:].reshape(-1, num_layers, hidden).transpose(0, 1)
        out = torch.zeros(num_layers, mask.shape[-1], hidden, dtype=self.dtype)
        out[:, mask[0]] = deepstack.to(self.dtype)
        return mask, out

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        """Prefill forward that feeds visual_pos_mask + deepstack to the prefill
        decoder; decode is unchanged (delegated to the base)."""
        if not model_input.is_prompt:
            return super().forward(model_input, **kwargs)

        prefill_kwargs = {
            "inputs_embeds": model_input.inputs_embeds,
            "position_embed": model_input.position_embed,
            "block_tables": model_input.block_tables,
            "cache_position": model_input.input_positions,
        }
        if model_input.visual_pos_mask is not None:
            prefill_kwargs["visual_pos_mask"] = model_input.visual_pos_mask
        if model_input.deepstack_embeds is not None:
            prefill_kwargs["deepstack_embeds"] = model_input.deepstack_embeds
        return self.model.prefill_decoder(**prefill_kwargs).logits


class RBLNOptimumQwen3VLMoeForConditionalGeneration(
    RBLNOptimumQwen3VLForConditionalGeneration
):
    """
    Qwen3-VL MoE model shares the same input structure as Qwen3-VL,
    so it inherits from RBLNOptimumQwen3VLForConditionalGeneration without changes.
    """

    pass


class RBLNOptimumQwen3_5ForConditionalGeneration(
    RBLNOptimumQwen2_5_VLForConditionalGeneration
):
    """
    Vision-language Qwen3.5 for RBLN.

    Qwen3.5 is a hybrid text backbone (GatedDeltaNet linear_attention layers + gated
    full_attention layers). Its vision encoder returns the merged image embeddings (a
    single tensor). It inherits the multimodal prefill path from Qwen2.5-VL.
    """

    def decode_batch_rows(
        self, cache_slot_ids: torch.Tensor, block_tables: torch.Tensor
    ) -> torch.Tensor:
        # The GatedDeltaNet linear_attention conv/recurrent state is a fixed
        # [max_num_seqs] on-device cache indexed by batch row, so each request
        # is pinned to its scheduler-assigned cache slot for its lifetime.
        return cache_slot_ids.to(torch.long)

    def _video_rope_kwargs(
        self, video_features: list[MultiModalFeatureSpec]
    ) -> dict[str, torch.Tensor]:
        return {}

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None = None,
    ):
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        """Prefill writes one state row, named by ``batch_idx``; decode arrives
        laid out by row (see decode_batch_rows) and the logits are gathered
        back to running order."""
        if model_input.is_prompt:
            assert model_input.cache_slot_ids is not None
            return self.model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                position_embed=model_input.position_embed,
                block_tables=model_input.block_tables,
                cache_position=model_input.input_positions,
                batch_idx=int(model_input.cache_slot_ids[0]),
            ).logits

        assert model_input.batch_rows is not None
        self.model.decoder = self.model.decoders[model_input.padded_batch_size]
        logits = self.model.decoder(
            inputs_embeds=self.model.embed_tokens(model_input.input_tokens),
            cache_position=model_input.input_positions,
            position_embed=model_input.position_embed,
            block_tables=model_input.block_tables,
        ).logits
        return logits[model_input.batch_rows]
