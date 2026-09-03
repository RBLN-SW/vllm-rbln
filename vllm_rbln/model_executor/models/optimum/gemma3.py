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
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma3_mm import (
    Gemma3ImageInputs,
    Gemma3ImagePixelInputs,
)

from .base import ModelInputForRBLN, version_error
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)

PAD_TOKEN_ID = 0


@dataclass
class MultimodalHybridAttentionEntry:
    pad_len: int
    attention_mask: torch.Tensor


class MultimodalHybridAttentionStateManager:
    """Per-request decode state that the prefill graph produces and the
    scheduler therefore cannot own: the padded cache length and the attention
    mask over the padded cache layout. Recorded at prefill; every decode step
    replays it, enables the newly generated token position, and stores the
    advanced mask for the next step. The model runner frees an entry when its
    request finishes.
    """

    def __init__(self) -> None:
        self.table: dict[str, MultimodalHybridAttentionEntry] = {}

    def add(self, request_id: str, pad_len: int, attention_mask: torch.Tensor) -> None:
        self.table[request_id] = MultimodalHybridAttentionEntry(
            pad_len=pad_len,
            attention_mask=attention_mask,
        )

    def build_decode_attention_inputs(
        self,
        running_requests_ids: list[str],
        position_ids: torch.Tensor,
        decoder_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch the recorded state in running order.

        `position_ids` must already be padded to `decoder_batch_size` rows.
        Returns (cache_position, position_ids, attention_mask): each row's
        cache_position offsets its position by the request's padded cache
        length. The advanced attention mask is written back to the table for
        the next step.
        """
        assert position_ids.shape[0] == decoder_batch_size
        entries = [self.table[request_id] for request_id in running_requests_ids]

        pad_lens = torch.zeros(decoder_batch_size, 1, dtype=torch.int16)
        pad_lens[: len(entries), 0] = torch.tensor(
            [entry.pad_len for entry in entries], dtype=torch.int16
        )
        cache_position = position_ids + pad_lens

        masks = [entry.attention_mask for entry in entries]
        attention_mask = torch.zeros(
            decoder_batch_size, masks[0].shape[1], dtype=masks[0].dtype
        )
        attention_mask[: len(masks)] = torch.cat(masks)
        # Enable attention for the token generated this step.
        rows = torch.arange(attention_mask.shape[0])
        attention_mask[rows, cache_position.squeeze(1)] = 1
        for idx, request_id in enumerate(running_requests_ids):
            self.table[request_id].attention_mask = attention_mask[idx : idx + 1]

        return cache_position, position_ids, attention_mask

    def pop(self, request_id: str) -> None:
        self.table.pop(request_id, None)

    def clear(self) -> None:
        self.table.clear()


class RBLNOptimumGemma3ForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
    RBLNOptimumDecoderMixin,
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<start_of_image>"

        raise ValueError("Only image modality is supported")

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        # NOTE:
        # model_config.vocab_size != tokenizer.vocab_size in Gemma3
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config.language_model,
                "use_multiple_decoder",
                False,
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )
        self.attention_manager = MultimodalHybridAttentionStateManager()

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        cache_slot_ids = model_input.cache_slot_ids
        assert cache_slot_ids is not None

        running_requests_ids = model_input.running_requests_ids
        request_nums = model_input.input_tokens.shape[0]

        if model_input.is_prompt:
            prefill_inputs = self.prepare_prefill_inputs(model_input)
            input_ids = prefill_inputs.input_ids
            # token_type_ids model_input != token_type_ids of gemma3
            # https://github.com/huggingface/transformers/blob/d0c9c66d1c09df3cd70bf036e813d88337b20d4c/src/transformers/models/gemma3/processing_gemma3.py#L143
            token_type_ids = torch.zeros_like(input_ids)
            # `_image_token_id()` resolves the placeholder id per model: Gemma3Config
            # has `image_token_index`, Gemma4Config (which inherits this forward) has
            # `image_token_id`. Subclasses override `_image_token_id()` accordingly.
            token_type_ids[input_ids == self._image_token_id()] = 1

            # Not an input of the compiled graph: optimum-rbln only uses this
            # mask to drop padding from the prefill inputs.
            attention_mask = (input_ids != PAD_TOKEN_ID).squeeze(0)
            inputs_embeds = model_input.inputs_embeds
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=prefill_inputs.cache_position,
                attention_mask=attention_mask,
                local_block_tables=cache_slot_ids,
                block_tables=prefill_inputs.block_tables,
                token_type_ids=token_type_ids,
            )
            logits = output.logits
            # The prefill graph computes the padded cache length and the
            # attention mask over the padded cache layout; keep them for the
            # decode steps of this request.
            self.attention_manager.add(
                running_requests_ids[0],
                pad_len=output.padded_cache_lengths,
                attention_mask=output.attention_mask,
            )
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            decode_inputs = self.prepare_decode_inputs(
                model_input, cache_slot_ids=cache_slot_ids
            )
            self.model.language_model.decoder = self.model.language_model.decoders[
                decode_inputs.padded_batch_size
            ]
            # `cache_position` and `position_ids` are distinguished due to the
            # padding space reserved in the cache during prefill.
            (
                cache_position,
                position_ids,
                attention_mask,
            ) = self.attention_manager.build_decode_attention_inputs(
                running_requests_ids,
                decode_inputs.cache_position,
                decode_inputs.padded_batch_size,
            )

            logits = self.model.language_model.decoder(
                input_ids=decode_inputs.input_ids,
                cache_position=cache_position,
                block_tables=decode_inputs.block_tables,
                local_block_tables=decode_inputs.local_block_tables,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
            logits = logits[:request_nums]
        return logits

    def get_language_model(self):
        return self.model.language_model

    def build_prefill_inputs_from_cache(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list,
        *,
        cache_position: torch.Tensor | None = None,
        running_requests_ids: list[str] | None = None,
        mrope_position_deltas: dict[str, float] | None = None,
    ) -> dict:
        # NOTE: this guard is currently unreachable — init_model() only enables
        # the EC path for "RBLNQwen3VLForConditionalGeneration", so Gemma3 never
        # enters here today. It documents the contract for when EC is extended.
        raise NotImplementedError(
            "EC disaggregation is not implemented for Gemma3: its hybrid "
            "sliding-window attention prefill needs the cache slot ids from "
            "ModelInputForRBLN and must record the prefill graph's attention "
            "mask, which build_prefill_inputs_from_cache does not support."
        )

    def _process_image_input(
        self, image_input: Gemma3ImageInputs
    ) -> list[torch.Tensor]:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]

        # Vision tower + multi-modal projector, compiled by optimum-rbln.
        # Returns (num_patches_total, mm_tokens_per_image, hidden_size).
        image_embeds = self.model.get_image_features(pixel_values)

        if num_patches is None:
            return list(image_embeds)
        return [e.flatten(0, 1) for e in image_embeds.split(num_patches.tolist())]

    def _embed_text_tokens(
        self, input_ids: torch.Tensor, is_multimodal: torch.Tensor
    ) -> torch.Tensor:
        # Gemma3's image token can be OOV; PAD-mask those positions before the
        # text embedding lookup (mirrors optimum-rbln's _preprocess_prefill).
        # `_image_token_id()` resolves per model (Gemma3 `image_token_index` /
        # Gemma4 `image_token_id`).
        if self._image_token_id() >= self.model.vocab_size:
            input_ids = input_ids.masked_fill(is_multimodal, PAD_TOKEN_ID)
        return self.model.get_input_embeddings()(input_ids)

    def _parse_and_validate_image_input(
        self, **kwargs: Any
    ) -> Gemma3ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config

        assert image_embeds is None, "Gemma3 does not support image_embeds."
        if pixel_values is None:
            return None

        image_size = config.vision_config.image_size
        return Gemma3ImagePixelInputs(
            pixel_values=pixel_values,
            num_patches=num_patches,
            resolve_bindings={"h": image_size, "w": image_size},
        )
