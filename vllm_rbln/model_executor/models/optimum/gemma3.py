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
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma3_mm import (
    Gemma3DummyInputsBuilder,
    Gemma3ImageInputs,
    Gemma3ImagePixelInputs,
    Gemma3MultiModalProcessor,
    Gemma3ProcessingInfo,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN, version_error
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)
from .optimum_attention import HybridAttentionImageManager, HybridAttentionImageStrategy

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.multimodal.processing.processor import (
        BaseMultiModalProcessor as _ProcessorBase,
    )
else:
    _ProcessorBase = object

PAD_TOKEN_ID = 0


class RBLNChunkedPrefillPadMixin(_ProcessorBase):
    """Left-pad ``prompt_token_ids`` so vLLM reserves enough KV-cache blocks.

    Why:
        vLLM sizes block allocation from the prompt length, but optimum-rbln's
        chunked prefill touches extra slots beyond the real tokens (trailing
        chunk write-extent + ``kvcache_partition_len`` alignment). We replay its
        planner (``_plan_prefill_chunks``) for the highest slot touched
        (``alloc_len``) and prepend ``alloc_len - query_length`` pad tokens.

    Placement:
        Pad tokens are masked out and stripped before attention, so only the
        count matters, not where they go.

    Subclasses override ``_image_buckets`` (gemma3: single bucket; gemma4: many)
    and, once video is supported, ``_token_types``. Bucket-selection and planning
    mirror optimum-rbln's ``RBLNDecoderOnly*`` mixins.
    """

    # MRO note: mix in BEFORE the HF ``*MultiModalProcessor`` so this ``apply``
    # wraps theirs (``super().apply`` resolves to the HF processor).
    #
    def apply(self, *args, **kwargs):
        output = super().apply(*args, **kwargs)
        return output


class RBLNGemma3MultiModalProcessor(
    RBLNChunkedPrefillPadMixin, Gemma3MultiModalProcessor
):
    def _image_buckets(self) -> list[int]:
        # gemma3: single image bucket.
        size = self._rbln_cfg().get("image_prefill_chunk_size")
        return [size] if size is not None else []


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma3MultiModalProcessor,
    info=Gemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class RBLNOptimumGemma3ForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
    RBLNOptimumDecoderMixin,
    VllmModelForTextGeneration,
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
        self.strategy = HybridAttentionImageStrategy(PAD_TOKEN_ID)
        self.attention_manager: HybridAttentionImageManager = (
            HybridAttentionImageManager(self.strategy)
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions
        block_tables = model_input.block_tables

        is_prompt = model_input.is_prompt

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids, padded_cache_lengths, attention_masks = (
            self.attention_manager.get(
                is_prompt,
                self.decoder_batch_size,
                running_requests_ids,
                finished_requests_ids,
                input_ids=input_ids,
            )
        )
        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, position_ids
        )

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx], dtype=torch.int16)
            # token_type_ids model_input != token_type_ids of gemma3
            # https://github.com/huggingface/transformers/blob/d0c9c66d1c09df3cd70bf036e813d88337b20d4c/src/transformers/models/gemma3/processing_gemma3.py#L143
            token_type_ids = torch.zeros_like(input_ids)
            # `_image_token_id()` resolves the placeholder id per model: Gemma3Config
            # has `image_token_index`, Gemma4Config (which inherits this forward) has
            # `image_token_id`. Subclasses override `_image_token_id()` accordingly.
            token_type_ids[input_ids == self._image_token_id()] = 1

            multimodal_embeddings = self.embed_multimodal(
                **(model_input.multi_modal_kwargs or {})
            )
            # Pass through as-is: `embed_input_ids` already treats None / len()==0 as
            # "text only". A bare `... or None` raises on Gemma4, whose multimodal
            # embeddings are a Tensor ("Boolean value of Tensor ... is ambiguous").
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings,
            )
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            assert attention_masks is not None
            attention_mask = attention_masks[0]
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                attention_mask=attention_mask,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=token_type_ids,
            )
            logits = output.logits
            updated_attention_mask = output.attention_mask
            left_pad = int((attention_mask == 0).sum().item())

            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=sliding_window_table_ids[0],
                pad_len=left_pad,
                attention_mask=updated_attention_mask,
            )
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.language_model.decoder = self.model.language_model.decoders[
                padded_batch_size
            ]
            (
                local_block_table_id,
                cache_position,
                position_ids,
                attention_mask,
            ) = self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
                pad_lens=padded_cache_lengths,
                attention_masks=attention_masks,
            )

            attention_mask = self.attention_manager.update(
                running_requests_ids,
                attention_mask,
                cache_position,
            )

            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_language_model(self):
        return self.model.language_model

    def build_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list,
        *,
        cache_position: torch.Tensor | None = None,
        running_requests_ids: list[str] | None = None,
    ) -> dict:
        # NOTE: this guard is currently unreachable — init_model() only enables
        # the EC path for "RBLNQwen3VLForConditionalGeneration", so Gemma3 never
        # enters here today. It documents the contract for when EC is extended.
        raise NotImplementedError(
            "EC disaggregation is not implemented for Gemma3: its hybrid "
            "sliding-window attention prefill needs attention_manager state "
            "that build_prefill_inputs does not yet provide."
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
