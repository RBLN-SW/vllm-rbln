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
import json
import os
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


def _run_length_from(token_types: list[int], start: int, value: int, cap: int) -> int:
    """Length of the run of ``value`` starting at ``start``, capped at ``cap``.

    Mirrors optimum-rbln's ``_run_length_from`` (decoderonly_runtime_utils.py) so
    the slot count this processor reserves matches what ``_plan_prefill_chunks``
    plans at runtime.

    Examples:
        tt = [0, 0, 0, 1, 1, 1, 1, 0, 0]
              ^ start=0
        _run_length_from(tt, 0, value=0, cap=256)  # -> 3  (leading text run)
        _run_length_from(tt, 3, value=1, cap=256)  # -> 4  (image run)
        _run_length_from(tt, 3, value=1, cap=2)    # -> 2  (capped at 2)
    """
    n = len(token_types)
    end = min(start + cap, n)
    i = start
    while i < end and token_types[i] == value:
        i += 1
    return i - start


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
    def apply(self, *args, **kwargs):
        output = super().apply(*args, **kwargs)
        output["prompt_token_ids"] = self._pad_image_boundaries(
            output["prompt_token_ids"]
        )
        return output

    def _rbln_cfg(self) -> dict:
        cached = getattr(self, "_rbln_cfg_cache", None)
        if cached is not None:
            return cached

        model_path = self.info.ctx.model_config.model
        cfg_path = os.path.join(model_path, "language_model", "rbln_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        self._rbln_cfg_cache = cfg
        return cfg

    def _image_buckets(self) -> list[int]:
        """Image-prefill bucket sizes, smallest-first-fit candidates.

        Overridden per model: gemma3 has a single ``image_prefill_chunk_size``,
        gemma4 has a list ``image_prefill_chunk_sizes``. Empty ⇒ no image prefill.
        """
        raise NotImplementedError

    def _use_image_prefill(self) -> bool:
        # Mirrors optimum's `use_tt = use_image_prefill and token_type_ids is not None`.
        # `use_image_prefill` is usually absent in rbln_config, so the presence of
        # image buckets is the proxy.
        cfg = self._rbln_cfg()
        if "use_image_prefill" in cfg:
            return bool(cfg["use_image_prefill"])
        return bool(self._image_buckets())

    def _token_types(self, prompt_ids: list[int]) -> list[int]:
        # 1 = image soft token, 0 = text. (Video=2 lands here once supported.)
        image_token_id = self.info.get_hf_processor().image_token_id
        return [1 if t == image_token_id else 0 for t in prompt_ids]

    def _resolve_image_chunk(
        self, token_types: list[int], step: int, start_type: int
    ) -> tuple[int, int]:
        """Return ``(run_len, chunk_size)`` for the image/video run at ``step``.

        Picks the smallest bucket that fits the run — identical to optimum-rbln's
        ``_resolve_image_chunk``. Reduces to the single bucket for gemma3.
        """
        buckets = self._image_buckets()
        max_bucket = max(buckets)
        run_len = _run_length_from(token_types, step, start_type, max_bucket + 1)
        if run_len > max_bucket:
            modality = "video" if start_type == 2 else "image"
            raise ValueError(
                f"{modality.capitalize()} run (token_type={start_type}) starting at "
                f"position {step} is longer than the largest image-prefill bucket "
                f"({max_bucket}); no bucket can hold it."
            )
        return run_len, min(b for b in buckets if b >= run_len)

    def _required_alloc_len(self, token_types: list[int]) -> int:
        cfg = self._rbln_cfg()
        prefill_chunk_size = cfg["prefill_chunk_size"]
        partition_len = cfg.get("kvcache_partition_len")
        use_tt = self._use_image_prefill()

        query_length = len(token_types)
        step = 0
        padded = 0
        alloc_len = query_length
        while step < query_length:
            start_type = token_types[step] if use_tt else 0
            is_image_prefill = use_tt and start_type > 0
            if is_image_prefill:
                run_len, chunk_size = self._resolve_image_chunk(
                    token_types, step, start_type
                )
            else:
                chunk_size = prefill_chunk_size
                run_len = (
                    _run_length_from(token_types, step, 0, prefill_chunk_size)
                    if use_tt
                    else prefill_chunk_size
                )

            if partition_len is not None:
                offset_in_partition = (step + padded) % partition_len
                if offset_in_partition + chunk_size > partition_len:
                    padded += partition_len - offset_in_partition

            is_last_chunk = step + run_len >= query_length
            if is_last_chunk:
                tail = query_length - step
                num_processed = min(tail, run_len) if run_len > 0 else tail
            else:
                num_processed = run_len

            alloc_len = max(alloc_len, step + padded + chunk_size)
            step += num_processed

        return alloc_len

    def _pad_image_boundaries(self, prompt_ids: list[int]) -> list[int]:
        prompt_ids = list(prompt_ids)
        token_types = self._token_types(prompt_ids)

        alloc_len = self._required_alloc_len(token_types)
        pad_len = alloc_len - len(prompt_ids)
        if pad_len <= 0:
            return prompt_ids
        return [PAD_TOKEN_ID] * pad_len + prompt_ids


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
            token_type_ids[input_ids == self.model.config.image_token_index] = 1

            multimodal_embeddings = self.embed_multimodal(
                **(model_input.multi_modal_kwargs or {})
            )
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings or None,
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
        config = self.model.config
        if config.image_token_index >= self.model.vocab_size:
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
