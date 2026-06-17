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
from vllm.config import VllmConfig
from vllm.model_executor.models.whisper import (
    WhisperAudioInputs,
    WhisperForConditionalGeneration,
)
from vllm.utils.jsontree import json_map_leaves

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase


class RBLNOptimumWhisperForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumDecoderMixin,
    WhisperForConditionalGeneration,
):
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        # WhisperForConditionalGeneration inherits SupportsLoRA, so vLLM
        # accepts a `--lora-modules`/`lora_config` for this model. The RBLN
        # backend does not support LoRA yet, so reject it explicitly instead
        # of silently ignoring the adapters.
        if vllm_config.lora_config is not None:
            raise NotImplementedError(
                "LoRA is not supported for Whisper on the RBLN backend. "
                "Please run the model without LoRA adapters."
            )
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=False,
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=[self.batch_size],
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )
        self.dec_max_seq_len = self.model_config.max_model_len
        self.dec_lengths = [0] * self.batch_size

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        block_tables = model_input.block_tables
        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt
        valid_block_ids = block_tables.flatten().to(torch.int32)

        if is_prompt:
            if model_input.multi_modal_kwargs:
                audio_input = self._parse_and_validate_audio_input(
                    **model_input.multi_modal_kwargs
                )
                input_features = audio_input["input_features"]
            if input_features is None:
                raise ValueError("Whisper requires `input_features` as an input.")
            _ = self.model.encoder(
                input_features=input_features,
                block_tables=block_tables.squeeze(0).to(torch.int16),
            )

        # Whisper model does not support bucketing.
        decoder_attention_mask = torch.zeros(
            self.batch_size, self.dec_max_seq_len, dtype=self.dtype
        )
        if is_prompt:
            # valid_block_ids has length 1 in prefill.
            assert valid_block_ids.shape[0] == 1, (
                "Whisper only supports batch_size=1 in prefill step."
            )
            batch_idx = valid_block_ids[0]
            token_sequence = input_ids[0].tolist()
            step_decoder_input_ids = torch.zeros(self.batch_size, 1, dtype=torch.long)
            # The decoder runtime is compiled at self.batch_size, so prefill
            # must still feed every slot. Point unused slots at a scratch
            # block so their K/V writes don't touch the active prefill block
            # or any other request's KV cache.
            if model_input.dummy_block != self.batch_size:
                raise RuntimeError(
                    f"Whisper prefill expects dummy_block to equal batch_size "
                    f"(got dummy_block={model_input.dummy_block}, "
                    f"batch_size={self.batch_size}). The scheduler should "
                    f"allocate the dummy block at index batch_size so unused "
                    f"slots don't collide with active KV cache blocks. "
                    f"This likely indicates a stale compiled artifact. "
                    f"Please recompile the model to regenerate the correct "
                    f"block layout."
                )
            decoder_block_tables = torch.full(
                (self.batch_size, 1), model_input.dummy_block, dtype=torch.int16
            )
            decoder_block_tables[batch_idx, 0] = batch_idx
            decoder_cache_position = torch.zeros(self.batch_size, 1, dtype=torch.int32)
            for step, token_id in enumerate(token_sequence):
                step_decoder_input_ids[batch_idx, 0] = token_id

                # cache_position: where in the KV cache this token's K/V is stored.
                # attention_mask: which positions this token may attend to.
                # Causal, so only past and current positions are visible; the
                # mask grows by one bit per step rather than being rebuilt.
                # e.g. step=2 -> cache_position=2, mask=[1,1,1,0,...,0]
                decoder_cache_position[batch_idx, 0] = step
                decoder_attention_mask[batch_idx, step] = 1
                decoder_output = self.model.decoder(
                    decoder_input_ids=step_decoder_input_ids.contiguous(),
                    decoder_attention_mask=decoder_attention_mask,
                    cache_position=decoder_cache_position,
                    block_tables=decoder_block_tables,
                )
            self.dec_lengths[batch_idx] = len(token_sequence)

        else:
            cache_position = torch.zeros(request_nums, 1, dtype=torch.int32)
            kwargs = self.preprocess_for_decoder(
                is_prompt=False,
                block_tables=block_tables,
                input_ids=input_ids,
                cache_position=cache_position,
                input_block_ids=valid_block_ids,
                dummy_block=model_input.dummy_block,
            )
            decoder_cache_position = kwargs.pop("cache_position")
            decoder_block_tables = kwargs.pop("block_tables")
            decoder_input_ids = kwargs.pop("input_ids")
            # Generate cache_position using dec_lengths
            for batch_idx in valid_block_ids:
                decoder_cache_position[batch_idx] = self.dec_lengths[batch_idx]
                decoder_attention_mask[
                    batch_idx, : decoder_cache_position[batch_idx] + 1
                ] = 1
                self.dec_lengths[batch_idx] += 1
            decoder_output = self.model.decoder(
                decoder_input_ids=decoder_input_ids.contiguous(),
                decoder_attention_mask=decoder_attention_mask,
                cache_position=decoder_cache_position,
                block_tables=decoder_block_tables,
            )

        lm_logits = decoder_output.logits
        lm_logits = lm_logits[valid_block_ids]
        return lm_logits

    def _parse_and_validate_audio_input(self, **kwargs: object) -> WhisperAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)
        return WhisperAudioInputs(input_features=input_features)
