# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    PromptType,
    TextPrompt,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.whisper import (
    ISO639_1_SUPPORTED_LANGS,
    WhisperAudioInputs,
)
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.jsontree import json_map_leaves

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumWhisperForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumDecoderMixin,
    SupportsTranscription,
    SupportsMultiModal,
):
    # Whisper only supports audio-conditioned generation.
    supports_transcription_only = True
    supports_segment_timestamp = True
    supports_explicit_language_detection = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.debug(
                "No language specified. Language will be auto-detected "
                "from audio. To skip detection, pass the `language` field "
                "in the TranscriptionRequest."
            )
            return None
        return super().validate_language(language)

    @classmethod
    def get_generation_prompt(
        cls,
        stt_params: SpeechToTextParams,
    ) -> PromptType:
        audio = stt_params.audio
        stt_config = stt_params.stt_config
        language = stt_params.language
        task_type = stt_params.task_type
        request_prompt = stt_params.request_prompt

        if language is None:
            raise ValueError(
                "Language must be specified when creating the Whisper prompt"
            )

        decoder_text = (
            f"<|prev|>{request_prompt}" if request_prompt else ""
        ) + f"<|startoftranscript|><|{language}|><|{task_type}|><|notimestamps|>"

        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",  # Whisper does not support encoder prompt.
                multi_modal_data={"audio": (audio, stt_config.sample_rate)},
            ),
            decoder_prompt=TextPrompt(prompt=decoder_text),
        )

    @classmethod
    def get_language_token_ids(
        cls,
        tokenizer: object,
    ) -> list[int]:
        """Return token IDs for all supported language tokens.

        Used with ``SamplingParams.allowed_token_ids`` to constrain
        language detection to only produce valid language tokens.
        """
        token_ids = [
            tokenizer.convert_tokens_to_ids(f"<|{lang_code}|>")  # type: ignore[attr-defined]
            for lang_code in cls.supported_languages
        ]
        return token_ids

    @classmethod
    def get_language_detection_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
    ) -> PromptType:
        """Return a prompt that elicits a single language token from Whisper.

        Feed only ``<|startoftranscript|>`` as the decoder input so the model
        predicts the most likely language token (e.g. ``<|de|>``).
        """
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={"audio": (audio, stt_config.sample_rate)},
            ),
            decoder_prompt=TextPrompt(prompt="<|startoftranscript|>"),
        )

    @classmethod
    def parse_language_detection_output(
        cls,
        token_ids: list[int],
        tokenizer: object,
    ) -> str | None:
        """Parse the language token predicted by Whisper.

        Decodes the first token ID and extracts the language code from the
        ``<|xx|>`` format. Expects a valid language token from constrained generation.
        """

        decoded = tokenizer.decode(  # type: ignore[attr-defined]
            [token_ids[0]],
            skip_special_tokens=False,
        )
        # Whisper language tokens have the form <|xx|>
        assert decoded.startswith("<|") and decoded.endswith("|>")
        lang_code = decoded[2:-2]
        assert lang_code in cls.supported_languages
        return lang_code

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None

        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)

        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        processor = cached_processor_from_config(model_config)
        hop_length = processor.feature_extractor.hop_length
        assert hop_length is not None
        # NOTE(NickLucche) user can't pass encoder
        # prompts directly at least not to Whisper.
        # One indicator of the encoder amount of processing
        # is the log-mel spectogram length.
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

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
