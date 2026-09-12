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
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    unpad_and_flat_audio_features,
)
from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)


class RBLNOptimumQwen3ASRForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
    RBLNOptimumDecoderMixin,
    SupportsTranscription,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        # vLLM reads both Qwen3-ASR checkpoint layouts, but optimum-rbln wraps
        # transformers' Qwen3ASRForConditionalGeneration, which only loads the
        # transformers-native one (the `-hf` repos): the original layout nests
        # everything under `thinker_config` and its weights are dropped.
        model_config = vllm_config.model_config
        config_dict = get_hf_file_to_dict(
            "config.json", model_config.model, model_config.revision
        )
        if config_dict is not None and "thinker_config" in config_dict:
            raise ValueError(
                f"Use the `-hf` Qwen3-ASR repo (e.g. `Qwen/Qwen3-ASR-1.7B-hf`) "
                f"instead of {model_config.model!r}. The RBLN optimum path loads "
                "the checkpoint through transformers' "
                "Qwen3ASRForConditionalGeneration, which needs the "
                "transformers-native layout; the original `thinker_config` "
                "layout cannot be loaded."
            )
        super().__init__(vllm_config=vllm_config)
        if vllm_config.lora_config is not None:
            raise NotImplementedError(
                "LoRA is not supported for Qwen3-ASR on the RBLN backend. "
                "Please run the model without LoRA adapters."
            )
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    # optimum-rbln's Qwen3-ASR subclasses RBLNQwen3ForCausalLM, so the decoder
    # runtimes live on the model itself, not under a `language_model` submodule.
    def get_prefill_decoder(self):
        return self.model.prefill_decoder

    def get_language_model(self):
        return self.model

    def _image_token_id(self) -> int:
        return self.model.config.audio_token_id

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Qwen2_5OmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None

        # inputs features from rust frontend is batched and padded
        # with shape [batch_size, n_mels, padded_seq_len], different
        # from python's shape [n_mels, batch_size * seq_len]
        if (
            isinstance(input_audio_features, torch.Tensor)
            and input_audio_features.dim() == 3
        ):
            input_audio_features = unpad_and_flat_audio_features(
                input_audio_features, audio_feature_lengths
            )

        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        return self._process_audio_input(audio_input)

    def _process_audio_input(
        self, audio_input: Qwen2_5OmniAudioFeatureInputs
    ) -> list[torch.Tensor]:
        # vLLM hands over the mel features of all audios concatenated along
        # time, while the compiled audio tower wants them batched, right-padded
        # to whole `chunk_len`-frame chunks, with a frame-validity mask.
        input_features = audio_input["input_features"]
        feature_lengths = audio_input["audio_feature_lengths"].tolist()
        chunk_len = self.model.audio_tower.chunk_len
        padded_len = -(-max(feature_lengths) // chunk_len) * chunk_len

        batched_features = input_features.new_zeros(
            len(feature_lengths), input_features.shape[0], padded_len
        )
        features_mask = torch.zeros(len(feature_lengths), padded_len, dtype=torch.long)
        per_audio_features = input_features.split(feature_lengths, dim=1)
        for i, (features, length) in enumerate(
            zip(per_audio_features, feature_lengths)
        ):
            batched_features[i, :, :length] = features
            features_mask[i, :length] = 1

        audio_embeds = self.model.audio_tower(
            batched_features.to(self.dtype), features_mask
        )

        # FIXME: embed_input_ids masked_scatters these into the text embeddings
        # without a cast, so both towers must agree on dtype. Both come from
        # rbln_config.dtype today; if that ever diverges, cast here (or override
        # embed_input_ids) instead of loosening this assert.
        text_dtype = self.model.get_input_embeddings().weight.dtype
        assert audio_embeds.dtype == text_dtype, (
            f"audio_tower emitted {audio_embeds.dtype} but embed_tokens is "
            f"{text_dtype}; embed_input_ids requires the same dtype."
        )

        output_lengths = _get_feat_extract_output_lengths(
            audio_input["audio_feature_lengths"]
        ).tolist()
        return list(audio_embeds.split(output_lengths))

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        assert len(model_input.running_requests_ids) == request_nums, (
            f"The number of running requests is "
            f"{len(model_input.running_requests_ids)}, "
            f"but the shape of input_ids is {input_ids.shape}"
        )

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            logits = self.model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                block_tables=block_tables,
                cache_position=cache_position,
            ).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.decoder = self.model.decoders[padded_batch_size]
            input_ids = kwargs.pop("input_ids")
            inputs_embeds = self.model.embed_tokens(input_ids)
            logits = self.model.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits
            logits = logits[:request_nums]
        return logits
