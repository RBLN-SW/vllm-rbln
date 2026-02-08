# SPDX-License-Identifier: Apache-2.0
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

from transformers import PretrainedConfig

# modified/customized models for RBLN
_RBLN_GENERATION_MODELS: dict[str, tuple[str, str]] = {
    "LlamaForCausalLM": (
        "llama",
        "RBLNLlamaForCausalLM",
    ),
    "GemmaForCausalLM": ("gemma", "RBLNGemmaForCausalLM"),
    "Gemma2ForCausalLM": ("gemma2", "RBLNGemma2ForCausalLM"),
    "PhiForCausalLM": ("phi", "RBLNPhiForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "RBLNGPT2LMHeadModel"),
    "MidmLMHeadModel": ("midm", "RBLNMidmLMHeadModel"),
    "MistralForCausalLM": ("mistral", "RBLNMistralForCausalLM"),
    "ExaoneForCausalLM": ("exaone", "RBLNExaoneForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "RBLNQwen2ForCausalLM"),
    "OPTForCausalLM": ("opt", "RBLNOPTForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "RBLNQwen3ForCausalLM"),
    "GptOssForCausalLM": ("gpt-oss", "RBLNGptOssForCausalLM"),
}

_RBLN_ENCODER_DECODER_MODELS: dict[str, tuple[str, str]] = {
    "WhisperForConditionalGeneration":
    ("whisper", "RBLNWhisperForConditionalGeneration"),
}

_RBLN_MULTIMODAL_MODELS = {
    "LlavaNextForConditionalGeneration":
    ("llava_next", "RBLNLlavaNextForConditionalGeneration"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl",
                                        "RBLNQwen2VLForConditionalGeneration"),
    "Qwen2_5_VLForConditionalGeneration":
    ("qwen2_5_vl", "RBLNQwen2_5_VLForConditionalGeneration"),
    "Idefics3ForConditionalGeneration":
    ("idefics3", "RBLNIdefics3ForConditionalGeneration"),
    "Blip2ForConditionalGeneration": ("blip2",
                                      "RBLNBlip2ForConditionalGeneration"),
    "Gemma3ForConditionalGeneration": ("gemma3",
                                       "RBLNGemma3ForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("llava",
                                      "RBLNLlavaForConditionalGeneration"),
    "PaliGemmaForConditionalGeneration":
    ("paligemma", "RBLNPaliGemmaForConditionalGeneration"),
}

_RBLN_EMBEDDING_MODELS = {
    "T5EncoderModel": ("t5_encoder", "RBLNT5EncoderModel"),
    "BertModel": ("bert_model", "RBLNBertModel"),
    "RobertaForSequenceClassification":
    ("roberta_classification", "RBLNRobertaForSequenceClassification"),
    "RobertaModel": ("roberta", "RBLNRobertaModel"),
    "XLMRobertaForSequenceClassification":
    ("xlm_roberta_classification", "RBLNXLMRobertaForSequenceClassification"),
    "XLMRobertaModel": ("xlm_roberta", "RBLNXLMRobertaModel"),
    "Qwen3Model": ("qwen3", "RBLNQwen3Model"),
}

_RBLN_SUPPORTED_MODELS = {
    **_RBLN_GENERATION_MODELS,
    **_RBLN_ENCODER_DECODER_MODELS,
    **_RBLN_MULTIMODAL_MODELS,
    **_RBLN_EMBEDDING_MODELS,
}


def is_multi_modal(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_MULTIMODAL_MODELS)


def is_pooling_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_EMBEDDING_MODELS)


def is_enc_dec_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_ENCODER_DECODER_MODELS)


def is_arch_supported(config: PretrainedConfig,
                      model_set: dict[str, tuple[str, str]]) -> bool:
    architectures = getattr(config, "architectures", [])
    return any(arch in _RBLN_SUPPORTED_MODELS and arch in model_set
               for arch in architectures)


def get_rbln_model_info(config: PretrainedConfig) -> tuple[str, str]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _RBLN_SUPPORTED_MODELS:
            model_name, model_cls_name = _RBLN_SUPPORTED_MODELS[arch]
            return model_name, model_cls_name

    raise ValueError(
        f"Model architectures {architectures} are not supported on RBLN "
        f"for now. Supported architectures: "
        f"{list(_RBLN_SUPPORTED_MODELS.keys())}")
