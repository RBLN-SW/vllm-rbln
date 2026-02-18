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

from pathlib import Path
from typing import Any

from optimum.rbln import (
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForSpeechSeq2Seq,
)
import optimum.rbln
from transformers import PretrainedConfig

from .multimodal import compile_multimodal

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
    "WhisperForConditionalGeneration": (
        "whisper",
        "RBLNWhisperForConditionalGeneration",
    ),
}

_RBLN_MULTIMODAL_MODELS = {
    "LlavaNextForConditionalGeneration": (
        "llava_next",
        "RBLNLlavaNextForConditionalGeneration",
    ),
    "Qwen2VLForConditionalGeneration": (
        "qwen2_vl",
        "RBLNQwen2VLForConditionalGeneration",
    ),
    "Qwen2_5_VLForConditionalGeneration": (
        "qwen2_5_vl",
        "RBLNQwen2_5_VLForConditionalGeneration",
    ),
    "Idefics3ForConditionalGeneration": (
        "idefics3",
        "RBLNIdefics3ForConditionalGeneration",
    ),
    "Blip2ForConditionalGeneration": ("blip2", "RBLNBlip2ForConditionalGeneration"),
    "Gemma3ForConditionalGeneration": ("gemma3", "RBLNGemma3ForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("llava", "RBLNLlavaForConditionalGeneration"),
    "PaliGemmaForConditionalGeneration": (
        "paligemma",
        "RBLNPaliGemmaForConditionalGeneration",
    ),
}

_RBLN_EMBEDDING_MODELS = {
    "T5EncoderModel": ("t5_encoder", "RBLNT5EncoderModel"),
    "BertModel": ("bert_model", "RBLNBertModel"),
    "RobertaForSequenceClassification": (
        "roberta_classification",
        "RBLNRobertaForSequenceClassification",
    ),
    "RobertaModel": ("roberta", "RBLNRobertaModel"),
    "XLMRobertaForSequenceClassification": (
        "xlm_roberta_classification",
        "RBLNXLMRobertaForSequenceClassification",
    ),
    "XLMRobertaModel": ("xlm_roberta", "RBLNXLMRobertaModel"),
    "Qwen3Model": ("qwen3", "RBLNQwen3Model"),
}

_RBLN_SUPPORTED_MODELS = {
    **_RBLN_GENERATION_MODELS,
    **_RBLN_ENCODER_DECODER_MODELS,
    **_RBLN_MULTIMODAL_MODELS,
    **_RBLN_EMBEDDING_MODELS,
}


def is_generation_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_GENERATION_MODELS)


def is_multi_modal(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_MULTIMODAL_MODELS)


def is_pooling_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_EMBEDDING_MODELS)


def is_enc_dec_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_ENCODER_DECODER_MODELS)


def is_arch_supported(
    config: PretrainedConfig, model_set: dict[str, tuple[str, str]]
) -> bool:
    architectures = getattr(config, "architectures", [])
    return any(
        arch in _RBLN_SUPPORTED_MODELS and arch in model_set for arch in architectures
    )


def get_rbln_model_info(config: PretrainedConfig) -> tuple[str, str]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _RBLN_SUPPORTED_MODELS:
            model_name, model_cls_name = _RBLN_SUPPORTED_MODELS[arch]
            return model_name, model_cls_name

    raise ValueError(
        f"Model architectures {architectures} are not supported on RBLN "
        f"for now. Supported architectures: "
        f"{list(_RBLN_SUPPORTED_MODELS.keys())}"
    )


def compile_model(
    hf_model_name: str,
    config: PretrainedConfig,
    batch_size: int,
    block_size: int,
    max_model_len: int,
    tp_size: int,
    model_path: Path,
) -> Any:
    architectures = getattr(config, "architectures", [])
    model_name, model_cls_name = get_rbln_model_info(
        config
    )  # check if the model is supported and get model info
    default_param: dict[str, Any] = {
        "export": True,
        "rbln_batch_size": batch_size,
        "rbln_tensor_parallel_size": tp_size,
    }
    if is_generation_arch(config):
        attn_impl = "flash_attn" if block_size != max_model_len else "eager"
        default_param["rbln_max_seq_len"] = max_model_len
        default_param["rbln_kvcache_partition_len"] = block_size
        default_param["rbln_attn_impl"] = attn_impl
        model = RBLNAutoModelForCausalLM.from_pretrained(
            hf_model_name,
            **default_param,
        )
    elif is_pooling_arch(config):
        model_cls_name = _RBLN_SUPPORTED_MODELS[architectures[0]][1]
        model_cls = getattr(optimum.rbln, model_cls_name)
        assert model_cls is not None
        default_param["rbln_max_seq_len"] = max_model_len
        if architectures[0] == "Qwen3Model":
            attn_impl = "flash_attn" if block_size != max_model_len else "eager"
            default_param["rbln_kvcache_partition_len"] = block_size
            default_param["rbln_attn_impl"] = attn_impl
        model = model_cls.from_pretrained(hf_model_name, **default_param)
    elif is_multi_modal(config):
        model = compile_multimodal(
            model_name=hf_model_name,
            architecture=architectures[0],
            model_alias=model_name,
            batch_size=batch_size,
            max_model_len=max_model_len,
            block_size=block_size,
            tp_size=tp_size,
        )
    elif is_enc_dec_arch(config):
        assert architectures[0] == "WhisperForConditionalGeneration"
        model = RBLNAutoModelForSpeechSeq2Seq.from_pretrained(
            hf_model_name,
            rbln_token_timestamps=False,
            **default_param,
        )
    else:
        raise NotImplementedError(
            "Compilation is not implemented for architecture *s",
            architectures[0],
        )
    model.save_pretrained(model_path)
    return model
