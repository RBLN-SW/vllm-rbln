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
"""Utilities for selecting and loading rbln models."""

import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN
from vllm_rbln.utils.optimum.registry import (_RBLN_MULTIMODAL_MODELS,
                                              is_enc_dec_arch, is_multi_modal,
                                              is_pooling_arch)

from .blip2 import RBLNOptimumBlip2ForConditionalGeneration  # noqa: F401
from .decoder_only import RBLNOptimumForCausalLM
from .encoder import RBLNOptimumForEncoderModel
from .gemma3 import RBLNOptimumGemma3ForConditionalGeneration  # noqa: F401
from .idefics3 import RBLNOptimumIdefics3ForConditionalGeneration  # noqa: F401
from .llava import RBLNOptimumLlavaForConditionalGeneration  # noqa: F401
from .llava_next import (  # noqa: F401
    RBLNOptimumLlavaNextForConditionalGeneration)
from .paligemma import (  # noqa: F401
    RBLNOptimumPaliGemmaForConditionalGeneration)
from .qwen_vl import (  # noqa: F401
    RBLNOptimumQwen2_5_VLForConditionalGeneration,
    RBLNOptimumQwen2VLForConditionalGeneration)
from .sliding_window import (  # noqa: F401
    RBLNOptimumSlidingWindowAttentionForCausalLM)
from .whisper import RBLNOptimumWhisperForConditionalGeneration  # noqa: F401

logger = init_logger(__name__)

_RBLN_OPTIMUM_MULTIMODAL_MODELS = {
    model_name: globals()[f"RBLNOptimum{model_name}"]
    for model_name in _RBLN_MULTIMODAL_MODELS
}


def load_model(vllm_config: VllmConfig) -> nn.Module:
    model_config = vllm_config.model_config

    if is_multi_modal(model_config.hf_config):
        assert vllm_config.cache_config.enable_prefix_caching in (
            False,
            None), ("Prefix caching is not supported with multimodal models. "
                    "Please set `enable_prefix_caching` to False.")
        architectures = getattr(model_config.hf_config, "architectures", [])
        if architectures[0] in _RBLN_OPTIMUM_MULTIMODAL_MODELS:
            rbln_model_arch = _RBLN_OPTIMUM_MULTIMODAL_MODELS[architectures[0]]
            rbln_model = rbln_model_arch(vllm_config)
        else:
            raise NotImplementedError(
                f"Model architectures {architectures} are "
                f"not supported on RBLN Optimum for now. "
                "Supported multimodal architectures: "
                f"{list(_RBLN_OPTIMUM_MULTIMODAL_MODELS.keys())}")
    elif is_enc_dec_arch(model_config.hf_config):
        assert vllm_config.cache_config.enable_prefix_caching in (
            False, None), (
                "Prefix caching is not supported with encoder-decoder models. "
                "Please set `enable_prefix_caching` to False.")
        architectures = getattr(model_config.hf_config, "architectures", [])
        if architectures[0] in ["WhisperForConditionalGeneration"]:
            rbln_model = RBLNOptimumWhisperForConditionalGeneration(
                vllm_config)
        else:
            raise NotImplementedError(
                "Encoder-decoder models are not supported"
                "since vLLM RBLN v0.10.1")
    elif is_pooling_arch(model_config.hf_config):
        assert vllm_config.cache_config.enable_prefix_caching in (
            False, None
        ), ("Prefix caching is not supported with pooling models. Please set "
            "`enable_prefix_caching` to False.")
        rbln_model = RBLNOptimumForEncoderModel(vllm_config)
    else:
        if getattr(model_config.hf_config,
                   "sliding_window", None) is not None and getattr(
                       model_config.hf_config, "use_sliding_window", True):
            logger.info(
                "The model is initialized with Sliding Window Attention.")
            assert vllm_config.cache_config.enable_prefix_caching in (
                False, None), (
                    "Prefix caching is not supported with sliding window "
                    "attention. Please set `enable_prefix_caching` to False.")
            rbln_model = RBLNOptimumSlidingWindowAttentionForCausalLM(
                vllm_config)
        else:
            rbln_model = RBLNOptimumForCausalLM(vllm_config)
    return rbln_model.eval()


__all__ = ["load_model", "ModelInputForRBLN", "RBLNOptimumForEncoderModel"]
