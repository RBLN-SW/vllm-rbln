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

from transformers import PretrainedConfig

import optimum.rbln
from optimum.rbln import (
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForSpeechSeq2Seq,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)

from .multimodal import (
    _COMPILE_MULTIMODAL_FNS,
    get_multimodal_cls,
)


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge ``overrides`` into ``base`` in place.

    Nested dicts are merged key-by-key; non-dict values overwrite. This
    preserves untouched sub-keys when the user only overrides part of a
    nested config (e.g. ``language_model.max_seq_len`` for multimodal).
    """
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value


@dataclass
class RBLNCompileSpec:
    """Resolved (model_cls, rbln_config) ready to feed optimum-rbln."""

    model_cls: Any
    rbln_config: dict[str, Any]

    @classmethod
    def for_architecture(
        cls,
        config: PretrainedConfig,
        *,
        batch_size: int,
        block_size: int,
        max_model_len: int,
        tp_size: int,
        rbln_overrides: dict[str, Any] | None = None,
    ) -> "RBLNCompileSpec":
        """Build a compile spec from vllm-rbln inputs, dispatched by architecture."""
        if is_generation_arch(config):
            spec = cls._for_decoder(batch_size, block_size, max_model_len, tp_size)
        elif is_pooling_arch(config):
            spec = cls._for_pooling(
                config, batch_size, block_size, max_model_len, tp_size
            )
        elif is_multi_modal(config):
            spec = cls._for_multimodal(
                config, batch_size, block_size, max_model_len, tp_size
            )
        elif is_enc_dec_arch(config):
            spec = cls._for_enc_dec(
                config, batch_size, block_size, max_model_len, tp_size
            )
        else:
            architectures = getattr(config, "architectures", [])
            raise NotImplementedError(
                f"Compilation is not implemented for architecture {architectures[0]}"
            )

        # FIXME: detect conflicts between spec.rbln_config and rbln_overrides
        # so we don't silently overwrite compile-critical fields.
        if rbln_overrides:
            _deep_merge(spec.rbln_config, rbln_overrides)
        return spec

    @classmethod
    def _for_decoder(
        cls,
        batch_size: int,
        block_size: int,
        max_model_len: int,
        tp_size: int,
    ) -> "RBLNCompileSpec":
        rbln_config: dict[str, Any] = {
            "tensor_parallel_size": tp_size,
            "batch_size": batch_size,
            "max_seq_len": max_model_len,
        }
        if block_size != max_model_len:
            rbln_config["kvcache_partition_len"] = block_size
            rbln_config["attn_impl"] = "flash_attn"
        return cls(model_cls=RBLNAutoModelForCausalLM, rbln_config=rbln_config)

    @classmethod
    def _for_pooling(
        cls,
        config: PretrainedConfig,
        batch_size: int,
        block_size: int,
        max_model_len: int,
        tp_size: int,
    ) -> "RBLNCompileSpec":
        _, model_cls_name = get_rbln_model_info(config)
        model_cls = getattr(optimum.rbln, model_cls_name)
        assert model_cls is not None

        rbln_config: dict[str, Any] = {
            "tensor_parallel_size": tp_size,
            "batch_size": batch_size,
            "max_seq_len": max_model_len,
        }
        # FIXME: We need a more generalized logic to specify block sizes
        # as the number of supported models continues to grow.
        architectures = getattr(config, "architectures", [])
        if architectures[0] == "Qwen3Model" and block_size != max_model_len:
            rbln_config["kvcache_partition_len"] = block_size
            rbln_config["attn_impl"] = "flash_attn"
        return cls(model_cls=model_cls, rbln_config=rbln_config)

    @classmethod
    def _for_multimodal(
        cls,
        config: PretrainedConfig,
        batch_size: int,
        block_size: int,
        max_model_len: int,
        tp_size: int,
    ) -> "RBLNCompileSpec":
        model_name, _ = get_rbln_model_info(config)
        compile_fn = _COMPILE_MULTIMODAL_FNS.get(model_name)
        if compile_fn is None:
            raise ValueError(
                f"Unknown multimodal model alias: {model_name}. "
                f"Supported aliases: {sorted(_COMPILE_MULTIMODAL_FNS.keys())}"
            )
        architectures = getattr(config, "architectures", [])
        return cls(
            model_cls=get_multimodal_cls(architectures[0]),
            rbln_config=compile_fn(batch_size, max_model_len, block_size, tp_size),
        )

    @classmethod
    def _for_enc_dec(
        cls,
        config: PretrainedConfig,
        batch_size: int,
        block_size: int,
        max_model_len: int,
        tp_size: int,
    ) -> "RBLNCompileSpec":
        architectures = getattr(config, "architectures", [])
        assert architectures[0] == "WhisperForConditionalGeneration"
        # Whisper does not support varying block_size or max_model_len.
        assert block_size == max_model_len, (
            "block_size must be equal to max_model_len for Whisper models."
        )
        assert max_model_len == config.max_length, (
            f"max_model_len ({max_model_len}) must match the Whisper model's "
            f"max_length ({config.max_length}) from the HuggingFace config."
        )
        return cls(
            model_cls=RBLNAutoModelForSpeechSeq2Seq,
            rbln_config={
                "tensor_parallel_size": tp_size,
                "batch_size": batch_size,
                "token_timestamps": False,
            },
        )
