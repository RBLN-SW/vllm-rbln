# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

ARCH = "AXK2ForCausalLM"
MODEL_TYPE = "axk2"
MODEL_CLASS_PATH = "vllm_rbln.patches.axk2.model:AXK2ForCausalLM"


def _upstream_has_axk2() -> bool:
    """True once the installed vLLM ships axk2 itself."""
    from vllm.model_executor.models import ModelRegistry

    return ARCH in ModelRegistry.get_supported_archs()


def _patch_condition() -> bool:
    return not _upstream_has_axk2()


def _register_is_deepseek_mla_patch() -> None:
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    from vllm_rbln.patches import register_patch

    original = ModelArchConfigConvertorBase.is_deepseek_mla

    @register_patch(
        target=(
            "vllm.transformers_utils.model_arch_config_convertor"
            ".ModelArchConfigConvertorBase.is_deepseek_mla"
        ),
        reason=(
            "axk2 is an MLA model but upstream decides MLA-ness from a hardcoded "
            "model_type tuple that does not list it"
        ),
        condition=_patch_condition,
    )
    def is_deepseek_mla(self) -> bool:
        hf_text_config = self.hf_text_config
        if getattr(hf_text_config, "model_type", None) == MODEL_TYPE:
            return getattr(hf_text_config, "kv_lora_rank", None) is not None
        return original(self)


_register_is_deepseek_mla_patch()

__all__ = ["ARCH", "MODEL_CLASS_PATH", "MODEL_TYPE"]
