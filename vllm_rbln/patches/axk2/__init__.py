# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.registry import add_registration

ARCH = "AXK2ForCausalLM"
MODEL_TYPE = "axk2"
MODEL_CLASS_PATH = "vllm_rbln.patches.axk2.model:AXK2ForCausalLM"


def _upstream_has_axk2() -> bool:
    """True once the installed vLLM ships axk2 itself."""
    from vllm.model_executor.models import ModelRegistry

    return ARCH in ModelRegistry.get_supported_archs()


def _patch_condition() -> bool:
    return not _upstream_has_axk2()


@add_registration(
    reason=(
        "A.X K2 is vendored here until upstream vLLM ships it. Registering "
        "through ModelRegistry keeps the architecture resolvable the way "
        "upstream resolves every other one."
    )
)
def register_axk2() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_rbln.patches.axk2.config import AXK2Config

    # Once upstream registers the architecture our copy would shadow it, so fail
    # loudly to get this vendored path removed instead of silently overriding it.
    if _upstream_has_axk2():
        raise RuntimeError(
            "upstream vLLM now ships the axk2 architecture; delete "
            "vllm_rbln/patches/axk2/ and this registration."
        )

    _CONFIG_REGISTRY[MODEL_TYPE] = AXK2Config
    ModelRegistry.register_model(ARCH, MODEL_CLASS_PATH)


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
        model_type = getattr(hf_text_config, "model_type", None)
        if model_type == MODEL_TYPE:
            return getattr(hf_text_config, "kv_lora_rank", None) is not None
        if model_type == "eagle":
            inner = getattr(hf_text_config, "model", None)
            if getattr(inner, "model_type", None) == MODEL_TYPE:
                return getattr(hf_text_config, "kv_lora_rank", None) is not None
        return original(self)


_register_is_deepseek_mla_patch()

__all__ = ["ARCH", "MODEL_CLASS_PATH", "MODEL_TYPE"]
