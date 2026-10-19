# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Config-time answers for the vendored MLA models.

vLLM decides MLA-ness in ``ModelArchConfigConvertorBase.is_deepseek_mla`` from
a hardcoded ``model_type`` tuple. The hook it offers for a model it does not
list is ``MODEL_ARCH_CONFIG_CONVERTORS``: one convertor class per model type,
looked up by ``ModelConfig.get_model_arch_config``. A vendored model registers
its convertor from its own registration, so no shared code has to name it.
"""

from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from vllm_rbln.patches.registry import add_registration


class KvLoraRankMLAConvertor(ModelArchConfigConvertorBase):
    """MLA iff the text config carries ``kv_lora_rank``, which is how upstream
    answers for the DeepSeek family it does list."""

    def is_deepseek_mla(self) -> bool:
        return getattr(self.hf_text_config, "kv_lora_rank", None) is not None


class EagleDraftConvertor(ModelArchConfigConvertorBase):
    """An EAGLE draft config carries its target under ``model``.

    Upstream checks that target against its own tuple, so a draft on a vendored
    MLA target would build without MLA. Ask the target's registered convertor
    instead; a target without one keeps upstream's answer."""

    def is_deepseek_mla(self) -> bool:
        target = getattr(self.hf_text_config, "model", None)
        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
            getattr(target, "model_type", None)
        )
        if convertor_cls is None:
            return super().is_deepseek_mla()
        return convertor_cls(target, target, revision=self.revision).is_deepseek_mla()


def _register_convertor(
    model_type: str, convertor_cls: type[ModelArchConfigConvertorBase]
) -> None:
    existing = MODEL_ARCH_CONFIG_CONVERTORS.get(model_type)
    if existing is not None and existing is not convertor_cls:
        # Upstream now has an opinion about this model type; shadowing it would
        # hide that, so fail loudly to get the vendored registration removed.
        raise RuntimeError(
            f"upstream vLLM now ships a model arch config convertor for "
            f"{model_type!r} ({existing.__qualname__}); drop the vendored one."
        )
    MODEL_ARCH_CONFIG_CONVERTORS[model_type] = convertor_cls


def register_mla_convertor(model_type: str) -> None:
    """Make ``model_type`` an MLA model by its ``kv_lora_rank``. Called from the
    model's own registration, next to its ``ModelRegistry.register_model``."""
    _register_convertor(model_type, KvLoraRankMLAConvertor)


@add_registration(
    reason=(
        "An EAGLE draft config names its target under `model`, and upstream "
        "decides the draft's MLA-ness from a hardcoded tuple of target types "
        "that omits the vendored ones; route the draft to its target's "
        "registered convertor."
    )
)
def register_eagle_draft_convertor() -> None:
    _register_convertor("eagle", EagleDraftConvertor)


__all__ = [
    "EagleDraftConvertor",
    "KvLoraRankMLAConvertor",
    "register_mla_convertor",
]
