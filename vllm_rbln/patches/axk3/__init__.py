# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.model_arch_config import register_mla_convertor
from vllm_rbln.patches.registry import add_registration

ARCH = "AXK3ForCausalLM"
MODEL_TYPE = "axk3"
MODEL_CLASS_PATH = "vllm_rbln.patches.axk3.model:AXK3ForCausalLM"


@add_registration(
    reason=(
        "A.X K3 is built on the vendored A.X K2 model until upstream vLLM ships "
        "it. Registering through ModelRegistry keeps the architecture "
        "resolvable the way upstream resolves every other one."
    )
)
def register_axk3() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_rbln.patches.axk3.config import AXK3Config

    registered = ModelRegistry.models.get(ARCH)
    if registered is not None and (
        getattr(registered, "module_name", None),
        getattr(registered, "class_name", None),
    ) != tuple(MODEL_CLASS_PATH.split(":")):
        raise RuntimeError(
            "upstream vLLM now ships the axk3 architecture; delete "
            "vllm_rbln/patches/axk3/ and this registration."
        )

    _CONFIG_REGISTRY[MODEL_TYPE] = AXK3Config
    ModelRegistry.register_model(ARCH, MODEL_CLASS_PATH)
    register_mla_convertor(MODEL_TYPE)


__all__ = ["ARCH", "MODEL_CLASS_PATH", "MODEL_TYPE"]
