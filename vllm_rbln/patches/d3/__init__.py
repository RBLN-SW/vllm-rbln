# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.registry import add_registration

ARCH = "D3ForSequenceClassification"
MODEL_CLASS_PATH = "vllm_rbln.patches.d3.model:D3ForSequenceClassification"


@add_registration(
    reason=(
        "Decision 3.0 is vendored here until the vLLM vllm-rbln pins ships it "
        "(upstream #61565, first in v0.32.1rc0). Registering through "
        "ModelRegistry and the model config map keeps it resolvable the way "
        "upstream resolves its own architectures."
    )
)
def register_d3() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_rbln.patches.d3.model import D3ForSequenceClassificationConfig

    # Once upstream registers the architecture our copy would shadow it, so fail
    # loudly to get this vendored path removed instead of silently overriding it.
    if ARCH in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            "upstream vLLM now ships Decision 3.0; delete vllm_rbln/patches/d3/ "
            "and this registration."
        )

    MODELS_CONFIG_MAP[ARCH] = D3ForSequenceClassificationConfig
    ModelRegistry.register_model(ARCH, MODEL_CLASS_PATH)
