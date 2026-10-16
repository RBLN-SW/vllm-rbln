# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.registry import add_registration

_MODULE = "vllm_rbln.model_executor.models.minimax_m3"


@add_registration(
    reason=(
        "Upstream's MiniMax-M3 model binds FlashInfer / CUDA / Triton kernels at "
        "import, so the RBLN copy (MSA sparse-attention custom ops, RBLN MoE "
        "runner) replaces both of its architectures. Its standalone text "
        "config also needs AutoConfig registration to load text-only checkpoints."
    )
)
def register_minimax_m3() -> None:
    from transformers import AutoConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    AutoConfig.register(
        MiniMaxM3TextConfig.model_type, MiniMaxM3TextConfig, exist_ok=True
    )

    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM", f"{_MODULE}:RBLNMiniMaxM3SparseForCausalLM"
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        f"{_MODULE}:RBLNMiniMaxM3SparseForConditionalGeneration",
    )
