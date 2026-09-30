# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.registry import add_registration

_MODULE = "vllm_rbln.model_executor.models.deepseek_v4"


@add_registration(
    reason=(
        "Upstream's DeepSeek-V4 binds FlashMLA / FlashInfer / DeepGEMM / tilelang kernels at "
        "import, so the RBLN copy (V4 attention custom ops, RBLN MoE runner) replaces it."
    )
)
def register_deepseek_v4() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", f"{_MODULE}:RBLNDeepseekV4ForCausalLM"
    )


@add_registration(
    reason=(
        "Override the built-in deepseek_v4_fp8 config so the block-FP8 linears keep their "
        "fp8 weights with a folded bf16 scale (W8A16) and the MXFP4 experts run through the "
        "RBLN group-dequantise MoE op instead of the CUDA backends."
    )
)
def register_rbln_deepseek_v4_fp8_config() -> None:
    from vllm.model_executor.layers.quantization import register_quantization_config

    from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import (
        RBLNDeepseekV4FP8Config,
    )

    register_quantization_config("deepseek_v4_fp8")(RBLNDeepseekV4FP8Config)
