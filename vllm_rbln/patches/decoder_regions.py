# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from vllm_rbln import envs
from vllm_rbln.patches import register_patch

if envs.VLLM_RBLN_USE_NESTED_COMPILE_REGION:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    from vllm.model_executor.models.gpt_oss import TransformerBlock
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    from vllm.model_executor.models.minimax_m2 import MiniMaxM2DecoderLayer
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    for layer in (
        DeepseekV2DecoderLayer,
        TransformerBlock,
        LlamaDecoderLayer,
        MiniMaxM2DecoderLayer,
        Qwen2DecoderLayer,
        Qwen3DecoderLayer,
        Qwen3MoeDecoderLayer,
    ):
        register_patch(
            target=f"{layer.__module__}.{layer.__qualname__}.forward",
            # Every nested_compile_region wrapper has the same __qualname__, so
            # the default key would be the same for every layer.
            key=f"{__name__}.{layer.__qualname__}.forward",
            owner_module=__name__,
            reason=(
                "Upstream decoder layers expose no compile-region hook. Preserve "
                "decoder bodies for RBLN loop-compilation bring-up using "
                "invoke_subgraph boundaries."
            ),
        )(torch.compiler.nested_compile_region(layer.forward))
