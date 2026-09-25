# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
from vllm.config import ModelConfig

from vllm_rbln.platform.optimum_impl import disable_unsupported_prefix_caching

# Hub checkpoints whose quantization_config carries a kv_cache_scheme: a
# "static" string in the fp8 (Quark) format, a dict in compressed-tensors.
QUANTIZED_KV_CACHE = [
    "amd/Llama-3.3-70B-Instruct-FP8-KV",
    "RedHatAI/Meta-Llama-3-8B-Instruct-FP8-KV",
    "novita/DeepSeek-R1-Distill-Llama-70B-w8a8kv8-s888",
]

# W8A8 checkpoints that quantize only the linear layers. Their
# kv_cache_scheme is absent or null.
QUANTIZED_LINEARS_ONLY = [
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8",
]

HYBRID = ["Qwen/Qwen3.5-0.8B"]


def _vllm_config(model_id: str) -> SimpleNamespace:
    """The fields the guard reads, around the ModelConfig vLLM would build."""
    return SimpleNamespace(
        model_config=ModelConfig(model_id, trust_remote_code=False),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )


@pytest.mark.parametrize("model_id", QUANTIZED_KV_CACHE)
def test_quantized_kv_cache_disables_prefix_caching(model_id):
    vllm_config = _vllm_config(model_id)
    disable_unsupported_prefix_caching(vllm_config)
    assert vllm_config.cache_config.enable_prefix_caching is False


@pytest.mark.parametrize("model_id", QUANTIZED_LINEARS_ONLY)
def test_quantized_linears_alone_keep_prefix_caching(model_id):
    vllm_config = _vllm_config(model_id)
    disable_unsupported_prefix_caching(vllm_config)
    assert vllm_config.cache_config.enable_prefix_caching is True


@pytest.mark.parametrize("model_id", HYBRID)
def test_hybrid_disables_prefix_caching(model_id):
    vllm_config = _vllm_config(model_id)
    disable_unsupported_prefix_caching(vllm_config)
    assert vllm_config.cache_config.enable_prefix_caching is False
