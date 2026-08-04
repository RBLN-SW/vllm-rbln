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

"""Optimum smoke tests for the RSD-1 runner: small pooling/encoder models
(kept full-size with a real ranking assertion) plus the hybrid-attention Gemma2
decoder. Run with ``pytest tests/optimum_compile/test_rsd1.py``."""

from test_base import DecoderSmoke, PoolingSmoke, requires_npu

pytestmark = requires_npu


class TestQwen3Embedding(PoolingSmoke.Test):
    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
    NUM_DEVICES = 1
    LLM_KWARGS = {
        "runner": "pooling",
        "max_model_len": 1024,
        "block_size": 1024,
        "max_num_seqs": 1,
    }


class TestBgeM3(PoolingSmoke.Test):
    MODEL_ID = "BAAI/bge-m3"
    NUM_DEVICES = 1
    LLM_KWARGS = {
        "runner": "pooling",
        "block_size": 1024,
        "max_model_len": 1024,
        "max_num_seqs": 4,
    }


class TestQwen3Reranker(DecoderSmoke.Test):
    # Reranker is a causal LM; a plain generate is enough for a smoke check.
    MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
    NUM_DEVICES = 1
    LLM_KWARGS = {"block_size": 1024, "max_model_len": 1024, "max_num_seqs": 2}


class TestGemma2(DecoderSmoke.Test):
    # Hybrid (sliding-window) attention decoder.
    MODEL_ID = "hf-internal-testing/tiny-random-Gemma2ForCausalLM"
    NUM_DEVICES = 1
    LLM_KWARGS = {"block_size": 1024, "max_model_len": 1024, "max_num_seqs": 4}
