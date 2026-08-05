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

"""Optimum smoke tests for the RSD-4 runner: decoder + multimodal.
Run with ``pytest tests/optimum_compile/test_rsd4.py``."""

import pytest
from test_base import DecoderSmoke, MultimodalSmoke, requires_npu

pytestmark = requires_npu

_GET_INPUT_EMBEDS_BUG = pytest.mark.xfail(
    reason="vllm-rbln get_input_embeddings() returns None in the MM embed path",
    strict=False,
)


class TestLlamaEager(DecoderSmoke.Test):
    # Single KV block (block_size == max_model_len) -> eager attention.
    MODEL_ID = "afmck/testing-llama-tiny"
    NUM_DEVICES = 4
    LLM_KWARGS = {"block_size": 2048, "max_model_len": 2048, "max_num_seqs": 2}


class TestLlamaFlash(DecoderSmoke.Test):
    # Multiple KV blocks (block_size < max_model_len) -> paged/flash attention.
    MODEL_ID = "afmck/testing-llama-tiny"
    NUM_DEVICES = 4
    LLM_KWARGS = {"block_size": 1024, "max_model_len": 2048, "max_num_seqs": 4}


class TestLlavaNext(MultimodalSmoke.Test):
    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    HF_OVERRIDES = {"text_config.num_hidden_layers": 1}
    NUM_DEVICES = 1
    LLM_KWARGS = {"block_size": 4096, "max_model_len": 4096, "max_num_seqs": 1}


class TestIdefics3(MultimodalSmoke.Test):
    MODEL_ID = "HuggingFaceM4/Idefics3-8B-Llama3"
    HF_OVERRIDES = {
        "text_config.num_hidden_layers": 1,
        "text_config.max_position_embeddings": 4096,
    }
    NUM_DEVICES = 1
    LLM_KWARGS = {"block_size": 4096, "max_model_len": 4096, "max_num_seqs": 1}


class TestBlip2(MultimodalSmoke.Test):
    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    HF_OVERRIDES = {
        "text_config.num_hidden_layers": 1,
        "qformer_config.num_hidden_layers": 1,
    }
    NUM_DEVICES = 1
    LLM_KWARGS = {"block_size": 2048, "max_model_len": 2048, "max_num_seqs": 1}
    USE_CHAT_TEMPLATE = False
    PROMPT = "Question: What is shown in this image? Answer:"


class TestPaligemma(MultimodalSmoke.Test):
    MODEL_ID = "google/paligemma2-3b-pt-224"
    HF_OVERRIDES = {"text_config.num_hidden_layers": 1}
    NUM_DEVICES = 1
    LLM_KWARGS = {"block_size": 8192, "max_model_len": 8192, "max_num_seqs": 1}
    USE_CHAT_TEMPLATE = False
    PROMPT = "caption en"
