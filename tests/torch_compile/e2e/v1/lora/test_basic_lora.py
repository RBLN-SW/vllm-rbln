# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SQL_LORA_MODEL_ID = "jeeejeee/llama32-3b-text2sql-spider"
PROMPT = (
    "[user] Write a SQL query to answer the question based on the table schema.\n\n "
    "context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n "
    "question: Name the ICAO for lilongwe international airport [/user] [assistant]"
)


@pytest.fixture(scope="module")
def llm(monkeypatch_module):
    monkeypatch_module.setenv("VLLM_RBLN_ENFORCE_MODEL_FP32", "1")
    monkeypatch_module.setenv("VLLM_RBLN_ENABLE_WARM_UP", "0")
    monkeypatch_module.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch_module.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    return LLM(
        model=MODEL_ID,
        max_model_len=8192,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=1,
    )


@pytest.fixture(scope="module")
def sql_lora_path() -> str:
    return snapshot_download(repo_id=SQL_LORA_MODEL_ID)


def test_basic_lora_sql_prompt(llm: LLM, sql_lora_path: str) -> None:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    lora_output = llm.generate(
        PROMPT,
        sampling_params=sampling_params,
        lora_request=LoRARequest("sql-lora", 1, sql_lora_path),
    )[0]
    base_output = llm.generate(PROMPT, sampling_params=sampling_params)[0]

    lora_text = lora_output.outputs[0].text.strip()
    base_text = base_output.outputs[0].text.strip()

    assert len(lora_text) > 0
    assert len(base_text) > 0
    assert base_text != lora_text
