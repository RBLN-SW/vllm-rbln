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

"""End-to-end smoke tests for Medusa speculative decoding."""

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams

from .utils import (
    DEFAULT_MEDUSA_MODEL_ID,
    DEFAULT_MODEL_ID,
    ensure_converted_medusa_adapter,
)

PROMPTS = [
    "A robot may not injure a human being",
    "The capital of France is",
]


@pytest.fixture(scope="module")
def llm_env(monkeypatch_module):
    monkeypatch_module.setenv("RBLN_USE_CUSTOM_KERNEL", "1")
    monkeypatch_module.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch_module.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch_module.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch_module.setenv("VLLM_RBLN_ENABLE_WARM_UP", "1")
    monkeypatch_module.setenv("VLLM_RBLN_SAMPLER", "0")


def _build_llm() -> LLM:
    medusa_model_id, num_speculative_tokens = ensure_converted_medusa_adapter(
        medusa_model_id=DEFAULT_MEDUSA_MODEL_ID,
        base_model_id=DEFAULT_MODEL_ID,
    )
    return LLM(
        model=DEFAULT_MODEL_ID,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        speculative_config={
            "method": "medusa",
            "model": medusa_model_id,
            "num_speculative_tokens": num_speculative_tokens,
        },
        disable_log_stats=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )


def test_basic_medusa_generation(llm_env) -> None:
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=32)

    llm = _build_llm()
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert output.prompt in PROMPTS
        assert len(output.outputs) == 1
        assert output.outputs[0].text.strip()

    metrics = llm.get_metrics()
    metric_names = {metric.name for metric in metrics}
    assert "vllm:spec_decode_num_drafts" in metric_names
    assert "vllm:spec_decode_num_draft_tokens" in metric_names
    assert "vllm:spec_decode_num_accepted_tokens" in metric_names
