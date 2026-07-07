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

"""End-to-end smoke tests for DeepSeek-V3 (with optional MTP)."""

from __future__ import annotations

import pytest
from vllm import SamplingParams

from ..utils import managed_llm

pytestmark = pytest.mark.skip(reason="DeepSeek-V3 e2e generation fails")

MODEL_ID = "deepseek-ai/DeepSeek-V3"

# num_hidden_layers must be >= 3 so first_k_dense_replace works and the
# MTP block doesn't fall back to dense (would hit weight-load KeyErrors).
NUM_HIDDEN_LAYERS = 3
NUM_SPECULATIVE_TOKENS = 1

PROMPTS = [
    "The capital of France is",
    "A robot may not injure a human being",
]


BASE_LLM_PARAMS: dict = {
    "model": MODEL_ID,
    "hf_overrides": {"num_hidden_layers": NUM_HIDDEN_LAYERS},
    "max_model_len": 4096,
    "block_size": 1024,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 8,
    "max_num_seqs": 1,
    "tensor_parallel_size": 4,
    "enable_expert_parallel": True,
    "trust_remote_code": True,
}


def test_deepseek_v3_basic_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    with managed_llm(monkeypatch, env=None, **BASE_LLM_PARAMS) as llm:
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert output.prompt in PROMPTS
        assert len(output.outputs) == 1
        assert output.outputs[0].text.strip()


def test_deepseek_v3_mtp_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    llm_params = {
        **BASE_LLM_PARAMS,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
    }

    with managed_llm(monkeypatch, env=None, **llm_params) as llm:
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

        metric_names = {metric.name for metric in llm.get_metrics()}

    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert output.prompt in PROMPTS
        assert len(output.outputs) == 1
        assert output.outputs[0].text.strip()

    assert "vllm:spec_decode_num_drafts" in metric_names
    assert "vllm:spec_decode_num_draft_tokens" in metric_names
    assert "vllm:spec_decode_num_accepted_tokens" in metric_names
