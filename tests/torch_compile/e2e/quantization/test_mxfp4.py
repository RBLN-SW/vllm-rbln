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

"""End-to-end test for MXFP4 MoE quantization (gpt-oss / swigluoai).

Exercises the RBLN ``Mxfp4MoEMethod`` path: expert weights ship as packed
FP4 blocks with E8M0 scales and are dequantized inside the fused
``custom_moe_glu_mxfp4`` op with the built-in swigluoai activation. The test
checks that the quantized MoE model still yields a coherent, factually
correct greedy completion.
"""

from __future__ import annotations

import pytest
from vllm import SamplingParams

from ..utils import managed_llm

MODEL_ID = "openai/gpt-oss-20b"

LLM_KWARGS = {
    "model": MODEL_ID,
    "max_model_len": 4096,
    "max_num_seqs": 4,
    "block_size": 1024,
    "max_num_batched_tokens": 128,
    "enable_chunked_prefill": True,
}

ENV = {
    "VLLM_RBLN_USE_VLLM_MODEL": "1",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "VLLM_RBLN_COMPILE_STRICT_MODE": "1",
    # gpt-oss alternates sliding-window and full attention, producing
    # multi-group KV caches, which RBLN sub-block prefix caching does not
    # support yet.
    "VLLM_RBLN_SUB_BLOCK_CACHE": "false",
}

# (prompt, expected substring) pairs verified by greedy decoding.
PROMPTS = [
    ("The capital of France is", "paris"),
    ("Water is made of hydrogen and", "oxygen"),
]


def test_mxfp4_greedy_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [p for p, _ in PROMPTS]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    with managed_llm(monkeypatch, ENV, **LLM_KWARGS) as llm:
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(outputs) == len(PROMPTS)
    for output, (prompt, expected) in zip(outputs, PROMPTS):
        text = output.outputs[0].text
        assert text.strip(), f"Empty completion for prompt: {prompt!r}"
        assert expected in text.lower(), (
            f"Expected {expected!r} in completion for {prompt!r}, got: {text!r}"
        )
