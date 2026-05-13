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

"""End-to-end smoke tests for EAGLE speculative decoding."""

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams

from .utils import (
    ensure_vllm_compatible_eagle_draft_model,
    get_default_eagle_test_model_ids,
)

PROMPTS = [
    "A robot may not injure a human being",
    "The capital of France is",
]
NUM_SPECULATIVE_TOKENS = 3


def _build_llm(method: str) -> LLM:
    base_model_id, draft_model_id = get_default_eagle_test_model_ids(method)
    draft_model_id = ensure_vllm_compatible_eagle_draft_model(
        eagle_model_id=draft_model_id,
        base_model_id=base_model_id,
    )
    return LLM(
        model=base_model_id,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        speculative_config={
            "method": method,
            "model": draft_model_id,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        disable_log_stats=False,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize("method", ["eagle", "eagle3"])
def test_basic_eagle_generation(method: str) -> None:
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=128)

    llm = _build_llm(method)
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
