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

"""Correctness tests for EAGLE speculative decoding."""

from __future__ import annotations

import gc

import pytest
from vllm import LLM, SamplingParams

from .utils import (
    ensure_vllm_compatible_eagle_draft_model,
    get_default_eagle_test_model_ids,
)

PROMPTS = [
    "The capital of France is",
]
NUM_SPECULATIVE_TOKENS = 3


def _build_base_llm(method: str) -> LLM:
    base_model_id, _ = get_default_eagle_test_model_ids(method)
    return LLM(
        model=base_model_id,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        disable_log_stats=False,
        tensor_parallel_size=1,
    )


def _build_eagle_llm(method: str) -> LLM:
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
        max_num_seqs=1,
        speculative_config={
            "method": method,
            "model": draft_model_id,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        disable_log_stats=False,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize(
    ("method", "max_tokens"),
    [("eagle", 8), ("eagle3", 32)],
)
def test_eagle_matches_base_generation(method: str, max_tokens: int) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    base_llm = _build_base_llm(method)
    base_outputs = base_llm.generate(PROMPTS, sampling_params=sampling_params)
    # Release the baseline engine before compiling the speculative variant.
    del base_llm
    gc.collect()

    eagle_llm = _build_eagle_llm(method)
    eagle_outputs = eagle_llm.generate(PROMPTS, sampling_params=sampling_params)

    assert len(base_outputs) == len(eagle_outputs)

    try:
        for base_output, eagle_output in zip(base_outputs, eagle_outputs, strict=True):
            assert base_output.prompt == eagle_output.prompt
            assert len(base_output.outputs) == len(eagle_output.outputs) == 1
            assert (
                base_output.outputs[0].finish_reason
                == eagle_output.outputs[0].finish_reason
            )
            assert base_output.outputs[0].text == eagle_output.outputs[0].text
            assert base_output.outputs[0].token_ids == eagle_output.outputs[0].token_ids
    finally:
        del eagle_llm
        gc.collect()
