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

"""Correctness tests for Medusa speculative decoding."""

from __future__ import annotations

import gc

from vllm import LLM, SamplingParams

from .utils import (
    DEFAULT_MEDUSA_MODEL_ID,
    DEFAULT_MODEL_ID,
    ensure_converted_medusa_adapter,
)

PROMPTS = [
    "The capital of France is",
]


def _build_base_llm() -> LLM:
    return LLM(
        model=DEFAULT_MODEL_ID,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        disable_log_stats=False,
        tensor_parallel_size=1,
    )


def _build_medusa_llm() -> LLM:
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
        max_num_seqs=1,
        speculative_config={
            "method": "medusa",
            "model": medusa_model_id,
            "num_speculative_tokens": num_speculative_tokens,
        },
        disable_log_stats=False,
        tensor_parallel_size=1,
    )


def test_medusa_matches_base_generation() -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        ignore_eos=True,
    )

    base_llm = _build_base_llm()
    base_outputs = base_llm.generate(PROMPTS, sampling_params=sampling_params)
    # Release the baseline engine before compiling the Medusa variant.
    del base_llm
    gc.collect()

    medusa_llm = _build_medusa_llm()
    medusa_outputs = medusa_llm.generate(PROMPTS, sampling_params=sampling_params)

    assert len(base_outputs) == len(medusa_outputs)

    for base_output, medusa_output in zip(base_outputs, medusa_outputs, strict=True):
        assert base_output.prompt == medusa_output.prompt
        assert len(base_output.outputs) == len(medusa_output.outputs) == 1
        assert (
            base_output.outputs[0].finish_reason
            == medusa_output.outputs[0].finish_reason
        )
        assert base_output.outputs[0].text == medusa_output.outputs[0].text
        assert base_output.outputs[0].token_ids == medusa_output.outputs[0].token_ids
