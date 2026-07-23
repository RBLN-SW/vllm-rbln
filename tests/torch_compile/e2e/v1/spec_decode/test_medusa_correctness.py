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

import pytest
from vllm import SamplingParams

from ...utils import managed_llm
from .utils import (
    DEFAULT_MEDUSA_MODEL_ID,
    DEFAULT_MODEL_ID,
    assert_spec_matches_base_within_noise,
    ensure_converted_medusa_adapter,
    make_batch_prompts,
)

# Batch sizes to exercise: batch=1 covers the single-sequence path, while
# batch=8/16 exercises the wider verify-kernel shapes where bf16 rounding can
# differ from the base path.
BATCH_SIZES = [1, 8, 16]


def _base_llm_kwargs(max_num_seqs: int) -> dict:
    return {
        "model": DEFAULT_MODEL_ID,
        "max_model_len": 2048,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 256,
        "max_num_seqs": max_num_seqs,
        "disable_log_stats": False,
        "tensor_parallel_size": 1,
    }


def _medusa_llm_kwargs(max_num_seqs: int) -> dict:
    medusa_model_id, num_speculative_tokens = ensure_converted_medusa_adapter(
        medusa_model_id=DEFAULT_MEDUSA_MODEL_ID,
        base_model_id=DEFAULT_MODEL_ID,
    )
    return {
        "model": DEFAULT_MODEL_ID,
        "max_model_len": 2048,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 256,
        "max_num_seqs": max_num_seqs,
        "speculative_config": {
            "method": "medusa",
            "model": medusa_model_id,
            "num_speculative_tokens": num_speculative_tokens,
        },
        "disable_log_stats": False,
        "tensor_parallel_size": 1,
    }


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_medusa_matches_base_generation(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        ignore_eos=True,
    )
    prompts = make_batch_prompts(batch_size)

    # Generate with Medusa speculative decoding first, then teacher-force the
    # resulting tokens through the base model and check, at every position,
    # that each spec token is the base model's greedy choice (up to bf16
    # near-tie flips). Only one engine is alive at a time to bound memory:
    # managed_llm shuts the medusa engine down deterministically before the
    # base engine is built.
    with managed_llm(
        monkeypatch, **_medusa_llm_kwargs(max_num_seqs=batch_size)
    ) as medusa_llm:
        spec_reqs = medusa_llm.generate(prompts, sampling_params=sampling_params)
        captured = [
            (list(req.prompt_token_ids), list(req.outputs[0].token_ids))
            for req in spec_reqs
        ]

    with managed_llm(
        monkeypatch, **_base_llm_kwargs(max_num_seqs=batch_size)
    ) as base_llm:
        for seq_idx, (prompt_token_ids, spec_token_ids) in enumerate(captured):
            assert_spec_matches_base_within_noise(
                base_llm, prompt_token_ids, spec_token_ids, seq_idx=seq_idx
            )
