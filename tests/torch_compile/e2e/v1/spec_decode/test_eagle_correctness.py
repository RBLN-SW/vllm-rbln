# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for EAGLE speculative decoding."""

from __future__ import annotations

import pytest
from vllm import SamplingParams

from ...utils import managed_llm
from .utils import (
    assert_spec_matches_base_within_noise,
    ensure_vllm_compatible_eagle_draft_model,
    get_default_eagle_test_model_ids,
    make_batch_prompts,
)

NUM_SPECULATIVE_TOKENS = 3

# Batch sizes to exercise: batch=1 covers the single-sequence path, while
# batch=8/16 exercises the wider verify-kernel shapes where bf16 rounding can
# differ from the base path.
BATCH_SIZES = [1, 8, 16]


def _base_llm_kwargs(method: str, max_num_seqs: int) -> dict:
    base_model_id, _ = get_default_eagle_test_model_ids(method)
    return {
        "model": base_model_id,
        "max_model_len": 2048,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 256,
        "max_num_seqs": max_num_seqs,
        "disable_log_stats": False,
        "tensor_parallel_size": 1,
    }


def _eagle_llm_kwargs(method: str, max_num_seqs: int) -> dict:
    base_model_id, draft_model_id = get_default_eagle_test_model_ids(method)
    draft_model_id = ensure_vllm_compatible_eagle_draft_model(
        eagle_model_id=draft_model_id,
        base_model_id=base_model_id,
    )
    return {
        "model": base_model_id,
        "max_model_len": 2048,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 256,
        "max_num_seqs": max_num_seqs,
        "speculative_config": {
            "method": method,
            "model": draft_model_id,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        "disable_log_stats": False,
        "tensor_parallel_size": 1,
    }


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize(
    ("method", "max_tokens"),
    [("eagle", 8), ("eagle3", 32)],
)
def test_eagle_matches_base_generation(
    monkeypatch: pytest.MonkeyPatch, method: str, max_tokens: int, batch_size: int
) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    prompts = make_batch_prompts(batch_size)

    # Generate with EAGLE speculative decoding first, then teacher-force the
    # resulting tokens through the base model and check, at every position,
    # that each spec token is the base model's greedy choice (up to bf16
    # near-tie flips). Only one engine is alive at a time to bound memory:
    # managed_llm shuts the eagle engine down deterministically before the base
    # engine is built.
    with managed_llm(
        monkeypatch, **_eagle_llm_kwargs(method, max_num_seqs=batch_size)
    ) as eagle_llm:
        spec_reqs = eagle_llm.generate(prompts, sampling_params=sampling_params)
        captured = [
            (list(req.prompt_token_ids), list(req.outputs[0].token_ids))
            for req in spec_reqs
        ]

    with managed_llm(
        monkeypatch, **_base_llm_kwargs(method, max_num_seqs=batch_size)
    ) as base_llm:
        for seq_idx, (prompt_token_ids, spec_token_ids) in enumerate(captured):
            assert_spec_matches_base_within_noise(
                base_llm, prompt_token_ids, spec_token_ids, seq_idx=seq_idx
            )
