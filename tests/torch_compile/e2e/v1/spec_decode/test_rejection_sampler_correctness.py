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

"""
End-to-end correctness for the RBLN rejection sampler (greedy path).
"""

from __future__ import annotations

import os

import pytest
from vllm import SamplingParams

from ...utils import managed_llm
from .utils import assert_spec_matches_base_within_noise

os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "1"
os.environ["VLLM_RBLN_SAMPLER"] = "1"
# Repeated n-grams so the ngram proposer finds prompt-lookup matches and drafts
# tokens; distinct content across requests keeps the batch heterogeneous.
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps "
    "over the lazy dog. The quick brown fox",
    "one two three four one two three four one two three four one two three",
]
MODEL_ID = "Qwen/Qwen3-0.6B"


def _base_llm_kwargs() -> dict:
    return {
        "model": MODEL_ID,
        "max_model_len": 2048,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 256,
        "max_num_seqs": len(PROMPTS),
        "tensor_parallel_size": 1,
    }


def _ngram_llm_kwargs() -> dict:
    return {
        **_base_llm_kwargs(),
        # get_metrics() asserts log_stats; the offline LLM class disables it by
        # default, so enable it explicitly for the spec engine.
        "disable_log_stats": False,
        "speculative_config": {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
        },
    }


def test_rejection_sampler_greedy_matches_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # temperature=0 routes the rejection sampler through its greedy path, where
    # the target distribution is one-hot and the output must equal the base
    # model's argmax token-for-token.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        ignore_eos=True,
    )

    # Generate with ngram spec decode first, capture the trajectories, then
    # build the non-speculative base engine. managed_llm shuts the spec engine
    # down before the base engine is built, so only one is alive at a time.
    with managed_llm(monkeypatch, **_ngram_llm_kwargs()) as spec_llm:
        spec_reqs = spec_llm.generate(PROMPTS, sampling_params=sampling_params)
        assert len(spec_reqs) == len(PROMPTS)
        captured = [
            (list(req.prompt_token_ids), list(req.outputs[0].token_ids))
            for req in spec_reqs
        ]

        metrics = spec_llm.get_metrics()
        metric_names = {metric.name for metric in metrics}
        # Sanity check: the proposer actually drafted, so accept/reject ran.
        assert "vllm:spec_decode_num_draft_tokens" in metric_names

    with managed_llm(monkeypatch, **_base_llm_kwargs()) as base_llm:
        for seq_idx, (prompt_token_ids, spec_token_ids) in enumerate(captured):
            assert spec_token_ids, f"seq {seq_idx} produced no tokens"
            assert_spec_matches_base_within_noise(
                base_llm, prompt_token_ids, spec_token_ids, seq_idx=seq_idx
            )
