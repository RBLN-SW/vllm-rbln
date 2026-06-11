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

import random
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Metric

_RNG = random.Random(0)
PREFIX = _RNG.sample(range(2000, 100000), 1600)
PROMPTS = [
    TokensPrompt(
        prompt_token_ids=PREFIX + random.Random(i + 1).sample(range(2000, 100000), 10)
    )
    for i in range(4)
]
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=32)
BLOCK_SIZE = 1024


def _get_counter(metrics: list[Metric], name: str) -> int:
    return sum(m.value for m in metrics if isinstance(m, Counter) and m.name == name)


def _build_llm(*, enable_prefix_caching: bool) -> LLM:
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=1,
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        enable_prefix_caching=enable_prefix_caching,
        # First LLM leaks device memory on teardown, so the second LLM
        # OOMs at default block count. 8 blocks × 1024 = 8192 tokens, enough
        # to hold the warm-up plus one query.
        num_gpu_blocks_override=8,
        seed=0,
    )
    # block_size is validated in the EngineArgs ctor; assign post-hoc to bypass.
    args.block_size = BLOCK_SIZE  # type: ignore[assignment]
    return LLM(**asdict(args))


def _generated_token_ids(outputs) -> list[list[int]]:
    return [list(o.outputs[0].token_ids) for o in outputs]


def _cleanup(llm: LLM) -> None:
    del llm
    try:
        cleanup_dist_env_and_memory()
    except RuntimeError as e:
        # ignore error when not using torch-rbln
        if "Cannot access accelerator device when none is available" not in str(e):
            raise


@pytest.mark.parametrize("use_device_tensor", [False, True])
def test_sub_block_prefix_cache_matches_baseline(
    monkeypatch: pytest.MonkeyPatch, use_device_tensor: bool
) -> None:
    monkeypatch.setenv("VLLM_RBLN_USE_DEVICE_TENSOR", "1" if use_device_tensor else "0")

    # Baseline
    baseline = _build_llm(enable_prefix_caching=False)
    try:
        baseline_tokens = _generated_token_ids(
            baseline.generate(PROMPTS, SAMPLING_PARAMS)
        )
    finally:
        _cleanup(baseline)

    # Prefix-cached
    cached = _build_llm(enable_prefix_caching=True)
    try:
        # Warm up prefix cache
        cached.generate(PROMPTS[0], SAMPLING_PARAMS)

        hits_before = _get_counter(cached.get_metrics(), "vllm:prefix_cache_hits")
        outputs = cached.generate(PROMPTS, SAMPLING_PARAMS)
        hits_after = _get_counter(cached.get_metrics(), "vllm:prefix_cache_hits")
        cached_tokens = _generated_token_ids(outputs)
    finally:
        _cleanup(cached)

    # Shared prefix is 1600 tokens (block_size=1024 + 576 trailing). Without
    # sub-block caching each prompt would hit at most one full block, capping
    # the total at len(PROMPTS) * BLOCK_SIZE. Sub-block caching extends the hit.
    hits = hits_after - hits_before
    assert hits > len(PROMPTS) * BLOCK_SIZE
    assert cached_tokens == baseline_tokens
