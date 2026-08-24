# Copyright 2026 Rebellions Inc. All rights reserved.
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

# Sub-block prefix caching on a real compiled engine: the KV copy execution the
# CPU tests cannot reach. Caching ON must match OFF greedily, and hits above
# len(PROMPTS) * BLOCK_SIZE prove the sub-block path really ran.

from __future__ import annotations

import random

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Metric

from tests.native.utils import TokensTextLogprobs, check_outputs_almost_equal

MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 1024
NUM_LOGPROBS = 5

# Deterministic 1600-token shared prefix (spans one 1024 block + 576 trailing),
# each prompt then diverging by 10 distinct tokens.
_PREFIX = random.Random(0).sample(range(2000, 100000), 1600)
PROMPTS = [
    TokensPrompt(
        prompt_token_ids=_PREFIX + random.Random(i + 1).sample(range(2000, 100000), 10)
    )
    for i in range(4)
]
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=32)

# On top of the shared runner defaults. A small KV budget keeps each engine
# cheap while the shared prefix still spans more than one block.
_ENGINE_OVERRIDES = dict(
    max_model_len=4096,
    num_gpu_blocks_override=8,
    seed=0,
    disable_log_stats=False,
)


def _get_counter(metrics: list[Metric], name: str) -> int:
    return sum(m.value for m in metrics if isinstance(m, Counter) and m.name == name)


def _generated_token_ids(outputs) -> list[list[int]]:
    return [list(o.outputs[0].token_ids) for o in outputs]


@pytest.mark.model_compile
def test_sub_block_prefix_cache_matches_baseline(vllm_runner) -> None:
    # Runs in whichever --device-tensor mode the session uses. The baseline
    # engine is torn down before the cached one is built, so they never coexist.
    with vllm_runner(
        MODEL, enable_prefix_caching=False, **_ENGINE_OVERRIDES
    ) as baseline:
        baseline_tokens = _generated_token_ids(
            baseline.llm.generate(PROMPTS, SAMPLING_PARAMS)
        )

    with vllm_runner(MODEL, enable_prefix_caching=True, **_ENGINE_OVERRIDES) as cached:
        # Warm the prefix cache with the first prompt only.
        cached.llm.generate(PROMPTS[0], SAMPLING_PARAMS)
        hits_before = _get_counter(cached.llm.get_metrics(), "vllm:prefix_cache_hits")
        outputs = cached.llm.generate(PROMPTS, SAMPLING_PARAMS)
        hits_after = _get_counter(cached.llm.get_metrics(), "vllm:prefix_cache_hits")
        cached_tokens = _generated_token_ids(outputs)

    # Sub-block hits push the total past what full-block hits alone could reach.
    assert hits_after - hits_before > len(PROMPTS) * BLOCK_SIZE
    # The copied KV must reproduce the uncached greedy output exactly.
    assert cached_tokens == baseline_tokens


def _run_to_completion(engine, tag, sampling, *, reset_at=None):
    """Drive the engine to completion, optionally resetting the prefix cache at
    a given step. Returns one (token_ids, text, logprobs) per prompt, in prompt
    order -- requests finish in whatever order the scheduler retires them."""
    for i, prompt in enumerate(PROMPTS):
        engine.add_request(f"{tag}_{i}", prompt, sampling)
    results: dict[str, TokensTextLogprobs] = {}
    step = 0
    while engine.has_unfinished_requests():
        if reset_at is not None and step == reset_at:
            # reset_running_requests=True preempts in-flight requests and clears
            # the sub-block index, forcing recompute.
            engine.reset_prefix_cache(reset_running_requests=True)
        for out in engine.step():
            if out.finished:
                completion = out.outputs[0]
                results[out.request_id] = (
                    list(completion.token_ids),
                    completion.text,
                    [
                        {tid: lp.logprob for tid, lp in step_logprobs.items()}
                        for step_logprobs in (completion.logprobs or [])
                    ],
                )
        step += 1
    return [results[f"{tag}_{i}"] for i in range(len(PROMPTS))]


@pytest.mark.model_compile
def test_reset_prefix_cache_mid_run_matches_baseline(vllm_runner) -> None:
    # Resetting mid-generation must not change the output: the preempted requests
    # recompute their KV and land on the same greedy tokens. Driven via the V1
    # engine so the reset can be injected between steps.
    #
    # The recompute re-runs the prefill arithmetic in a different shape than the
    # cached continuation it replaces, so the two agree only up to rounding.
    # Only near-tie flips are tolerated.
    sampling = SamplingParams(temperature=0.0, max_tokens=16, logprobs=NUM_LOGPROBS)
    with vllm_runner(MODEL, enable_prefix_caching=True, **_ENGINE_OVERRIDES) as runner:
        engine = runner.llm.llm_engine
        baseline = _run_to_completion(engine, "gt", sampling)
        after_reset = _run_to_completion(engine, "rs", sampling, reset_at=10)

    check_outputs_almost_equal(
        outputs_0_lst=baseline,
        outputs_1_lst=after_reset,
        name_0="baseline",
        name_1="after_reset",
    )
