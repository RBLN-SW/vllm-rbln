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

# Block prefix caching on a real compiled engine, for a cache that holds more
# than KV: a GatedDeltaNet layer resumes its state and its conv ring from the
# cached block. Caching ON must reproduce OFF, and every request must hit.

from __future__ import annotations

import random

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.vllm.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from vllm_rbln import envs

BLOCK_SIZE = 256

MODELS = [
    CompileModelSpec(
        "Qwen/Qwen3.5-4B",
        # text-only; the block size at which a GatedDeltaNet state fills a
        # full-attention page. The pool holds the cached prefix beside the
        # running request.
        {
            "language_model_only": True,
            "block_size": BLOCK_SIZE,
            "max_model_len": 2048,
            "num_gpu_blocks_override": 64,
            "seed": 0,
        },
    ),
]

# Deterministic 600-token shared prefix (two 256 blocks + 88 trailing), each
# prompt then diverging by 10 distinct tokens.
_PREFIX = random.Random(0).sample(range(2000, 100000), 600)
PROMPTS = [
    TokensPrompt(
        prompt_token_ids=_PREFIX + random.Random(i + 1).sample(range(2000, 100000), 10)
    )
    for i in range(4)
]
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=32, logprobs=5)


def _tokens_and_logprobs(outputs) -> list[tuple[list[int], list[dict[int, float]]]]:
    results = []
    for output in outputs:
        completion = output.outputs[0]
        logprobs = [
            {tid: lp.logprob for tid, lp in step.items()}
            for step in completion.logprobs
        ]
        results.append((list(completion.token_ids), logprobs))
    return results


@pytest.mark.model_compile
@pytest.mark.parametrize("spec", spec_params(MODELS))
def test_prefix_cache_matches_baseline(
    vllm_runner, spec: CompileModelSpec, monkeypatch
) -> None:
    apply_spec_envs(spec, monkeypatch)
    # The baseline engine is torn down before the cached one is built, so they
    # never coexist.
    with vllm_runner(
        spec.model, enable_prefix_caching=False, **spec.engine_kwargs
    ) as baseline:
        baseline_outputs = _tokens_and_logprobs(
            baseline.llm.generate(PROMPTS, SAMPLING_PARAMS)
        )

    with vllm_runner(
        spec.model, enable_prefix_caching=True, **spec.engine_kwargs
    ) as cached:
        # Warm the prefix cache with the first prompt only.
        cached.llm.generate(PROMPTS[0], SAMPLING_PARAMS)
        outputs = cached.llm.generate(PROMPTS, SAMPLING_PARAMS)
        cached_outputs = _tokens_and_logprobs(outputs)
        # Per request, as in test_sub_block_prefix_caching_e2e: the engine drops
        # the stats of a step that produced no request output.
        num_cached = [o.num_cached_tokens for o in outputs]

    # The state is kept at the last block boundary below a prompt's end, and
    # the shared prefix covers it: every prompt resumes from two blocks.
    assert num_cached == [2 * BLOCK_SIZE] * len(PROMPTS)
    # A request resumed from the stored state and conv ring runs the uncached
    # one's arithmetic, so the logprobs are equal, not merely close. The tokens
    # alone say little: a truncated model repeats one.
    assert cached_outputs == baseline_outputs


# A hit inside a block: --prefix-match-unit at the prefill chunk keeps one
# partial entry at the prompt's last unit boundary, and a hit copies that block
# (KV, state and conv ring) before the request writes past it.
CHUNK = 128
MTP = {"method": "mtp", "num_speculative_tokens": 1}


def _prompts_sharing(prefix_len: int) -> list[TokensPrompt]:
    prefix = random.Random(0).sample(range(2000, 100000), prefix_len)
    return [
        TokensPrompt(
            prompt_token_ids=prefix
            + random.Random(i + 1).sample(range(2000, 100000), 10)
        )
        for i in range(4)
    ]


@pytest.mark.model_compile
@pytest.mark.parametrize(
    "speculative_config, prefix_len, hit",
    [
        # The first prompt's last unit boundary, 710 // 128 * 128. A whole-block
        # hit stops at 512.
        (None, 700, 640),
        # An MTP lookup drops one unit, as the block drop does a block: one
        # unit below 838 // 128 * 128. A whole-block hit stops at 512.
        (MTP, 828, 640),
    ],
    ids=["mtp-off", "mtp-on"],
)
@pytest.mark.parametrize("spec", spec_params(MODELS))
def test_fine_grained_prefix_hit_matches_baseline(
    vllm_runner,
    spec: CompileModelSpec,
    monkeypatch,
    speculative_config,
    prefix_len,
    hit,
) -> None:
    apply_spec_envs(spec, monkeypatch)
    if 0 < envs.VLLM_RBLN_NUM_HIDDEN_LAYERS < 8:
        # Below four layers Qwen3.5 has no attention group, and a single group
        # hashes whole blocks; the compiler's layer rolling rejects one period
        # of four (see test_spec_decode_e2e). Two periods roll.
        monkeypatch.setenv("VLLM_RBLN_NUM_HIDDEN_LAYERS", "8")
    prompts = _prompts_sharing(prefix_len)
    engine_kwargs = {
        **spec.engine_kwargs,
        "max_num_batched_tokens": CHUNK,
        "speculative_config": speculative_config,
    }
    with vllm_runner(
        spec.model, enable_prefix_caching=False, **engine_kwargs
    ) as baseline:
        baseline_outputs = _tokens_and_logprobs(
            baseline.llm.generate(prompts, SAMPLING_PARAMS)
        )

    with vllm_runner(
        spec.model,
        enable_prefix_caching=True,
        prefix_match_unit=CHUNK,
        **engine_kwargs,
    ) as cached:
        cached.llm.generate(prompts[0], SAMPLING_PARAMS)
        outputs = cached.llm.generate(prompts, SAMPLING_PARAMS)
        cached_outputs = _tokens_and_logprobs(outputs)
        num_cached = [o.num_cached_tokens for o in outputs]

    assert num_cached == [hit] * len(prompts)
    assert cached_outputs == baseline_outputs
