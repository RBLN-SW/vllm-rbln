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
"""Sampling through a compiled model: what only a whole engine can get wrong.

That is the runner's grammar bitmask, rows mapped back to requests, and the
padded decode bucket. None depends on output quality, so the truncated model
of the default --num-hidden-layers is enough.
"""

import re

import pytest
import torch
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from tests.vllm.model_specs import CompileModelSpec, apply_spec_envs

from .utils import assert_samples_match, expected_probs

pytestmark = pytest.mark.model_compile

SPEC = CompileModelSpec("Qwen/Qwen3-0.6B")
PROMPT = "The capital of France is"
# Coarse on purpose: enough to catch an ignored filter or a misrouted row.
NUM_SAMPLES = 2048


def generate(
    runner, prompts: list[str], params: SamplingParams | list[SamplingParams]
) -> list[CompletionOutput]:
    """The first completion of each prompt."""
    return [out.outputs[0] for out in runner.llm.generate(prompts, params)]


@pytest.fixture(scope="module")
def runner(vllm_runner):
    engine_kwargs = {**SPEC.engine_kwargs, "max_num_seqs": 32, "max_logprobs": -1}
    with pytest.MonkeyPatch.context() as monkeypatch:
        apply_spec_envs(SPEC, monkeypatch)
        with vllm_runner(SPEC.model, **engine_kwargs) as runner:
            yield runner


@pytest.fixture(scope="module")
def greedy_token(runner) -> int:
    """What the model picks first: the token the bans below have to move."""
    (completion,) = generate(
        runner, [PROMPT], SamplingParams(temperature=0.0, max_tokens=1)
    )
    return completion.token_ids[0]


def test_allowed_token_ids_bound_every_token(runner, greedy_token):
    allowed = [t for t in range(1000, 1016) if t != greedy_token]
    params = SamplingParams(
        temperature=1.0, allowed_token_ids=allowed, max_tokens=8, ignore_eos=True
    )

    for completion in generate(runner, [PROMPT] * 8, params):
        assert set(completion.token_ids) <= set(allowed)


def test_bad_words_never_sampled(runner, greedy_token):
    word = runner.llm.get_tokenizer().decode([greedy_token]).strip()
    params = SamplingParams(temperature=0.0, bad_words=[word], max_tokens=4)

    (completion,) = generate(runner, [PROMPT], params)

    assert greedy_token not in completion.token_ids


def test_min_tokens_holds_back_stop_tokens(runner, greedy_token):
    params = SamplingParams(
        temperature=0.0, min_tokens=4, max_tokens=6, stop_token_ids=[greedy_token]
    )

    (completion,) = generate(runner, [PROMPT], params)

    assert len(completion.token_ids) >= 4
    assert greedy_token not in completion.token_ids[:4]


CHOICE = StructuredOutputsParams(choice=["yes", "no"])
DIGITS = StructuredOutputsParams(regex=r"[0-9]{3}")


# No set may bring back a token the runner's grammar mask dropped. The bias
# lifts tokens neither grammar allows far above the rest.
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"temperature": 0.0}, id="greedy"),
        pytest.param({"temperature": 2.0}, id="temperature_2.0"),
        pytest.param({"temperature": 1.0, "top_k": 5}, id="top_k"),
        pytest.param({"temperature": 1.0, "top_p": 0.5}, id="top_p"),
        pytest.param({"temperature": 1.0, "min_p": 0.1}, id="min_p"),
        pytest.param(
            {
                "temperature": 1.0,
                "presence_penalty": 1.0,
                "frequency_penalty": 1.0,
                "repetition_penalty": 1.5,
            },
            id="penalties",
        ),
        pytest.param(
            {"temperature": 1.0, "logit_bias": {t: 100.0 for t in range(1000, 1016)}},
            id="logit_bias",
        ),
    ],
)
def test_sampling_params_stay_inside_the_grammar(runner, kwargs):
    params = [
        SamplingParams(structured_outputs=CHOICE, max_tokens=4, **kwargs),
        SamplingParams(structured_outputs=DIGITS, max_tokens=8, **kwargs),
    ] * 4

    completions = generate(runner, [PROMPT] * len(params), params)

    for p, completion in zip(params, completions):
        if p.structured_outputs is CHOICE:
            assert completion.text in ("yes", "no")
        else:
            assert re.fullmatch(r"[0-9]{3}", completion.text)


def test_mixed_decode_batch_keeps_each_request_to_its_params(runner, greedy_token):
    """Greedy, filtered and grammar-bound requests share the decode steps after
    the first token. Each must keep its own constraint."""
    allowed = [t for t in range(1000, 1016) if t != greedy_token]
    params = [
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        SamplingParams(temperature=1.0, allowed_token_ids=allowed, max_tokens=4),
        SamplingParams(temperature=1.0, top_k=1, max_tokens=4, ignore_eos=True),
        SamplingParams(temperature=1.0, structured_outputs=CHOICE, max_tokens=4),
    ] * 8

    completions = generate(runner, [PROMPT] * len(params), params)

    for i in range(0, len(completions), 4):
        greedy, allowed_only, top_1, choice = completions[i : i + 4]
        assert set(allowed_only.token_ids) <= set(allowed)
        assert choice.text in ("yes", "no")
        # Same prompt, same batch: greedy and top_k=1 both follow the argmax.
        assert greedy.token_ids == top_1.token_ids


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=1.0, max_tokens=1),
        SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1),
        SamplingParams(top_k=20, max_tokens=1),
        SamplingParams(temperature=1.2, min_p=0.05, max_tokens=1),
        SamplingParams(temperature=1.0, structured_outputs=DIGITS, max_tokens=1),
        SamplingParams(
            temperature=1.0, top_k=3, structured_outputs=DIGITS, max_tokens=1
        ),
    ],
    ids=[
        "temperature",
        "temperature_top_p",
        "top_k",
        "temperature_min_p",
        "grammar",
        "grammar_top_k",
    ],
)
def test_first_token_distribution_matches_upstream(runner, params):
    """Upstream's sampling steps on the model's raw logprobs predict the
    first-token distribution."""
    probe = params.clone()
    probe.logprobs = -1
    (completion,) = generate(runner, [PROMPT], probe)
    vocab_size = runner.llm.llm_engine.model_config.get_vocab_size()
    raw = torch.full((vocab_size,), float("-inf"))
    for token_id, logprob in completion.logprobs[0].items():
        raw[token_id] = logprob.logprob
    expected = expected_probs(raw.unsqueeze(0), [params])[0]

    completions = generate(runner, [PROMPT] * NUM_SAMPLES, params)

    assert_samples_match(torch.tensor([c.token_ids[0] for c in completions]), expected)


# The grammar fixes the first token to "x", so the digit after it comes from a
# decode step, where the measured rows share the batch with the rows below.
X_THEN_DIGIT = StructuredOutputsParams(regex=r"x[0-9]")
NEIGHBORS = [
    SamplingParams(temperature=0.0, max_tokens=2),
    SamplingParams(
        temperature=1.0,
        presence_penalty=1.0,
        frequency_penalty=1.0,
        repetition_penalty=1.5,
        max_tokens=2,
    ),
]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 1.0},
        {"temperature": 0.7, "top_k": 3},
        {"temperature": 1.2, "top_p": 0.8},
    ],
    ids=["temperature", "top_k", "top_p"],
)
def test_decode_distribution_holds_in_a_mixed_batch(runner, kwargs):
    """Greedy and penalized rows beside a random row must not change its
    decode-step distribution."""
    probe = SamplingParams(
        temperature=0.0, structured_outputs=X_THEN_DIGIT, max_tokens=2, logprobs=-1
    )
    (completion,) = generate(runner, [PROMPT], probe)
    vocab_size = runner.llm.llm_engine.model_config.get_vocab_size()
    raw = torch.full((vocab_size,), float("-inf"))
    for token_id, logprob in completion.logprobs[1].items():
        raw[token_id] = logprob.logprob
    measured = SamplingParams(structured_outputs=X_THEN_DIGIT, max_tokens=2, **kwargs)
    expected = expected_probs(raw.unsqueeze(0), [measured])[0]
    params = [measured, measured, *NEIGHBORS] * (NUM_SAMPLES // 2)

    completions = generate(runner, [PROMPT] * len(params), params)

    digits = [c for c, p in zip(completions, params) if p is measured]
    assert {c.token_ids[0] for c in digits} == {completion.token_ids[0]}
    assert_samples_match(torch.tensor([c.token_ids[1] for c in digits]), expected)
