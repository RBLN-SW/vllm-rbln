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

# Speculative decoding is a pure optimization: drafts are verified against the
# target, so greedy output with a method ON must equal spec OFF. Covers ngram,
# suffix, eagle, eagle3 and medusa.

import pytest

from tests.native.utils import check_outputs_almost_equal
from tests.native.v1.spec_decode.utils import (
    _DRAFT,
    MEDUSA_DRAFT,
    MEDUSA_TARGET,
    TARGET_MODEL,
)
from tests.native.vllm_config import local_weights_path

# A small, compilable target for the draft-free methods; ngram/suffix speculate
# by matching repeats, so the prompt is deliberately repetitive.
_NGRAM_TARGET = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
MAX_TOKENS = 16
NUM_LOGPROBS = 5

# method -> (target model, extra LLM kwargs, speculative_config). The eagle/medusa
# drafts cap at 2048 positions, so max_model_len is lowered for them.
SPEC_METHODS = {
    "ngram": (
        _NGRAM_TARGET,
        {},
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
    ),
    "suffix": (
        _NGRAM_TARGET,
        {},
        {
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
            "num_speculative_tokens": 3,
        },
    ),
    "eagle": (
        TARGET_MODEL,
        {"max_model_len": 2048, "tensor_parallel_size": 2},
        {"method": "eagle", "model": _DRAFT["eagle"], "num_speculative_tokens": 3},
    ),
    "eagle3": (
        TARGET_MODEL,
        {"max_model_len": 2048, "tensor_parallel_size": 2},
        {"method": "eagle3", "model": _DRAFT["eagle3"], "num_speculative_tokens": 3},
    ),
    "medusa": (
        MEDUSA_TARGET,
        {"max_model_len": 2048},
        {"method": "medusa", "model": MEDUSA_DRAFT, "num_speculative_tokens": 3},
    ),
}

# Methods whose draft path should land accepted tokens on the repetitive prompt.
# medusa is excluded: its only small ungated checkpoint is random-init, so
# acceptance there would measure model quality, not the code path.
_EXPECT_ACCEPTANCE = {"ngram", "suffix", "eagle", "eagle3"}


@pytest.fixture(autouse=True)
def _use_reference_sampler(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")


@pytest.mark.model_compile
@pytest.mark.parametrize("method", SPEC_METHODS)
def test_speculative_decoding_matches_reference(
    vllm_runner, method: str, whole_model: bool
) -> None:
    target, extra_kwargs, spec_config = SPEC_METHODS[method]
    if draft := spec_config.get("model"):
        spec_config = {**spec_config, "model": local_weights_path(draft)}

    with vllm_runner(target, **extra_kwargs) as ref_model:
        ref_outputs = ref_model.generate_greedy_logprobs(
            [PROMPT], MAX_TOKENS, NUM_LOGPROBS
        )

    with vllm_runner(
        target, **extra_kwargs, speculative_config=spec_config
    ) as spec_model:
        spec_outputs = spec_model.generate_greedy_logprobs(
            [PROMPT], MAX_TOKENS, NUM_LOGPROBS
        )
        # Read while the engine is alive (__exit__ shuts down the EngineCore).
        accepted = spec_model.spec_decode_accepted_tokens()

    # Rejection sampling is lossless in exact arithmetic only: the spec run
    # verifies K+1 tokens in one target pass where the reference decodes one at
    # a time, so the two differ by rounding. Only near-tie flips are tolerated.
    check_outputs_almost_equal(
        outputs_0_lst=ref_outputs,
        outputs_1_lst=spec_outputs,
        name_0="reference",
        name_1=f"spec:{method}",
    )
    # Equivalence alone passes even if every draft is rejected, so where drafts
    # can realistically hit, require at least one acceptance. Truncated layers
    # leave the draft's predictions unrelated to the target's, so nothing hits.
    if method in _EXPECT_ACCEPTANCE and whole_model:
        assert accepted > 0, f"spec:{method} accepted no draft tokens"
