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

from tests.native.utils import check_outputs_equal
from tests.native.v1.spec_decode.utils import (
    _DRAFT,
    MEDUSA_DRAFT,
    MEDUSA_TARGET,
    TARGET_MODEL,
)

# A small, compilable target for the draft-free methods; ngram/suffix speculate
# by matching repeats, so the prompt is deliberately repetitive.
_NGRAM_TARGET = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
MAX_TOKENS = 16

# method -> (target model, extra LLM kwargs, speculative_config). The eagle/medusa
# drafts cap at 2048 positions, so max_model_len is lowered for them.
SPEC_METHODS = {
    "ngram": (
        _NGRAM_TARGET,
        {"num_gpu_blocks_override": 32},
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
    ),
    "suffix": (
        _NGRAM_TARGET,
        {"num_gpu_blocks_override": 32},
        {
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
            "num_speculative_tokens": 3,
        },
    ),
    "eagle": (
        TARGET_MODEL,
        {"max_model_len": 2048, "tensor_parallel_size": 4},
        {"method": "eagle", "model": _DRAFT["eagle"], "num_speculative_tokens": 3},
    ),
    "eagle3": (
        TARGET_MODEL,
        {"max_model_len": 2048, "tensor_parallel_size": 4},
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
_XFAIL_REASON = {
    "eagle3": "RBLNCompileError: RblnTensorAllocateDevTensorKey pass fails on the "
    "eagle3 aux-hidden-state graph",
}


def _method_param(method: str):
    if method in _XFAIL_REASON:
        return pytest.param(
            method,
            marks=pytest.mark.xfail(reason=_XFAIL_REASON[method], run=False),
        )
    return method


@pytest.fixture(autouse=True)
def _use_reference_sampler(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")


@pytest.mark.model_compile
@pytest.mark.parametrize("method", [_method_param(m) for m in SPEC_METHODS])
def test_speculative_decoding_matches_reference(vllm_runner, method: str) -> None:
    target, extra_kwargs, spec_config = SPEC_METHODS[method]

    with vllm_runner(target, **extra_kwargs) as ref_model:
        ref_outputs = ref_model.generate_greedy([PROMPT], MAX_TOKENS)

    with vllm_runner(
        target, **extra_kwargs, speculative_config=spec_config
    ) as spec_model:
        spec_outputs = spec_model.generate_greedy([PROMPT], MAX_TOKENS)
        # Read while the engine is alive (__exit__ shuts down the EngineCore).
        accepted = spec_model.spec_decode_accepted_tokens()

    check_outputs_equal(
        outputs_0_lst=ref_outputs,
        outputs_1_lst=spec_outputs,
        name_0="reference",
        name_1=f"spec:{method}",
    )
    # Equivalence alone passes even if every draft is rejected, so where drafts
    # can realistically hit, require at least one acceptance.
    if method in _EXPECT_ACCEPTANCE:
        assert accepted > 0, f"spec:{method} accepted no draft tokens"
