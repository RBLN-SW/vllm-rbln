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

"""Basic correctness: greedy output tracks an HF reference, compared via logprobs
(greedy flips on near-tied logits). Models use the shared CompileModelSpec."""

import pytest

from tests.native.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from tests.native.utils import check_logprobs_close

MODELS = [
    CompileModelSpec("meta-llama/Llama-3.2-1B-Instruct"),
    CompileModelSpec("Qwen/Qwen3-0.6B"),
    # SWA models are incompatible with sub-block prefix caching.
    CompileModelSpec("google/gemma-3-1b-it", envs={"VLLM_RBLN_SUB_BLOCK_CACHE": "0"}),
]

MAX_TOKENS = 5
NUM_LOGPROBS = 5


@pytest.mark.model_compile
@pytest.mark.parametrize("spec", spec_params(MODELS))
def test_models(hf_runner, vllm_runner, spec: CompileModelSpec, monkeypatch) -> None:
    apply_spec_envs(spec, monkeypatch)
    # ~1600 tokens so the context exceeds a 1024 sliding window and prefill spans
    # many chunks; a repeated sentence keeps the continuation confident.
    prompt = "The quick brown fox jumps over the lazy dog. " * 160
    example_prompts = [prompt]

    with hf_runner(spec.model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(
            example_prompts, MAX_TOKENS, NUM_LOGPROBS
        )

    with vllm_runner(spec.model, **spec.engine_kwargs) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, MAX_TOKENS, NUM_LOGPROBS
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
