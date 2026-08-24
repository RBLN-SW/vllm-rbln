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

# Whole-model compile-and-run smoke on a real NPU: each model compiles and
# generates tokens.

import os

import pytest

from tests.native.compile.models import MODELS
from tests.native.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from tests.native.runners import DPRequest
from tests.native.utils import devices_needed, rbln_device_count

PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_TOKENS = 8

# Compiling a whole model outlasts any default timeout; thread method because
# SIGALRM cannot fire while the compiler is down in C.
_TIMEOUT_S = int(os.environ.get("VLLM_RBLN_TEST_COMPILE_TIMEOUT_S", 3600))


def _generate(spec: CompileModelSpec, vllm_runner, async_vllm_runner) -> list:
    """One output per DP rank, or one output for a non-DP spec. DP has to go
    through AsyncLLM: the sync LLM rejects data_parallel_size > 1."""
    dp = spec.engine_kwargs.get("data_parallel_size", 1)
    if dp == 1:
        with vllm_runner(spec.model, **spec.engine_kwargs) as model:
            return model.generate_greedy([PROMPT], MAX_TOKENS)
    with async_vllm_runner(spec.model, **spec.engine_kwargs) as model:
        return model.generate_greedy(
            [DPRequest(PROMPT, MAX_TOKENS, dp_rank=rank) for rank in range(dp)]
        )


@pytest.mark.model_compile
@pytest.mark.timeout(_TIMEOUT_S, method="thread")
@pytest.mark.parametrize("spec", spec_params(MODELS))
def test_compile_and_generate(
    vllm_runner, async_vllm_runner, spec: CompileModelSpec, monkeypatch
) -> None:
    needed = devices_needed(spec.engine_kwargs, spec.rsd)
    available = rbln_device_count()
    if available < needed:
        pytest.skip(f"{spec.test_id} needs {needed} NPUs, host has {available}")

    apply_spec_envs(spec, monkeypatch)
    outputs = _generate(spec, vllm_runner, async_vllm_runner)

    assert len(outputs) == spec.engine_kwargs.get("data_parallel_size", 1)
    for rank, (token_ids, _text) in enumerate(outputs):
        assert token_ids, f"{spec.test_id} rank{rank} compiled but generated no tokens"
