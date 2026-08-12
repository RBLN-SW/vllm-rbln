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
# generates tokens (no HF reference -- too large).

import pytest

from tests.native.compile.models import MODELS
from tests.native.model_specs import CompileModelSpec, apply_spec_envs

# TODO: skipped until each model's compilable engine_kwargs are pinned on hardware.
pytestmark = pytest.mark.skip(reason="TODO: pin compilable engine_kwargs per model")

PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_TOKENS = 8


@pytest.mark.model_compile
@pytest.mark.parametrize("spec", MODELS, ids=lambda spec: spec.model)
def test_compile_and_generate(vllm_runner, spec: CompileModelSpec, monkeypatch) -> None:
    apply_spec_envs(spec, monkeypatch)
    with vllm_runner(spec.model, **spec.engine_kwargs) as model:
        outputs = model.generate_greedy([PROMPT], MAX_TOKENS)

    assert len(outputs) == 1
    token_ids, _text = outputs[0]
    assert token_ids, "model compiled but generated no tokens"
