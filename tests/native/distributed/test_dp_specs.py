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

from __future__ import annotations

import os

import pytest
from vllm.engine.arg_utils import EngineArgs

from tests.native.distributed.test_dp_e2e import DP_MODELS
from tests.native.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from tests.native.runners import rbln_engine_args

_STANDALONE = "RBLN_CTX_STANDALONE"


@pytest.fixture(autouse=True)
def isolated_standalone_env(monkeypatch):
    # The hook writes it directly; monkeypatch can only roll back a key it recorded.
    monkeypatch.setenv(_STANDALONE, "")


@pytest.mark.parametrize("spec", spec_params(DP_MODELS))
def test_every_dp_spec_passes_the_platform_rules(spec: CompileModelSpec, monkeypatch):
    apply_spec_envs(spec, monkeypatch)
    EngineArgs(
        model=spec.model, **rbln_engine_args(spec.model, **spec.engine_kwargs)
    ).create_engine_config()

    assert os.environ[_STANDALONE] == "1"
