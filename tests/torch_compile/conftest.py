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

import pytest
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.plugins import load_general_plugins


@pytest.fixture(scope="session", autouse=True)
def initialize_environment():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch.setenv("VLLM_USE_V1", "1")
    load_general_plugins()
    return


@pytest.fixture(scope="class")
def monkeypatch_class():
    monkeypatch = pytest.MonkeyPatch()
    yield monkeypatch
    monkeypatch.undo()


@pytest.fixture(scope="module")
def monkeypatch_module():
    monkeypatch = pytest.MonkeyPatch()
    yield monkeypatch
    monkeypatch.undo()


@pytest.fixture
def vllm_config():
    scheduler_config = SchedulerConfig.default_factory()
    model_config = ModelConfig(model="facebook/opt-125m")
    cache_config = CacheConfig(
        block_size=1024,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(data_parallel_size=2)
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config
