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

import os

import pytest
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.plugins import load_general_plugins


def pytest_configure(config):
    # Must run before test collection so that monkey patches applied by
    # `register_ops()` are in place before any test module does
    # `from vllm.xxx import yyy` at import time and captures the original symbol.
    os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
    # Running torch.compile-based tests in this tree leaves hundreds of
    # background threads alive in the pytest process (we saw ~2400 before
    # the EngineCore spawn). POSIX fork() clones only the calling thread
    # but copies every other thread's mutex into the child in its locked
    # state with no owner, so vLLM's default fork-based EngineCore spawn
    # deadlocks on the first inherited lock it touches. Force spawn for a
    # fresh interpreter in the child. Cost: ~seconds of extra startup per
    # EngineCore.
    #
    # Upstream vLLM forces spawn at conftest scope for similar hazards
    # (e.g. `tests/compile/fusions_e2e/conftest.py`, though their motivation
    # is subprocess-log capture, not thread deadlocks), and marks individual
    # compile-touching tests `@pytest.mark.forked` to isolate them. Tree-
    # level here (vs per-test) so new tests that instantiate an engine
    # don't silently reintroduce the hang.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    load_general_plugins()


@pytest.fixture(autouse=True)
def _isolate_rbln_ctx_standalone():
    # `RblnPlatform.validate_and_setup_prerequisite` sets
    # `RBLN_CTX_STANDALONE=1` in the process env whenever it sees a config
    # with TP/DP/PP/EP > 1, and never clears it. The flag is read by the
    # rebel runtime on every context creation, so once any test's
    # `VllmConfig` triggers it, every subsequent test in the session
    # (and every forked child) creates exclusive contexts and any second
    # compile on the same device fails. Clear it before each test so
    # tests don't depend on collection order.
    os.environ.pop("RBLN_CTX_STANDALONE", None)
    yield


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
