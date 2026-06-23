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

# SOC markers: a test marked with one of these runs only when the current
# RBLN target SOC matches; on any other SOC it is skipped. Matching is by
# substring so it works regardless of the vendor prefix (e.g. REBEL-CR03,
# RBLN-CA25).
SOC_MARKERS = ("CR03", "CA25")


def _current_target_soc() -> str | None:
    """Resolve the current RBLN target SOC (upper-cased) for marker skipping.

    Prefer the ``RBLN_TARGET_SOC`` env var (set on CPU-only/compile workers
    and in CI); fall back to the live NPU name when an NPU is mounted.
    Returns ``None`` when the SOC cannot be determined.
    """
    soc = os.environ.get("RBLN_TARGET_SOC")
    if soc:
        return soc.upper()
    try:
        import rebel

        name = rebel.get_npu_name()
        if name:
            return name.upper()
    except Exception:
        pass
    return None


def pytest_collection_modifyitems(config, items):
    """Run ``test_rbln_envs`` first, then apply SOC-marker skips.

    Ordering: ``vllm_rbln.platform`` mutates module-level ``rbln_envs``
    attributes at runtime (e.g. sets ``VLLM_RBLN_SAMPLER = False`` when
    ``speculative_config`` is present, see ``platform.py:260``).  Any
    earlier test that instantiates a vLLM config with spec-decode
    enabled leaves the env-defaults test asserting on dirty state and
    failing.  ``importlib.reload`` does not undo the mutation because
    it lives in the module ``__dict__`` and shadows the lazy
    ``__getattr__`` lambda.  Run the env-defaults test first instead of
    polluting the test source with reset boilerplate.

    SOC markers: tests marked with a SOC name (see ``SOC_MARKERS``) only
    run when the current target SOC matches; otherwise they are skipped.
    """
    items.sort(key=lambda item: 0 if item.name == "test_rbln_envs" else 1)

    target_soc = _current_target_soc()
    for item in items:
        required = [soc for soc in SOC_MARKERS if item.get_closest_marker(soc)]
        if not required:
            continue
        if target_soc is None:
            reason = (
                f"requires SOC {'/'.join(required)} but the target SOC could "
                "not be determined (set RBLN_TARGET_SOC)"
            )
            item.add_marker(pytest.mark.skip(reason=reason))
        elif not any(soc in target_soc for soc in required):
            reason = (
                f"requires SOC {'/'.join(required)}; current target is "
                f"{target_soc}"
            )
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_configure(config):
    for soc in SOC_MARKERS:
        config.addinivalue_line(
            "markers",
            f"{soc}: run only when the RBLN target SOC is {soc} "
            "(skipped on any other SOC)",
        )
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
