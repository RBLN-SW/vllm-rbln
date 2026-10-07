# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import vllm.profiler.wrapper as profiler_wrapper
from vllm.config import ProfilerConfig
from vllm.v1.worker.worker_base import WorkerBase

import vllm_rbln.v1.worker.utils as worker_utils
from vllm_rbln.v1.worker.optimum_worker import RBLNOptimumWorker


def _fake_super_init(
    self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False
):
    # Stand-in for WorkerBase.__init__ that skips real device setup.
    self.vllm_config = vllm_config
    self.local_rank = local_rank
    self.rank = rank


@pytest.fixture
def make_worker(monkeypatch):
    monkeypatch.setattr(WorkerBase, "__init__", _fake_super_init)

    def _make(profiler_config: ProfilerConfig) -> RBLNOptimumWorker:
        return RBLNOptimumWorker(
            vllm_config=SimpleNamespace(
                profiler_config=profiler_config, instance_id="test"
            ),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
        )

    return _make


class _TorchWrapperStub:
    """Stands in for TorchProfilerWrapper: a real one starts kineto, and with
    torch-rbln loaded that reaches into the NPU from this process."""

    def __init__(self, profiler_config, *, worker_name, local_rank, activities):
        self.activities = activities


@pytest.fixture
def rbln_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        worker_utils.rbln_profiler, "start", lambda: calls.append("start")
    )
    monkeypatch.setattr(
        worker_utils.rbln_profiler, "done", lambda: calls.append("done")
    )
    return calls


@pytest.mark.parametrize(
    ("torch_on", "rbln_on", "expected"),
    [
        (False, False, None),
        (True, False, _TorchWrapperStub),
        (False, True, worker_utils.RblnProfilerWrapper),
        # kineto's RBLN bridge opens and flushes the session itself while a
        # torch profile runs, so the RBLN wrapper stays out of it.
        (True, True, _TorchWrapperStub),
    ],
)
def test_each_profiler_combination_picks_its_wrapper(
    make_worker, monkeypatch, tmp_path, torch_on, rbln_on, expected
):
    monkeypatch.setattr(profiler_wrapper, "TorchProfilerWrapper", _TorchWrapperStub)
    if rbln_on:
        monkeypatch.setenv("RBLN_PROFILER", "1")
    else:
        monkeypatch.delenv("RBLN_PROFILER", raising=False)
    config = (
        ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path))
        if torch_on
        else ProfilerConfig()
    )

    worker = make_worker(config)

    if expected is None:
        assert worker.profiler is None
        with pytest.raises(RuntimeError, match="not enabled"):
            worker.profile(is_start=True)
    else:
        assert type(worker.profiler) is expected


def test_the_rbln_profiler_alone_flushes_at_stop(make_worker, monkeypatch, rbln_calls):
    monkeypatch.setenv("RBLN_PROFILER", "1")
    worker = make_worker(ProfilerConfig())

    worker.profile(is_start=True)
    worker.profile(is_start=False)

    assert rbln_calls == ["start", "done"]


def test_shutdown_flushes_a_profile_that_was_never_stopped(
    make_worker, monkeypatch, rbln_calls
):
    monkeypatch.setenv("RBLN_PROFILER", "1")
    worker = make_worker(ProfilerConfig())
    worker.model_runner = SimpleNamespace()  # shutdown reads it for metrics only
    monkeypatch.setattr(
        "vllm_rbln.v1.worker.optimum_worker.envs.VLLM_RBLN_METRICS", False
    )

    worker.profile(is_start=True)
    worker.shutdown()

    assert rbln_calls == ["start", "done"]
