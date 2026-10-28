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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import ProfilerConfig
from vllm.v1.worker.worker_base import WorkerBase

import vllm_rbln.v1.worker.optimum_worker as optimum_worker
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


@pytest.fixture
def rbln_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        worker_utils,
        "rbln_profiler",
        SimpleNamespace(
            start=lambda: calls.append("start"),
            done=lambda: calls.append("done"),
        ),
    )
    return calls


# These two assert what tests/vllm/v1/worker/test_rbln_worker.py::TestProfile
# asserts for the vllm model path: both workers promise the same behaviour. With
# RBLN_PROFILER on, kineto's bridge opens and flushes the RBLN session for the
# torch profile, so the RBLN wrapper has to stay out of it. The flag is patched on
# the worker, not exported: a unit test must not switch the device profiler on.
@pytest.mark.parametrize("rbln_profiler", [False, True])
def test_torch_profiler_keeps_the_rbln_session_to_itself(
    make_worker, monkeypatch, rbln_calls, tmp_path, rbln_profiler
):
    monkeypatch.setattr(
        "vllm_rbln.v1.worker.optimum_worker.rbln_flags",
        SimpleNamespace(RBLN_PROFILER=rbln_profiler),
    )
    worker = make_worker(
        ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_dump_cuda_time_total=False,
        )
    )

    worker.profile(is_start=True)
    worker.profile(is_start=False)

    assert rbln_calls == []


def test_rbln_profiler_starts_and_flushes_at_stop(make_worker, monkeypatch, rbln_calls):
    monkeypatch.setattr(
        "vllm_rbln.v1.worker.optimum_worker.rbln_flags",
        SimpleNamespace(RBLN_PROFILER=True),
    )
    worker = make_worker(ProfilerConfig())

    worker.profile(is_start=True)
    worker.profile(is_start=False)

    assert rbln_calls == ["start", "done"]
    assert isinstance(worker.profiler, worker_utils.RblnProfilerWrapper)


def test_discard_rejects_selective_memory_release(make_worker):
    worker = make_worker(ProfilerConfig())

    with pytest.raises(
        RuntimeError, match="Selective discard is not supported on RBLN"
    ):
        worker.discard(("kv_cache",))


def test_rbln_profiler_does_not_stand_in_for_another_profiler(make_worker, monkeypatch):
    monkeypatch.setattr(
        "vllm_rbln.v1.worker.optimum_worker.rbln_flags",
        SimpleNamespace(RBLN_PROFILER=True),
    )
    worker = make_worker(ProfilerConfig(profiler="cuda"))

    assert worker.profiler is None


def test_shutdown_flushes_a_profile_that_was_never_stopped(
    make_worker, monkeypatch, rbln_calls
):
    monkeypatch.setattr(
        "vllm_rbln.v1.worker.optimum_worker.rbln_flags",
        SimpleNamespace(RBLN_PROFILER=True),
    )
    monkeypatch.setenv("VLLM_RBLN_METRICS", "1")
    worker = make_worker(ProfilerConfig())
    worker.model_runner = SimpleNamespace(
        _metrics_ctx=SimpleNamespace(print_stats=lambda: rbln_calls.append("stats")),
    )

    worker.profile(is_start=True)
    worker.shutdown()

    assert rbln_calls == ["start", "stats", "done"]


def _scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_new_reqs=[object()],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[object(), object()]),
    )


def test_a_profiled_step_is_annotated_through_the_wrapper(make_worker, tmp_path):
    # The worker used to hold a bare torch.profiler.profile, which has no
    # annotate_context_manager, so the first profiled step raised.
    worker = make_worker(
        ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path))
    )

    with worker.annotate_profile(_scheduler_output()):
        pass


@pytest.mark.parametrize("should_annotate", [False, True])
def test_annotation_uses_profiler_state_after_step(make_worker, should_annotate):
    worker = make_worker(ProfilerConfig())
    profiler = SimpleNamespace(
        should_annotate=not should_annotate,
        annotate_context_manager=Mock(return_value=nullcontext()),
    )
    profiler.step = Mock(
        side_effect=lambda: setattr(profiler, "should_annotate", should_annotate)
    )
    worker.profiler = profiler
    scheduler_output = _scheduler_output() if should_annotate else object()

    with worker.annotate_profile(scheduler_output):
        pass

    profiler.step.assert_called_once_with()
    if should_annotate:
        profiler.annotate_context_manager.assert_called_once_with(
            "execute_new_1_cached_2"
        )
    else:
        profiler.annotate_context_manager.assert_not_called()


class TestInitHostThreads:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        saved = torch.get_num_threads()
        for var in (
            "RBLN_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
        ):
            # setenv records the original value so teardown undoes the
            # writes _init_host_threads makes straight to os.environ.
            monkeypatch.setenv(var, "0")
            monkeypatch.delenv(var)
        monkeypatch.setattr(
            optimum_worker,
            "numba",
            SimpleNamespace(
                set_num_threads=lambda n: None,
                get_num_threads=torch.get_num_threads,
            ),
        )
        monkeypatch.setattr(
            optimum_worker.os, "sched_getaffinity", lambda pid: set(range(126))
        )
        yield
        torch.set_num_threads(saved)

    @staticmethod
    def _worker(make_worker, num_local_ranks=1):
        worker = make_worker(ProfilerConfig())
        worker.vllm_config.parallel_config = SimpleNamespace(
            local_world_size=num_local_ranks, data_parallel_size_local=1
        )
        return worker

    def test_capped_by_cgroup_quota(self, make_worker, monkeypatch):
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: 14.0)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: 14.0)
        self._worker(make_worker)._init_host_threads()
        assert torch.get_num_threads() == 7
        assert optimum_worker.os.environ["OMP_NUM_THREADS"] == "7"

    def test_no_quota_uses_half_affinity(self, make_worker, monkeypatch):
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: None)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: None)
        self._worker(make_worker)._init_host_threads()
        assert torch.get_num_threads() == 63

    def test_rbln_num_threads_env_wins(self, make_worker, monkeypatch):
        monkeypatch.setenv("RBLN_NUM_THREADS", "5")
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: 14.0)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: 14.0)
        self._worker(make_worker)._init_host_threads()
        assert torch.get_num_threads() == 5
        assert optimum_worker.os.environ["RBLN_NUM_THREADS"] == "5"
        assert optimum_worker.os.environ["OMP_NUM_THREADS"] == "5"

    def test_quota_split_across_local_ranks(self, make_worker, monkeypatch):
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: 64.0)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: 64.0)
        self._worker(make_worker, num_local_ranks=4)._init_host_threads()
        assert torch.get_num_threads() == 8
