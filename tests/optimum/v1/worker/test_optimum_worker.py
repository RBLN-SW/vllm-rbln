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
        model_performance_tracker=SimpleNamespace(
            print_final_stats=lambda: rbln_calls.append("stats")
        ),
        sampler_performance_tracker=None,
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
