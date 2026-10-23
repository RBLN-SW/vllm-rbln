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

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.v1.worker.optimum_worker as optimum_worker
import vllm_rbln.v1.worker.utils as worker_utils
from vllm_rbln.v1.worker.optimum_worker import RBLNOptimumWorker


class ProfilerConfig:
    pass


@pytest.fixture
def make_worker():
    def _make(_profiler_config) -> RBLNOptimumWorker:
        worker = object.__new__(RBLNOptimumWorker)
        worker.rank = 0
        worker.local_rank = 0
        return worker

    return _make


class TestInitHostThreads:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        saved = torch.get_num_threads()
        monkeypatch.delenv("RBLN_NUM_THREADS", raising=False)
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
        ):
            monkeypatch.delenv(var, raising=False)
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

    def test_capped_by_cgroup_quota(self, make_worker, monkeypatch):
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: 14.0)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: 14.0)
        make_worker(ProfilerConfig())._init_host_threads()
        assert torch.get_num_threads() == 7
        assert optimum_worker.os.environ["OMP_NUM_THREADS"] == "7"

    def test_no_quota_uses_half_affinity(self, make_worker, monkeypatch):
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: None)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: None)
        make_worker(ProfilerConfig())._init_host_threads()
        assert torch.get_num_threads() == 63

    def test_rbln_num_threads_env_wins(self, make_worker, monkeypatch):
        monkeypatch.setenv("RBLN_NUM_THREADS", "5")
        monkeypatch.setattr(worker_utils, "get_cgroup_cpu_quota", lambda: 14.0)
        monkeypatch.setattr(optimum_worker, "get_cgroup_cpu_quota", lambda: 14.0)
        make_worker(ProfilerConfig())._init_host_threads()
        assert torch.get_num_threads() == 5
        assert optimum_worker.os.environ["RBLN_NUM_THREADS"] == "5"
        assert optimum_worker.os.environ["OMP_NUM_THREADS"] == "5"
