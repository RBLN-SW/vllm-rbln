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
from vllm.config import ProfilerConfig
from vllm.v1.worker.worker_base import WorkerBase

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


def test_discard_rejects_selective_memory_release(make_worker):
    worker = make_worker(ProfilerConfig())

    with pytest.raises(
        RuntimeError, match="Selective discard is not supported on RBLN"
    ):
        worker.discard(("kv_cache",))


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
