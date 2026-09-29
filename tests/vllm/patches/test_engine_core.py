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

"""The post-step draft fetch waits for a request that can verify drafts.

A stub engine only -- the guard reads `scheduler.running` and the three flags
`post_step` already branches on, so no config, checkpoint or device is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.core import EngineCore

from vllm_rbln.patches.engine_core import patched_post_step

PREFILLING = SimpleNamespace(is_prefill_chunk=True, num_output_tokens=0)
LAST_CHUNK_SCHEDULED = SimpleNamespace(is_prefill_chunk=False, num_output_tokens=0)
DECODING = SimpleNamespace(is_prefill_chunk=False, num_output_tokens=1)


class _Executor:
    def __init__(self, drafts="drafts"):
        self.drafts = drafts
        self.calls = 0

    def take_draft_token_ids(self):
        self.calls += 1
        return self.drafts


class _Scheduler:
    def __init__(self, running):
        self.running = list(running)
        self.updated = []

    def update_draft_token_ids(self, draft_token_ids):
        self.updated.append(draft_token_ids)


def _engine(running, *, spec=True, async_scheduling=False, drafts="drafts"):
    return SimpleNamespace(
        check_for_draft_tokens=spec,
        async_scheduling=async_scheduling,
        model_executor=_Executor(drafts),
        scheduler=_Scheduler(running),
    )


def test_the_patch_is_the_one_installed():
    assert EngineCore.post_step is patched_post_step


@pytest.mark.parametrize(
    "running",
    [
        [PREFILLING],
        [PREFILLING] * 4,
        [LAST_CHUNK_SCHEDULED],
        [PREFILLING, LAST_CHUNK_SCHEDULED],
        [LAST_CHUNK_SCHEDULED, PREFILLING],
    ],
)
def test_a_batch_that_cannot_verify_drafts_skips_the_fetch(running):
    engine = _engine(running)

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 0
    assert engine.scheduler.updated == []


@pytest.mark.parametrize(
    "running",
    [
        [DECODING],
        [PREFILLING, DECODING],
        [LAST_CHUNK_SCHEDULED, DECODING],
    ],
)
def test_a_decoding_request_still_fetches(running):
    engine = _engine(running)

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1
    assert engine.scheduler.updated == ["drafts"]


def test_an_empty_running_queue_still_fetches():
    # The guard must not turn a no-op queue into a skip: `any()` is False on an
    # empty list, so the emptiness check is what keeps this path unchanged.
    engine = _engine([])

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1


@pytest.mark.parametrize(
    ("spec", "async_scheduling", "model_executed"),
    [
        (False, False, True),  # no spec decode
        (True, True, True),  # async scheduling updates in the worker
        (True, False, False),  # nothing ran
    ],
)
def test_the_upstream_conditions_are_untouched(spec, async_scheduling, model_executed):
    engine = _engine([DECODING], spec=spec, async_scheduling=async_scheduling)

    patched_post_step(engine, model_executed=model_executed)

    assert engine.model_executor.calls == 0


def test_a_none_result_is_not_forwarded():
    engine = _engine([DECODING], drafts=None)

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1
    assert engine.scheduler.updated == []


def test_a_request_boundary_fetches_once_instead_of_every_step():
    """The regression this guard exists for, in miniature.

    `_update_after_schedule` clears `is_prefill_chunk` when it schedules a
    request's last chunk, but the request keeps its slot in `running` until the
    output that retires it comes back out of `batch_queue` -- a full queue
    later. Keying on the flag alone pays for the round-trip on every one of
    those steps, and under PP each one costs a traversal of the whole pipeline.
    """
    engine = _engine([])
    batch_queue_size = 8

    for _ in range(4):
        engine.scheduler.running = [PREFILLING]
        patched_post_step(engine, model_executed=True)

    for _ in range(batch_queue_size):
        engine.scheduler.running = [LAST_CHUNK_SCHEDULED]
        patched_post_step(engine, model_executed=True)

    engine.scheduler.running = [DECODING]
    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1
    assert engine.scheduler.updated == ["drafts"]


def test_the_guard_is_per_step_not_sticky():
    # A prefilling request must not suppress the fetch for a decode request
    # that joins the batch later in the same run.
    engine = _engine([PREFILLING])

    patched_post_step(engine, model_executed=True)
    assert engine.model_executor.calls == 0

    engine.scheduler.running.append(DECODING)
    patched_post_step(engine, model_executed=True)
    assert engine.model_executor.calls == 1
