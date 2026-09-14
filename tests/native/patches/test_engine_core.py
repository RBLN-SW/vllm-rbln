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

"""The post-step draft fetch stays off the prefill path.

A stub engine only -- the guard reads `scheduler.running` and the three flags
`post_step` already branches on, so no config, checkpoint or device is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.core import EngineCore

from vllm_rbln.patches.engine_core import patched_post_step


class _Executor:
    def __init__(self, drafts="drafts"):
        self.drafts = drafts
        self.calls = 0

    def take_draft_token_ids(self):
        self.calls += 1
        return self.drafts


class _Scheduler:
    def __init__(self, prefill_flags):
        self.running = [
            SimpleNamespace(is_prefill_chunk=flag) for flag in prefill_flags
        ]
        self.updated = []

    def update_draft_token_ids(self, draft_token_ids):
        self.updated.append(draft_token_ids)


def _engine(prefill_flags, *, spec=True, async_scheduling=False, drafts="drafts"):
    return SimpleNamespace(
        check_for_draft_tokens=spec,
        async_scheduling=async_scheduling,
        model_executor=_Executor(drafts),
        scheduler=_Scheduler(prefill_flags),
    )


def test_the_patch_is_the_one_installed():
    assert EngineCore.post_step is patched_post_step


@pytest.mark.parametrize("prefill_flags", [[True], [True, True], [True] * 4])
def test_all_prefill_skips_the_fetch(prefill_flags):
    engine = _engine(prefill_flags)

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 0
    assert engine.scheduler.updated == []


@pytest.mark.parametrize(
    "prefill_flags",
    [
        [False],  # a decode request
        [True, False],  # the step that scheduled a request's last chunk
        [False, True],
    ],
)
def test_any_non_prefill_request_still_fetches(prefill_flags):
    engine = _engine(prefill_flags)

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1
    assert engine.scheduler.updated == ["drafts"]


def test_an_empty_running_queue_still_fetches():
    # The guard must not turn a no-op queue into a skip: `all()` is True on an
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
    engine = _engine([False], spec=spec, async_scheduling=async_scheduling)

    patched_post_step(engine, model_executed=model_executed)

    assert engine.model_executor.calls == 0


def test_a_none_result_is_not_forwarded():
    engine = _engine([False], drafts=None)

    patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1
    assert engine.scheduler.updated == []


def test_a_chunked_prefill_run_fetches_once_instead_of_every_step():
    """The regression this guard exists for, in miniature.

    A long prompt under chunked prefill is many intermediate chunks and one
    that finishes it. Only the last one produces an anchor token, so only its
    drafts are ever verified -- yet unpatched `post_step` pays for the
    round-trip on every step. Under PP that round-trip is a wait on the last
    stage, which is what stops `batch_queue` from refilling.
    """
    engine = _engine([])
    n_chunks = 101

    for step in range(n_chunks):
        last = step == n_chunks - 1
        # `_update_after_schedule` advances num_computed_tokens before the step
        # runs, so the request already reads as non-prefill on its last chunk.
        engine.scheduler.running = [SimpleNamespace(is_prefill_chunk=not last)]
        patched_post_step(engine, model_executed=True)

    assert engine.model_executor.calls == 1
    assert engine.scheduler.updated == ["drafts"]


def test_the_guard_is_per_step_not_sticky():
    # A prefilling request must not suppress the fetch for a decode request
    # that joins the batch later in the same run.
    engine = _engine([True])

    patched_post_step(engine, model_executed=True)
    assert engine.model_executor.calls == 0

    engine.scheduler.running.append(SimpleNamespace(is_prefill_chunk=False))
    patched_post_step(engine, model_executed=True)
    assert engine.model_executor.calls == 1
