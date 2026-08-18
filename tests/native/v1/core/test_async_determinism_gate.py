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

"""Regression tests for the RBLNAsyncScheduler cold-start determinism gate.

Async (optimistic) scheduling steps non-blocking, so it can begin stepping
before all in-flight requests have been ingested from the EngineCore input
queue. The number admitted before the first prefill then varies run-to-run with
IPC timing, which varies the DP batch composition. RBLNAsyncScheduler holds the
first step until request ingestion quiesces (see rbln_scheduler.py). These tests
guard that gate at the scheduler level only; no device is involved.
"""

import pytest

from .utils import create_requests, create_scheduler

# The gate sleeps ~10ms per held step; neutralize it so the unit test is fast.
STABLE_STEPS = 3


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("vllm_rbln.v1.core.rbln_scheduler.time.sleep", lambda _s: None)


def _num_scheduled(output) -> int:
    return output.total_num_scheduled_tokens


def test_async_gate_holds_first_step_until_ingestion_quiesces():
    """With pending requests but no new arrivals, the async scheduler must
    schedule nothing for STABLE_STEPS checks, then proceed."""
    scheduler = create_scheduler(async_scheduling=True, max_num_seqs=8)
    for req in create_requests(num_requests=4, num_tokens=8):
        scheduler.add_request(req)

    # Held while quiescing: schedules nothing.
    for _ in range(STABLE_STEPS):
        out = scheduler.schedule()
        assert scheduler._det_hold is True
        assert _num_scheduled(out) == 0

    # Quiesced: now schedules real work.
    out = scheduler.schedule()
    assert scheduler._det_hold is False
    assert _num_scheduled(out) > 0


def test_async_gate_resets_when_new_request_arrives_mid_quiesce():
    """A request arriving during the hold window resets the stability counter,
    so the gate keeps waiting for the request set to settle."""
    scheduler = create_scheduler(async_scheduling=True, max_num_seqs=8)
    reqs = create_requests(num_requests=4, num_tokens=8)
    scheduler.add_request(reqs[0])
    scheduler.add_request(reqs[1])

    scheduler.schedule()  # stable -> 0
    scheduler.schedule()  # stable -> 1
    assert scheduler._det_stable == 1

    # New arrival: waiting count changes -> counter resets, still holding.
    scheduler.add_request(reqs[2])
    out = scheduler.schedule()
    assert scheduler._det_stable == 0
    assert scheduler._det_hold is True
    assert _num_scheduled(out) == 0


def test_async_gate_rearms_when_engine_goes_idle():
    """Once the engine drains to idle, the gate re-arms so the *next* cold start
    is covered too (not just the first)."""
    scheduler = create_scheduler(async_scheduling=True, max_num_seqs=8)
    scheduler._det_warm = True  # pretend a previous batch already warmed it

    # Fresh scheduler has no waiting/running -> idle branch re-arms and holds
    # nothing.
    assert scheduler._det_quiesced() is True
    assert scheduler._det_warm is False


def test_sync_scheduler_never_holds():
    """Sync scheduling gets determinism for free via its blocking first prefill;
    the gate must stay inert (never sets _det_hold, schedules immediately)."""
    scheduler = create_scheduler(async_scheduling=False, max_num_seqs=8)
    for req in create_requests(num_requests=4, num_tokens=8):
        scheduler.add_request(req)

    out = scheduler.schedule()
    assert scheduler._det_hold is False
    assert _num_scheduled(out) > 0
