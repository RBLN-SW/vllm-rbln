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

"""Translation of scheduler-output block ids from pages to kernel blocks.

The worker addresses KV by kernel block, so a wrong delta here does not crash -- it
silently points attention at the wrong memory. These tests pin the delta
semantics directly.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler


@dataclass
class FakeNewReq:
    req_id: str
    block_ids: tuple = ()


@dataclass
class FakeCached:
    req_ids: list = field(default_factory=list)
    resumed_req_ids: set = field(default_factory=set)
    new_block_ids: list = field(default_factory=list)


@dataclass
class FakeOutput:
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_cached_reqs: FakeCached = field(default_factory=FakeCached)


class FakeManager:
    """Stands in for RBLNPageLayoutKVCacheManager's block_table()."""

    def __init__(self, tables):
        self.tables = tables

    def block_table(self, req_id):
        return list(self.tables.get(req_id, []))


@pytest.fixture
def scheduler(monkeypatch):
    """A bare RBLNScheduler with only what the translation touches."""
    sched = RBLNScheduler.__new__(RBLNScheduler)
    sched._sent_kernel_block_counts = {}
    # The translation is gated on the manager's type; swap the check for one
    # that accepts the fake.
    monkeypatch.setattr(
        "vllm_rbln.v1.core.rbln_scheduler.RBLNPageLayoutKVCacheManager",
        FakeManager,
    )
    return sched


def translate(sched, output):
    RBLNScheduler._rewrite_block_ids_to_kernel_blocks(sched, output)


class TestNewRequests:
    def test_new_request_gets_its_full_kernel_block_table(self, scheduler):
        scheduler.kv_cache_manager = FakeManager({"a": [5, 6]})
        out = FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([0, 1, 2],))])
        translate(scheduler, out)
        assert out.scheduled_new_reqs[0].block_ids == ([5, 6],)
        assert scheduler._sent_kernel_block_counts["a"] == 2


class TestCachedRequests:
    def test_no_new_kernel_block_yields_no_delta(self, scheduler):
        # Pages accumulate within one kernel block, so the worker needs nothing new.
        scheduler.kv_cache_manager = FakeManager({"a": [5]})
        scheduler._sent_kernel_block_counts = {"a": 1}
        out = FakeOutput(
            scheduled_cached_reqs=FakeCached(req_ids=["a"], new_block_ids=[([9],)])
        )
        translate(scheduler, out)
        assert out.scheduled_cached_reqs.new_block_ids[0] is None

    def test_crossing_a_kernel_block_boundary_emits_one_kernel_block(self, scheduler):
        scheduler.kv_cache_manager = FakeManager({"a": [5, 6]})
        scheduler._sent_kernel_block_counts = {"a": 1}
        out = FakeOutput(
            scheduled_cached_reqs=FakeCached(req_ids=["a"], new_block_ids=[([9],)])
        )
        translate(scheduler, out)
        assert out.scheduled_cached_reqs.new_block_ids[0] == ([6],)
        assert scheduler._sent_kernel_block_counts["a"] == 2

    def test_first_sight_of_a_cached_request_sends_everything(self, scheduler):
        scheduler.kv_cache_manager = FakeManager({"a": [5, 6]})
        out = FakeOutput(
            scheduled_cached_reqs=FakeCached(req_ids=["a"], new_block_ids=[None])
        )
        translate(scheduler, out)
        assert out.scheduled_cached_reqs.new_block_ids[0] == ([5, 6],)

    def test_resumed_request_gets_the_whole_table_not_a_delta(self, scheduler):
        # Resumed requests replace their block table rather than appending.
        scheduler.kv_cache_manager = FakeManager({"a": [5, 6, 7]})
        scheduler._sent_kernel_block_counts = {"a": 2}
        out = FakeOutput(
            scheduled_cached_reqs=FakeCached(
                req_ids=["a"], resumed_req_ids={"a"}, new_block_ids=[None]
            )
        )
        translate(scheduler, out)
        assert out.scheduled_cached_reqs.new_block_ids[0] == ([5, 6, 7],)
        assert scheduler._sent_kernel_block_counts["a"] == 3

    def test_multiple_requests_are_tracked_independently(self, scheduler):
        scheduler.kv_cache_manager = FakeManager({"a": [1, 2], "b": [3]})
        scheduler._sent_kernel_block_counts = {"a": 1, "b": 1}
        out = FakeOutput(
            scheduled_cached_reqs=FakeCached(
                req_ids=["a", "b"], new_block_ids=[None, None]
            )
        )
        translate(scheduler, out)
        assert out.scheduled_cached_reqs.new_block_ids == [([2],), None]

    def test_repeated_steps_never_resend_a_kernel_block(self, scheduler):
        # Whole-sequence property: concatenating every delta reproduces the
        # table exactly once, which is what the worker's append assumes.
        tables = {"a": [10]}
        scheduler.kv_cache_manager = FakeManager(tables)
        out = FakeOutput(scheduled_new_reqs=[FakeNewReq("a")])
        translate(scheduler, out)
        seen = list(out.scheduled_new_reqs[0].block_ids[0])

        for grown in ([10, 11], [10, 11], [10, 11, 12]):
            tables["a"] = grown
            out = FakeOutput(
                scheduled_cached_reqs=FakeCached(req_ids=["a"], new_block_ids=[None])
            )
            translate(scheduler, out)
            delta = out.scheduled_cached_reqs.new_block_ids[0]
            if delta is not None:
                seen.extend(delta[0])
        assert seen == [10, 11, 12]


class TestDisabled:
    def test_other_managers_are_left_alone(self, scheduler):
        scheduler.kv_cache_manager = SimpleNamespace()  # not the kernel block manager
        out = FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([0, 1],))])
        translate(scheduler, out)
        assert out.scheduled_new_reqs[0].block_ids == ([0, 1],)


def test_rewrite_runs_before_the_connector_reads_the_output():
    """The rewrite must precede `build_connector_meta` in `schedule()`.

    Both consumers index the same ids: the worker builds its block table from
    `scheduled_*_reqs`, and a KV connector builds its transfer descriptors from
    the same output. Whichever runs first decides the unit each one sees.

    It used to run last, so the worker got kernel blocks and the connector kept
    pages. Nothing complained until a request was long enough to move KV across
    a PD pair -- NIXL then prepped descriptors over 49 kernel blocks and was
    handed 53 page ids for a ~20k prompt ("transfer_setup_failed ...
    num_local_blocks: 53", measured 2026-08-22 on MiniMax-M2.5 / R100 2P2D).
    Short requests kept answering correctly the whole time, which is what makes
    the ordering worth pinning rather than leaving to review.
    """
    import inspect

    src = inspect.getsource(RBLNScheduler.schedule)
    rewrite = src.index("_rewrite_block_ids_to_kernel_blocks(scheduler_output)")
    connector = src.index("self.connector.build_connector_meta(")
    assert rewrite < connector, (
        "_rewrite_block_ids_to_kernel_blocks must run before "
        "build_connector_meta, or the KV connector indexes its descriptors "
        "with page ids while the worker uses kernel blocks"
    )
