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


def test_rewrite_runs_after_the_connector_reads_the_output():
    """Connectors keep **pages**; only the worker's block table is converted.

    The NIXL transfer is page-granular (see `_page_transfer_geometry`), so the
    ids it indexes descriptors with must be pages. LMCache needs kernel blocks
    instead -- it indexes the KV tensor directly -- and gets them from
    `_connector_blocks`, a different accessor. That split is the only reason two
    connectors can disagree about the unit without one of them corrupting KV.

    This ordering was inverted once, to give connectors kernel blocks. It fixed
    NIXL's descriptor math at the time and broke LMCache's, and the version that
    folded ids for both silently corrupted generation (a2b0a118).
    """
    import inspect

    src = inspect.getsource(RBLNScheduler.schedule)
    rewrite = src.index("_rewrite_block_ids_to_kernel_blocks(scheduler_output)")
    connector = src.index("self.connector.build_connector_meta(")
    assert rewrite > connector, (
        "the connector must see page ids; only the worker block table is "
        "rewritten to kernel blocks"
    )


class _FakeConnector:
    """Non-HMA connector: records what `request_finished` was handed."""

    def __init__(self):
        self.seen = None

    def request_finished(self, request, block_ids):
        self.seen = block_ids
        return False, None


def _page_layout_manager(kernel_blocks):
    """A real page-layout manager instance (isinstance matters) without __init__."""
    from vllm_rbln.v1.core.rbln_page_layout_kv_cache_manager import (
        RBLNPageLayoutKVCacheManager,
    )

    mgr = RBLNPageLayoutKVCacheManager.__new__(RBLNPageLayoutKVCacheManager)
    mgr.remove_skipped_blocks = lambda **_: None
    mgr.block_table = lambda _req_id: kernel_blocks
    # what upstream would have used -- pages, deliberately different
    mgr.get_block_ids = lambda _req_id: ([0, 1, 2, 3, 4, 5, 6, 7],)
    # what upstream hands the connector: pages, plus an attribute that must
    # survive the wrapper.
    real = SimpleNamespace(
        get_block_ids=lambda *_a, **_k: ([0, 1, 2, 3, 4, 5, 6, 7],), marker=object()
    )
    mgr.get_blocks = lambda _req_id: real
    return mgr


def _bare_scheduler(**attrs):
    sched = RBLNScheduler.__new__(RBLNScheduler)
    for k, v in attrs.items():
        setattr(sched, k, v)
    return sched


def test_connector_finished_leaves_pages_alone():
    """A prefill's `remote_block_ids` are pages -- the decode indexes its
    page-granular descriptors with them."""
    called = {}
    import vllm.v1.core.sched.scheduler as up

    sched = _bare_scheduler(kv_cache_manager=_page_layout_manager([3]),
                            connector=_FakeConnector())
    request = SimpleNamespace(request_id="r0", num_computed_tokens=0)
    original = up.Scheduler._connector_finished
    up.Scheduler._connector_finished = lambda self, req: called.setdefault("hit", True)
    try:
        sched._connector_finished(request)
    finally:
        up.Scheduler._connector_finished = original
    assert called.get("hit"), "must defer to upstream, which yields pages"


def test_connector_blocks_reports_kernel_blocks():
    """`update_state_after_alloc` is the last id path outside scheduler_output.

    Connectors read `get_block_ids()` off this object and index device memory
    with it. Pages here made LMCache's retrieve scatter on the wrong stride --
    "unflatten: Provided sizes [8, 4096] ... dim 2 (4096)" (measured 2026-08-22).
    """
    kernel_blocks = [3, 4]
    sched = _bare_scheduler(kv_cache_manager=_page_layout_manager(kernel_blocks))
    view = sched._connector_blocks("r0")
    assert view.get_block_ids() == ([3, 4],)


def test_connector_blocks_delegates_everything_else():
    """Only the id view is swapped; the real block object still answers."""
    sched = _bare_scheduler(kv_cache_manager=_page_layout_manager([3]))
    original = sched.kv_cache_manager.get_blocks("r0")
    view = sched._connector_blocks("r0")
    assert view.marker is original.marker


def test_connector_blocks_untouched_without_page_layout():
    manager = SimpleNamespace(get_blocks=lambda _r: "the-real-blocks")
    sched = _bare_scheduler(kv_cache_manager=manager)
    assert sched._connector_blocks("r0") == "the-real-blocks"


def test_update_state_after_alloc_uses_the_converted_blocks():
    """The call site must pass `_connector_blocks`, not the raw manager blocks.

    Testing the helper alone would not catch a call site that goes back to
    `kv_cache_manager.get_blocks()` -- which is exactly the shape of the bug.
    """
    import inspect

    src = inspect.getsource(RBLNScheduler.schedule)
    call = src.index("self.connector.update_state_after_alloc(")
    window = src[call : call + 300]
    assert "_connector_blocks(" in window, (
        "update_state_after_alloc must receive _connector_blocks(request_id); "
        "kv_cache_manager.get_blocks() yields pages under page layout"
    )
