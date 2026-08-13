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

"""Behavioural tests for extent binding: attach, copy-on-write, seal."""

import pytest

from vllm_rbln.v1.core.page_extent import (
    ExtentCopyOp,
    ExtentGeometry,
    OutOfExtents,
    PageExtentManager,
)

PAGES_PER_EXTENT = 4


@pytest.fixture
def manager():
    # 4 pages per extent keeps the fixtures readable; the real ratio is 8.
    geo = ExtentGeometry(page_size=16, extent_size=16 * PAGES_PER_EXTENT)
    return PageExtentManager(geo, num_extents=8)


def pages(*ids):
    return list(ids)


class TestFreshRequest:
    def test_no_cache_means_no_copies(self, manager):
        ops = manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        assert ops == []
        assert len(manager.block_table("r1")) == 1

    def test_spans_multiple_extents(self, manager):
        ops = manager.bind("r1", list(range(10, 20)), num_cached_pages=0)
        assert ops == []
        # 10 pages over 4-page extents -> 3 extents.
        assert len(manager.block_table("r1")) == 3

    def test_full_extent_is_sealed(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        assert manager.table.require(extent_id).sealed

    def test_partial_extent_is_not_sealed(self, manager):
        manager.bind("r1", pages(10, 11), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        assert not manager.table.require(extent_id).sealed

    def test_out_of_extents_is_raised(self, manager):
        for i in range(8):
            manager.bind(f"r{i}", pages(100 + i), num_cached_pages=0)
        with pytest.raises(OutOfExtents):
            manager.bind("overflow", pages(999), num_cached_pages=0)


class TestFullExtentAttach:
    def test_whole_extent_hit_attaches_without_copying(self, manager):
        producer = pages(10, 11, 12, 13)
        manager.bind("r1", producer, num_cached_pages=0)
        src_extent = manager.block_table("r1")[0]

        # r2 hits the whole extent and continues past it.
        ops = manager.bind("r2", producer + pages(20), num_cached_pages=4)
        assert ops == [], "a fully matched extent must be reused by reference"
        assert manager.block_table("r2")[0] == src_extent
        assert manager.table.require(src_extent).ref_cnt == 2

    def test_unsealed_extent_is_never_attached(self, manager):
        # r1's extent holds 2 of 4 pages, so it is still open for appending.
        manager.bind("r1", pages(10, 11), num_cached_pages=0)
        ops = manager.bind("r2", pages(10, 11), num_cached_pages=2)
        assert manager.block_table("r2")[0] != manager.block_table("r1")[0]
        assert len(ops) == 1, "a partial extent must be copied, not attached"


class TestPartialMerge:
    def test_partial_hit_copies_into_a_private_extent(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        src_extent = manager.block_table("r1")[0]

        # r2 matches the first two pages only, then writes its own.
        ops = manager.bind("r2", pages(10, 11, 30, 31), num_cached_pages=2)
        dst_extent = manager.block_table("r2")[0]

        assert dst_extent != src_extent
        assert ops == [
            ExtentCopyOp(
                src_extent_id=src_extent,
                dst_extent_id=dst_extent,
                src_start=0,
                dst_start=0,
                num_tokens=2 * 16,
            )
        ], "contiguous copies must coalesce into one run"

    def test_copy_is_bounded_by_one_extent(self, manager):
        # A long producer: two full extents plus two pages.
        producer = list(range(10, 20))
        manager.bind("r1", producer, num_cached_pages=0)

        # r2 matches all 10 pages: the two full extents attach by reference,
        # only the straddling third extent is copied. This is the hybrid-FTL
        # partial merge -- a full merge never happens.
        ops = manager.bind("r2", producer, num_cached_pages=10)
        assert sum(op.num_tokens for op in ops) <= PAGES_PER_EXTENT * 16
        assert manager.block_table("r2")[:2] == manager.block_table("r1")[:2]

    def test_source_extent_is_untouched_by_the_copy(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        src_extent = manager.block_table("r1")[0]
        manager.bind("r2", pages(10, 11, 30, 31), num_cached_pages=2)
        # Invariant I4: the producer's pages are never appended over.
        assert manager.table.require(src_extent).page_ids == [10, 11, 12, 13]


class TestAdoption:
    """I5b: a retained partial extent is extended in place, not copied.

    The multi-turn shape -- a request ends mid-extent and the next one shares
    that prefix and appends to it -- which is where this design beats the
    sub-block overlay, since the overlay must copy the matched remainder into a
    freshly allocated block.
    """

    def test_resumed_prefix_is_extended_in_place(self, manager):
        manager.bind("r1", pages(10, 11, 12), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        manager.free_request("r1")

        ops = manager.bind("r2", pages(10, 11, 12, 13), num_cached_pages=3)
        assert ops == []
        assert manager.block_table("r2") == [extent_id]
        assert manager.table.require(extent_id).page_ids == [10, 11, 12, 13]
        assert manager.num_pages_copied == 0

    def test_adopted_extent_is_reowned_and_referenced(self, manager):
        manager.bind("r1", pages(10, 11), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        manager.free_request("r1")

        manager.bind("r2", pages(10, 11, 12), num_cached_pages=2)
        extent = manager.table.require(extent_id)
        assert extent.ref_cnt == 1 and extent.owner == "r2"

    def test_second_branch_off_the_same_prefix_still_copies(self, manager):
        # Two continuations cannot share one write pointer; the loser pays CoW.
        manager.bind("r1", pages(10, 11, 12), num_cached_pages=0)
        manager.free_request("r1")
        manager.bind("r2", pages(10, 11, 12, 13), num_cached_pages=3)

        ops = manager.bind("r3", pages(10, 11, 12, 14), num_cached_pages=3)
        assert len(ops) == 1 and ops[0].num_tokens == 3 * 16
        assert manager.block_table("r3") != manager.block_table("r2")

    def test_live_sharer_is_never_adopted(self, manager):
        # I4 still governs: r1 has not finished, so its extent is off limits.
        manager.bind("r1", pages(10, 11, 12), num_cached_pages=0)
        src = manager.block_table("r1")[0]

        ops = manager.bind("r2", pages(10, 11, 12, 13), num_cached_pages=3)
        assert len(ops) == 1 and ops[0].src_extent_id == src
        assert manager.table.require(src).page_ids == [10, 11, 12]

    def test_divergent_prefix_is_not_adopted(self, manager):
        # A shared first page is not a shared prefix; slots must line up (I3).
        manager.bind("r1", pages(10, 11, 12), num_cached_pages=0)
        manager.free_request("r1")

        ops = manager.bind("r2", pages(10, 20, 21, 22), num_cached_pages=1)
        assert len(ops) == 1 and ops[0].num_tokens == 16
        assert manager.block_table("r2") != manager.block_table("r1")

    def test_real_turn_still_copies_because_the_tail_slot_is_stale(self, manager):
        # The shape a real turn leaves behind, and the limit of adoption today:
        # turn 0 ends mid-page, so its extent holds a partial page (16) upstream
        # never hashed. Turn 1 matches only 14, 15 and gets a fresh trailing
        # block (99), so the extent is not a prefix of the group and 14, 15 are
        # copied. Fixing this needs the write pointer rewound past slot 2, which
        # is only safe for pages nothing else has cached -- see doc step 5.
        manager.bind("r1", pages(10, 11, 12, 13, 14, 15, 16), num_cached_pages=0)
        head, tail = manager.block_table("r1")
        manager.free_request("r1")

        ops = manager.bind("r2", pages(10, 11, 12, 13, 14, 15, 99), num_cached_pages=6)
        assert manager.table.require(tail).page_ids == [14, 15, 16]
        assert len(ops) == 1 and ops[0].num_tokens == 2 * 16
        # The leading full extent still attaches by reference, as it should.
        assert manager.block_table("r2")[0] == head

    def test_uncached_prefix_is_not_adopted(self, manager):
        # Upstream reported no hit, so those page ids carry no reusable content.
        manager.bind("r1", pages(10, 11), num_cached_pages=0)
        retained = manager.block_table("r1")[0]
        manager.free_request("r1")

        manager.bind("r2", pages(10, 11, 12), num_cached_pages=0)
        assert manager.block_table("r2") != [retained]


class TestGrowth:
    def test_open_extent_grows_across_steps(self, manager):
        manager.bind("r1", pages(10, 11), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        manager.bind("r1", pages(10, 11, 12), num_cached_pages=0)
        assert manager.block_table("r1") == [extent_id]
        assert manager.table.require(extent_id).page_ids == [10, 11, 12]

    def test_growth_seals_and_opens_a_new_extent(self, manager):
        manager.bind("r1", pages(10, 11, 12), num_cached_pages=0)
        first = manager.block_table("r1")[0]
        manager.bind("r1", pages(10, 11, 12, 13, 14), num_cached_pages=0)
        table = manager.block_table("r1")
        assert len(table) == 2 and table[0] == first
        assert manager.table.require(first).sealed


class TestLifetime:
    def test_finished_request_leaves_extents_retained(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        assert manager.free_request("r1") == []
        # Retained: unreferenced but still resident and revivable.
        assert manager.table.require(extent_id).ref_cnt == 0
        assert extent_id in manager.retained_extents()

    def test_preempted_request_reclaims_immediately(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        assert manager.free_request("r1", preempted=True) == [extent_id]
        assert manager.table.get(extent_id) is None
        assert manager.allocator.num_free == manager.allocator.num_extents

    def test_reclaim_refuses_a_referenced_extent(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        assert manager.reclaim(extent_id) is False

    def test_shared_extent_survives_one_owner_finishing(self, manager):
        producer = pages(10, 11, 12, 13)
        manager.bind("r1", producer, num_cached_pages=0)
        manager.bind("r2", producer, num_cached_pages=4)
        extent_id = manager.block_table("r1")[0]
        manager.free_request("r1", preempted=True)
        # Still referenced by r2, so preemption must not reclaim it.
        assert manager.table.get(extent_id) is not None
        assert manager.table.require(extent_id).ref_cnt == 1

    def test_reclaimed_extent_is_reusable(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        extent_id = manager.block_table("r1")[0]
        manager.free_request("r1", preempted=True)
        manager.bind("r2", pages(20, 21), num_cached_pages=0)
        assert manager.block_table("r2") == [extent_id]


class TestMetrics:
    def test_copy_amplification_counts_copied_vs_written(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        assert manager.copy_amplification == 0.0
        manager.bind("r2", pages(10, 11, 30, 31), num_cached_pages=2)
        # 2 pages copied; 6 pages written fresh across both requests.
        assert manager.num_pages_copied == 2
        assert manager.num_pages_written == 6
        assert manager.copy_amplification == pytest.approx(2 / 6)

    def test_reset_clears_state(self, manager):
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        manager.reset()
        assert manager.block_table("r1") == []
        assert manager.allocator.num_free == manager.allocator.num_extents
        assert manager.copy_amplification == 0.0


class TestOverProvisioning:
    def test_reserve_is_available_to_copy_on_write(self):
        geo = ExtentGeometry(page_size=16, extent_size=16 * PAGES_PER_EXTENT)
        manager = PageExtentManager(geo, num_extents=3, num_reserved=1)
        manager.bind("r1", pages(10, 11, 12, 13), num_cached_pages=0)
        manager.bind("r2", pages(20, 21, 22, 23), num_cached_pages=0)
        assert manager.allocator.num_allocatable == 0
        # An ordinary allocation is refused...
        with pytest.raises(OutOfExtents):
            manager.bind("r3", pages(30), num_cached_pages=0)
        # ...but a copy-on-write destination may use the reserve.
        ops = manager.bind("r4", pages(10, 11), num_cached_pages=2)
        assert len(ops) == 1


class TestVictimOrder:
    def test_deepest_extent_is_reclaimed_first(self, manager):
        # Hashes are chained, so reclaiming a shallow extent strands every
        # deeper one; the tail must go first.
        manager.bind("r1", list(range(10, 22)), num_cached_pages=0)  # 3 extents
        shallow, middle, deep = manager.block_table("r1")
        manager.free_request("r1")

        assert manager.retained_extents() == [deep, middle, shallow]
        assert manager.reclaim_retained(1) == 1
        assert manager.table.get(deep) is None
        assert manager.table.get(shallow) is not None

    def test_depth_records_chain_position(self, manager):
        manager.bind("r1", list(range(10, 22)), num_cached_pages=0)
        depths = [manager.table.require(e).depth for e in manager.block_table("r1")]
        assert depths == [0, 1, 2]
