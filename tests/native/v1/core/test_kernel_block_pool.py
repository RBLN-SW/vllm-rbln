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

"""The pool itself enforces what a second allocator used to try to.

Contiguity and one-writer-per-block become properties of allocation, and
`get_num_free_blocks` -- which `KVCacheManager` consults before admitting work --
counts what is actually backable. That last point is the bug this class exists to
remove: upstream admitted against free pages while kernel blocks were exhausted,
and binding then raised into the engine core.
"""

import pytest

from vllm_rbln.v1.core.kernel_block_pool import KernelBlockPool, RBLNKVCacheBlock

PPE = 4  # pages per kernel block


def pool(num_pages=16, ppe=PPE, enable_caching=True):
    return KernelBlockPool(
        num_gpu_blocks=num_pages,
        enable_caching=enable_caching,
        hash_block_size=16,
        pages_per_kernel_block=ppe,
    )


def ids(blocks):
    return [b.block_id for b in blocks]


class TestGeometry:
    def test_page_id_encodes_its_location(self):
        p = pool()
        assert p.kernel_block_of(0) == 0 and p.slot_of(0) == 0
        assert p.kernel_block_of(5) == 1 and p.slot_of(5) == 1
        assert list(p.page_ids_of(2)) == [8, 9, 10, 11]

    def test_a_kernel_block_holds_its_pages(self):
        p = pool()
        group = p.kernel_blocks[1]
        assert [page.block_id for page in group.pages] == [4, 5, 6, 7]
        assert all(isinstance(page, RBLNKVCacheBlock) for page in group.pages)
        assert all(page.kernel_block is group for page in group.pages)
        assert p.page(5).kernel_block is group
        assert group.pages[p.slot_of(5)] is p.blocks[5]

    def test_trailing_partial_run_is_dropped(self):
        # 14 pages is 3 whole kernel blocks plus 2 that could never be used
        # without breaking contiguity.
        p = pool(num_pages=14)
        assert p.num_kernel_blocks == 3
        assert p.num_gpu_blocks == 12

    def test_pool_smaller_than_one_kernel_block_is_rejected(self):
        with pytest.raises(ValueError, match="less than one kernel block"):
            pool(num_pages=3)


class TestAllocation:
    def test_a_request_fills_one_kernel_block_before_opening_another(self):
        p = pool()
        with p.allocating_for("r1"):
            got = ids(p.get_new_blocks(3))
        assert [p.kernel_block_of(i) for i in got] == [1, 1, 1], got
        assert [p.slot_of(i) for i in got] == [0, 1, 2]

    def test_growth_continues_in_the_same_kernel_block(self):
        p = pool()
        with p.allocating_for("r1"):
            first = ids(p.get_new_blocks(2))
            second = ids(p.get_new_blocks(1))
        assert {p.kernel_block_of(i) for i in first + second} == {1}
        assert p.slot_of(second[0]) == 2

    def test_a_second_request_never_shares_an_open_kernel_block(self):
        p = pool()
        with p.allocating_for("r1"):
            mine = ids(p.get_new_blocks(1))
        with p.allocating_for("r2"):
            theirs = ids(p.get_new_blocks(1))
        assert p.kernel_block_of(mine[0]) != p.kernel_block_of(theirs[0])

    def test_a_request_spanning_blocks_gets_consecutive_runs(self):
        p = pool()
        with p.allocating_for("r1"):
            got = ids(p.get_new_blocks(PPE + 1))
        blocks = [p.kernel_block_of(i) for i in got]
        assert blocks[:PPE] == [blocks[0]] * PPE
        assert blocks[PPE] != blocks[0]
        assert p.slot_of(got[PPE]) == 0


class TestAdmissionAccounting:
    def test_free_count_is_reported_in_backable_pages(self):
        # 4 kernel blocks, but block 0 is unusable: page 0 is upstream's null
        # block, so a request's run could never start at its slot 0.
        p = pool()
        assert p.get_num_free_blocks() == 3 * PPE

    def test_an_open_block_still_offers_its_free_slots(self):
        p = pool()
        with p.allocating_for("r1"):
            p.get_new_blocks(1)
        # r1 opened a kernel block and used one slot of it.
        assert p.get_num_free_blocks() == (PPE - 1) + 2 * PPE

    def test_pinning_every_kernel_block_reports_only_their_leftovers(self):
        # The shape that crashed the engine: each request pins a whole kernel
        # block while using one page of it. Upstream used to see plenty of free
        # pages here; now it sees only what those open blocks can still take.
        p = pool()
        for r in range(3):
            with p.allocating_for(f"r{r}"):
                p.get_new_blocks(1)
        assert p.get_num_free_blocks() == 3 * (PPE - 1)
        # ...and a fourth request cannot be backed by a fresh block.
        assert p.num_idle_kernel_blocks() == 0
        with p.allocating_for("r4"):
            assert p.kernel_blocks_needed(1) > p.num_idle_kernel_blocks()

    def test_refuses_more_than_it_reports(self):
        p = pool()
        with pytest.raises(ValueError, match="Cannot get"), p.allocating_for("r1"):
            p.get_new_blocks(p.get_num_free_blocks() + 1)

    def test_need_is_asked_per_owner_because_a_page_sum_cannot_say_it(self):
        p = pool()
        for r in ("r1", "r2"):
            with p.allocating_for(r):
                p.get_new_blocks(PPE)  # each fills a whole group
        # A page short each, so a whole group apiece. Their page total says 1,
        # which is why in-flight reservations round per request, not after.
        assert p.kernel_blocks_needed(1, owner="r1") == 1
        assert p.kernel_blocks_needed(1, owner="r2") == 1
        assert p.kernel_blocks_needed(2) == 1

    def test_need_is_counted_in_groups_after_the_open_run_is_used_up(self):
        p = pool()
        with p.allocating_for("r1"):
            p.get_new_blocks(1)
            # The open block's remaining slots cost nothing new; only what
            # spills past them opens another group.
            assert p.kernel_blocks_needed(PPE - 1) == 0
            assert p.kernel_blocks_needed(PPE) == 1
            assert p.kernel_blocks_needed(PPE - 1 + PPE + 1) == 2


class TestRelease:
    def test_a_kernel_block_reopens_once_all_its_pages_are_free(self):
        p = pool()
        with p.allocating_for("r1"):
            blocks = p.get_new_blocks(2)
        group = blocks[0].kernel_block
        assert group.owner == "r1"

        p.free_blocks(blocks)
        assert group.owner is None
        with p.allocating_for("r2"):
            reused = p.get_new_blocks(1)
        assert reused[0].kernel_block is group

    def test_a_partly_freed_block_stays_with_its_owner(self):
        p = pool()
        with p.allocating_for("r1"):
            blocks = p.get_new_blocks(2)
        group = blocks[0].kernel_block
        p.free_blocks(blocks[:1])
        assert group.owner == "r1"
        with p.allocating_for("r2"):
            other = p.get_new_blocks(1)
        assert other[0].kernel_block is not group
