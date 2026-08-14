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

"""Unit tests for the page/kernel block addressing layer."""

import pytest

from vllm_rbln.v1.core.page_layout import (
    INVALID_PAGE,
    KernelBlockAllocator,
    KernelBlockTable,
    OutOfKernelBlocks,
    PageLayout,
    reserve_kernel_blocks,
    resolve_config,
    validate_fragmentation,
)


class TestPageLayout:
    def test_ratio(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        assert geo.pages_per_kernel_block == 8
        assert not geo.is_degenerate

    def test_degenerate(self):
        assert PageLayout(page_size=512, kernel_block_size=512).is_degenerate

    @pytest.mark.parametrize(
        "page_size, kernel_block_size",
        [(0, 4096), (512, 0), (-1, 4096), (512, 4097), (513, 4096)],
    )
    def test_rejects_bad_sizes(self, page_size, kernel_block_size):
        with pytest.raises(ValueError):
            PageLayout(page_size=page_size, kernel_block_size=kernel_block_size)

    def test_validate_chunk_accepts_divisor(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        geo.validate_chunk(512)  # equality: the usual default
        geo.validate_chunk(256)  # page = 2 * chunk

    def test_validate_chunk_rejects_chunk_larger_than_page(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        with pytest.raises(ValueError, match="never spans"):
            geo.validate_chunk(1024)

    def test_validate_chunk_rejects_non_divisor(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        with pytest.raises(ValueError):
            geo.validate_chunk(300)

    def test_slot_is_derived_from_logical_index(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        # Invariant I3: slot is a pure function of the logical page index.
        for page_index in range(20):
            assert geo.slot(page_index) == page_index % 8

    def test_num_kernel_blocks_for_pages(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        assert geo.num_kernel_blocks_for_pages(0) == 0
        assert geo.num_kernel_blocks_for_pages(1) == 1
        assert geo.num_kernel_blocks_for_pages(8) == 1
        assert geo.num_kernel_blocks_for_pages(9) == 2


class TestKernelBlockAllocator:
    def test_allocate_and_free(self):
        alloc = KernelBlockAllocator(num_kernel_blocks=4)
        ids = [alloc.allocate() for _ in range(4)]
        assert sorted(ids) == [0, 1, 2, 3]
        assert alloc.num_free == 0
        with pytest.raises(OutOfKernelBlocks):
            alloc.allocate()
        alloc.free(ids[0])
        assert alloc.num_free == 1
        assert alloc.allocate() == ids[0]

    def test_free_rejects_unallocated(self):
        alloc = KernelBlockAllocator(num_kernel_blocks=2)
        with pytest.raises(ValueError):
            alloc.free(0)

    def test_reserve_is_withheld_from_ordinary_allocation(self):
        alloc = KernelBlockAllocator(num_kernel_blocks=4, num_reserved=2)
        assert alloc.num_allocatable == 2
        alloc.allocate()
        alloc.allocate()
        assert alloc.num_allocatable == 0
        with pytest.raises(OutOfKernelBlocks):
            alloc.allocate()
        # The CoW destination path may dip into the reserve.
        assert alloc.allocate(urgent=True) in (2, 3)

    def test_rejects_bad_construction(self):
        with pytest.raises(ValueError):
            KernelBlockAllocator(num_kernel_blocks=0)
        with pytest.raises(ValueError):
            KernelBlockAllocator(num_kernel_blocks=4, num_reserved=4)


class TestKernelBlockTable:
    @pytest.fixture
    def table(self):
        return KernelBlockTable(pages_per_kernel_block=4)

    def test_append_assigns_sequential_slots(self, table):
        table.create(0, owner="r1")
        for expected_slot, page_id in enumerate([7, 3, 91, 12]):
            assert table.append_page(0, page_id, fresh=True) == expected_slot
        assert table.require(0).page_ids == [7, 3, 91, 12]
        assert table.locate(91) == (0, 2)

    def test_append_past_capacity_is_rejected(self, table):
        table.create(0)
        for page_id in range(4):
            table.append_page(0, page_id, fresh=True)
        with pytest.raises(ValueError, match="is full"):
            table.append_page(0, 99, fresh=True)

    def test_recycled_page_poisons_the_stale_slot(self, table):
        # Upstream reissues a freed page id for new content, so the older
        # kernel block's bytes are stale under that id.
        table.create(0)
        table.create(1)
        table.append_page(0, 7, fresh=True)
        table.append_page(0, 8, fresh=True)
        table.append_page(1, 7, fresh=True)
        # Slot 0 of kernel block 0 is poisoned, not removed: later slots must keep
        # their positions (invariant I3).
        assert table.require(0).page_ids == [INVALID_PAGE, 8]
        assert table.locate(8) == (0, 1)
        assert table.locate(7) == (1, 0)
        assert table.holders(7) == {1}

    def test_copy_on_write_keeps_both_holders(self, table):
        table.create(0)
        table.create(1)
        table.append_page(0, 7, fresh=True)
        # A CoW duplicate: the source stays valid under the same page id.
        table.append_page(1, 7, fresh=False)
        assert table.holders(7) == {0, 1}
        assert table.require(0).page_ids == [7]
        assert table.require(1).page_ids == [7]

    def test_locate_ignores_invalid_page(self, table):
        assert table.locate(INVALID_PAGE) is None

    def test_remove_drops_page_mappings(self, table):
        table.create(0)
        table.append_page(0, 7, fresh=True)
        table.remove(0)
        assert table.locate(7) is None
        assert table.get(0) is None
        assert table.holders(7) == set()

    def test_remove_keeps_surviving_copy(self, table):
        table.create(0)
        table.create(1)
        table.append_page(0, 7, fresh=True)
        table.append_page(1, 7, fresh=False)
        table.remove(0)
        assert table.locate(7) == (1, 0)

    def test_refcount_tracks_request_attachment(self, table):
        table.create(0)
        table.attach_to_request("r1", 0)
        table.attach_to_request("r2", 0)
        assert table.require(0).ref_cnt == 2
        assert table.detach_request("r1") == []  # still referenced
        assert table.detach_request("r2") == [0]  # now retained
        assert table.require(0).ref_cnt == 0

    def test_attach_is_idempotent_per_request(self, table):
        table.create(0)
        table.attach_to_request("r1", 0)
        table.attach_to_request("r1", 0)
        assert table.require(0).ref_cnt == 1

    def test_release_underflow_is_rejected(self, table):
        table.create(0)
        with pytest.raises(ValueError, match="underflow"):
            table.release(0)

    def test_seal_requires_full_kernel_block(self, table):
        table.create(0, owner="r1")
        table.append_page(0, 1, fresh=True)
        with pytest.raises(ValueError, match="only a"):
            table.seal(0)
        for page_id in (2, 3, 4):
            table.append_page(0, page_id, fresh=True)
        table.seal(0)
        assert table.require(0).sealed
        assert table.require(0).owner is None

    def test_require_missing_kernel_block(self, table):
        with pytest.raises(KeyError):
            table.require(42)

    def test_reset_clears_everything(self, table):
        table.create(0, owner="r1")
        table.append_page(0, 7, fresh=True)
        table.attach_to_request("r1", 0)
        table.reset()
        assert table.locate(7) is None
        assert table.request_kernel_blocks("r1") == []


class TestPageLayoutConfig:
    def test_resolves_geometry_and_pool(self):
        cfg = resolve_config(page_size=512, kernel_block_size=4096, num_pages=800)
        assert cfg.enabled
        assert cfg.geometry.pages_per_kernel_block == 8
        assert cfg.num_kernel_blocks == 100
        assert cfg.num_reserved == 5

    def test_no_published_kernel_block_size_is_degenerate(self):
        # A model that does not declare an kernel block size behaves exactly as
        # upstream: one page per kernel block, layer is a no-op.
        cfg = resolve_config(page_size=512, kernel_block_size=None, num_pages=800)
        assert not cfg.enabled
        assert cfg.geometry.pages_per_kernel_block == 1
        assert cfg.num_reserved == 0

    def test_reserve_is_at_least_one_kernel_block(self):
        # A single CoW still needs a destination.
        assert reserve_kernel_blocks(4) == 1
        assert reserve_kernel_blocks(1) == 0
        assert reserve_kernel_blocks(0) == 0

    def test_reserve_never_consumes_the_whole_pool(self):
        assert reserve_kernel_blocks(4, fraction=10.0) == 3

    def test_fragmentation_rejects_pool_smaller_than_concurrency(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        with pytest.raises(ValueError, match="kernel blocks per request"):
            validate_fragmentation(geo, max_num_seqs=8, num_kernel_blocks=8)

    def test_fragmentation_warns_when_most_of_the_pool_can_be_pinned(self, caplog):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        validate_fragmentation(geo, max_num_seqs=7, num_kernel_blocks=10)
        assert "kernel blocks can be pinned at once" in caplog.text

    def test_fragmentation_is_silent_when_degenerate(self):
        geo = PageLayout(page_size=512, kernel_block_size=512)
        validate_fragmentation(geo, max_num_seqs=1000, num_kernel_blocks=1)
