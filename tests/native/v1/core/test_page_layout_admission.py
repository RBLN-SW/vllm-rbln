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

"""Admission is gated on kernel blocks, the resource that actually runs out.

Upstream's page pool drives scheduling, but a request pins every kernel block it
spans while filling only part of their page capacity, so kernel blocks empty
while pages still look free. Before this gate existed, upstream kept admitting
and `bind` raised `OutOfKernelBlocks` into the engine core, killing the worker
(MiniMax-M2.5 DP4+EP, conversations spanning two kernel blocks, 2026-08-14).
"""

import types

import pytest

from vllm_rbln.v1.core.page_layout import PageLayout, PageLayoutManager
from vllm_rbln.v1.core.rbln_page_layout_kv_cache_manager import (
    RBLNPageLayoutKVCacheManager,
)

PAGE = 512
KERNEL_BLOCK = 8192  # 16 pages
BLOCKS = 4


def manager(num_kernel_blocks=BLOCKS, num_reserved=1):
    mgr = RBLNPageLayoutKVCacheManager.__new__(RBLNPageLayoutKVCacheManager)
    mgr.geometry = PageLayout(page_size=PAGE, kernel_block_size=KERNEL_BLOCK)
    mgr.kernel_blocks = PageLayoutManager(
        geometry=mgr.geometry,
        num_kernel_blocks=num_kernel_blocks,
        num_reserved=num_reserved,
    )
    return mgr


def request(rid="r1"):
    return types.SimpleNamespace(request_id=rid)


def have(mgr, req, total_tokens):
    return RBLNPageLayoutKVCacheManager._have_kernel_blocks_for(
        mgr, req, total_tokens
    )


class TestAdmission:
    def test_admits_what_fits(self):
        mgr = manager()
        assert have(mgr, request(), KERNEL_BLOCK)

    def test_admits_a_multi_block_request(self):
        mgr = manager()
        # 3 blocks needed, 4 in the pool with 1 reserved -> exactly fits.
        assert have(mgr, request(), 3 * KERNEL_BLOCK)

    def test_refuses_when_the_pool_cannot_cover_the_request(self):
        mgr = manager()
        # 4 blocks needed but only 3 are allocatable outside the reserve.
        assert not have(mgr, request(), 4 * KERNEL_BLOCK)

    def test_refuses_once_other_requests_hold_the_pool(self):
        mgr = manager()
        for i in range(3):
            mgr.kernel_blocks.bind(f"other{i}", [100 + i], num_cached_pages=0)
        assert not have(mgr, request(), KERNEL_BLOCK)

    def test_growth_counts_only_the_shortfall(self):
        mgr = manager()
        # r1 already spans one block; growing inside it needs nothing new.
        mgr.kernel_blocks.bind("r1", list(range(10, 26)), num_cached_pages=0)
        assert len(mgr.kernel_blocks.block_table("r1")) == 1
        for other in range(2):  # drains the rest of the non-reserve pool
            mgr.kernel_blocks.bind(f"o{other}", [200 + other], num_cached_pages=0)
        assert not mgr.kernel_blocks.allocator.can_allocate(1)
        assert have(mgr, request("r1"), KERNEL_BLOCK)
        # ...but crossing into a second block does need one, and none is free.
        assert not have(mgr, request("r1"), KERNEL_BLOCK + 1)

    def test_reclaims_retained_blocks_before_refusing(self):
        mgr = manager()
        mgr._reclaim_retained = types.MethodType(
            RBLNPageLayoutKVCacheManager._reclaim_retained, mgr
        )
        # the real one also drops the upstream page; only reclaim is under test
        mgr._reclaim_kernel_block = lambda kb: int(mgr.kernel_blocks.reclaim(kb))
        for i in range(3):
            mgr.kernel_blocks.bind(f"done{i}", [300 + i], num_cached_pages=0)
        assert not mgr.kernel_blocks.allocator.can_allocate(1)
        for i in range(3):
            mgr.kernel_blocks.free_request(f"done{i}")  # retained, not reclaimed
        assert have(mgr, request(), KERNEL_BLOCK)

    def test_the_reserve_is_not_spent_on_admission(self):
        # The reserve exists so an already-admitted request's partial match can
        # find a copy destination; admitting against it would deadlock that.
        mgr = manager(num_kernel_blocks=2, num_reserved=1)
        assert have(mgr, request(), KERNEL_BLOCK)
        mgr.kernel_blocks.bind("other", [1], num_cached_pages=0)
        assert mgr.kernel_blocks.allocator.num_free == 1
        assert not have(mgr, request(), KERNEL_BLOCK)


class TestFragmentationGuard:
    def test_multi_block_requests_are_counted(self):
        from vllm_rbln.v1.core.page_layout import validate_fragmentation

        geo = PageLayout(page_size=PAGE, kernel_block_size=KERNEL_BLOCK)
        # 8 seqs over 16 blocks: one block per sequence fits, two does not.
        validate_fragmentation(geo, max_num_seqs=8, num_kernel_blocks=16)
        with pytest.raises(ValueError, match="kernel blocks per"):
            validate_fragmentation(
                geo, max_num_seqs=8, num_kernel_blocks=16, max_model_len=16384
            )

    def test_short_contexts_still_pass(self):
        from vllm_rbln.v1.core.page_layout import validate_fragmentation

        geo = PageLayout(page_size=PAGE, kernel_block_size=KERNEL_BLOCK)
        validate_fragmentation(
            geo, max_num_seqs=8, num_kernel_blocks=27, max_model_len=8192
        )
