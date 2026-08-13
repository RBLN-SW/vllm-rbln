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

"""Reclaiming an extent must take upstream's claim on its pages with it."""

from types import SimpleNamespace

import pytest

from vllm_rbln.v1.core.page_extent import ExtentGeometry, PageExtentManager
from vllm_rbln.v1.core.rbln_page_extent_kv_cache_manager import (
    RBLNPageExtentKVCacheManager,
)

PAGES_PER_EXTENT = 4


@pytest.fixture
def manager():
    """A manager with only the pieces the eviction path touches."""
    mgr = RBLNPageExtentKVCacheManager.__new__(RBLNPageExtentKVCacheManager)
    geo = ExtentGeometry(page_size=16, extent_size=16 * PAGES_PER_EXTENT)
    mgr.geometry = geo
    mgr.extents = PageExtentManager(geo, num_extents=8)
    mgr.block_pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=f"h{i}") for i in range(64)]
    )
    mgr.evicted_upstream = []
    mgr._evict_cached_block = lambda block: mgr.evicted_upstream.append(block)
    return mgr


def test_siblings_are_evicted_upstream(manager):
    # Without this, upstream keeps reporting the siblings as cache hits while
    # their bytes are gone: nothing recomputes them and nothing copies them.
    manager.extents.bind("r1", [10, 11, 12, 13], num_cached_pages=0)
    manager.extents.free_request("r1")

    manager._on_page_evicted(10)

    evicted = {b.block_hash for b in manager.evicted_upstream}
    assert evicted == {"h11", "h12", "h13"}, "siblings must lose their hashes"
    assert manager.extents.table.locate(11) is None


def test_the_evicted_page_is_not_re_evicted(manager):
    manager.extents.bind("r1", [10, 11], num_cached_pages=0)
    manager.extents.free_request("r1")
    manager._on_page_evicted(10)
    assert [b.block_hash for b in manager.evicted_upstream] == ["h11"]


def test_a_surviving_copy_keeps_the_page_cached(manager):
    # r2 copy-on-writes page 11, so its bytes outlive r1's extent.
    manager.extents.bind("r1", [10, 11, 12, 13], num_cached_pages=0)
    manager.extents.bind("r2", [11], num_cached_pages=1)
    manager.extents.free_request("r1")

    manager._on_page_evicted(10)

    evicted = {b.block_hash for b in manager.evicted_upstream}
    assert "h11" not in evicted, "a page with a surviving copy stays cached"
    assert evicted == {"h12", "h13"}


def test_a_referenced_extent_is_left_alone(manager):
    manager.extents.bind("r1", [10, 11, 12, 13], num_cached_pages=0)
    manager._on_page_evicted(10)  # r1 still running
    assert manager.evicted_upstream == []
    assert manager.extents.table.locate(11) is not None


def test_reclaim_under_pressure_also_evicts_upstream(manager):
    manager.extents.bind("r1", [10, 11, 12, 13], num_cached_pages=0)
    manager.extents.free_request("r1")
    assert manager._reclaim_retained(1) == 1
    assert {b.block_hash for b in manager.evicted_upstream} == {
        "h10",
        "h11",
        "h12",
        "h13",
    }
