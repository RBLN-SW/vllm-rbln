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

"""Page-native KV cache manager with extent backing.

See docs/page_extent_kv_manager.md. Once ``--block-size`` is the page, upstream
does all the matching natively, so this only maps those pages onto contiguous
extents and emits the copies that implies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.page_extent import (
    INVALID_PAGE,
    ExtentCopyOp,
    OutOfExtents,
    PageExtentConfig,
    PageExtentManager,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

__all__ = ["RBLNPageExtentKVCacheManager"]


class RBLNPageExtentKVCacheManager(KVCacheManager):
    """``KVCacheManager`` whose pages are backed by contiguous extents."""

    @staticmethod
    def can_use_page_extent(
        kv_cache_config: KVCacheConfig,
        config: PageExtentConfig,
    ) -> bool:
        """I10 keeps the MVP to a single full-attention group."""
        if not config.enabled:
            return False
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            return False
        block_size = groups[0].kv_cache_spec.block_size
        return block_size == config.geometry.page_size

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        page_extent_config: PageExtentConfig,
        **kwargs,
    ) -> None:
        assert self.can_use_page_extent(kv_cache_config, page_extent_config)
        super().__init__(kv_cache_config=kv_cache_config, **kwargs)

        self.page_extent_config = page_extent_config
        self.geometry = page_extent_config.geometry
        self.extents = PageExtentManager(
            geometry=page_extent_config.geometry,
            num_extents=page_extent_config.num_extents,
            num_reserved=page_extent_config.num_reserved,
        )
        # Copies the worker must perform before the next forward pass. Each op
        # keeps its source extent referenced until the worker is done.
        self.pending_copy_ops: list[ExtentCopyOp] = []
        self._install_eviction_hook()

        logger.info(
            "Page/extent KV cache: page=%d, extent=%d (%d pages), "
            "extents=%d (%d reserved for copy-on-write)",
            self.geometry.page_size,
            self.geometry.extent_size,
            self.geometry.pages_per_extent,
            page_extent_config.num_extents,
            page_extent_config.num_reserved,
        )

    # -- allocation ---------------------------------------------------------

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        *args,
        **kwargs,
    ) -> KVCacheBlocks | None:
        # computed + just-matched pages have content; the rest gets written
        cached_tokens = request.num_computed_tokens + num_new_computed_tokens

        result = super().allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            *args,
            **kwargs,
        )
        if result is None:
            return None

        self._bind_extents(request, cached_tokens)
        return result

    def _bind_extents(self, request: Request, cached_tokens: int) -> None:
        """Back the request's pages with extents, queueing any copies."""
        page_ids = self._page_ids(request)
        if not page_ids:
            return
        num_cached_pages = min(cached_tokens // self.geometry.page_size, len(page_ids))
        try:
            ops = self.extents.bind(request.request_id, page_ids, num_cached_pages)
        except OutOfExtents:
            # retained extents are only given up under pressure
            needed = self.geometry.num_extents_for_pages(len(page_ids))
            if self._reclaim_retained(needed) == 0:
                raise
            ops = self.extents.bind(request.request_id, page_ids, num_cached_pages)

        for op in ops:
            self.extents.table.acquire(op.src_extent_id)
        self.pending_copy_ops.extend(ops)

    def _page_ids(self, request: Request) -> list[int]:
        blocks: list[KVCacheBlock] = self.coordinator.get_blocks(request.request_id)[0]
        return [block.block_id for block in blocks]

    # -- copy op plumbing ---------------------------------------------------

    def drain_pending_copy_ops(self) -> list[ExtentCopyOp]:
        """Copies for this step; sources stay referenced until released."""
        ops = self.pending_copy_ops
        self.pending_copy_ops = []
        return ops

    def release_copy_ops(self, ops: list[ExtentCopyOp]) -> None:
        """Drop the source references held by drained copy ops."""
        for op in ops:
            if self.extents.table.get(op.src_extent_id) is not None:
                self.extents.table.release(op.src_extent_id)

    def block_table(self, request_id: str) -> list[int]:
        """The worker's block table: extent ids backing a request."""
        return self.extents.block_table(request_id)

    # -- lifetime -----------------------------------------------------------

    def free(self, request: Request) -> None:
        preempted = request.num_computed_tokens == 0
        self.extents.free_request(request.request_id, preempted=preempted)
        super().free(request)

    def reset_prefix_cache(self) -> bool:
        result = super().reset_prefix_cache()
        if result:
            self.extents.reset()
            self.pending_copy_ops.clear()
        return result

    def _on_page_evicted(self, page_id: int) -> None:
        """I7 has no way to punch a hole, so losing one page costs the extent.

        CoW can leave several holders; referenced ones are skipped.
        """
        for extent_id in self.extents.table.holders(page_id):
            self._reclaim_extent(extent_id, already_evicted=page_id)

    def _reclaim_extent(self, extent_id: int, *, already_evicted: int = -1) -> bool:
        """Reclaim an extent and drop upstream's claim on the pages it held.

        Reclaiming takes down every page in the extent, but upstream still
        lists the siblings as cached and would report hits for bytes that are
        gone -- a miss that recomputes nothing. Evict them there too, unless a
        copy survives in another extent.
        """
        extent = self.extents.table.get(extent_id)
        if extent is None or extent.ref_cnt > 0:
            return False
        siblings = [
            page_id
            for page_id in extent.page_ids
            if page_id not in (already_evicted, INVALID_PAGE)
        ]
        if not self.extents.reclaim(extent_id):
            return False
        for page_id in siblings:
            if not self.extents.table.holders(page_id):
                self._evict_upstream_page(page_id)
        return True

    def _reclaim_retained(self, count: int) -> int:
        reclaimed = 0
        for extent_id in self.extents.retained_extents():
            if reclaimed >= count:
                break
            reclaimed += self._reclaim_extent(extent_id)
        return reclaimed

    def _evict_upstream_page(self, page_id: int) -> None:
        block = self.block_pool.blocks[page_id]
        if block.block_hash is not None:
            # the unhooked original, so this does not recurse into our hook
            self._evict_cached_block(block)

    def _install_eviction_hook(self) -> None:
        # no upstream callback, and we need to know whether it actually evicted
        original_evict = self._evict_cached_block = (
            self.block_pool._maybe_evict_cached_block
        )

        def evict_with_extent_reclaim(block: KVCacheBlock) -> bool:
            page_id = block.block_id
            evicted = original_evict(block)
            if evicted:
                self._on_page_evicted(page_id)
            return evicted

        self.block_pool._maybe_evict_cached_block = evict_with_extent_reclaim

    # -- metrics ------------------------------------------------------------

    @property
    def copy_amplification(self) -> float:
        return self.extents.copy_amplification
