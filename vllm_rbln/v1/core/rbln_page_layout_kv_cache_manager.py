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

"""Page-native KV cache manager with kernel block backing.

See docs/page_layout_kv_manager.md. Once ``--block-size`` is the page, upstream
does all the matching natively, so this only maps those pages onto contiguous
kernel blocks and emits the copies that implies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.page_layout import (
    INVALID_PAGE,
    KernelBlockCopyOp,
    OutOfKernelBlocks,
    PageLayoutConfig,
    PageLayoutManager,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

__all__ = ["RBLNPageLayoutKVCacheManager"]


class RBLNPageLayoutKVCacheManager(KVCacheManager):
    """``KVCacheManager`` whose pages are backed by contiguous kernel blocks."""

    @staticmethod
    def can_use_page_layout(
        kv_cache_config: KVCacheConfig,
        config: PageLayoutConfig,
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
        page_layout_config: PageLayoutConfig,
        **kwargs,
    ) -> None:
        assert self.can_use_page_layout(kv_cache_config, page_layout_config)
        super().__init__(kv_cache_config=kv_cache_config, **kwargs)

        self.page_layout_config = page_layout_config
        self.geometry = page_layout_config.geometry
        self.kernel_blocks = PageLayoutManager(
            geometry=page_layout_config.geometry,
            num_kernel_blocks=page_layout_config.num_kernel_blocks,
            num_reserved=page_layout_config.num_reserved,
        )
        # Copies the worker must perform before the next forward pass. Each op
        # keeps its source kernel block referenced until the worker is done.
        self.pending_copy_ops: list[KernelBlockCopyOp] = []
        self._install_eviction_hook()

        logger.info(
            "Page/kernel block KV cache: page=%d, kernel block=%d (%d pages), "
            "kernel blocks=%d (%d reserved for copy-on-write)",
            self.geometry.page_size,
            self.geometry.kernel_block_size,
            self.geometry.pages_per_kernel_block,
            page_layout_config.num_kernel_blocks,
            page_layout_config.num_reserved,
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

        # Kernel blocks are the scarcer resource: a request pins whole blocks
        # while filling only part of their page capacity, so they run out while
        # the page pool still looks free. Gate on them here, before upstream
        # commits page ids, so exhaustion becomes a scheduling outcome upstream
        # can act on -- preempt and retry -- instead of an exception out of
        # `bind` that has nowhere to go.
        if not self._have_kernel_blocks_for(request, cached_tokens + num_new_tokens):
            return None

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

        self._bind_kernel_blocks(request, cached_tokens)
        return result

    def _have_kernel_blocks_for(self, request: Request, total_tokens: int) -> bool:
        """Can this request reach ``total_tokens`` without exhausting the pool?

        Counts what it already holds, then reclaims retained blocks before
        giving up. The reserve stays out of it: that share exists so a partial
        match of an already-admitted request can find a copy destination, and
        spending it on admission would deadlock exactly the case it protects.
        """
        pages = -(-total_tokens // self.geometry.page_size)
        shortfall = self.geometry.num_kernel_blocks_for_pages(pages) - len(
            self.kernel_blocks.block_table(request.request_id)
        )
        if shortfall <= 0:
            return True
        if self.kernel_blocks.allocator.can_allocate(shortfall):
            return True
        self._reclaim_retained(shortfall)
        return self.kernel_blocks.allocator.can_allocate(shortfall)

    def _bind_kernel_blocks(self, request: Request, cached_tokens: int) -> None:
        """Back the request's pages with kernel blocks, queueing any copies."""
        page_ids = self._page_ids(request)
        if not page_ids:
            return
        num_cached_pages = min(cached_tokens // self.geometry.page_size, len(page_ids))
        try:
            ops = self.kernel_blocks.bind(
                request.request_id, page_ids, num_cached_pages
            )
        except OutOfKernelBlocks:
            # retained kernel blocks are only given up under pressure
            needed = self.geometry.num_kernel_blocks_for_pages(len(page_ids))
            if self._reclaim_retained(needed) == 0:
                raise
            ops = self.kernel_blocks.bind(
                request.request_id, page_ids, num_cached_pages
            )

        for op in ops:
            self.kernel_blocks.table.acquire(op.src_kernel_block_id)
        self.pending_copy_ops.extend(ops)

    def _page_ids(self, request: Request) -> list[int]:
        blocks: list[KVCacheBlock] = self.coordinator.get_blocks(request.request_id)[0]
        return [block.block_id for block in blocks]

    # -- copy op plumbing ---------------------------------------------------

    def drain_pending_copy_ops(self) -> list[KernelBlockCopyOp]:
        """Copies for this step; sources stay referenced until released."""
        ops = self.pending_copy_ops
        self.pending_copy_ops = []
        return ops

    def release_copy_ops(self, ops: list[KernelBlockCopyOp]) -> None:
        """Drop the source references held by drained copy ops."""
        for op in ops:
            if self.kernel_blocks.table.get(op.src_kernel_block_id) is not None:
                self.kernel_blocks.table.release(op.src_kernel_block_id)

    def block_table(self, request_id: str) -> list[int]:
        """The worker's block table: kernel block ids backing a request."""
        return self.kernel_blocks.block_table(request_id)

    # -- lifetime -----------------------------------------------------------

    def free(self, request: Request) -> None:
        preempted = request.num_computed_tokens == 0
        self.kernel_blocks.free_request(request.request_id, preempted=preempted)
        super().free(request)

    def reset_prefix_cache(self) -> bool:
        result = super().reset_prefix_cache()
        if result:
            self.kernel_blocks.reset()
            self.pending_copy_ops.clear()
        return result

    def _on_page_evicted(self, page_id: int) -> None:
        """I7 has no way to punch a hole, so losing one page costs the kernel block.

        CoW can leave several holders; referenced ones are skipped.
        """
        for kernel_block_id in self.kernel_blocks.table.holders(page_id):
            self._reclaim_kernel_block(kernel_block_id, already_evicted=page_id)

    def _reclaim_kernel_block(
        self, kernel_block_id: int, *, already_evicted: int = -1
    ) -> bool:
        """Reclaim an kernel block and drop upstream's claim on the pages it held.

        Reclaiming takes down every page in the kernel block, but upstream still
        lists the siblings as cached and would report hits for bytes that are
        gone -- a miss that recomputes nothing. Evict them there too, unless a
        copy survives in another kernel block.
        """
        kernel_block = self.kernel_blocks.table.get(kernel_block_id)
        if kernel_block is None or kernel_block.ref_cnt > 0:
            return False
        siblings = [
            page_id
            for page_id in kernel_block.page_ids
            if page_id not in (already_evicted, INVALID_PAGE)
        ]
        if not self.kernel_blocks.reclaim(kernel_block_id):
            return False
        for page_id in siblings:
            if not self.kernel_blocks.table.holders(page_id):
                self._evict_upstream_page(page_id)
        return True

    def _reclaim_retained(self, count: int) -> int:
        reclaimed = 0
        for kernel_block_id in self.kernel_blocks.retained_kernel_blocks():
            if reclaimed >= count:
                break
            reclaimed += self._reclaim_kernel_block(kernel_block_id)
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

        def evict_with_kernel_block_reclaim(block: KVCacheBlock) -> bool:
            page_id = block.block_id
            evicted = original_evict(block)
            if evicted:
                self._on_page_evicted(page_id)
            return evicted

        self.block_pool._maybe_evict_cached_block = evict_with_kernel_block_reclaim

    # -- metrics ------------------------------------------------------------

    @property
    def copy_amplification(self) -> float:
        return self.kernel_blocks.copy_amplification
