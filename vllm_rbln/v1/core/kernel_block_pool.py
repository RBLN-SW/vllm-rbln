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

"""A block pool that hands out pages in kernel-block-aligned runs.

Page layout needs two things upstream's pool does not provide: a request's
pages must land contiguously inside one kernel block (I3), and the capacity
that gates admission must be the kernel block, because that is what runs out
first. Keeping those rules in a second allocator beside the pool is what let
the two disagree -- upstream kept admitting against free pages while kernel
blocks were exhausted, and the failure surfaced as an exception with nowhere
to go. Putting them *in* the pool makes upstream's own accounting correct:
`get_num_free_blocks` is what `KVCacheManager` consults before allocating.

Page ids are partitioned statically: kernel block ``k`` owns page ids
``[k * ppe, (k + 1) * ppe)``. A page id therefore encodes its own physical
location, so no separate page -> (block, slot) table is needed:

    kernel_block = page_id // pages_per_kernel_block
    slot         = page_id %  pages_per_kernel_block
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING

from vllm.v1.core.block_pool import BlockPool

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class KernelBlockPool(BlockPool):
    """``BlockPool`` whose allocation unit is the kernel block.

    Pages are still the unit upstream schedules, hashes and matches on. Only
    *where* a page lives, and *how many* are considered free, change.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        pages_per_kernel_block: int,
        **kwargs,
    ) -> None:
        if pages_per_kernel_block < 1:
            raise ValueError(
                f"pages_per_kernel_block must be >= 1, got {pages_per_kernel_block}"
            )
        # Whole kernel blocks only: a trailing partial run could never satisfy
        # a request without breaking contiguity, so it is not offered at all.
        usable = (num_gpu_blocks // pages_per_kernel_block) * pages_per_kernel_block
        if usable < pages_per_kernel_block:
            raise ValueError(
                f"{num_gpu_blocks} pages is less than one kernel block of "
                f"{pages_per_kernel_block} pages"
            )
        super().__init__(usable, enable_caching, hash_block_size, **kwargs)

        self.pages_per_kernel_block = pages_per_kernel_block
        self.num_kernel_blocks = usable // pages_per_kernel_block
        # Which request each open kernel block belongs to. A kernel block is
        # open while it holds live pages of exactly one request; I4 forbids a
        # second writer, so ownership is single-valued.
        self._owner: dict[int, str] = {}
        self._allocating_for: str | None = None
        if usable != num_gpu_blocks:
            logger.info(
                "Kernel block pool: dropped %d trailing pages that cannot form "
                "a whole kernel block of %d pages.",
                num_gpu_blocks - usable,
                pages_per_kernel_block,
            )

    # -- geometry -----------------------------------------------------------

    def kernel_block_of(self, page_id: int) -> int:
        return page_id // self.pages_per_kernel_block

    def slot_of(self, page_id: int) -> int:
        return page_id % self.pages_per_kernel_block

    def page_ids_of(self, kernel_block: int) -> range:
        base = kernel_block * self.pages_per_kernel_block
        return range(base, base + self.pages_per_kernel_block)

    # -- allocation ---------------------------------------------------------

    @contextmanager
    def allocating_for(self, request_id: str) -> Iterator[None]:
        """Attribute the pages taken inside this block to ``request_id``.

        `get_new_blocks` is called several frames below `allocate_slots` and is
        not told who is asking, but ownership is what keeps one kernel block to
        one writer, so the caller states it here.
        """
        previous = self._allocating_for
        self._allocating_for = request_id
        try:
            yield
        finally:
            self._allocating_for = previous

    def _free_page_ids_in(self, kernel_block: int) -> list[int]:
        # Cached-but-unreferenced pages count: they sit in the free queue and
        # `get_new_blocks` evicts them, exactly as upstream does.
        return [
            page_id
            for page_id in self.page_ids_of(kernel_block)
            if self.blocks[page_id].ref_cnt == 0 and not self.blocks[page_id].is_null
        ]

    # -- adoption -----------------------------------------------------------

    def owner_of(self, kernel_block: int) -> str | None:
        return self._owner.get(kernel_block)

    def can_resume(self, kernel_block: int, from_slot: int) -> bool:
        """May a request that already holds slots ``[0, from_slot)`` append here?

        Only when nobody owns the block and its remaining slots hold nothing
        worth keeping. A cached tail means the producer filled past this point
        and other requests can still match it, so appending would destroy live
        cache; the caller copies instead.
        """
        if kernel_block in self._owner:
            return False
        return all(
            self.blocks[page_id].ref_cnt == 0
            and not self.blocks[page_id].is_null
            and self.blocks[page_id].block_hash is None
            for page_id in self.page_ids_of(kernel_block)[from_slot:]
        )

    def open_run(self, request_id: str, kernel_block: int) -> None:
        """Claim ``kernel_block`` so this request's next pages continue in it."""
        self._owner[kernel_block] = request_id

    def _open_run_for(
        self, request_id: str | None, taken: set[int] | None = None
    ) -> int | None:
        """The kernel block this request is already writing into, if any."""
        if request_id is None:
            return None
        taken = taken or set()
        for kernel_block, owner in self._owner.items():
            if owner != request_id:
                continue
            if any(p not in taken for p in self._free_page_ids_in(kernel_block)):
                return kernel_block
        return None

    def _idle_kernel_blocks(self) -> list[int]:
        """Kernel blocks with no live page and no owner: allocatable whole."""
        idle = []
        for kernel_block in range(self.num_kernel_blocks):
            if kernel_block in self._owner:
                continue
            if all(
                self.blocks[p].ref_cnt == 0 and not self.blocks[p].is_null
                for p in self.page_ids_of(kernel_block)
            ):
                idle.append(kernel_block)
        return idle

    def get_num_free_blocks(self) -> int:
        """Pages that can actually be handed out, in kernel block terms.

        Upstream's `KVCacheManager` gates admission on this, so reporting the
        raw free-page count is what let it admit work the kernel block pool
        could not back. Inside `allocating_for` the answer is narrowed to what
        *that* request can take: another request's open run has free slots I4
        forbids it from touching, and counting them would admit work that
        `get_new_blocks` then cannot serve.
        """
        free_in_open = sum(
            len(self._free_page_ids_in(kernel_block))
            for kernel_block, owner in self._owner.items()
            if self._allocating_for is None or owner == self._allocating_for
        )
        return free_in_open + len(self._idle_kernel_blocks()) * (
            self.pages_per_kernel_block
        )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Serve pages from this request's open kernel block, then fresh ones."""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        request_id = self._allocating_for
        chosen: list[int] = []
        taken: set[int] = set()  # ref_cnt is only bumped below, so track here
        while len(chosen) < num_blocks:
            kernel_block = self._open_run_for(request_id, taken)
            if kernel_block is None:
                idle = [
                    k
                    for k in self._idle_kernel_blocks()
                    if any(p not in taken for p in self.page_ids_of(k))
                ]
                if not idle:
                    raise ValueError(
                        f"Cannot get {num_blocks} free blocks from the pool"
                    )
                kernel_block = idle[0]
                if request_id is not None:
                    self._owner[kernel_block] = request_id
            available = [
                p for p in self._free_page_ids_in(kernel_block) if p not in taken
            ]
            if not available:
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            picked = available[: num_blocks - len(chosen)]
            chosen.extend(picked)
            taken.update(picked)

        blocks = [self.blocks[page_id] for page_id in chosen]
        for block in blocks:
            self.free_block_queue.remove(block)
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return blocks

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Release pages, and the kernel block once none of them are live."""
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        for kernel_block in {self.kernel_block_of(b.block_id) for b in blocks}:
            if all(self.blocks[p].ref_cnt == 0 for p in self.page_ids_of(kernel_block)):
                self._owner.pop(kernel_block, None)

    def release_owner(self, request_id: str) -> None:
        """Give up the request's open runs so the next turn can adopt them.

        A block's cached head often outlives its writer, so waiting for every
        page to go free would pin ownership for as long as the prefix stays
        cached and make the block unadoptable.
        """
        for kernel_block in [
            kernel_block
            for kernel_block, owner in self._owner.items()
            if owner == request_id
        ]:
            del self._owner[kernel_block]

    def reset_ownership(self) -> None:
        self._owner.clear()
        self._allocating_for = None
