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

"""A block pool whose allocation unit is the group (one kernel block).

Page ids are partitioned statically: kernel block ``k`` owns page ids
``[k * ppe, (k + 1) * ppe)``, so a page id encodes its own physical location and
no page -> (block, slot) table is needed:

    kernel_block = page_id // pages_per_kernel_block
    slot         = page_id %  pages_per_kernel_block

Two things upstream's pool cannot provide: a request's pages must land
contiguously inside one kernel block with a single writer, and the capacity that
gates admission must be the group, because that is what runs out first. Keeping
those rules in a second allocator beside the pool is what let the two disagree --
upstream kept admitting against free pages while kernel blocks were exhausted,
and the failure surfaced as an exception with nowhere to go. Putting them *in*
the pool makes upstream's own accounting correct: `get_num_free_blocks` is what
`KVCacheManager` consults before allocating.

`KernelBlock` is that group: it holds the ``ppe`` pages, and each page is an
``RBLNKVCacheBlock`` that points back at it. The pointer is the object form of
``page_id // ppe``, not a second map.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import vllm.v1.core.block_pool as block_pool_mod
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

logger = init_logger(__name__)


@dataclass(slots=True)
class RBLNKVCacheBlock(KVCacheBlock):
    """A page that knows which kernel block it lives in.

    Upstream ``KVCacheBlock`` is slotted, so the back-reference cannot be added
    there. The pool constructs these instead; ``kernel_block`` is filled in as
    soon as the ``KernelBlock`` exists.
    """

    kernel_block: KernelBlock = field(init=False)


class KernelBlock:
    """One kernel block: the ``ppe`` consecutive pages that share a DMA unit.

    ``pages`` is the group. Slot ``i`` is ``pages[i]``, and because page ids are
    partitioned statically that page's id is ``index * ppe + i``. State is not
    stored -- it is read off those pages, which is what keeps it from drifting
    out of sync with upstream's refcounts:

        Open      an owner is writing it
        Sealed    no owner, every page cached; shareable, matchable
        Retired   no owner, nothing live; reallocatable

    ``owner`` is single-valued: an Open kernel block has one writer.
    """

    __slots__ = ("index", "owner", "pages")

    def __init__(self, index: int, pages: list[RBLNKVCacheBlock]) -> None:
        self.index = index
        self.pages = pages
        self.owner: str | None = None
        for page in pages:
            page.kernel_block = self

    def free_pages(self) -> list[RBLNKVCacheBlock]:
        """Pages nobody holds. Cached ones count -- allocation evicts them."""
        return [page for page in self.pages if page.ref_cnt == 0 and not page.is_null]

    @property
    def num_cached(self) -> int:
        return sum(page.block_hash is not None for page in self.pages)

    @property
    def is_retired(self) -> bool:
        return self.owner is None and all(
            page.ref_cnt == 0 and not page.is_null for page in self.pages
        )

    def tail_is_uncached(self, from_slot: int) -> bool:
        """Is everything from ``from_slot`` on both free and unpublished?

        A published tail means the block is some *other* continuation of the same
        prefix, so resume is refused.
        """
        return all(
            page.ref_cnt == 0 and not page.is_null and page.block_hash is None
            for page in self.pages[from_slot:]
        )

    def can_resume(self, from_slot: int) -> bool:
        """May a request that already holds slots ``[0, from_slot)`` append here?

        Nobody owns the block, nothing live sits in the tail, and -- the part that
        is easy to get wrong -- the tail carries no cache. A cached tail means the
        block belongs to some *other* continuation of the same prefix, and
        appending would overwrite pages that continuation still matches.

        Dropping this condition looked like a win (it took copies to nearly zero)
        and was in fact catastrophic: with a short shared prefix, `from_slot` is
        1, the block holding it belongs to another conversation, and every turn
        resumed a stranger's block and destroyed its cache. Measured on
        MiniMax-M2.5 multi-turn: every eviction in the run came from that path,
        88% of published hashes were lost, and matches collapsed to one page.
        """
        return self.owner is None and self.tail_is_uncached(from_slot)

    def open(self, request_id: str) -> None:
        """Claim this block so ``request_id``'s next pages continue in it.

        The tail keeps whatever cache it holds; `get_new_blocks` evicts a page as
        it hands it out, which is late enough to be correct and leaves cache that
        the request never reaches intact.
        """
        self.owner = request_id

    def release(self) -> None:
        self.owner = None


class KernelBlockHashToBlockMap(BlockHashToBlockMap):
    """``BlockHashToBlockMap`` that can list every page carrying a hash.

    Upstream's map returns *a* page for a hash. A copy publishes a second page
    under the same hash, and prefix match has to see both so it can pick the
    kernel block that actually continues the run.
    """

    def get_blocks(self, key: BlockHashWithGroupId) -> Iterable[KVCacheBlock]:
        """Every cached page carrying ``key`` (one, or several after a copy)."""
        blocks = self._cache.get(key)
        if blocks is None:
            return ()
        if isinstance(blocks, KVCacheBlock):
            return (blocks,)
        if isinstance(blocks, dict):
            return blocks.values()
        self._unexpected_blocks_type(blocks)
        return ()


class KernelBlockPool(BlockPool):
    """``BlockPool`` whose allocation unit is the kernel block.

    Pages are still the unit upstream schedules, hashes and matches on. Only
    *where* a page lives, and *how many* are considered free, change.
    """

    cached_block_hash_to_block: KernelBlockHashToBlockMap

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
        # BlockPool constructs KVCacheBlock(idx) from its own import. Swap in
        # the subclass so every page can hold the kernel-block back-reference.
        original = block_pool_mod.KVCacheBlock
        block_pool_mod.KVCacheBlock = RBLNKVCacheBlock
        try:
            super().__init__(usable, enable_caching, hash_block_size, **kwargs)
        finally:
            block_pool_mod.KVCacheBlock = original
        self.cached_block_hash_to_block = KernelBlockHashToBlockMap()

        self.pages_per_kernel_block = pages_per_kernel_block
        self.num_kernel_blocks = usable // pages_per_kernel_block
        pages = cast(list[RBLNKVCacheBlock], self.blocks)
        self.kernel_blocks = [
            KernelBlock(index, pages[base : base + pages_per_kernel_block])
            for index, base in enumerate(range(0, usable, pages_per_kernel_block))
        ]
        self._allocating_for: str | None = None
        # Mirror of `KernelBlock.owner`, so the questions asked per allocation --
        # which group is this request writing into, how many slots does it still
        # have -- cost the size of one request's run instead of the whole pool.
        # `open_group`/`release_group` are the only two mutators of either side.
        self._groups_by_owner: dict[str, set[KernelBlock]] = {}
        self.num_hash_inserts = 0
        self.num_evictions = 0
        if usable != num_gpu_blocks:
            logger.info(
                "Kernel block pool: dropped %d trailing pages that cannot form "
                "a whole kernel block of %d pages.",
                num_gpu_blocks - usable,
                pages_per_kernel_block,
            )

    # -- accounting probes ---------------------------------------------------

    def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens) -> None:
        self.num_hash_inserts += 1
        super()._insert_block_hash(block_hash_with_group_id, block, num_tokens)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        evicted = super()._maybe_evict_cached_block(block)
        self.num_evictions += evicted
        return evicted

    # -- geometry -----------------------------------------------------------

    def kernel_block_of(self, page_id: int) -> int:
        return page_id // self.pages_per_kernel_block

    def slot_of(self, page_id: int) -> int:
        return page_id % self.pages_per_kernel_block

    def page_ids_of(self, kernel_block: int) -> range:
        base = kernel_block * self.pages_per_kernel_block
        return range(base, base + self.pages_per_kernel_block)

    def page(self, page_id: int) -> RBLNKVCacheBlock:
        """The page at ``page_id``, already bound to its kernel block."""
        block = self.blocks[page_id]
        assert isinstance(block, RBLNKVCacheBlock)
        return block

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

    def _open_run_for(
        self, request_id: str | None, taken: set[int] | None = None
    ) -> KernelBlock | None:
        """The kernel block this request is already writing into, if any."""
        if request_id is None:
            return None
        taken = taken or set()
        for group in self._groups_by_owner.get(request_id, ()):
            if any(page.block_id not in taken for page in group.free_pages()):
                return group
        return None

    def num_idle_kernel_blocks(self) -> int:
        """Retired kernel blocks -- the capacity a new run can be opened in.

        Counting them all costs the pool, and `is_retired` reads every page in a
        group, so prefer `has_idle_kernel_blocks` wherever a comparison is what
        the caller actually wants. This exact count is for stats and for
        `get_usage`, which run once per tick rather than per allocation.
        """
        return sum(group.is_retired for group in self.kernel_blocks)

    def has_idle_kernel_blocks(self, at_least: int) -> bool:
        """Are there ``at_least`` retired groups? Stops as soon as it knows.

        Every admission check is a comparison, not a census, and `at_least` is
        almost always 0 or 1 -- so this returns after the first idle group
        instead of reading all `pool x ppe` pages. Only a pool with nothing to
        give walks the whole way, which is the path that refuses anyway.
        """
        if at_least <= 0:
            return True
        found = 0
        for group in self.kernel_blocks:
            if group.is_retired:
                found += 1
                if found >= at_least:
                    return True
        return False

    def num_free_pages_in_open(self, owner: str | None = None) -> int:
        """Free slots in Open groups ``owner`` is allowed to write into.

        Defaults to whoever `allocating_for` names, and that is only *its*
        groups: another request's Open group already has a writer, so its free
        slots cannot be touched, and counting them would admit work
        `get_new_blocks` then cannot serve. Outside an allocation, with no owner
        named, every Open group counts -- that is the whole-pool view
        `get_usage` reads.
        """
        owner = owner if owner is not None else self._allocating_for
        return sum(len(group.free_pages()) for group in self._owned_groups(owner))

    def kernel_blocks_needed(self, num_pages: int, owner: str | None = None) -> int:
        """Idle groups that serving ``owner`` ``num_pages`` more would open.

        The owner's own Open groups are used up first, so this is not
        `num_pages / ppe`: a request with three free slots left needs one new
        group for five pages, not two. It has to be asked per request -- one
        request a page short needs a whole group, and four of them need four,
        which no sum of their page counts can express.
        """
        outside = max(0, num_pages - self.num_free_pages_in_open(owner))
        return math.ceil(outside / self.pages_per_kernel_block)

    def num_owned(self) -> int:
        return sum(len(groups) for groups in self._groups_by_owner.values())

    def open_group(self, group: KernelBlock, request_id: str) -> None:
        """Claim ``group`` for ``request_id``, keeping the owner index in step."""
        if group.owner is not None:
            self._groups_by_owner[group.owner].discard(group)
        group.open(request_id)
        self._groups_by_owner.setdefault(request_id, set()).add(group)

    def release_group(self, group: KernelBlock) -> None:
        if group.owner is not None:
            owned = self._groups_by_owner.get(group.owner)
            if owned is not None:
                owned.discard(group)
                if not owned:
                    del self._groups_by_owner[group.owner]
        group.release()

    def _owned_groups(self, owner: str | None):
        """Open groups ``owner`` may write into; every Open group when unnamed."""
        if owner is not None:
            return self._groups_by_owner.get(owner, ())
        return [g for groups in self._groups_by_owner.values() for g in groups]

    def _next_idle_kernel_block(self, taken: set[int]) -> KernelBlock | None:
        """The idle kernel block it costs least to take.

        Taking a block destroys every cached page in it, so a block holding no
        cache is free and one holding cache is not. Prefer the former outright;
        among the rest take the least-cached, lowest index first.

        Breaking the tie by recency instead was tried and measured -- upstream's
        free list already orders cached pages LRU, so it cost nothing to consult.
        Over 12 vs 5 repetitions of a fixed workload it moved the hit rate by
        +0.16 points, 95% CI [-3.60, +3.92]: no effect this instrument can see.
        The lowest-index rule stays because it is the simpler of two equals.

        Ordering by anything else was catastrophic for hit rate: the pool sat 25
        of 27 blocks idle while holding barely 20 cached pages, because each new
        request kept claiming a block that still held cache.
        """
        fallback: tuple[int, KernelBlock] | None = None
        for group in self.kernel_blocks:
            if not group.is_retired:
                continue
            if all(page.block_id in taken for page in group.pages):
                continue
            if group.num_cached == 0:
                return group
            if fallback is None or group.num_cached < fallback[0]:
                fallback = (group.num_cached, group)
        return fallback[1] if fallback else None

    def get_num_free_blocks(self) -> int:
        """Pages that can actually be handed out, in kernel block terms.

        Upstream states capacity in pages -- `get_usage` and `get_new_blocks`
        both read it that way -- so this keeps that unit and only makes the
        number honest: reporting the raw free-page count is what let
        `KVCacheManager` admit work the kernel block pool could not back.
        Admission itself is decided in groups, by `kernel_blocks_needed`.
        """
        return (
            self.num_free_pages_in_open()
            + self.num_idle_kernel_blocks() * self.pages_per_kernel_block
        )

    def _claim(self, group: KernelBlock) -> None:
        """Drop the cache still held in a block a request is about to write into.

        Leaving it would let another request match a page *ahead* of the owner's
        write pointer and take a reference on it. The owner would then have to
        skip that slot, putting a hole in its run, and the block table addresses
        slots positionally (slot = page index % ppe), so a hole is silent
        corruption rather than waste. Upstream evicts the same span when it takes
        a block at this size, so this is no coarser than the alternative.
        """
        if not self.enable_caching:
            return
        for page in group.pages:
            if page.block_hash is not None:
                self._maybe_evict_cached_block(page)

    def get_new_blocks(self, num_blocks: int) -> list[RBLNKVCacheBlock]:
        """Serve pages from this request's open kernel block, then fresh ones."""
        if not self.has_idle_kernel_blocks(self.kernel_blocks_needed(num_blocks)):
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        request_id = self._allocating_for
        chosen: list[int] = []
        taken: set[int] = set()  # ref_cnt is only bumped below, so track here
        while len(chosen) < num_blocks:
            group = self._open_run_for(request_id, taken)
            if group is None:
                group = self._next_idle_kernel_block(taken)
                if group is None:
                    raise ValueError(
                        f"Cannot get {num_blocks} free blocks from the pool"
                    )
                self._claim(group)
                if request_id is not None:
                    self.open_group(group, request_id)
            available = [
                page.block_id
                for page in group.free_pages()
                if page.block_id not in taken
            ]
            if not available:
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            # A gap means someone took a slot ahead of this request's write
            # pointer. Slots are addressed positionally, so serving across the
            # gap would silently point attention at the wrong page.
            if available != list(range(available[0], available[0] + len(available))):
                raise ValueError(
                    f"kernel block {group.index} has a hole in its free run: "
                    f"{available}"
                )
            picked = available[: num_blocks - len(chosen)]
            chosen.extend(picked)
            taken.update(picked)

        blocks = [self.page(page_id) for page_id in chosen]
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
        for group in {cast(RBLNKVCacheBlock, block).kernel_block for block in blocks}:
            if all(page.ref_cnt == 0 for page in group.pages):
                self.release_group(group)

    def publish_copy(
        self, source: RBLNKVCacheBlock, destination: RBLNKVCacheBlock
    ) -> None:
        """Give a copied page the identity of the page it was copied from.

        Without this the copy is unmatchable: upstream counts the range as
        already cached, so it never hashes the destination, and the only page
        carrying that hash stays in the producer's block. A later turn of the same
        conversation then matches the producer's page at index 0 and its own page
        at index 1, which is not a run, so the whole match collapses to one page.
        Upstream keeps several blocks per hash by design, so publishing a second
        one is a supported state.
        """
        if not self.enable_caching:
            return
        block_hash = source.block_hash
        if block_hash is None:
            return
        self._insert_block_hash(block_hash, destination, source.block_hash_num_tokens)

    def release_owner(self, request_id: str) -> None:
        """Give up the request's open runs so the next turn can adopt them.

        A block's cached head often outlives its writer, so waiting for every
        page to go free would pin ownership for as long as the prefix stays
        cached and make the block unadoptable.
        """
        for group in list(self._groups_by_owner.get(request_id, ())):
            self.release_group(group)

    def reset_ownership(self) -> None:
        for group in self.kernel_blocks:
            group.release()
        self._groups_by_owner.clear()
        self._allocating_for = None
