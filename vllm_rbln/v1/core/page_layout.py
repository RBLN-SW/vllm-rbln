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

"""Page/kernel block KV addressing. See docs/page_layout_kv_manager.md.

Two maps, and conflating them is a correctness bug:
  request -> [kernel_block_id]   the address the worker uses
  page_id -> {kernel_block_id}   a content locator for finding a copy source;
                           many-valued, because CoW duplicates content
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

__all__ = [
    "ATTN_BLOCK_SIZE_KEY",
    "INVALID_PAGE",
    "KernelBlock",
    "KernelBlockAllocator",
    "KernelBlockCopyOp",
    "PageLayout",
    "KernelBlockTable",
    "OutOfKernelBlocks",
    "PageLayoutConfig",
    "PageLayoutManager",
    "kernel_block_size_from_config",
    "reserve_kernel_blocks",
    "resolve_config",
    "validate_fragmentation",
]

# Slot whose page id upstream recycled: the bytes no longer identify a live page.
INVALID_PAGE = -1

# Where the compiled model publishes its kernel block size.
ATTN_BLOCK_SIZE_KEY = "attn_block_size"

# Pool share withheld for CoW destinations; without one, a partial match cannot
# be serviced at all.
DEFAULT_RESERVE_FRACTION = 0.05


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageLayout:
    page_size: int
    kernel_block_size: int

    def __post_init__(self) -> None:
        if self.page_size <= 0 or self.kernel_block_size <= 0:
            raise ValueError(
                f"sizes must be positive, got page={self.page_size} "
                f"kernel_block={self.kernel_block_size}"
            )
        if self.kernel_block_size % self.page_size != 0:
            raise ValueError(
                f"kernel_block_size ({self.kernel_block_size}) must be a multiple of "
                f"page_size ({self.page_size})"
            )

    @property
    def pages_per_kernel_block(self) -> int:
        return self.kernel_block_size // self.page_size

    @property
    def is_degenerate(self) -> bool:
        """One page per kernel block: the layer is a no-op."""
        return self.pages_per_kernel_block == 1

    def validate_chunk(self, chunk_size: int) -> None:
        """I2: a prefill step must never straddle a page boundary."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.page_size % chunk_size != 0:
            raise ValueError(
                f"page_size ({self.page_size}) must be a multiple of the "
                f"prefill chunk ({chunk_size}) so a prefill step never spans "
                f"two pages"
            )

    def num_kernel_blocks_for_pages(self, num_pages: int) -> int:
        return -(-num_pages // self.pages_per_kernel_block)

    def slot(self, page_index: int) -> int:
        """I3: sequential writes make the slot a function of the index alone."""
        return page_index % self.pages_per_kernel_block


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageLayoutConfig:
    geometry: PageLayout
    num_kernel_blocks: int
    num_reserved: int

    @property
    def enabled(self) -> bool:
        return not self.geometry.is_degenerate


def reserve_kernel_blocks(
    num_kernel_blocks: int, fraction: float = DEFAULT_RESERVE_FRACTION
) -> int:
    """At least one (a CoW needs a destination), never the whole pool."""
    if num_kernel_blocks <= 1:
        return 0
    return max(1, min(num_kernel_blocks - 1, int(num_kernel_blocks * fraction)))


def resolve_config(
    page_size: int,
    kernel_block_size: int | None,
    num_pages: int,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> PageLayoutConfig:
    """``kernel_block_size=None`` yields a no-op geometry."""
    geometry = PageLayout(page_size, kernel_block_size or page_size)
    num_kernel_blocks = max(1, num_pages // geometry.pages_per_kernel_block)
    return PageLayoutConfig(
        geometry,
        num_kernel_blocks,
        0
        if geometry.is_degenerate
        else reserve_kernel_blocks(num_kernel_blocks, reserve_fraction),
    )


def kernel_block_size_from_config(vllm_config: VllmConfig) -> int | None:
    additional: dict[str, Any] | None = getattr(vllm_config, "additional_config", None)
    value = additional.get(ATTN_BLOCK_SIZE_KEY) if additional else None
    return int(value) if value else None


def validate_fragmentation(
    geometry: PageLayout,
    max_num_seqs: int,
    num_kernel_blocks: int,
    max_model_len: int | None = None,
) -> None:
    """A running request pins every kernel block it spans, not just one.

    Only its last block is partly filled, but all of them are held, so demand
    scales with request length. Sizing for one block per sequence under-counts
    by that factor and lets a pool through that then runs out at runtime.
    """
    if geometry.is_degenerate:
        return
    blocks_per_seq = (
        1 if not max_model_len else -(-max_model_len // geometry.kernel_block_size)
    )
    peak = max_num_seqs * blocks_per_seq
    if peak >= num_kernel_blocks:
        raise ValueError(
            f"max_num_seqs ({max_num_seqs}) x {blocks_per_seq} kernel blocks per "
            f"request at max_model_len needs {peak} kernel blocks, but the pool "
            f"holds only {num_kernel_blocks} of {geometry.kernel_block_size} "
            f"tokens; lower max_num_seqs or max_model_len, or raise the KV cache size"
        )
    if peak / num_kernel_blocks > 0.5:
        logger.warning(
            "Page layout: up to %d of %d kernel blocks can be pinned at once "
            "(%d requests x %d blocks each at max_model_len).",
            peak,
            num_kernel_blocks,
            max_num_seqs,
            blocks_per_seq,
        )


# --------------------------------------------------------------------------- #
# KernelBlock pool
# --------------------------------------------------------------------------- #


class OutOfKernelBlocks(RuntimeError):
    """No kernel block can be allocated, reserve included."""


class KernelBlockAllocator:
    """Free list over a fixed pool; I7 makes reclaim whole-kernel block."""

    def __init__(self, num_kernel_blocks: int, num_reserved: int = 0) -> None:
        if num_kernel_blocks <= 0:
            raise ValueError(
                f"num_kernel_blocks must be positive, got {num_kernel_blocks}"
            )
        if not 0 <= num_reserved < num_kernel_blocks:
            raise ValueError(
                f"num_reserved ({num_reserved}) must be in [0, {num_kernel_blocks})"
            )
        self.num_kernel_blocks = num_kernel_blocks
        self.num_reserved = num_reserved
        self._free: deque[int] = deque(range(num_kernel_blocks))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocatable(self) -> int:
        return max(0, len(self._free) - self.num_reserved)

    def can_allocate(self, count: int, *, urgent: bool = False) -> bool:
        return (self.num_free if urgent else self.num_allocatable) >= count

    def allocate(self, *, urgent: bool = False) -> int:
        """``urgent`` (the CoW destination path) may dip into the reserve."""
        if not self.can_allocate(1, urgent=urgent):
            raise OutOfKernelBlocks(
                f"no kernel block available (free={self.num_free}, "
                f"reserved={self.num_reserved}, urgent={urgent})"
            )
        kernel_block_id = self._free.popleft()
        self._allocated.add(kernel_block_id)
        return kernel_block_id

    def free(self, kernel_block_id: int) -> None:
        if kernel_block_id not in self._allocated:
            raise ValueError(f"kernel block {kernel_block_id} is not allocated")
        self._allocated.discard(kernel_block_id)
        self._free.appendleft(kernel_block_id)

    def reset(self) -> None:
        self._free = deque(range(self.num_kernel_blocks))
        self._allocated.clear()


# --------------------------------------------------------------------------- #
# Page -> kernel block map
# --------------------------------------------------------------------------- #


@dataclass
class KernelBlock:
    kernel_block_id: int
    # slot -> page id; append-only (I3), so len() is the write pointer
    page_ids: list[int] = field(default_factory=list)
    # memory lifetime only, not the hash multiplicity that drives KV events
    ref_cnt: int = 0
    owner: str | None = None
    sealed: bool = False
    # position in the prefix chain; hashes are chained, so reclaiming a shallow
    # kernel block strands every deeper one
    depth: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)


class KernelBlockTable:
    """Page <-> kernel block map plus per-request ownership.

    Lifetime: live (ref_cnt > 0), retained (0 but a hit can revive it),
    reclaimed (evicted, id returned to the pool).
    """

    def __init__(self, pages_per_kernel_block: int) -> None:
        self.pages_per_kernel_block = pages_per_kernel_block
        self._kernel_blocks: dict[int, KernelBlock] = {}
        self._page_holders: dict[int, set[int]] = {}
        self._request_kernel_blocks: dict[str, list[int]] = {}

    # -- kernel blocks ------------------------------------------------------------

    def create(
        self, kernel_block_id: int, owner: str | None = None, depth: int = 0
    ) -> KernelBlock:
        if kernel_block_id in self._kernel_blocks:
            raise ValueError(f"kernel block {kernel_block_id} already exists")
        kernel_block = self._kernel_blocks[kernel_block_id] = KernelBlock(
            kernel_block_id, owner=owner, depth=depth
        )
        return kernel_block

    def get(self, kernel_block_id: int) -> KernelBlock | None:
        return self._kernel_blocks.get(kernel_block_id)

    def require(self, kernel_block_id: int) -> KernelBlock:
        if (kernel_block := self._kernel_blocks.get(kernel_block_id)) is None:
            raise KeyError(f"kernel block {kernel_block_id} is not mapped")
        return kernel_block

    def remove(self, kernel_block_id: int) -> KernelBlock | None:
        kernel_block = self._kernel_blocks.pop(kernel_block_id, None)
        if kernel_block is not None:
            for page_id in kernel_block.page_ids:
                if holders := self._page_holders.get(page_id):
                    holders.discard(kernel_block_id)
                    if not holders:
                        del self._page_holders[page_id]
        return kernel_block

    def seal(self, kernel_block_id: int) -> None:
        kernel_block = self.require(kernel_block_id)
        if kernel_block.num_pages != self.pages_per_kernel_block:
            raise ValueError(
                f"kernel block {kernel_block_id} holds "
                f"{kernel_block.num_pages} pages; only a full kernel block "
                f"({self.pages_per_kernel_block}) may be sealed"
            )
        kernel_block.sealed = True
        kernel_block.owner = None

    # -- pages --------------------------------------------------------------

    def append_page(self, kernel_block_id: int, page_id: int, *, fresh: bool) -> int:
        """Bind ``page_id`` to the next slot.

        ``fresh``: upstream just handed the id out for new content, so any other
        kernel block claiming it holds stale bytes.
        """
        kernel_block = self.require(kernel_block_id)
        if kernel_block.num_pages >= self.pages_per_kernel_block:
            raise ValueError(
                f"kernel block {kernel_block_id} is full "
                f"({self.pages_per_kernel_block} pages); seal it and open a new one"
            )
        if fresh:
            self._revoke_stale_claims(page_id, kernel_block_id)
        slot = kernel_block.num_pages
        kernel_block.page_ids.append(page_id)
        self._page_holders.setdefault(page_id, set()).add(kernel_block_id)
        return slot

    def truncate(self, kernel_block_id: int, keep: int) -> None:
        """Rewind the write pointer to ``keep`` slots, dropping their claims.

        Only ever called for slots upstream handed out for fresh writing, so no
        one can be matching the pages dropped here (same reasoning as
        ``_revoke_stale_claims``).
        """
        kernel_block = self.require(kernel_block_id)
        for page_id in kernel_block.page_ids[keep:]:
            if holders := self._page_holders.get(page_id):
                holders.discard(kernel_block_id)
                if not holders:
                    del self._page_holders[page_id]
        del kernel_block.page_ids[keep:]

    def _revoke_stale_claims(self, page_id: int, new_kernel_block_id: int) -> None:
        """Poison, don't remove: removing shifts later slots and breaks I3."""
        holders = self._page_holders.get(page_id)
        if not holders:
            return
        for holder_id in list(holders):
            if holder_id == new_kernel_block_id:
                continue
            if (holder := self._kernel_blocks.get(holder_id)) is not None:
                holder.page_ids = [
                    INVALID_PAGE if pid == page_id else pid for pid in holder.page_ids
                ]
            holders.discard(holder_id)

    def locate(self, page_id: int) -> tuple[int, int] | None:
        """A copy source, not an address; any holder will do."""
        if page_id == INVALID_PAGE:
            return None
        for kernel_block_id in self._page_holders.get(page_id, ()):
            kernel_block = self._kernel_blocks.get(kernel_block_id)
            if kernel_block is not None and page_id in kernel_block.page_ids:
                return kernel_block_id, kernel_block.page_ids.index(page_id)
        return None

    def holders(self, page_id: int) -> set[int]:
        return set(self._page_holders.get(page_id, ()))

    # -- references and ownership -------------------------------------------

    def acquire(self, kernel_block_id: int) -> None:
        self.require(kernel_block_id).ref_cnt += 1

    def release(self, kernel_block_id: int) -> int:
        kernel_block = self.require(kernel_block_id)
        if kernel_block.ref_cnt <= 0:
            raise ValueError(f"kernel block {kernel_block_id} refcount underflow")
        kernel_block.ref_cnt -= 1
        return kernel_block.ref_cnt

    def attach_to_request(self, request_id: str, kernel_block_id: int) -> None:
        kernel_blocks = self._request_kernel_blocks.setdefault(request_id, [])
        if kernel_block_id not in kernel_blocks:
            kernel_blocks.append(kernel_block_id)
            self.acquire(kernel_block_id)

    def request_kernel_blocks(self, request_id: str) -> list[int]:
        return list(self._request_kernel_blocks.get(request_id, ()))

    def detach_request(self, request_id: str) -> list[int]:
        """Release a request's kernel blocks; returns those now retained."""
        retained = []
        for kernel_block_id in self._request_kernel_blocks.pop(request_id, []):
            kernel_block = self._kernel_blocks.get(kernel_block_id)
            if kernel_block is not None and self.release(kernel_block_id) == 0:
                kernel_block.owner = None
                retained.append(kernel_block_id)
        return retained

    def reset(self) -> None:
        self._kernel_blocks.clear()
        self._page_holders.clear()
        self._request_kernel_blocks.clear()


# --------------------------------------------------------------------------- #
# Binding policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelBlockCopyOp:
    """A token range to copy before the forward pass.

    In tokens, not slots: page size is scheduler-side and the worker never
    needs it.
    """

    src_kernel_block_id: int
    dst_kernel_block_id: int
    src_start: int
    dst_start: int
    num_tokens: int


class PageLayoutManager:
    """Owns the kernel block pool and decides attach-vs-copy per request.

    I4-I7: a fully matched kernel block attaches by reference, an unreferenced partial
    one is adopted and extended in place, anything else is copied into a private
    kernel block, reclaim is whole-block. Splitting the match at kernel-block
    boundaries bounds the copy by one block however long the match.
    """

    def __init__(
        self, geometry: PageLayout, num_kernel_blocks: int, num_reserved: int = 0
    ) -> None:
        self.geometry = geometry
        self.allocator = KernelBlockAllocator(num_kernel_blocks, num_reserved)
        self.table = KernelBlockTable(geometry.pages_per_kernel_block)
        self.num_pages_copied = 0
        self.num_pages_written = 0

    def bind(
        self, request_id: str, page_ids: list[int], num_cached_pages: int
    ) -> list[KernelBlockCopyOp]:
        """Back a request's pages with kernel blocks, returning the copies implied.

        ``num_cached_pages``: leading pages whose content already exists; the
        rest the forward pass writes.
        """
        ppe = self.geometry.pages_per_kernel_block
        existing = self.table.request_kernel_blocks(request_id)
        ops: list[KernelBlockCopyOp] = []

        for ext_idx in range(self.geometry.num_kernel_blocks_for_pages(len(page_ids))):
            group = page_ids[ext_idx * ppe : (ext_idx + 1) * ppe]
            base = ext_idx * ppe

            if ext_idx < len(existing):
                kernel_block_id = existing[ext_idx]
            else:
                attach_id = self._attachable(group, base, num_cached_pages)
                if attach_id is not None:
                    self.table.attach_to_request(request_id, attach_id)
                    continue
                adopt_id = self._adoptable(group, base, num_cached_pages)
                if adopt_id is None:
                    kernel_block_id = self.allocator.allocate(
                        urgent=base < num_cached_pages
                    )
                    self.table.create(kernel_block_id, depth=ext_idx)
                else:
                    kernel_block_id = adopt_id
                    # Rewind to the cached prefix; the rest this request rewrites.
                    self.table.truncate(
                        kernel_block_id,
                        self._cached_in_group(group, base, num_cached_pages),
                    )
                self.table.attach_to_request(request_id, kernel_block_id)
                self.table.require(kernel_block_id).owner = request_id

            written = self.table.require(kernel_block_id).num_pages
            if written >= len(group):
                continue
            pending = [
                op
                for slot in range(written, len(group))
                for op in self._place(
                    group[slot], kernel_block_id, slot, base + slot, num_cached_pages
                )
            ]
            self._maybe_seal(kernel_block_id)
            ops.extend(_coalesce(pending))

        return ops

    def _place(
        self,
        page_id: int,
        dst_kernel_block_id: int,
        dst_slot: int,
        page_index: int,
        num_cached_pages: int,
    ) -> list[KernelBlockCopyOp]:
        """Bind one page, copying when its data lives elsewhere."""
        fresh = page_index >= num_cached_pages
        ops: list[KernelBlockCopyOp] = []
        if fresh:
            self.num_pages_written += 1
        elif (source := self.table.locate(page_id)) is not None:
            src_kernel_block_id, src_slot = source
            if src_kernel_block_id != dst_kernel_block_id:
                page = self.geometry.page_size
                ops.append(
                    KernelBlockCopyOp(
                        src_kernel_block_id,
                        dst_kernel_block_id,
                        src_slot * page,
                        dst_slot * page,
                        page,
                    )
                )
                self.num_pages_copied += 1
        self.table.append_page(dst_kernel_block_id, page_id, fresh=fresh)
        return ops

    def _attachable(
        self, group: list[int], base: int, num_cached_pages: int
    ) -> int | None:
        """I5: a sealed kernel block holding exactly this group, shared read-only."""
        if len(group) != self.geometry.pages_per_kernel_block:
            return None
        if base + len(group) > num_cached_pages:
            return None
        for candidate in self.table.holders(group[0]):
            kernel_block = self.table.get(candidate)
            if (
                kernel_block is not None
                and kernel_block.sealed
                and kernel_block.page_ids == group
            ):
                return candidate
        return None

    def _adoptable(
        self, group: list[int], base: int, num_cached_pages: int
    ) -> int | None:
        """I5b: a retained partial kernel block this request can keep writing into.

        The multi-turn case: a request ended mid-kernel block and the next one resumes
        that prefix and extends it. Appending in place is safe precisely when
        nobody else references the kernel block -- the resumed pages keep their slots,
        so I3 holds and the sharer I4 protects does not exist. Two requests
        extending one prefix still copy: the first adopts, the second sees a live
        refcount.

        A slot may hold the group's page or ``INVALID_PAGE`` (poisoned when that
        page was last written elsewhere); either way the request rewrites it,
        since only its cached pages are kept.
        """
        cached = self._cached_in_group(group, base, num_cached_pages)
        if cached == 0:
            return None
        for candidate in self.table.holders(group[0]):
            kernel_block = self.table.get(candidate)
            if (
                kernel_block is None
                or kernel_block.ref_cnt > 0
                or kernel_block.sealed
                or kernel_block.num_pages > len(group)
                or kernel_block.num_pages < cached
            ):
                continue
            if all(
                held in (group[slot], INVALID_PAGE)
                for slot, held in enumerate(kernel_block.page_ids)
            ):
                return candidate
        return None

    def _cached_in_group(
        self, group: list[int], base: int, num_cached_pages: int
    ) -> int:
        """Leading pages of this group whose content already exists."""
        return max(0, min(len(group), num_cached_pages - base))

    def _maybe_seal(self, kernel_block_id: int) -> None:
        kernel_block = self.table.get(kernel_block_id)
        if (
            kernel_block is not None
            and not kernel_block.sealed
            and kernel_block.num_pages == self.geometry.pages_per_kernel_block
        ):
            self.table.seal(kernel_block_id)

    # -- lifetime -----------------------------------------------------------

    def block_table(self, request_id: str) -> list[int]:
        return self.table.request_kernel_blocks(request_id)

    def free_request(self, request_id: str, *, preempted: bool = False) -> list[int]:
        """Preempted recomputes anyway, so reclaim; finished stays retained."""
        retained = self.table.detach_request(request_id)
        if not preempted:
            return []
        return [e for e in retained if self.reclaim(e)]

    def reclaim(self, kernel_block_id: int) -> bool:
        kernel_block = self.table.get(kernel_block_id)
        if kernel_block is None or kernel_block.ref_cnt > 0:
            return False
        self.table.remove(kernel_block_id)
        self.allocator.free(kernel_block_id)
        return True

    def reclaim_retained(self, count: int) -> int:
        """Give up unreferenced kernel blocks under pressure."""
        reclaimed = 0
        for kernel_block_id in self.retained_kernel_blocks():
            if reclaimed >= count:
                break
            reclaimed += self.reclaim(kernel_block_id)
        return reclaimed

    def retained_kernel_blocks(self) -> list[int]:
        """Unreferenced kernel blocks, deepest first: reclaiming a shallow kernel block
        strands every deeper one on the same chain."""
        retained = [
            kernel_block
            for kernel_block_id in range(self.allocator.num_kernel_blocks)
            if (kernel_block := self.table.get(kernel_block_id)) is not None
            and kernel_block.ref_cnt == 0
        ]
        retained.sort(key=lambda e: e.depth, reverse=True)
        return [e.kernel_block_id for e in retained]

    def reset(self) -> None:
        self.table.reset()
        self.allocator.reset()
        self.num_pages_copied = self.num_pages_written = 0

    @property
    def copy_amplification(self) -> float:
        """Copied pages per freshly computed page."""
        if self.num_pages_written == 0:
            return 0.0
        return self.num_pages_copied / self.num_pages_written


def _coalesce(ops: list[KernelBlockCopyOp]) -> list[KernelBlockCopyOp]:
    """Merge copies contiguous in both source and destination."""
    merged: list[KernelBlockCopyOp] = []
    for op in ops:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.src_kernel_block_id == op.src_kernel_block_id
            and prev.dst_kernel_block_id == op.dst_kernel_block_id
            and prev.src_start + prev.num_tokens == op.src_start
            and prev.dst_start + prev.num_tokens == op.dst_start
        ):
            merged[-1] = KernelBlockCopyOp(
                prev.src_kernel_block_id,
                prev.dst_kernel_block_id,
                prev.src_start,
                prev.dst_start,
                prev.num_tokens + op.num_tokens,
            )
        else:
            merged.append(op)
    return merged
