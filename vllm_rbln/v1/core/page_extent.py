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

"""Page/extent KV addressing. See docs/page_extent_kv_manager.md.

Two maps, and conflating them is a correctness bug:
  request -> [extent_id]   the address the worker uses
  page_id -> {extent_id}   a content locator for finding a copy source;
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
    "Extent",
    "ExtentAllocator",
    "ExtentCopyOp",
    "ExtentGeometry",
    "ExtentTable",
    "OutOfExtents",
    "PageExtentConfig",
    "PageExtentManager",
    "extent_size_from_config",
    "reserve_extents",
    "resolve_config",
    "validate_fragmentation",
]

# Slot whose page id upstream recycled: the bytes no longer identify a live page.
INVALID_PAGE = -1

# Where the compiled model publishes its extent size.
ATTN_BLOCK_SIZE_KEY = "attn_block_size"

# Pool share withheld for CoW destinations; without one, a partial match cannot
# be serviced at all.
DEFAULT_RESERVE_FRACTION = 0.05


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExtentGeometry:
    page_size: int
    extent_size: int

    def __post_init__(self) -> None:
        if self.page_size <= 0 or self.extent_size <= 0:
            raise ValueError(
                f"sizes must be positive, got page={self.page_size} "
                f"extent={self.extent_size}"
            )
        if self.extent_size % self.page_size != 0:
            raise ValueError(
                f"extent_size ({self.extent_size}) must be a multiple of "
                f"page_size ({self.page_size})"
            )

    @property
    def pages_per_extent(self) -> int:
        return self.extent_size // self.page_size

    @property
    def is_degenerate(self) -> bool:
        """One page per extent: the layer is a no-op."""
        return self.pages_per_extent == 1

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

    def num_extents_for_pages(self, num_pages: int) -> int:
        return -(-num_pages // self.pages_per_extent)

    def slot(self, page_index: int) -> int:
        """I3: sequential writes make the slot a function of the index alone."""
        return page_index % self.pages_per_extent


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageExtentConfig:
    geometry: ExtentGeometry
    num_extents: int
    num_reserved: int

    @property
    def enabled(self) -> bool:
        return not self.geometry.is_degenerate


def reserve_extents(
    num_extents: int, fraction: float = DEFAULT_RESERVE_FRACTION
) -> int:
    """At least one (a CoW needs a destination), never the whole pool."""
    if num_extents <= 1:
        return 0
    return max(1, min(num_extents - 1, int(num_extents * fraction)))


def resolve_config(
    page_size: int,
    extent_size: int | None,
    num_pages: int,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> PageExtentConfig:
    """``extent_size=None`` yields a no-op geometry."""
    geometry = ExtentGeometry(page_size, extent_size or page_size)
    num_extents = max(1, num_pages // geometry.pages_per_extent)
    return PageExtentConfig(
        geometry,
        num_extents,
        0 if geometry.is_degenerate else reserve_extents(num_extents, reserve_fraction),
    )


def extent_size_from_config(vllm_config: VllmConfig) -> int | None:
    additional: dict[str, Any] | None = getattr(vllm_config, "additional_config", None)
    value = additional.get(ATTN_BLOCK_SIZE_KEY) if additional else None
    return int(value) if value else None


def validate_fragmentation(
    geometry: ExtentGeometry, max_num_seqs: int, num_extents: int
) -> None:
    """Every running request pins one partly filled extent."""
    if geometry.is_degenerate:
        return
    if max_num_seqs >= num_extents:
        raise ValueError(
            f"max_num_seqs ({max_num_seqs}) needs at least one extent each, but "
            f"the pool holds only {num_extents} extents of {geometry.extent_size} "
            f"tokens; lower max_num_seqs or raise the KV cache size"
        )
    if max_num_seqs / num_extents > 0.5:
        logger.warning(
            "Page/extent: up to %d of %d extents can be pinned by partially "
            "filled extents, one per running request.",
            max_num_seqs,
            num_extents,
        )


# --------------------------------------------------------------------------- #
# Extent pool
# --------------------------------------------------------------------------- #


class OutOfExtents(RuntimeError):
    """No extent can be allocated, reserve included."""


class ExtentAllocator:
    """Free list over a fixed pool; I7 makes reclaim whole-extent."""

    def __init__(self, num_extents: int, num_reserved: int = 0) -> None:
        if num_extents <= 0:
            raise ValueError(f"num_extents must be positive, got {num_extents}")
        if not 0 <= num_reserved < num_extents:
            raise ValueError(
                f"num_reserved ({num_reserved}) must be in [0, {num_extents})"
            )
        self.num_extents = num_extents
        self.num_reserved = num_reserved
        self._free: deque[int] = deque(range(num_extents))
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
            raise OutOfExtents(
                f"no extent available (free={self.num_free}, "
                f"reserved={self.num_reserved}, urgent={urgent})"
            )
        extent_id = self._free.popleft()
        self._allocated.add(extent_id)
        return extent_id

    def free(self, extent_id: int) -> None:
        if extent_id not in self._allocated:
            raise ValueError(f"extent {extent_id} is not allocated")
        self._allocated.discard(extent_id)
        self._free.appendleft(extent_id)

    def reset(self) -> None:
        self._free = deque(range(self.num_extents))
        self._allocated.clear()


# --------------------------------------------------------------------------- #
# Page -> extent map
# --------------------------------------------------------------------------- #


@dataclass
class Extent:
    extent_id: int
    # slot -> page id; append-only (I3), so len() is the write pointer
    page_ids: list[int] = field(default_factory=list)
    # memory lifetime only, not the hash multiplicity that drives KV events
    ref_cnt: int = 0
    owner: str | None = None
    sealed: bool = False
    # position in the prefix chain; hashes are chained, so reclaiming a shallow
    # extent strands every deeper one
    depth: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)


class ExtentTable:
    """Page <-> extent map plus per-request ownership.

    Lifetime: live (ref_cnt > 0), retained (0 but a hit can revive it),
    reclaimed (evicted, id returned to the pool).
    """

    def __init__(self, pages_per_extent: int) -> None:
        self.pages_per_extent = pages_per_extent
        self._extents: dict[int, Extent] = {}
        self._page_holders: dict[int, set[int]] = {}
        self._request_extents: dict[str, list[int]] = {}

    # -- extents ------------------------------------------------------------

    def create(
        self, extent_id: int, owner: str | None = None, depth: int = 0
    ) -> Extent:
        if extent_id in self._extents:
            raise ValueError(f"extent {extent_id} already exists")
        extent = self._extents[extent_id] = Extent(extent_id, owner=owner, depth=depth)
        return extent

    def get(self, extent_id: int) -> Extent | None:
        return self._extents.get(extent_id)

    def require(self, extent_id: int) -> Extent:
        if (extent := self._extents.get(extent_id)) is None:
            raise KeyError(f"extent {extent_id} is not mapped")
        return extent

    def remove(self, extent_id: int) -> Extent | None:
        extent = self._extents.pop(extent_id, None)
        if extent is not None:
            for page_id in extent.page_ids:
                if holders := self._page_holders.get(page_id):
                    holders.discard(extent_id)
                    if not holders:
                        del self._page_holders[page_id]
        return extent

    def seal(self, extent_id: int) -> None:
        extent = self.require(extent_id)
        if extent.num_pages != self.pages_per_extent:
            raise ValueError(
                f"extent {extent_id} holds {extent.num_pages} pages; only a "
                f"full extent ({self.pages_per_extent}) may be sealed"
            )
        extent.sealed = True
        extent.owner = None

    # -- pages --------------------------------------------------------------

    def append_page(self, extent_id: int, page_id: int, *, fresh: bool) -> int:
        """Bind ``page_id`` to the next slot.

        ``fresh``: upstream just handed the id out for new content, so any other
        extent claiming it holds stale bytes.
        """
        extent = self.require(extent_id)
        if extent.num_pages >= self.pages_per_extent:
            raise ValueError(
                f"extent {extent_id} is full ({self.pages_per_extent} pages); "
                "seal it and open a new one"
            )
        if fresh:
            self._revoke_stale_claims(page_id, extent_id)
        slot = extent.num_pages
        extent.page_ids.append(page_id)
        self._page_holders.setdefault(page_id, set()).add(extent_id)
        return slot

    def _revoke_stale_claims(self, page_id: int, new_extent_id: int) -> None:
        """Poison, don't remove: removing shifts later slots and breaks I3."""
        holders = self._page_holders.get(page_id)
        if not holders:
            return
        for holder_id in list(holders):
            if holder_id == new_extent_id:
                continue
            if (holder := self._extents.get(holder_id)) is not None:
                holder.page_ids = [
                    INVALID_PAGE if pid == page_id else pid for pid in holder.page_ids
                ]
            holders.discard(holder_id)

    def locate(self, page_id: int) -> tuple[int, int] | None:
        """A copy source, not an address; any holder will do."""
        if page_id == INVALID_PAGE:
            return None
        for extent_id in self._page_holders.get(page_id, ()):
            extent = self._extents.get(extent_id)
            if extent is not None and page_id in extent.page_ids:
                return extent_id, extent.page_ids.index(page_id)
        return None

    def holders(self, page_id: int) -> set[int]:
        return set(self._page_holders.get(page_id, ()))

    # -- references and ownership -------------------------------------------

    def acquire(self, extent_id: int) -> None:
        self.require(extent_id).ref_cnt += 1

    def release(self, extent_id: int) -> int:
        extent = self.require(extent_id)
        if extent.ref_cnt <= 0:
            raise ValueError(f"extent {extent_id} refcount underflow")
        extent.ref_cnt -= 1
        return extent.ref_cnt

    def attach_to_request(self, request_id: str, extent_id: int) -> None:
        extents = self._request_extents.setdefault(request_id, [])
        if extent_id not in extents:
            extents.append(extent_id)
            self.acquire(extent_id)

    def request_extents(self, request_id: str) -> list[int]:
        return list(self._request_extents.get(request_id, ()))

    def detach_request(self, request_id: str) -> list[int]:
        """Release a request's extents; returns those now retained."""
        retained = []
        for extent_id in self._request_extents.pop(request_id, []):
            extent = self._extents.get(extent_id)
            if extent is not None and self.release(extent_id) == 0:
                extent.owner = None
                retained.append(extent_id)
        return retained

    def reset(self) -> None:
        self._extents.clear()
        self._page_holders.clear()
        self._request_extents.clear()


# --------------------------------------------------------------------------- #
# Binding policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExtentCopyOp:
    """A token range to copy before the forward pass.

    In tokens, not slots: page size is scheduler-side and the worker never
    needs it.
    """

    src_extent_id: int
    dst_extent_id: int
    src_start: int
    dst_start: int
    num_tokens: int


class PageExtentManager:
    """Owns the extent pool and decides attach-vs-copy per request.

    I4-I7: a fully matched extent attaches by reference, anything less is
    copied into a private one, only sealed extents are attach targets, reclaim
    is whole-extent. Splitting the match at extent boundaries bounds the copy
    by one extent however long the match.
    """

    def __init__(
        self, geometry: ExtentGeometry, num_extents: int, num_reserved: int = 0
    ) -> None:
        self.geometry = geometry
        self.allocator = ExtentAllocator(num_extents, num_reserved)
        self.table = ExtentTable(geometry.pages_per_extent)
        self.num_pages_copied = 0
        self.num_pages_written = 0

    def bind(
        self, request_id: str, page_ids: list[int], num_cached_pages: int
    ) -> list[ExtentCopyOp]:
        """Back a request's pages with extents, returning the copies implied.

        ``num_cached_pages``: leading pages whose content already exists; the
        rest the forward pass writes.
        """
        ppe = self.geometry.pages_per_extent
        existing = self.table.request_extents(request_id)
        ops: list[ExtentCopyOp] = []

        for ext_idx in range(self.geometry.num_extents_for_pages(len(page_ids))):
            group = page_ids[ext_idx * ppe : (ext_idx + 1) * ppe]
            base = ext_idx * ppe

            if ext_idx < len(existing):
                extent_id = existing[ext_idx]
                extent = self.table.require(extent_id)
                if extent.num_pages >= len(group):
                    continue
                pending = [
                    op
                    for slot in range(extent.num_pages, len(group))
                    for op in self._place(
                        group[slot], extent_id, slot, base + slot, num_cached_pages
                    )
                ]
            else:
                attach_id = self._attachable(group, base, num_cached_pages)
                if attach_id is not None:
                    self.table.attach_to_request(request_id, attach_id)
                    continue
                extent_id = self.allocator.allocate(urgent=base < num_cached_pages)
                self.table.create(extent_id, owner=request_id, depth=ext_idx)
                self.table.attach_to_request(request_id, extent_id)
                pending = [
                    op
                    for slot, page_id in enumerate(group)
                    for op in self._place(
                        page_id, extent_id, slot, base + slot, num_cached_pages
                    )
                ]

            self._maybe_seal(extent_id)
            ops.extend(_coalesce(pending))

        return ops

    def _place(
        self,
        page_id: int,
        dst_extent_id: int,
        dst_slot: int,
        page_index: int,
        num_cached_pages: int,
    ) -> list[ExtentCopyOp]:
        """Bind one page, copying when its data lives elsewhere."""
        fresh = page_index >= num_cached_pages
        ops: list[ExtentCopyOp] = []
        if fresh:
            self.num_pages_written += 1
        elif (source := self.table.locate(page_id)) is not None:
            src_extent_id, src_slot = source
            if src_extent_id != dst_extent_id:
                page = self.geometry.page_size
                ops.append(
                    ExtentCopyOp(
                        src_extent_id,
                        dst_extent_id,
                        src_slot * page,
                        dst_slot * page,
                        page,
                    )
                )
                self.num_pages_copied += 1
        self.table.append_page(dst_extent_id, page_id, fresh=fresh)
        return ops

    def _attachable(
        self, group: list[int], base: int, num_cached_pages: int
    ) -> int | None:
        """I5: a sealed extent holding exactly this group."""
        if len(group) != self.geometry.pages_per_extent:
            return None
        if base + len(group) > num_cached_pages:
            return None
        for candidate in self.table.holders(group[0]):
            extent = self.table.get(candidate)
            if extent is not None and extent.sealed and extent.page_ids == group:
                return candidate
        return None

    def _maybe_seal(self, extent_id: int) -> None:
        extent = self.table.get(extent_id)
        if (
            extent is not None
            and not extent.sealed
            and extent.num_pages == self.geometry.pages_per_extent
        ):
            self.table.seal(extent_id)

    # -- lifetime -----------------------------------------------------------

    def block_table(self, request_id: str) -> list[int]:
        return self.table.request_extents(request_id)

    def free_request(self, request_id: str, *, preempted: bool = False) -> list[int]:
        """Preempted recomputes anyway, so reclaim; finished stays retained."""
        retained = self.table.detach_request(request_id)
        if not preempted:
            return []
        return [e for e in retained if self.reclaim(e)]

    def reclaim(self, extent_id: int) -> bool:
        extent = self.table.get(extent_id)
        if extent is None or extent.ref_cnt > 0:
            return False
        self.table.remove(extent_id)
        self.allocator.free(extent_id)
        return True

    def reclaim_retained(self, count: int) -> int:
        """Give up unreferenced extents under pressure."""
        reclaimed = 0
        for extent_id in self.retained_extents():
            if reclaimed >= count:
                break
            reclaimed += self.reclaim(extent_id)
        return reclaimed

    def retained_extents(self) -> list[int]:
        """Unreferenced extents, deepest first: reclaiming a shallow extent
        strands every deeper one on the same chain."""
        retained = [
            extent
            for extent_id in range(self.allocator.num_extents)
            if (extent := self.table.get(extent_id)) is not None and extent.ref_cnt == 0
        ]
        retained.sort(key=lambda e: e.depth, reverse=True)
        return [e.extent_id for e in retained]

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


def _coalesce(ops: list[ExtentCopyOp]) -> list[ExtentCopyOp]:
    """Merge copies contiguous in both source and destination."""
    merged: list[ExtentCopyOp] = []
    for op in ops:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.src_extent_id == op.src_extent_id
            and prev.dst_extent_id == op.dst_extent_id
            and prev.src_start + prev.num_tokens == op.src_start
            and prev.dst_start + prev.num_tokens == op.dst_start
        ):
            merged[-1] = ExtentCopyOp(
                prev.src_extent_id,
                prev.dst_extent_id,
                prev.src_start,
                prev.dst_start,
                prev.num_tokens + op.num_tokens,
            )
        else:
            merged.append(op)
    return merged
