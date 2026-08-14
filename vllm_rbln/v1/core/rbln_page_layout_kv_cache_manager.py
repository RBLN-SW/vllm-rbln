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
does all the matching natively, so what is left is making a page id name its own
physical home:

    kernel block = page_id // pages_per_kernel_block
    slot         = page_id %  pages_per_kernel_block

`KernelBlockPool` allocates so that this identity holds for pages it hands out,
and prefix matching preserves it for pages it does not, because a cache hit
returns the page produced at the same sequence position. The block table the
worker wants then falls out arithmetically -- no page -> location map to keep.

That leaves one case with no legal answer: a group whose match ends part-way
while the producer's block is still live. Its remaining slots must be written,
R1 forbids writing them in the producer's block, and the group is one block table
entry so they cannot go elsewhere. The group is then re-allocated whole as a
private run and the matched head copied into it -- the copied pages get *fresh*
ids naming the new block, which is what keeps the identity intact.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.kernel_block_pool import KernelBlockPool
from vllm_rbln.v1.core.page_layout import KernelBlockCopyOp, PageLayoutConfig

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

__all__ = ["RBLNPageLayoutKVCacheManager"]


@contextmanager
def _kernel_block_pool(pages_per_kernel_block: int) -> Iterator[None]:
    """Make the coordinator build a `KernelBlockPool` instead of a `BlockPool`.

    The pool is created several frames down and captured by every single-type
    manager (including its null block), so substituting the object afterwards
    would leave those references pointing at the pool it replaced.
    """

    def factory(**kwargs) -> BlockPool:
        return KernelBlockPool(pages_per_kernel_block=pages_per_kernel_block, **kwargs)

    original = kv_cache_coordinator.BlockPool
    kv_cache_coordinator.BlockPool = factory
    try:
        yield
    finally:
        kv_cache_coordinator.BlockPool = original


@dataclass
class _Adoption:
    """The request continues writing into the block its match ended in."""

    kernel_block: int


@dataclass
class _Refused:
    """No private block for the copy, so the request waits."""


@dataclass
class _PrivateCopy:
    """The matched head of a group, re-issued in a private block."""

    first_page_index: int
    source: list[KVCacheBlock]
    destination: list[KVCacheBlock]


class RBLNPageLayoutKVCacheManager(KVCacheManager):
    """``KVCacheManager`` whose pages are backed by contiguous kernel blocks."""

    @staticmethod
    def can_use_page_layout(
        kv_cache_config: KVCacheConfig,
        config: PageLayoutConfig,
    ) -> bool:
        """Eligibility: the MVP groups a single full-attention group only."""
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
        self.page_layout_config = page_layout_config
        self.geometry = page_layout_config.geometry
        ppe = self.geometry.pages_per_kernel_block
        with _kernel_block_pool(ppe):
            super().__init__(kv_cache_config=kv_cache_config, **kwargs)
        assert isinstance(self.block_pool, KernelBlockPool)
        self.pool: KernelBlockPool = self.block_pool

        # Copies the worker must perform before the next forward pass, and the
        # source pages each keeps alive until it has run.
        self.pending_copy_ops: list[KernelBlockCopyOp] = []
        self._pending_sources: list[KVCacheBlock] = []
        self._in_flight_sources: list[list[KVCacheBlock]] = []
        self.num_pages_copied = 0
        self.num_pages_written = 0
        self.num_whole_groups = 0
        self.num_resumes = 0
        self.num_copies = 0
        self.num_refusals = 0
        self.num_truncated = 0
        self.num_pages_truncated = 0

        logger.info(
            "Page/kernel block KV cache: page=%d, kernel block=%d (%d pages), "
            "kernel blocks=%d",
            self.geometry.page_size,
            self.geometry.kernel_block_size,
            ppe,
            self.pool.num_kernel_blocks,
        )

    # -- prefix match -------------------------------------------------------

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Drop any tail of the match whose pages are not laid out as a run.

        A hash can name several blocks (upstream does not de-duplicate), so a
        lookup can stitch one group out of two producers' pages. The group is a
        single block table entry, so such a match cannot be addressed at all.
        """
        blocks, num_computed_tokens = super().get_computed_blocks(request)
        if num_computed_tokens == 0:
            return blocks, num_computed_tokens

        pages = blocks.blocks[0]
        if self._laid_out_prefix(pages) == len(pages):
            return blocks, num_computed_tokens

        realigned = self._realign(pages)
        if len(realigned) < len(pages):
            self.num_truncated += 1
            self.num_pages_truncated += len(pages) - len(realigned)
        return (
            self.create_kv_cache_blocks((realigned,)),
            min(num_computed_tokens, len(realigned) * self.geometry.page_size),
        )

    def _realign(self, pages: Sequence[KVCacheBlock]) -> list[KVCacheBlock]:
        """Re-pick the match so each group sits in one kernel block.

        A copy publishes a second page under a hash the original already holds,
        and upstream deliberately keeps both and returns an arbitrary one. So a
        long match routinely arrives as one conversation's page 0 followed by
        another's page 1 -- addressable nowhere, since a group is one block table
        entry. The duplicate that *does* continue the run usually exists; find it
        by asking, for each kernel block, whether it holds this group's hashes at
        the right slots.
        """
        ppe = self.geometry.pages_per_kernel_block
        cache = self.pool.cached_block_hash_to_block
        out: list[KVCacheBlock] = []
        for start in range(0, len(pages), ppe):
            group = pages[start : start + ppe]
            hashes = [page.block_hash for page in group]
            if any(block_hash is None for block_hash in hashes):
                break
            best: list[int] = []
            for kernel_block in range(self.pool.num_kernel_blocks):
                base = kernel_block * ppe
                run: list[int] = []
                for slot, block_hash in enumerate(hashes):
                    if not cache.contain(block_hash, base + slot):
                        break
                    run.append(base + slot)
                if len(run) > len(best):
                    best = run
                    if len(best) == len(group):
                        break
            out.extend(self.pool.blocks[page_id] for page_id in best)
            if len(best) < len(group):
                break
        return out

    def _laid_out_prefix(self, pages: Sequence[KVCacheBlock]) -> int:
        """Length of the leading run of pages that satisfy the identity map."""
        ppe = self.geometry.pages_per_kernel_block
        for index, block in enumerate(pages):
            slot = index % ppe
            if block.block_id % ppe != slot:
                return index
            if slot and block.block_id != pages[index - 1].block_id + 1:
                return index
        return len(pages)

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
        plan = self._plan_partial_group(request, new_computed_blocks)
        if isinstance(plan, _Refused):
            return None
        with self.pool.allocating_for(request.request_id):
            result = super().allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                *args,
                **kwargs,
            )
        if result is None:
            self._abandon(request, plan)
            return None
        if isinstance(plan, _PrivateCopy):
            self._redirect(request, plan)
        self.num_pages_written += sum(len(group) for group in result.blocks)
        return result

    def _plan_partial_group(
        self, request: Request, new_computed_blocks: KVCacheBlocks | None
    ) -> _Adoption | _PrivateCopy | _Refused | None:
        """Decide how a match that ends mid-group is continued.

        Runs before the allocation so the pool knows which block the request's
        next pages belong in: upstream allocates the group's unmatched tail
        without knowing it must land at a particular slot of a particular block.
        """
        if new_computed_blocks is None:
            return None
        matched = new_computed_blocks.blocks[0]
        ppe = self.geometry.pages_per_kernel_block
        head = len(matched) % ppe
        if not matched or head == 0:
            self.num_whole_groups += bool(matched)
            return None

        producer = self.pool.kernel_block_of(matched[-head].block_id)
        if self.pool.can_resume(producer, head):
            self.pool.open_run(request.request_id, producer, head)
            self.num_resumes += 1
            return _Adoption(producer)

        with self.pool.allocating_for(request.request_id):
            if self.pool.get_num_free_blocks() < head:
                self.num_refusals += 1
                return _Refused()
            destination = self.pool.get_new_blocks(head)
        return _PrivateCopy(len(matched) - head, list(matched[-head:]), destination)

    def log_binding_stats(self) -> None:
        """Why the copies happen, in one line. Called per scheduler stats tick."""
        logger.info(
            "Page layout: whole-group=%d resume=%d copy=%d refused=%d | "
            "pages copied=%d written=%d CA=%.4f | match truncated=%d (%d pages)",
            self.num_whole_groups,
            self.num_resumes,
            self.num_copies,
            self.num_refusals,
            self.num_pages_copied,
            self.num_pages_written,
            self.copy_amplification,
            self.num_truncated,
            self.num_pages_truncated,
        )

    def _redirect(self, request: Request, plan: _PrivateCopy) -> None:
        """Swap the matched head for the private copies and queue the transfer.

        The originals keep the reference `allocate_slots` took on them; that is
        what holds the copy source in place until the worker has read it.
        """
        blocks = self.coordinator.get_blocks(request.request_id)[0]
        first, last = plan.first_page_index, plan.first_page_index + len(plan.source)
        assert [b.block_id for b in blocks[first:last]] == [
            b.block_id for b in plan.source
        ]
        blocks[first:last] = plan.destination

        page = self.geometry.page_size
        self.pending_copy_ops.append(
            KernelBlockCopyOp(
                src_kernel_block_id=self.pool.kernel_block_of(plan.source[0].block_id),
                dst_kernel_block_id=self.pool.kernel_block_of(
                    plan.destination[0].block_id
                ),
                src_start=0,
                dst_start=0,
                num_tokens=len(plan.source) * page,
            )
        )
        for source, destination in zip(plan.source, plan.destination):
            self.pool.publish_copy(source, destination)
        self._pending_sources.extend(plan.source)
        self.num_pages_copied += len(plan.source)
        self.num_copies += 1

    def _abandon(self, request: Request, plan: _Adoption | _PrivateCopy | None) -> None:
        """Undo a plan whose allocation the scheduler then refused."""
        if isinstance(plan, _Adoption):
            self.pool.release_owner(request.request_id)
        elif isinstance(plan, _PrivateCopy):
            self.pool.free_blocks(plan.destination)

    # -- copy op plumbing ---------------------------------------------------

    def drain_pending_copy_ops(self) -> list[KernelBlockCopyOp]:
        """Copies for this step; sources stay referenced until released."""
        ops = self.pending_copy_ops
        self.pending_copy_ops = []
        self._in_flight_sources.append(self._pending_sources)
        self._pending_sources = []
        if ops:
            self.log_binding_stats()
        return ops

    def release_copy_ops(self, ops: list[KernelBlockCopyOp]) -> None:
        """Drop the source references held by drained copy ops."""
        del ops
        if self._in_flight_sources:
            self.pool.free_blocks(self._in_flight_sources.pop(0))

    def block_table(self, request_id: str) -> list[int]:
        """The worker's block table: kernel block ids backing a request."""
        ppe = self.geometry.pages_per_kernel_block
        pages = self.coordinator.get_blocks(request_id)[0]
        return [
            self.pool.kernel_block_of(pages[i].block_id)
            for i in range(0, len(pages), ppe)
        ]

    # -- lifetime -----------------------------------------------------------

    def free(self, request: Request) -> None:
        # Before the pages go back: an owned block stays unadoptable, and its
        # cached head can outlive the writer by any number of turns.
        self.pool.release_owner(request.request_id)
        super().free(request)

    def reset_prefix_cache(self) -> bool:
        result = super().reset_prefix_cache()
        if result:
            self.pool.reset_ownership()
            self.pending_copy_ops.clear()
        return result

    # -- metrics ------------------------------------------------------------

    @property
    def copy_amplification(self) -> float:
        if self.num_pages_written == 0:
            return 0.0
        return self.num_pages_copied / self.num_pages_written
