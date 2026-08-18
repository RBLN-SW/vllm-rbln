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

Design: https://github.com/RBLN-SW/vllm-rbln/issues/928

Once ``--block-size`` is the page, upstream does all the matching natively, so what is left is making a page id name its own
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

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.request import RequestStatus

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.kernel_block_pool import (
    KernelBlock,
    KernelBlockPool,
    RBLNKVCacheBlock,
    as_page,
)
from vllm_rbln.v1.core.page_layout import KernelBlockCopyOp, PageLayoutConfig

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

__all__ = ["RBLNKVCacheBlocks", "RBLNPageLayoutKVCacheManager"]


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


class RBLNKVCacheBlocks(KVCacheBlocks):
    """``KVCacheBlocks`` whose pages are ``RBLNKVCacheBlock``."""

    @property
    def pages(self) -> list[RBLNKVCacheBlock]:
        """Pages of the single full-attention group."""
        return [as_page(page) for page in self.blocks[0]]


@dataclass
class _PrivateCopy:
    """The matched head of a group, re-issued in a private block."""

    first_page_index: int
    source: list[RBLNKVCacheBlock]
    destination: list[RBLNKVCacheBlock]


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
        # R2 replaces upstream's matcher, so anything that changes what a match
        # means has to be refused rather than silently ignored: EAGLE drops the
        # last matched block, and context parallelism scales the block size.
        for name in ("use_eagle", "dcp_world_size", "pcp_world_size"):
            value = kwargs.get(name)
            if value not in (None, False, 0, 1):
                raise ValueError(
                    f"page layout does not support {name}={value}; its group "
                    f"lookup does not reproduce that matching rule"
                )
        self.page_layout_config = page_layout_config
        self.geometry = page_layout_config.geometry
        ppe = self.geometry.pages_per_kernel_block
        with _kernel_block_pool(ppe):
            super().__init__(kv_cache_config=kv_cache_config, **kwargs)
        assert isinstance(self.block_pool, KernelBlockPool)
        self.pool: KernelBlockPool = self.block_pool
        # Upstream sizes the watermark in pages, but it is headroom for admitting
        # the *next* request, and a waiting or preempted one can only be served
        # out of an idle kernel block -- pages free inside another request's Open
        # group are exactly the ones R1 forbids it from taking. A reserve smaller
        # than one group therefore reserves nothing, so round it up to whole
        # groups. It stays expressed in pages, so `get_num_free_blocks` remains
        # the single admission gate; and since the watermark only applies to
        # requests that hold no Open group, that count is `idle * ppe` exactly,
        # which makes the page comparison mean "keep N kernel blocks idle".
        self.watermark_blocks = math.ceil(self.watermark_blocks / ppe) * ppe
        self.empty_kv_cache_blocks = RBLNKVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # Copies the worker must perform before the next forward pass, and the
        # source pages each keeps alive until it has run.
        self.pending_copy_ops: list[KernelBlockCopyOp] = []
        self._pending_sources: list[RBLNKVCacheBlock] = []
        self._in_flight_sources: list[list[RBLNKVCacheBlock]] = []
        self.num_pages_copied = 0
        self.num_pages_written = 0
        self.num_whole_groups = 0
        self.num_resumes = 0
        self.num_copies = 0
        self.num_refusals = 0

        logger.info(
            "Page/kernel block KV cache: page=%d, kernel block=%d (%d pages), "
            "kernel blocks=%d",
            self.geometry.page_size,
            self.geometry.kernel_block_size,
            ppe,
            self.pool.num_kernel_blocks,
        )

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> RBLNKVCacheBlocks:
        typed = tuple([as_page(page) for page in group] for group in blocks)
        if not any(typed):
            empty = self.empty_kv_cache_blocks
            assert isinstance(empty, RBLNKVCacheBlocks)
            return empty
        return RBLNKVCacheBlocks(typed)

    # -- prefix match -------------------------------------------------------

    def get_computed_blocks(self, request: Request) -> tuple[RBLNKVCacheBlocks, int]:
        """R2: match a group at a time, over kernel blocks rather than pages.

        Upstream matches page by page and asks the hash table for *a* block
        carrying each hash. That is the wrong question here. A copy publishes a
        second page under an existing hash, upstream deliberately keeps both and
        returns an arbitrary one, so a page-at-a-time walk routinely stitches one
        group out of two producers' blocks -- and a group is one block table
        entry, so the result names no address that exists.

        Asking instead "which kernel block holds the longest prefix of this
        group's hashes, at the right slots?" answers attach, partial match and
        duplicate resolution in one pass, and cannot return an unaddressable
        match by construction.
        """
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # The last token must be recomputed to produce logits.
        max_pages = (request.num_tokens - 1) // self.geometry.page_size
        pages = self._match(request.block_hashes, max_pages)
        if self.log_stats and self.prefix_cache_stats is not None:
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=len(pages) * self.geometry.page_size,
                preempted=request.num_preemptions > 0,
            )
        if not pages:
            return self.empty_kv_cache_blocks, 0
        return (
            self.create_kv_cache_blocks((pages,)),
            len(pages) * self.geometry.page_size,
        )

    def _match(
        self, block_hashes: Sequence[BlockHash], max_pages: int
    ) -> list[RBLNKVCacheBlock]:
        """The longest addressable prefix: whole groups, then one partial tail."""
        ppe = self.geometry.pages_per_kernel_block
        cache = self.pool.cached_block_hash_to_block
        matched: list[RBLNKVCacheBlock] = []
        for start in range(0, min(max_pages, len(block_hashes)), ppe):
            group = [
                make_block_hash_with_group_id(block_hash, 0)
                for block_hash in block_hashes[start : min(start + ppe, max_pages)]
            ]
            best: range | None = None
            for kernel_block in range(self.pool.num_kernel_blocks):
                base = kernel_block * ppe
                length = 0
                while length < len(group) and cache.contain(
                    group[length], base + length
                ):
                    length += 1
                if best is None or length > len(best):
                    best = range(base, base + length)
                    if length == len(group):
                        break
            if best is None or not best:
                break
            matched.extend(self.pool.page(page_id) for page_id in best)
            if len(best) < len(group):
                break  # a partial group ends the match
        return matched

    # -- allocation ---------------------------------------------------------

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> RBLNKVCacheBlocks | None:
        """Allocate pages for ``request``, laid out as kernel-block runs.

        Upstream's ``allocate_slots`` plus the partial-group plan: a match that
        ends mid-group is resumed in place or copied to a private block before
        the rest of the tokens are allocated, so the tail lands at the right
        slot. Returns ``None`` when the request cannot be admitted.
        """
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        matched: list[RBLNKVCacheBlock] = []
        if new_computed_blocks is not None:
            matched = [as_page(page) for page in new_computed_blocks.blocks[0]]
        computed = (
            new_computed_blocks.blocks
            if new_computed_blocks is not None
            else self.empty_kv_cache_blocks.blocks
        )

        # How a match that ends mid-group is continued. Decided here, before the
        # allocation, because the pool has to know which block the request's next
        # pages belong in -- upstream would allocate the group's unmatched tail
        # without knowing it must land at a particular slot of a particular block.
        adopted: KernelBlock | None = None
        copy: _PrivateCopy | None = None
        head = len(matched) % self.geometry.pages_per_kernel_block
        if matched and head == 0:
            self.num_whole_groups += 1
        elif matched:
            producer = matched[-head].kernel_block
            if producer.can_resume(head):
                self.pool.open_group(producer, request.request_id)
                self.num_resumes += 1
                adopted = producer
            else:
                # One idle group, no more and no less: `head` is a partial group
                # so it never spans two, and `_redirect` copies it to slot 0, so
                # it cannot come from a group already part-written. Nothing else
                # is available to a request that reached here anyway -- it lost
                # the adoption branch, and a waiting one holds no Open group.
                if not self.pool.has_idle_kernel_blocks(1):
                    # No private block for the copy, so the request waits.
                    self.num_refusals += 1
                    return None
                with self.pool.allocating_for(request.request_id):
                    destination = self.pool.get_new_blocks(head)
                assert self.pool.slot_of(destination[0].block_id) == 0
                copy = _PrivateCopy(
                    len(matched) - head, list(matched[-head:]), destination
                )

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        watermark_blocks = 0
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        with self.pool.allocating_for(request.request_id):
            if full_sequence_must_fit:
                full_num_tokens = min(request.num_tokens, self.max_model_len)
                required_page = (
                    self.coordinator.get_num_blocks_to_allocate(
                        request_id=request.request_id,
                        num_tokens=full_num_tokens,
                        new_computed_blocks=computed,
                        num_encoder_tokens=num_encoder_tokens,
                        total_computed_tokens=total_computed_tokens,
                        num_tokens_main_model=full_num_tokens,
                        apply_admission_cap=True,
                    )
                    + watermark_blocks
                )
                required_kernel_block = self.pool.kernel_blocks_needed(required_page)
                if not self.pool.has_idle_kernel_blocks(required_kernel_block):
                    self._abandon(request, adopted, copy)
                    return None

            self.coordinator.remove_skipped_blocks(
                request.request_id, total_computed_tokens
            )
            # Everything this step must find room for, in pages: the tokens
            # themselves, the watermark held back for the next admission, and
            # the pages in-flight prefills have already spoken for.
            #
            # Both reserves arrive already rounded to whole groups -- the
            # watermark in `__init__`, `reserved_blocks` in the scheduler's
            # `_inflight_prefill_reserved_blocks` -- and a multiple of ppe passes
            # through the ceil untouched. That is what keeps them from sharing
            # this request's last partial group, which is the whole point of
            # reserving. Rounding them here instead would not work: they are sums
            # over other requests, and four prefills a page short each need four
            # groups, which their page total cannot say.
            required_page = (
                self.coordinator.get_num_blocks_to_allocate(
                    request_id=request.request_id,
                    num_tokens=num_tokens_need_slot,
                    new_computed_blocks=computed,
                    num_encoder_tokens=num_encoder_tokens,
                    total_computed_tokens=(
                        num_local_computed_tokens + num_external_computed_tokens
                    ),
                    num_tokens_main_model=num_tokens_main_model,
                )
                + watermark_blocks
                + reserved_blocks
            )
            required_kernel_block = self.pool.kernel_blocks_needed(required_page)
            if not self.pool.has_idle_kernel_blocks(required_kernel_block):
                self._abandon(request, adopted, copy)
                return None

            if computed is not self.empty_kv_cache_blocks.blocks or (
                num_external_computed_tokens > 0
            ):
                self.coordinator.allocate_new_computed_blocks(
                    request_id=request.request_id,
                    new_computed_blocks=computed,
                    num_local_computed_tokens=num_local_computed_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                )
            if copy is not None:
                self._redirect(request, copy)
            new_blocks = self.coordinator.allocate_new_blocks(
                request.request_id,
                num_tokens_need_slot,
                num_tokens_main_model,
                num_encoder_tokens,
            )

        if self.enable_caching and not delay_cache_blocks:
            self.coordinator.cache_blocks(
                request,
                min(total_computed_tokens + num_new_tokens, request.num_tokens),
            )
        result = self.create_kv_cache_blocks(new_blocks)
        self.num_pages_written += sum(len(group) for group in result.blocks)
        return result

    def log_binding_stats(self) -> None:
        """Why the copies happen, in one line. Called per scheduler stats tick."""
        logger.info(
            "Page layout: whole-group=%d resume=%d copy=%d refused=%d | "
            "pages copied=%d written=%d CA=%.4f",
            self.num_whole_groups,
            self.num_resumes,
            self.num_copies,
            self.num_refusals,
            self.num_pages_copied,
            self.num_pages_written,
            self.copy_amplification,
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
                src_kernel_block_id=plan.source[0].kernel_block.index,
                dst_kernel_block_id=plan.destination[0].kernel_block.index,
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

    def _abandon(
        self,
        request: Request,
        adopted: KernelBlock | None,
        copy: _PrivateCopy | None,
    ) -> None:
        """Undo the partial-group plan whose allocation was then refused."""
        if adopted is not None:
            self.pool.release_owner(request.request_id)
        elif copy is not None:
            self.pool.free_blocks(copy.destination)

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
            as_page(pages[i]).kernel_block.index for i in range(0, len(pages), ppe)
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
