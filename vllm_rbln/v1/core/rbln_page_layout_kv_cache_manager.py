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

Once ``--block-size`` is the page, upstream does all the matching natively, so
what is left is making a page id name its own physical home:

    kernel block = page_id // pages_per_kernel_block
    slot         = page_id %  pages_per_kernel_block

`KernelBlockPool` allocates so that this identity holds for pages it hands out,
and prefix matching preserves it for pages it does not, because a cache hit
returns the page produced at the same sequence position. The block table the
worker wants then falls out arithmetically -- no page -> location map to keep.

That leaves one case with no legal answer: a group whose match ends part-way
while the producer's block is still live. Its remaining slots must be written,
but an Open kernel block has one writer so they cannot go in the producer's
block, and the group is one block table entry so they cannot go elsewhere. The
group is then re-allocated whole as a private run and the matched tail copied
into it -- the copied pages get *fresh* ids naming the new block, which is what
keeps the identity intact.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import RequestStatus

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.kernel_block_pool import (
    KernelBlock,
    KernelBlockPool,
    RBLNKVCacheBlock,
)
from vllm_rbln.v1.core.kv_cache_copy import CopyOpMixin, KVCacheCopyOp
from vllm_rbln.v1.core.page_layout import PageLayoutConfig

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
        return [cast(RBLNKVCacheBlock, page) for page in self.blocks[0]]


@dataclass
class _TailCopy:
    """The match's partial tail, to be copied into a new kernel block."""

    tail_start: int
    source: list[RBLNKVCacheBlock]
    destination: list[RBLNKVCacheBlock]


class RBLNPageLayoutKVCacheManager(CopyOpMixin, KVCacheManager):
    """``KVCacheManager`` whose pages are backed by contiguous kernel blocks."""

    @staticmethod
    def can_use_page_layout(
        kv_cache_config: KVCacheConfig,
        config: PageLayoutConfig,
    ) -> bool:
        """Eligibility: the MVP groups a single full-attention group only.

        `_match` assumes upstream's own hit-finding never returns a null
        prefix -- true for `FullAttentionSpec` but not for
        `SlidingWindowSpec` / `ChunkedLocalAttentionSpec`, whose managers
        skip positions outside the window. Nothing upstream of this call
        currently produces that combination (RBLN's own
        `RBLNSlidingWindowManager` disables prefix-cache hits outright, and
        `disable_unsupported_prefix_caching` turns off prefix caching
        entirely for sliding-window models), but the guard belongs here,
        at the one place that assumes it, rather than in those other call
        sites.
        """
        if not config.enabled:
            return False
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            return False
        spec = groups[0].kv_cache_spec
        if not isinstance(spec, FullAttentionSpec):
            return False
        return spec.block_size == config.geometry.page_size

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        page_layout_config: PageLayoutConfig,
        **kwargs,
    ) -> None:
        assert self.can_use_page_layout(kv_cache_config, page_layout_config)
        # Prefix matching walks kernel-block groups, not pages, so anything that
        # changes what a match means has to be refused rather than silently
        # ignored: EAGLE drops the last matched block, and context parallelism
        # scales the block size.
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
        # group already have a writer and cannot be taken. A reserve smaller
        # than one group therefore reserves nothing, so round it up to whole
        # groups. It stays expressed in pages, so `get_num_free_blocks` remains
        # the single admission gate; and since the watermark only applies to
        # requests that hold no Open group, that count is `idle * ppe` exactly,
        # which makes the page comparison mean "keep N kernel blocks idle".
        self.watermark_blocks = math.ceil(self.watermark_blocks / ppe) * ppe  # type: ignore[has-type]
        self.empty_kv_cache_blocks = RBLNKVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

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
        typed = tuple(
            [cast(RBLNKVCacheBlock, page) for page in group] for group in blocks
        )
        if not any(typed):
            empty = self.empty_kv_cache_blocks
            assert isinstance(empty, RBLNKVCacheBlocks)
            return empty
        return RBLNKVCacheBlocks(typed)

    # -- prefix match -------------------------------------------------------

    def get_computed_blocks(self, request: Request) -> tuple[RBLNKVCacheBlocks, int]:
        """Match a group at a time, over kernel blocks rather than pages.

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
            # Groups are kernel-block aligned, so the first hash's cached
            # pages *are* the candidate starts. Copies make a few extras.
            for page in cache.get_blocks(group[0]):
                if self.pool.slot_of(page.block_id) != 0:
                    continue
                base = page.block_id
                length = 1
                while length < len(group) and cache.contain(
                    group[length], base + length
                ):
                    length += 1
                if best is None or length > len(best):
                    best = range(base, base + length)
                    if length == len(group):
                        break
            if best is None:
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
        the rest of the tokens are allocated, so the unmatched pages land at the
        right slot. Returns ``None`` when the request cannot be admitted.
        """
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        matched: list[RBLNKVCacheBlock] = []
        if new_computed_blocks is not None:
            matched = [
                cast(RBLNKVCacheBlock, page) for page in new_computed_blocks.blocks[0]
            ]
        computed = (
            new_computed_blocks.blocks
            if new_computed_blocks is not None
            else self.empty_kv_cache_blocks.blocks
        )

        # How a match that ends mid-group is continued. Decided here, before the
        # allocation, because the pool has to know which block the request's next
        # pages belong in -- upstream would allocate the group's unmatched tail
        # without knowing it must land at a particular slot of a particular block.
        resume_block: KernelBlock | None = None
        tail_copy: _TailCopy | None = None
        n_tail = len(matched) % self.geometry.pages_per_kernel_block
        if n_tail:
            tail_pages = matched[-n_tail:]
            matched_block = tail_pages[0].kernel_block
            if matched_block.can_resume(n_tail):
                self.pool.open_group(matched_block, request.request_id)
                resume_block = matched_block
            else:
                # One idle group, no more and no less: `tail_pages` is a partial
                # group so it never spans two, and `_redirect` copies it to slot
                # 0, so it cannot come from a group already part-written. Nothing
                # else is available to a request that reached here anyway -- it
                # lost the resume branch, and a waiting one holds no Open group.
                if not self.pool.has_idle_kernel_blocks(1):
                    # No private block for the copy, so the request waits.
                    return None
                with self.pool.allocating_for(request.request_id):
                    destination = self.pool.get_new_blocks(n_tail)
                assert self.pool.slot_of(destination[0].block_id) == 0
                tail_copy = _TailCopy(
                    tail_start=len(matched) - n_tail,
                    source=tail_pages,
                    destination=destination,
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
                    self._abandon(request, resume_block, tail_copy)
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
                self._abandon(request, resume_block, tail_copy)
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
            if tail_copy is not None:
                self._redirect(request, tail_copy)
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
        return self.create_kv_cache_blocks(new_blocks)

    def _redirect(self, request: Request, tail_copy: _TailCopy) -> None:
        """Swap the matched tail for the copies and queue the transfer.

        The originals keep the reference `allocate_slots` took on them; that is
        what holds the copy source in place until the worker has read it.
        """
        blocks = self.coordinator.get_blocks(request.request_id)[0]
        tail_start = tail_copy.tail_start
        tail_end = tail_start + len(tail_copy.source)
        assert [b.block_id for b in blocks[tail_start:tail_end]] == [
            b.block_id for b in tail_copy.source
        ]
        blocks[tail_start:tail_end] = tail_copy.destination

        page = self.geometry.page_size
        self.queue_copy(
            KVCacheCopyOp(
                src_block_id=tail_copy.source[0].kernel_block.index,
                dst_block_id=tail_copy.destination[0].kernel_block.index,
                num_tokens=len(tail_copy.source) * page,
            ),
            tail_copy.source,
        )
        for source, destination in zip(tail_copy.source, tail_copy.destination):
            self.pool.publish_copy(source, destination)

    def _abandon(
        self,
        request: Request,
        resume_block: KernelBlock | None,
        tail_copy: _TailCopy | None,
    ) -> None:
        """Undo the partial-group resume or copy whose allocation was then refused."""
        if resume_block is not None:
            self.pool.release_owner(request.request_id)
        elif tail_copy is not None:
            self.pool.free_blocks(tail_copy.destination)

    def block_table(self, request_id: str) -> list[int]:
        """The worker's block table: kernel block ids backing a request."""
        ppe = self.geometry.pages_per_kernel_block
        pages = self.coordinator.get_blocks(request_id)[0]
        return [
            cast(RBLNKVCacheBlock, pages[i]).kernel_block.index
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
            self.clear_copy_ops()
        return result
