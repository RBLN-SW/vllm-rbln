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

"""Sub-block prefix caching for RBLN KV cache management.

See docs/sub_block_prefix_caching.md for design overview and rationale.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    make_block_hash_with_group_id,
    need_extra_keys,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Attention spec types whose KV cache stores per-token data and can be
# partially copied at sub-block granularity.  Types NOT in this set (e.g.
# MambaSpec — stores recurrent state per block, not per-token KV) are
# incompatible with sub-block caching.
_SUB_BLOCK_ELIGIBLE_SPECS: tuple[type[KVCacheSpec], ...] = (
    FullAttentionSpec,  # includes MLAAttentionSpec (subclass)
    SlidingWindowSpec,
    ChunkedLocalAttentionSpec,
)

logger = init_logger(__name__)


@dataclass
class KVCacheCopyOp:
    """Describes a sub-block KV cache copy from a cached block to a new block."""

    group_id: int
    # NOTE: While pending, it holds a ref count of the source block to prevent eviction.
    src_block_id: int
    dst_block_id: int
    # Number of tokens to copy (= num_matched_sub_blocks * sub_block_size).
    num_tokens: int


class SubBlockHasher:
    """Computes chained sub-block hashes from token IDs.

    Uses the same ``hash_block_tokens`` as upstream, but at sub-block
    granularity.  When a *request* is provided, per-sub-block
    ``extra_keys`` (cache_salt, LoRA, multimodal, prompt_embeds) are
    mixed in, mirroring upstream full-block hashing.
    """

    def __init__(
        self,
        hash_fn: Callable,
        sub_block_size: int,
    ) -> None:
        self.hash_fn = hash_fn
        self.sub_block_size = sub_block_size

    def hash_tokens(
        self,
        token_ids: Sequence[int],
        *,
        parent_hash: BlockHash | None = None,
        num_hashed_tokens: int = 0,
        request: Request | None = None,
        start_mm_idx: int = 0,
    ) -> tuple[list[BlockHash], int]:
        """Return sub-block hashes for *full* sub-blocks in ``token_ids``.

        Args:
            token_ids: Full token sequence of the request.
            parent_hash: Hash of the last sub-block before the range we
                are hashing (``None`` for the very first sub-block).
            num_hashed_tokens: Number of tokens already hashed (i.e. the
                start offset into ``token_ids``).
            request: When provided and the request carries extra hash
                keys (LoRA, cache_salt, multimodal, prompt_embeds),
                those keys are mixed into each sub-block hash.
            start_mm_idx: Starting multimodal feature index for
                incremental hashing with multimodal requests.

        Returns:
            ``(hashes, mm_idx)`` — list of ``BlockHash`` values (one
            per full sub-block) and the updated multimodal index.
        """
        # Based on upstream request_block_hasher() in get_request_block_hasher().

        sbs = self.sub_block_size
        hashes: list[BlockHash] = []
        use_extra = request is not None and need_extra_keys(request)
        # NOTE: We can't simply use `mm_idx=-1`,
        # because it means the last mm input in the entire prompt,
        # which is meant to be used during decode phase.
        mm_idx = start_mm_idx
        start = num_hashed_tokens
        for i in range(start, len(token_ids) - sbs + 1, sbs):
            extra_keys = None
            if use_extra:
                extra_keys, mm_idx = generate_block_hash_extra_keys(
                    request, i, i + sbs, mm_idx
                )
            parent_hash = hash_block_tokens(
                self.hash_fn,
                parent_hash,
                token_ids[i : i + sbs],
                extra_keys,
            )
            hashes.append(parent_hash)
        return hashes, mm_idx


class SubBlockIndex:
    """Index mapping sub-block hashes to physical block IDs that contain that
    sub-block as a prefix."""

    def __init__(self) -> None:
        # sub_block_hash → set of block IDs with that prefix cached.
        self._hash_to_blocks: dict[BlockHash, set[int]] = {}
        # Reverse index: block_id → list of sub-block hashes (for removal).
        self._block_hashes: dict[int, list[BlockHash]] = {}

    def update(self, block_id: int, sub_block_hashes: list[BlockHash]) -> None:
        """Index or extend a block's sub-block hashes (idempotent).

        If the block is not yet indexed, inserts all hashes.
        If already indexed, appends any new hashes beyond what is recorded.
        """
        existing = self._block_hashes.setdefault(block_id, [])
        for i in range(len(existing), len(sub_block_hashes)):
            h = sub_block_hashes[i]
            existing.append(h)
            self._hash_to_blocks.setdefault(h, set()).add(block_id)

    def pop(self, block_id: int) -> None:
        """Remove a block from the index (called on eviction)."""
        hashes = self._block_hashes.pop(block_id, None)
        if hashes is None:
            return
        for h in hashes:
            s = self._hash_to_blocks.get(h)
            if s is not None:
                s.discard(block_id)
                if not s:
                    del self._hash_to_blocks[h]

    def contains(self, block_id: int) -> bool:
        """Return True if the block is indexed."""
        return block_id in self._block_hashes

    def longest_match(
        self, sub_block_hashes: Sequence[BlockHash]
    ) -> tuple[int | None, int]:
        """Find a block with the longest prefix match.

        Returns:
            ``(block_id, num_matched)`` where ``block_id`` is any block
            matching that prefix, or ``(None, 0)`` if no match.
        """
        best_block_id: int | None = None
        best_depth = 0
        for depth, h in enumerate(sub_block_hashes, start=1):
            blocks = self._hash_to_blocks.get(h)
            if not blocks:
                break
            # Pick an arbitrary block_id from the set.
            best_block_id = next(iter(blocks))
            best_depth = depth
        return best_block_id, best_depth


# ---------------------------------------------------------------------------
# RBLNKVCacheManager
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SubHashState:
    """Per-request cached sub-block hashes and multimodal index."""

    hashes: list[BlockHash]
    mm_idx: int = 0


@dataclass(slots=True)
class SubBlockMatch:
    """Result of a sub-block partial match lookup.

    Returned by ``RBLNKVCacheManager.get_computed_blocks_sub_block``.
    The caller must either pass it to ``apply_sub_block_match`` (to create
    copy ops) or ``release_sub_block_match`` (to discard it and free the
    source-block references).
    """

    request: Request
    num_tokens: int
    group_matches: tuple[_GroupPartialMatch, ...]


@dataclass(slots=True)
class _GroupPartialMatch:
    """Per-group partial match info."""

    src_block: KVCacheBlock
    # Index of the block (in the request's block list) that will receive
    # the copied sub-blocks.
    dst_req_block_index: int


@dataclass(slots=True)
class _GroupInfo:
    """Per-group configuration for sub-block caching."""

    block_size: int
    sub_blocks_per_block: int
    sub_block_index: SubBlockIndex


class RBLNKVCacheManager(KVCacheManager):
    """KVCacheManager with sub-block prefix caching for RBLN.

    It first applies the upstream full-block prefix matching, then extends
    matches at sub-block granularity, which are reused via copy operations
    scheduled to the model runner.
    """

    @staticmethod
    def can_use_sub_block_caching(
        kv_cache_config: KVCacheConfig,
        sub_block_size: int,
    ) -> bool:
        """Return True if sub-block caching is applicable to this config.

        Sub-block caching requires:
        1. Every group's spec type must store per-token KV data (allow-listed
           in ``_SUB_BLOCK_ELIGIBLE_SPECS``).
        2. Every group must have block_size > sub_block_size and
           block_size % sub_block_size == 0.
        """
        if sub_block_size <= 0:
            return False
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if not isinstance(spec, _SUB_BLOCK_ELIGIBLE_SPECS):
                return False
            bs = spec.block_size
            if bs <= sub_block_size or bs % sub_block_size != 0:
                return False
        return True

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        sub_block_size: int,
        hash_fn: Callable,
        **kwargs,
    ) -> None:
        assert self.can_use_sub_block_caching(kv_cache_config, sub_block_size)

        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            hash_block_size=hash_block_size,
            enable_caching=True,
            **kwargs,
        )

        self.sub_block_size = sub_block_size

        # Build per-group info.
        self._group_infos: list[_GroupInfo] = []
        for group in kv_cache_config.kv_cache_groups:
            bs = group.kv_cache_spec.block_size
            self._group_infos.append(
                _GroupInfo(
                    block_size=bs,
                    sub_blocks_per_block=bs // sub_block_size,
                    sub_block_index=SubBlockIndex(),
                )
            )

        self.sub_block_hasher = SubBlockHasher(hash_fn, sub_block_size)

        # Per-request sub-block hash cache (group-independent).
        self._req_sub_hashes: dict[str, _SubHashState] = {}

        # Copy operations accumulated during a scheduling step; the scheduler
        # drains this list when building SchedulerOutput. While pending, each
        # copy op holds a ref count of its source block to prevent eviction.
        self.pending_copy_ops: list[KVCacheCopyOp] = []

        # Requests for which sub-block indexing is pending.
        # Each entry stores the per-group full-block count snapshot
        # taken before allocate_slots, used to narrow the scan range.
        self._pending_indexing: dict[str, tuple[Request, tuple[int, ...]]] = {}

        self._install_eviction_hook()

    def get_computed_blocks_sub_block(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> SubBlockMatch | None:
        """Discover a sub-block partial match after full-block matching.

        Looks for sub-block prefix matches in the block immediately after the
        full-block boundary.  Returns a ``SubBlockMatch`` handle if found, or
        ``None``.

        The caller must pass the result to ``apply_sub_block_match`` (to
        create copy ops) or ``release_sub_block_match`` (to free the
        source-block references it holds).
        """
        if request.skip_reading_prefix_cache:
            return None

        sub_hashes = self._get_or_compute_sub_hashes(request)

        min_matched: int | None = None
        group_matches: list[_GroupPartialMatch] = []

        for gi in self._group_infos:
            num_full_blocks = num_computed_tokens // gi.block_size
            sbpb = gi.sub_blocks_per_block
            next_sub_start = num_full_blocks * sbpb

            # We need at least one sub-block hash beyond the full-block matches
            # AND fewer than a full block (otherwise upstream would have matched).
            query = sub_hashes[next_sub_start : next_sub_start + sbpb - 1]
            if not query:
                min_matched = 0
                break

            src_block_id, num_matched = gi.sub_block_index.longest_match(query)
            if src_block_id is None:
                min_matched = 0
                break

            if min_matched is None or num_matched < min_matched:
                min_matched = num_matched

            group_matches.append(
                _GroupPartialMatch(
                    src_block=self.block_pool.blocks[src_block_id],
                    dst_req_block_index=num_full_blocks,
                )
            )

        if not min_matched:
            return None

        # We have a partial match!
        num_matched = min_matched
        extra_tokens = num_matched * self.sub_block_size

        # NOTE: Upstream guarantees num_computed_tokens <= num_tokens - 1,
        # because at least the last token must be recomputed for logits.
        # We enforce the same condition here.
        max_allowed_extra_tokens = (request.num_tokens - 1) - num_computed_tokens
        max_allowed_sub_blocks = max_allowed_extra_tokens // self.sub_block_size
        num_matched = min(num_matched, max_allowed_sub_blocks)
        if num_matched == 0:
            return None
        extra_tokens = num_matched * self.sub_block_size

        # Touch source blocks to prevent eviction during allocate_slots.
        # The SubBlockMatch owns these references until apply or release.
        self.block_pool.touch([gm.src_block for gm in group_matches])
        match = SubBlockMatch(
            request=request,
            num_tokens=extra_tokens,
            group_matches=tuple(group_matches),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Sub-block partial match for request %s: %d sub-blocks (%d tokens); %s",
                request.request_id,
                num_matched,
                extra_tokens,
                ", ".join(
                    f"group {i} from block {gm.src_block.block_id}"
                    for i, gm in enumerate(group_matches)
                ),
            )

        return match

    def apply_sub_block_match(self, match: SubBlockMatch) -> None:
        """Create copy ops from a sub-block match.

        Call this after ``allocate_slots`` succeeds. The source-block refs
        owned by *match* are transferred to the pending copy ops (released
        by ``release_copy_ops``).
        """
        request = match.request
        blocks = self.coordinator.get_blocks(request.request_id)

        for i, gm in enumerate(match.group_matches):
            block_list = blocks[i]
            # By this point, the destination block must have been allocated.
            dst_block = block_list[gm.dst_req_block_index]
            self.pending_copy_ops.append(
                KVCacheCopyOp(
                    group_id=i,
                    src_block_id=gm.src_block.block_id,
                    dst_block_id=dst_block.block_id,
                    num_tokens=match.num_tokens,
                )
            )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=0,  # already counted in get_computed_blocks
                num_hits=match.num_tokens,
                preempted=request.num_preemptions > 0,
            )

    def release_sub_block_match(self, match: SubBlockMatch) -> None:
        """Release source-block references held by a sub-block match.

        Call this when discarding a match (e.g. connector provides better
        coverage, or allocation failed).
        """
        self.block_pool.free_blocks([gm.src_block for gm in match.group_matches])

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
    ) -> KVCacheBlocks | None:
        num_full_blocks_before = tuple(
            request.num_computed_tokens // gi.block_size for gi in self._group_infos
        )

        result = super().allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
        )

        if result is not None:
            # Defer sub-block indexing until after execute_model writes KV cache,
            # so that concurrent prefills in the same step cannot match sub-blocks
            # whose KV data does not yet exist.
            self._pending_indexing[request.request_id] = (
                request,
                num_full_blocks_before,
            )

        return result

    def drain_pending_copy_ops(self) -> list[KVCacheCopyOp]:
        """Return and clear all pending copy operations.

        Source-block refs are **retained** so the data remains valid until
        the model runner finishes the copies.  The caller must call
        :meth:`release_copy_ops` afterwards to free the refs.
        """
        ops = self.pending_copy_ops
        self.pending_copy_ops = []
        return ops

    def release_copy_ops(self, ops: list[KVCacheCopyOp]) -> None:
        """Release source-block refs held by previously drained copy ops."""
        if ops:
            self.block_pool.free_blocks(
                [self.block_pool.blocks[op.src_block_id] for op in ops]
            )

    def do_pending_indexing(self) -> None:
        """Index sub-blocks for requests whose indexing was deferred.

        Must be called **after** ``super().update_from_output()`` so that
        ``num_computed_tokens`` is up-to-date and ``free()`` has already
        consumed its own pending entries.
        """
        for request, num_full_blocks_before in self._pending_indexing.values():
            self._index_newly_cached_blocks(request, num_full_blocks_before)
            self._index_partial_block(request, False)
        self._pending_indexing.clear()

    def free(self, request: Request) -> None:
        """Index any not-yet-indexed sub-blocks, free blocks, and clean up
        sub-block state for the request."""
        # Consume this request's pending indexing entry and index all blocks
        # (full + partial) before releasing them.
        pending = self._pending_indexing.pop(request.request_id, None)
        if pending is not None:
            _, num_full_blocks_before = pending
            self._index_newly_cached_blocks(request, num_full_blocks_before)
        self._index_partial_block(request, True)

        # Clean up request states
        del self._req_sub_hashes[request.request_id]
        super().free(request)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache including all per-group sub-block indices."""
        result = super().reset_prefix_cache()
        if result:
            for gi in self._group_infos:
                gi.sub_block_index = SubBlockIndex()
            self._pending_indexing.clear()
        return result

    # -- sub-block hash helpers ---------------------------------------------

    def _get_or_compute_sub_hashes(self, request: Request) -> list[BlockHash]:
        """Return the sub-block hashes for the request, computing new ones if
        the request has grown since the last call.  Group-independent."""
        state = self._req_sub_hashes.get(request.request_id)
        if state is None:
            state = _SubHashState(hashes=[])
            self._req_sub_hashes[request.request_id] = state

        num_hashed_tokens = len(state.hashes) * self.sub_block_size
        parent_hash = state.hashes[-1] if state.hashes else None

        new_hashes, new_mm_idx = self.sub_block_hasher.hash_tokens(
            request.all_token_ids,
            parent_hash=parent_hash,
            num_hashed_tokens=num_hashed_tokens,
            request=request,
            start_mm_idx=state.mm_idx,
        )
        if new_hashes:
            state.hashes.extend(new_hashes)
            state.mm_idx = new_mm_idx
        return state.hashes

    # -- sub-block index maintenance ----------------------------------------

    def _index_newly_cached_blocks(
        self, request: Request, num_full_blocks_before: tuple[int, ...]
    ) -> None:
        """After allocate_slots caches new full blocks, index their sub-blocks."""
        blocks = self.coordinator.get_blocks(request.request_id)
        for gi, block_list, before in zip(
            self._group_infos, blocks, num_full_blocks_before
        ):
            for blk_idx in range(before, len(block_list)):
                blk = block_list[blk_idx]
                # Only for newly cached full blocks.
                # NOTE: For a new request, num_full_blocks_before is zero
                # because num_computed_tokens is set after scheduling.
                # So this doesn't tell us which blocks are newly cached.
                # _on_block_cached uses update() which is idempotent,
                # so re-processing already-indexed blocks is fine.
                if blk.block_hash is not None:
                    self._on_block_cached(request, blk_idx, blk, gi)

    def _on_block_cached(
        self,
        request: Request,
        blk_idx: int,
        blk: KVCacheBlock,
        gi: _GroupInfo,
    ) -> None:
        """Index a newly cached full block's sub-blocks."""
        sub_hashes = self._get_or_compute_sub_hashes(request)
        sbpb = gi.sub_blocks_per_block
        sub_start = blk_idx * sbpb
        sub_end = min(sub_start + sbpb, len(sub_hashes))
        if sub_end <= sub_start:
            return
        blk_sub_hashes = sub_hashes[sub_start:sub_end]
        gi.sub_block_index.update(blk.block_id, blk_sub_hashes)

    def _index_partial_block(self, request: Request, mark_cached: bool) -> None:
        """Index sub-blocks of the last partial block per group.

        Args:
            mark_cached: Whether to assign a synthetic ``block_hash`` so the
                upstream LRU preserves the block.
        """
        num_computed_tokens = request.num_computed_tokens
        sub_hashes = self._get_or_compute_sub_hashes(request)
        blocks = self.coordinator.get_blocks(request.request_id)

        for gid, gi in enumerate(self._group_infos):
            remainder = num_computed_tokens % gi.block_size
            if remainder == 0:
                continue  # No partial block.

            num_sub_blocks = remainder // self.sub_block_size
            if num_sub_blocks == 0:
                continue

            block_list = blocks[gid]
            last_blk_idx = num_computed_tokens // gi.block_size
            if last_blk_idx >= len(block_list):
                continue
            blk = block_list[last_blk_idx]

            sbpb = gi.sub_blocks_per_block
            sub_start = last_blk_idx * sbpb
            sub_end = min(sub_start + num_sub_blocks, len(sub_hashes))
            if sub_end <= sub_start:
                continue
            partial_sub_hashes = sub_hashes[sub_start:sub_end]

            # Index in the group's sub-block index.
            gi.sub_block_index.update(blk.block_id, partial_sub_hashes)

            if mark_cached:
                # Give the block a synthetic block_hash so the upstream block pool
                # keeps it in the LRU cache instead of immediately reusing it.
                # The actual value doesn't matter as long as it' unique so that it
                # doesn't collide with any real full-block hash.
                synthetic_hash = make_block_hash_with_group_id(
                    BlockHash(
                        b"partial_block_"
                        + str(blk.block_id).encode("ascii")
                        + b"_"
                        + partial_sub_hashes[-1]
                    ),
                    gid,
                )
                assert blk.block_hash is None
                blk.block_hash = synthetic_hash
                self.block_pool.cached_block_hash_to_block.insert(synthetic_hash, blk)

    def _on_block_evicted(self, block_id: int) -> None:
        """Called when a block is evicted — remove from index."""
        for gi in self._group_infos:
            gi.sub_block_index.pop(block_id)

    def _install_eviction_hook(self) -> None:
        # Monkey-patch the block pool's eviction method to also clean up the
        # sub-block index. We can't simply override evict_blocks because we
        # need to know if eviction actually happened.
        original_evict = self.block_pool._maybe_evict_cached_block

        def evict_with_index_cleanup(block: KVCacheBlock) -> bool:
            block_id = block.block_id
            result = original_evict(block)
            if result:
                self._on_block_evicted(block_id)
            return result

        self.block_pool._maybe_evict_cached_block = evict_with_index_cleanup
