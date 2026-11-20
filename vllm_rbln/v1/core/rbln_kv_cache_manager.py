# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    to_event_extra_keys,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    iter_layer_specs,
)

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

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
    granularity.  Per-sub-block ``extra_keys`` (cache_salt, LoRA,
    multimodal, prompt_embeds) are mixed in, mirroring upstream
    full-block hashing.
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
        request: Request,
        *,
        parent_hash: BlockHash | None = None,
        num_hashed_tokens: int = 0,
        start_mm_idx: int = 0,
    ) -> tuple[list[BlockHash], list[tuple[Any, ...] | None], int]:
        """Return sub-block hashes for *full* sub-blocks in ``token_ids``.

        Args:
            token_ids: Full token sequence of the request.
            request: Source of the extra hash keys (LoRA, cache_salt,
                multimodal, prompt_embeds) mixed into each sub-block hash.
            parent_hash: Hash of the last sub-block before the range we
                are hashing (``None`` for the very first sub-block).
            num_hashed_tokens: Number of tokens already hashed (i.e. the
                start offset into ``token_ids``).
            start_mm_idx: Starting multimodal feature index for
                incremental hashing with multimodal requests.

        Returns:
            ``(hashes, extra_keys, mm_idx)``.  ``extra_keys`` is a list
            (aligned with ``hashes``) of the per-sub-block extra-keys
            tuples actually mixed into the hash, or ``None`` for
            sub-blocks that had no extra keys — returned so callers can
            reproduce the hash when emitting KV events.
        """
        # Based on upstream request_block_hasher() in get_request_block_hasher().

        sbs = self.sub_block_size
        hashes: list[BlockHash] = []
        extra_keys_list: list[tuple[Any, ...] | None] = []
        # NOTE: We can't simply use `mm_idx=-1`,
        # because it means the last mm input in the entire prompt,
        # which is meant to be used during decode phase.
        mm_idx = start_mm_idx
        start = num_hashed_tokens
        for i in range(start, len(token_ids) - sbs + 1, sbs):
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
            extra_keys_list.append(extra_keys)
        return hashes, extra_keys_list, mm_idx


class SubBlockIndex:
    """Index mapping sub-block hashes to physical block IDs that contain that
    sub-block as a prefix."""

    def __init__(self) -> None:
        # sub_block_hash → set of block IDs with that prefix cached.
        self._hash_to_blocks: dict[BlockHash, set[int]] = {}
        # Reverse index: block_id → list of sub-block hashes (for removal).
        self._block_hashes: dict[int, list[BlockHash]] = {}

    def update(self, block_id: int, sub_block_hashes: list[BlockHash]) -> int:
        """Index or extend a block's sub-block hashes (idempotent).

        Returns ``first_fresh_idx``: the index in ``sub_block_hashes`` at
        which the first globally-fresh (previously absent from the whole
        index) hash was added by this call.
        ``first_fresh_idx == len(sub_block_hashes)`` means nothing was fresh.
        """
        existing = self._block_hashes.setdefault(block_id, [])
        first_fresh_idx = len(sub_block_hashes)
        for i in range(len(existing), len(sub_block_hashes)):
            h = sub_block_hashes[i]
            bucket = self._hash_to_blocks.setdefault(h, set())
            if not bucket and i < first_fresh_idx:
                first_fresh_idx = i
            existing.append(h)
            bucket.add(block_id)
        return first_fresh_idx

    def pop(self, block_id: int) -> list[BlockHash]:
        """Remove a block from the index (called on eviction).

        Returns the list of hashes whose bucket became empty as a result.
        An empty list means nothing was removed or every removed hash is still
        held by another block.
        """
        hashes = self._block_hashes.pop(block_id, None)
        if hashes is None:
            return []
        fully_removed: list[BlockHash] = []
        for h in hashes:
            s = self._hash_to_blocks.get(h)
            if s is not None:
                s.discard(block_id)
                if not s:
                    del self._hash_to_blocks[h]
                    fully_removed.append(h)
        return fully_removed

    def all_hashes(self) -> list[BlockHash]:
        """All currently-indexed hashes (order unspecified)."""
        return list(self._hash_to_blocks.keys())

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
    """Per-request cached sub-block hashes and multimodal index.

    ``extra_keys`` is aligned 1:1 with ``hashes`` and records the
    extra-keys tuple mixed into each sub-block's hash (or ``None`` when
    the sub-block had no extras). ``num_indexed_full_blocks`` counts, per
    group, the leading full blocks whose sub-blocks are already indexed.
    """

    hashes: list[BlockHash]
    extra_keys: list[tuple[Any, ...] | None] = field(default_factory=list)
    mm_idx: int = 0
    num_indexed_full_blocks: list[int] = field(default_factory=list)
    retry_without_copy: bool = False


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
    group_id: int
    num_tokens: int
    src_block: KVCacheBlock
    dst_req_block_index: int


@dataclass(slots=True)
class _GroupInfo:
    """Per-group configuration for sub-block caching."""

    group_id: int
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
        1. Every layer's spec type must store per-token KV data (allow-listed
           in ``_SUB_BLOCK_ELIGIBLE_SPECS``).
        2. Every group must have block_size > sub_block_size and
           block_size % sub_block_size == 0.
        """
        if sub_block_size <= 0:
            return False
        for group in kv_cache_config.kv_cache_groups:
            for spec in iter_layer_specs(group.kv_cache_spec):
                # The shift kernel overwrites the prefix with the latest window.
                if isinstance(spec, RBLNSlidingWindowSpec):
                    return False
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
        draft_shares_target_kv: bool = False,
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
        # Sliding-window retention keeps only the blocks a hit at this alignment
        # can reach, and a sub-block hit ends at any sub-block boundary.
        for manager in self.coordinator.single_type_managers:
            manager.cache_hit_alignment_tokens = sub_block_size

        self.sub_block_size = sub_block_size

        # Build per-group info.
        self._group_infos: list[_GroupInfo] = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            bs = group.kv_cache_spec.block_size
            self._group_infos.append(
                _GroupInfo(
                    group_id=gid,
                    block_size=bs,
                    sub_blocks_per_block=bs // sub_block_size,
                    sub_block_index=SubBlockIndex(),
                )
            )

        block_sizes = {group.block_size for group in self._group_infos}
        self._max_sub_block_extension = max_model_len
        # A drafter writing lookahead KV needs upstream's trailing block drop.
        if len(block_sizes) == 1 or (self.use_eagle and not draft_shares_target_kv):
            self._max_sub_block_extension = min(block_sizes) - sub_block_size

        self.sub_block_hasher = SubBlockHasher(hash_fn, sub_block_size)

        # Per-request sub-block hash cache (group-independent).
        self._req_sub_hashes: dict[str, _SubHashState] = {}

        # Copy operations accumulated during a scheduling step; the scheduler
        # drains this list when building SchedulerOutput. While pending, each
        # copy op holds a ref count of its source block to prevent eviction.
        self.pending_copy_ops: list[KVCacheCopyOp] = []

        # Requests whose newly cached blocks still need sub-block indexing.
        self._pending_indexing: dict[str, Request] = {}

        # Sub-block-granular KV event queue. We intercept the upstream pool's
        # big-block event emission and publish sub-block events instead.
        self.enable_kv_cache_events = self.block_pool.enable_kv_cache_events
        self.block_pool.enable_kv_cache_events = False
        self._sub_block_event_queue: list[KVCacheEvent] = []

        self._install_eviction_hook()

    def get_computed_blocks_sub_block(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> SubBlockMatch | None:
        """Discover a sub-block partial match after full-block matching.

        The caller must apply the match after allocation or release its source
        block references.
        """
        if request.skip_reading_prefix_cache:
            return None
        # Copies start at offset zero in newly allocated destination blocks.
        if any(num_computed_tokens % group.block_size for group in self._group_infos):
            return None

        sub_size = self.sub_block_size
        first = num_computed_tokens // sub_size
        common_end = (
            min(
                request.num_tokens - 1,
                num_computed_tokens + self._max_sub_block_extension,
            )
            // sub_size
        )
        if common_end <= first:
            return None
        state = self._get_or_compute_sub_hashes(request)
        if state.retry_without_copy:
            state.retry_without_copy = False
            return None
        candidates: list[_GroupPartialMatch] = []
        for group in self._group_infos:
            pos = first
            while pos < common_end:
                end = min(pos + group.sub_blocks_per_block, common_end)
                block_id, depth = group.sub_block_index.longest_match(
                    state.hashes[pos:end]
                )
                if block_id is None:
                    break
                candidates.append(
                    _GroupPartialMatch(
                        group_id=group.group_id,
                        src_block=self.block_pool.blocks[block_id],
                        dst_req_block_index=pos // group.sub_blocks_per_block,
                        num_tokens=depth * sub_size,
                    )
                )
                pos += depth
                if pos < end:
                    break
            if pos == first:
                return None
            common_end = pos

        group_matches = []
        for span in candidates:
            block_size = self._group_infos[span.group_id].block_size
            start = span.dst_req_block_index * block_size
            remaining = common_end * sub_size - start
            if remaining > 0:
                span.num_tokens = min(span.num_tokens, remaining)
                group_matches.append(span)
        extra_tokens = (common_end - first) * sub_size

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
                common_end - first,
                extra_tokens,
                ", ".join(
                    f"group {gm.group_id} block {gm.src_block.block_id} -> "
                    f"slot {gm.dst_req_block_index} ({gm.num_tokens} tokens)"
                    for gm in group_matches
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

        for gm in match.group_matches:
            dst_block = blocks[gm.group_id][gm.dst_req_block_index]
            if dst_block.is_null:
                self.block_pool.free_blocks([gm.src_block])
                continue
            self.pending_copy_ops.append(
                KVCacheCopyOp(
                    group_id=gm.group_id,
                    src_block_id=gm.src_block.block_id,
                    dst_block_id=dst_block.block_id,
                    num_tokens=gm.num_tokens,
                )
            )

    def release_sub_block_match(
        self, match: SubBlockMatch, *, retry_without_copy: bool = False
    ) -> None:
        """Release source-block references held by a sub-block match.

        Call this when discarding a match (e.g. connector provides better
        coverage, or allocation failed).
        """
        self.block_pool.free_blocks([gm.src_block for gm in match.group_matches])
        if retry_without_copy and len(match.group_matches) > len(self._group_infos):
            self._req_sub_hashes[match.request.request_id].retry_without_copy = True

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
    ) -> KVCacheBlocks | None:
        result = super().allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )

        if result is not None and not delay_cache_blocks:
            # When delay_cache_blocks=True, the caller is responsible for
            # calling schedule_sub_block_indexing() after cache_blocks().
            self.schedule_sub_block_indexing(request)

        return result

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        # The hybrid coordinator rounds down to its block-aligned hit length,
        # which leaves a group with smaller blocks nothing to index for a
        # sub-block hit; the base coordinator caches every full block.
        if self.enable_caching:
            KVCacheCoordinator.cache_blocks(
                self.coordinator, request, num_computed_tokens
            )

    def schedule_sub_block_indexing(self, request: Request) -> None:
        """Record that *request* needs sub-block indexing in the next
        ``do_pending_indexing`` call.

        When ``allocate_slots`` is called with ``delay_cache_blocks=False``,
        this is called automatically.  Otherwise the caller must call it
        """
        self._pending_indexing[request.request_id] = request

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
        for request in self._pending_indexing.values():
            self._index_newly_cached_blocks(request)
            self._index_partial_block(request, False)
        self._pending_indexing.clear()

    def _finalize_sub_block_state(self, request: Request) -> None:
        """Index what this request still owes, then drop its per-request state.

        Runs before the blocks leave this manager, since indexing reads them
        through the coordinator.
        """
        # Consume this request's pending indexing entry and index all blocks
        # (full + partial) before releasing them.
        if self._pending_indexing.pop(request.request_id, None) is not None:
            self._index_newly_cached_blocks(request)
        self._index_partial_block(request, True)

        # Clean up request states
        del self._req_sub_hashes[request.request_id]

    def free(self, request: Request) -> None:
        """Index any not-yet-indexed sub-blocks, free blocks, and clean up
        sub-block state for the request."""
        self._finalize_sub_block_state(request)
        super().free(request)

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Same finalization as ``free()`` on the deferred path.

        The scheduler takes this one when an in-flight step may still write the
        blocks, so it holds them and returns them to the pool later. What settles
        here is tied to the blocks leaving, not to the request ending: a preempted
        request comes through too and goes back on the waiting queue, where the
        state it needs is rebuilt from its tokens on resume.
        """
        self._finalize_sub_block_state(request)
        return super().pop_blocks_for_free(request)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache including all per-group sub-block indices."""
        result = super().reset_prefix_cache()
        if result:
            if self.enable_kv_cache_events:
                self._sub_block_event_queue.append(AllBlocksCleared())
            for gi in self._group_infos:
                gi.sub_block_index = SubBlockIndex()
            self._pending_indexing.clear()
        return result

    def take_events(self) -> list[KVCacheEvent]:
        """Return and clear the sub-block KV event queue.

        Overrides upstream to return sub-block-granular events (big-block
        emission from the underlying pool is suppressed in ``__init__``).
        """
        if not self.enable_kv_cache_events:
            return []
        events = self._sub_block_event_queue
        self._sub_block_event_queue = []
        return events

    # -- sub-block hash helpers ---------------------------------------------

    def _get_or_compute_sub_hashes(self, request: Request) -> _SubHashState:
        """Return the sub-block hash state for the request, computing new ones if
        the request has grown since the last call. Group-independent."""
        state = self._req_sub_hashes.get(request.request_id)
        if state is None:
            state = _SubHashState(hashes=[])
            self._req_sub_hashes[request.request_id] = state

        num_hashed_tokens = len(state.hashes) * self.sub_block_size
        parent_hash = state.hashes[-1] if state.hashes else None

        new_hashes, new_extra_keys, new_mm_idx = self.sub_block_hasher.hash_tokens(
            request.all_token_ids,
            request,
            parent_hash=parent_hash,
            num_hashed_tokens=num_hashed_tokens,
            start_mm_idx=state.mm_idx,
        )
        if new_hashes:
            state.hashes.extend(new_hashes)
            state.extra_keys.extend(new_extra_keys)
            state.mm_idx = new_mm_idx
        return state

    # -- sub-block index maintenance ----------------------------------------

    def _index_newly_cached_blocks(self, request: Request) -> None:
        """Index sub-blocks for full blocks cached since the last call."""
        # Resume from what this request has indexed, not from num_computed_tokens:
        # KV loaded remotely (NIXL) or externally (LMCache) is counted as computed
        # before its blocks are cached, so that would skip them. Blocks another
        # request already indexed are re-visited; update() emits only fresh hashes.
        state = self._get_or_compute_sub_hashes(request)
        if not state.num_indexed_full_blocks:
            state.num_indexed_full_blocks = [0] * len(self._group_infos)
        blocks = self.coordinator.get_blocks(request.request_id)
        for gid, (gi, block_list) in enumerate(zip(self._group_infos, blocks)):
            num_indexed = state.num_indexed_full_blocks[gid]
            num_full_blocks = min(
                len(block_list), request.num_computed_tokens // gi.block_size
            )
            for blk_idx in range(num_indexed, num_full_blocks):
                blk = block_list[blk_idx]
                if blk.is_null or blk.block_hash is None:
                    continue
                self._on_block_cached(request, blk_idx, blk, gi)
                num_indexed = blk_idx + 1
            state.num_indexed_full_blocks[gid] = num_indexed

    def _on_block_cached(
        self,
        request: Request,
        blk_idx: int,
        blk: KVCacheBlock,
        gi: _GroupInfo,
    ) -> None:
        """Index a newly cached full block's sub-blocks."""
        state = self._get_or_compute_sub_hashes(request)
        sbpb = gi.sub_blocks_per_block
        sub_start = blk_idx * sbpb
        sub_end = min(sub_start + sbpb, len(state.hashes))
        if sub_end <= sub_start:
            return
        blk_sub_hashes = state.hashes[sub_start:sub_end]
        first_fresh_idx = gi.sub_block_index.update(blk.block_id, blk_sub_hashes)
        self._emit_block_stored(request, state, sub_start, sub_end, first_fresh_idx, gi)

    def _index_partial_block(self, request: Request, mark_cached: bool) -> None:
        """Index sub-blocks of the last partial block per group.

        Args:
            mark_cached: Whether to assign a synthetic ``block_hash`` so the
                upstream LRU preserves the block.
        """
        num_computed_tokens = request.num_computed_tokens
        state = self._get_or_compute_sub_hashes(request)
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
            if blk.is_null:
                continue

            sbpb = gi.sub_blocks_per_block
            sub_start = last_blk_idx * sbpb
            sub_end = min(sub_start + num_sub_blocks, len(state.hashes))
            if sub_end <= sub_start:
                continue
            partial_sub_hashes = state.hashes[sub_start:sub_end]

            # Index in the group's sub-block index.
            first_fresh_idx = gi.sub_block_index.update(
                blk.block_id, partial_sub_hashes
            )
            self._emit_block_stored(
                request, state, sub_start, sub_end, first_fresh_idx, gi
            )

            # A block upstream already keyed (its partial-tail entry, when
            # hash_block_size < block_size) stays in the LRU cache on its own.
            if mark_cached and blk.block_hash is None:
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
                blk.set_block_hash(synthetic_hash, num_tokens=num_computed_tokens)
                self.block_pool.cached_block_hash_to_block.insert(synthetic_hash, blk)

    def _on_block_evicted(self, block_id: int) -> None:
        """Called when a block is evicted — remove from index and emit
        ``BlockRemoved`` for hashes whose last holder was this block."""
        for gi in self._group_infos:
            fully_removed = gi.sub_block_index.pop(block_id)
            self._emit_block_removed(fully_removed, gi)

    def _install_eviction_hook(self) -> None:
        # Running partial blocks are indexed before they have an upstream hash.
        # Reuse and explicit invalidation must clear their sub-block index even
        # when upstream has no cached hash to evict.
        original_evict = self.block_pool._maybe_evict_cached_block

        def evict_with_index_cleanup(block: KVCacheBlock) -> bool:
            block_id = block.block_id
            result = original_evict(block)
            self._on_block_evicted(block_id)
            return result

        self.block_pool._maybe_evict_cached_block = evict_with_index_cleanup

    # -- event emission ------------------------------------------------------

    def _emit_block_stored(
        self,
        request: Request,
        state: _SubHashState,
        sub_start: int,
        sub_end: int,
        first_fresh_idx: int,
        gi: _GroupInfo,
    ) -> None:
        """Emit a single ``BlockStored`` for the fresh suffix of one update.

        The caller has just indexed ``state.hashes[sub_start:sub_end]`` via
        ``SubBlockIndex.update``, which returned ``first_fresh_idx``.
        The fresh suffix we emit is
        ``state.hashes[sub_start + first_fresh_idx : sub_end]``.
        """
        if not self.enable_kv_cache_events:
            return
        slice_len = sub_end - sub_start
        if first_fresh_idx >= slice_len:
            return
        fresh_start = sub_start + first_fresh_idx
        fresh_hashes = state.hashes[fresh_start:sub_end]
        parent_hash = state.hashes[fresh_start - 1] if fresh_start > 0 else None
        tok_start = fresh_start * self.sub_block_size
        tok_end = sub_end * self.sub_block_size
        token_ids = request.all_token_ids[tok_start:tok_end]
        extras = state.extra_keys[fresh_start:sub_end]
        kind, sliding_window = self.kv_cache_event_metadata[gi.group_id]
        self._sub_block_event_queue.append(
            BlockStored(
                block_hashes=[maybe_convert_block_hash(h) for h in fresh_hashes],
                parent_block_hash=(
                    maybe_convert_block_hash(parent_hash)
                    if parent_hash is not None
                    else None
                ),
                token_ids=list(token_ids),
                block_size=self.sub_block_size,
                lora_id=(
                    request.lora_request.adapter_id if request.lora_request else None
                ),
                medium=MEDIUM_GPU,
                lora_name=(request.lora_request.name if request.lora_request else None),
                extra_keys=(
                    to_event_extra_keys(extras)
                    if any(e is not None for e in extras)
                    else None
                ),
                group_idx=gi.group_id,
                kv_cache_spec_kind=kind,
                kv_cache_spec_sliding_window=sliding_window,
            )
        )

    def _emit_block_removed(
        self, fully_removed: list[BlockHash], gi: _GroupInfo
    ) -> None:
        """Emit a single ``BlockRemoved`` for hashes whose last holder is gone."""
        if not self.enable_kv_cache_events or not fully_removed:
            return
        self._sub_block_event_queue.append(
            BlockRemoved(
                block_hashes=[maybe_convert_block_hash(h) for h in fully_removed],
                medium=MEDIUM_GPU,
                group_idx=gi.group_id,
            )
        )
