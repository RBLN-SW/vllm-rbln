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


from collections.abc import Callable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.optimum_kv_cache_coordinator import RBLNKVCacheCoordinator
from vllm_rbln.v1.core.sub_block import (
    KVCacheCopyOp,
    SubBlockHasher,
    SubBlockIndex,
)

logger = init_logger(__name__)


@dataclass(slots=True)
class _SubHashState:
    """Per-request cached sub-block hashes and multimodal index."""

    hashes: list[BlockHash]
    mm_idx: int = 0


@dataclass(slots=True)
class SubBlockMatch:
    """Result of a sub-block partial match lookup.

    Returned by ``RBLNKVCacheManager.get_computed_blocks_sub_block``.
    The caller must either pass it to ``apply_sub_block_match`` (to create a
    copy op) or ``release_sub_block_match`` (to discard it and free the
    source-block reference).
    """

    request: Request
    num_tokens: int
    src_block: KVCacheBlock
    # Index of the block (in the request's block list) that will receive
    # the copied sub-blocks.
    dst_req_block_index: int


class RBLNKVCacheManager(KVCacheManager):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_num_batched_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
        is_encoder_decoder: bool = False,
        prefill_chunk_size: int | None = None,
        image_prefill_chunk_size: list[int] | None = None,
        needs_chunked_prefill_pad: bool = False,
        sub_block_size: int | None = None,
        hash_fn: Callable | None = None,
    ) -> None:
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        self.scheduler_block_size = scheduler_block_size
        assert watermark == 0.0, "watermark is not supported on the RBLN optimum path"
        self.watermark_blocks = 0
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None
        # NOTE(eunji.lee):
        # This is the scheduler's per-step token budget, not
        # scheduler_config.max_num_batched_tokens (which on the optimum path is
        # the compiled prefill chunk size). It only feeds the recycling-aware
        # admission cap for SWA / chunked-local specs, clamped there by
        # max_model_len; full/cross-attention block allocation (e.g. Whisper)
        # sizes purely off the request's own tokens.
        assert max_num_batched_tokens is not None, "max_num_batched_tokens must be set."
        self.coordinator = RBLNKVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            is_encoder_decoder=is_encoder_decoder,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        # gemma3/gemma4: optimum-rbln's chunked prefill touches extra KV-cache
        # slots beyond the prompt length (partition-alignment + trailing chunk
        # write-extent). `block_size` here equals `kvcache_partition_len`
        # (compilation sets `kvcache_partition_len = block_size`), so the same
        # value drives the boundary check in `allocate_slots`.
        self.block_size = block_size
        self.needs_chunked_prefill_pad = needs_chunked_prefill_pad
        self.prefill_chunk_size = prefill_chunk_size
        # gemma3: single image bucket; gemma4: descending list of buckets.
        # Used to size the per-image chunk in `_chunked_prefill_pad`.
        self.image_prefill_chunk_size = image_prefill_chunk_size
        if needs_chunked_prefill_pad:
            assert prefill_chunk_size is not None, (
                "prefill_chunk_size is required when needs_chunked_prefill_pad "
                "is set (gemma3/gemma4)."
            )
        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
        # Cache the chunked-prefill padding per request. It depends only on the
        # prompt (fixed for the request's lifetime), so compute it once at the
        # first allocate_slots and reuse it on every later (decode) call.
        self._chunked_prefill_pad_cache: dict[str, int] = {}

        # Sub-block prefix caching: extends full-block hits at sub-block
        # granularity via copy ops executed by the model runner. See
        # docs/sub_block_prefix_caching.md; unlike the native manager this is
        # single-group (the optimum coordinator is Unitary) and emits no KV
        # events.
        self.sub_block_size = sub_block_size
        self.pending_copy_ops: list[KVCacheCopyOp] = []
        self._pending_indexing: dict[str, Request] = {}
        self._req_sub_hashes: dict[str, _SubHashState] = {}
        if sub_block_size is not None:
            assert enable_caching, "sub-block caching requires prefix caching"
            assert hash_fn is not None, "sub-block caching requires a hash_fn"
            assert block_size > sub_block_size and block_size % sub_block_size == 0, (
                "sub_block_size must strictly divide block_size"
            )
            self.sub_blocks_per_block = block_size // sub_block_size
            self.sub_block_index = SubBlockIndex()
            self.sub_block_hasher = SubBlockHasher(hash_fn, sub_block_size)
            self.block_pool.evicted_block_hook = self._on_block_evicted

    def _image_embed_segments(
        self, request: Request, query_len: int
    ) -> list[tuple[int, int]]:
        """Contiguous image-embed token runs (start, end), sorted by start.

        Uses `mm_position.is_embed` so runs are the actual image tokens, not the
        whole placeholder (which also holds text-like boi/eoi/\\n\\n tokens).
        """
        segments: list[tuple[int, int]] = []
        for f in request.mm_features:
            pos = f.mm_position
            start = pos.offset
            assert pos.is_embed is not None, (
                "mm_position.is_embed must be set for image placeholders"
            )
            mask = pos.is_embed.tolist()  # per-position embed flags within placeholder
            i, n = 0, len(mask)
            while i < n:
                if mask[i]:
                    # Start of an embed run; extend `j` to its end.
                    j = i
                    while j < n and mask[j]:
                        j += 1
                    # Record the run in absolute prompt positions (clamped).
                    if start + i < query_len:
                        segments.append((start + i, min(start + j, query_len)))
                    i = j  # jump past this run
                else:
                    i += 1  # text token, skip
        # Sort by start position (tuple order: by `start`, then `end`) so the
        # runs are returned in prompt order; features may arrive out of order.
        segments.sort()
        return segments

    def _image_chunk_size(self, run_len: int) -> int:
        buckets = self.image_prefill_chunk_size
        if not buckets:
            assert self.prefill_chunk_size is not None, (
                "prefill_chunk_size must be set when image_prefill_chunk_size is empty"
            )
            return self.prefill_chunk_size
        # buckets is descending, so `reversed` is ascending: the first bucket
        # that is >= run_len is the smallest one that fits.
        chunk = next((b for b in reversed(buckets) if b >= run_len), None)
        if chunk is None:
            raise ValueError(
                f"image run of {run_len} tokens exceeds the largest "
                f"image-prefill bucket ({buckets[0]})"
            )
        return chunk

    def _chunked_prefill_pad(self, request: Request, query_len: int) -> int:
        # FIXME chunk size?????
        text_chunk = self.prefill_chunk_size
        assert text_chunk is not None, (
            "prefill_chunk_size must be set when needs_chunked_prefill_pad is True"
        )
        block_size = self.block_size
        image_segments = self._image_embed_segments(request, query_len)
        # `step`: next prompt token to process (excludes alignment padding).
        # `align_pad`: alignment padding so far; the token sits at cache slot
        #   `step + align_pad`.
        #
        # Each run is one chunk: an image run uses its bucket as the chunk size
        step = 0
        align_pad = 0
        while step < query_len:
            seg_end = next((e for s, e in image_segments if s <= step < e), None)
            if seg_end is not None:
                # image run: processed as one bucket-sized chunk.
                run_len = seg_end - step
                chunk_size = self._image_chunk_size(run_len)
            else:
                # text run: up to the next image, in `text_chunk` pieces.
                run_end = min(
                    (s for s, _ in image_segments if s > step), default=query_len
                )
                run_len = min(run_end - step, text_chunk)
                chunk_size = text_chunk

            # Pad to the block boundary if this chunk would straddle one.
            # `offset_in_block`: this chunk's first cache slot within its block.
            offset_in_block = (step + align_pad) % block_size
            if offset_in_block + chunk_size > block_size:
                align_pad += block_size - offset_in_block
            step += run_len
        return align_pad

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request."""
        if self.sub_block_size is not None:
            self._finalize_sub_block_state(request)
        super().free(request)
        self._chunked_prefill_pad_cache.pop(request.request_id, None)

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
        assert num_lookahead_tokens == 0
        assert num_external_computed_tokens == 0
        assert not delay_cache_blocks
        assert num_encoder_tokens == 0
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # In prefill,
        # `request.num_computed_tokens` = 0,
        # `num_new_computed_tokens` = the prefix-cache hit length,
        # `num_new_tokens` = the rest of the prompt.
        # In decode,
        # `request.num_computed_tokens` = the length of prompt + generated
        # text, `num_new_tokens` = 1.
        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_main_model = min(
            num_computed_tokens + num_new_tokens, self.max_model_len
        )
        num_tokens_need_slot = num_tokens_main_model
        if self.needs_chunked_prefill_pad:
            # gemma3/gemma4: reserve the partition-alignment + trailing-chunk
            # slots optimum-rbln's chunked prefill touches beyond the prompt.
            # The padding is fixed by the prompt; later decode tokens append on
            # top of it, so compute it once and reuse it on later calls.
            pad = self._chunked_prefill_pad_cache.get(request.request_id)
            if pad is None:
                pad = self._chunked_prefill_pad(
                    request, min(request.num_prompt_tokens, self.max_model_len)
                )
                self._chunked_prefill_pad_cache[request.request_id] = pad
            num_tokens_need_slot += pad

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=0,
            total_computed_tokens=num_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        if new_computed_block_list is not self.empty_kv_cache_blocks.blocks:
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_computed_tokens,
                num_external_computed_tokens=0,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_tokens_main_model
        )

        if not self.enable_caching:
            return self.create_kv_cache_blocks(new_blocks)

        # Cache the full blocks covered by tokens computed after this step.
        # The cap at `request.num_tokens` keeps chunked-prefill pad slots out
        # of the hash.
        num_tokens_to_cache = min(num_tokens_main_model, request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)
        if self.sub_block_size is not None:
            self.schedule_sub_block_indexing(request)
        return self.create_kv_cache_blocks(new_blocks)

    def get_dummy_block(self) -> int:
        # Reserved by RBLNBlockPool for encoder-decoder models and for prefix
        # caching (decode-batch padding must not write into cached blocks).
        assert self.block_pool.dummy_block is not None, (
            "No dummy block is reserved in the block pool"
        )
        # In V1, block ID 0 is the null_block, so scheduler-side block
        # IDs start at 1. The compiler expects valid blocks to start at
        # 0, so shift by -1 to translate into compiler-space.
        return self.block_pool.dummy_block.block_id - 1

    # -- sub-block prefix caching ---------------------------------------------

    def get_computed_blocks_sub_block(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> SubBlockMatch | None:
        """Discover a sub-block partial match after full-block matching.

        Looks for sub-block prefix matches in the block immediately after the
        full-block boundary. Returns a ``SubBlockMatch`` handle (holding a
        reference on the source block) or ``None``.
        """
        if self.sub_block_size is None or request.skip_reading_prefix_cache:
            return None

        sub_hashes = self._get_or_compute_sub_hashes(request).hashes
        num_full_blocks = num_computed_tokens // self.block_size
        next_sub_start = num_full_blocks * self.sub_blocks_per_block

        # At least one sub-block beyond the full-block match AND fewer than a
        # full block (a full-block-sized match is upstream's job).
        query = sub_hashes[
            next_sub_start : next_sub_start + self.sub_blocks_per_block - 1
        ]
        if not query:
            return None
        src_block_id, num_matched = self.sub_block_index.longest_match(query)
        if src_block_id is None:
            return None

        # Upstream guarantees num_computed_tokens <= num_tokens - 1, because
        # at least the last token must be recomputed for logits. Enforce the
        # same condition here.
        max_allowed_sub_blocks = (
            request.num_tokens - 1 - num_computed_tokens
        ) // self.sub_block_size
        num_matched = min(num_matched, max_allowed_sub_blocks)
        if num_matched <= 0:
            return None

        # Touch the source block to prevent eviction during allocate_slots.
        # The SubBlockMatch owns this reference until apply or release.
        src_block = self.block_pool.blocks[src_block_id]
        self.block_pool.touch([src_block])
        return SubBlockMatch(
            request=request,
            num_tokens=num_matched * self.sub_block_size,
            src_block=src_block,
            dst_req_block_index=num_full_blocks,
        )

    def apply_sub_block_match(self, match: SubBlockMatch) -> None:
        """Create a copy op from a sub-block match.

        Call this after ``allocate_slots`` succeeds. The source-block ref
        owned by *match* is transferred to the pending copy op (released by
        ``release_copy_ops``).
        """
        blocks = self.coordinator.get_blocks(match.request.request_id)[0]
        dst_block = blocks[match.dst_req_block_index]
        self.pending_copy_ops.append(
            KVCacheCopyOp(
                group_id=0,
                src_block_id=match.src_block.block_id,
                dst_block_id=dst_block.block_id,
                num_tokens=match.num_tokens,
            )
        )
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=0,  # already counted in get_computed_blocks
                num_hits=match.num_tokens,
                preempted=match.request.num_preemptions > 0,
            )

    def release_sub_block_match(self, match: SubBlockMatch) -> None:
        """Release the source-block reference held by a discarded match."""
        self.block_pool.free_blocks([match.src_block])

    def drain_pending_copy_ops(self) -> list[KVCacheCopyOp]:
        """Return and clear all pending copy operations.

        Source-block refs are retained so the data remains valid until the
        model runner finishes the copies. The caller must call
        ``release_copy_ops`` afterwards to free the refs.
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

    def schedule_sub_block_indexing(self, request: Request) -> None:
        """Record that *request* needs sub-block indexing in the next
        ``do_pending_indexing`` call."""
        self._pending_indexing[request.request_id] = request

    def do_pending_indexing(self) -> None:
        """Index sub-blocks for requests whose indexing was deferred.

        Must be called after ``super().update_from_output()`` so that
        ``num_computed_tokens`` covers the KV the forward pass just wrote and
        ``free()`` has already consumed its own pending entries.
        """
        for request in self._pending_indexing.values():
            self._index_request_blocks(request, mark_cached=False)
        self._pending_indexing.clear()

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache including the sub-block index."""
        result = super().reset_prefix_cache()
        if result and self.sub_block_size is not None:
            self.sub_block_index = SubBlockIndex()
            self._pending_indexing.clear()
        return result

    def _get_or_compute_sub_hashes(self, request: Request) -> _SubHashState:
        """Return the sub-block hash state for the request, extending it if
        the request has grown since the last call."""
        state = self._req_sub_hashes.get(request.request_id)
        if state is None:
            state = _SubHashState(hashes=[])
            self._req_sub_hashes[request.request_id] = state

        assert self.sub_block_size is not None
        num_hashed_tokens = len(state.hashes) * self.sub_block_size
        parent_hash = state.hashes[-1] if state.hashes else None
        new_hashes, _, new_mm_idx = self.sub_block_hasher.hash_tokens(
            request.all_token_ids,
            parent_hash=parent_hash,
            num_hashed_tokens=num_hashed_tokens,
            request=request,
            start_mm_idx=state.mm_idx,
        )
        if new_hashes:
            state.hashes.extend(new_hashes)
            state.mm_idx = new_mm_idx
        return state

    def _index_request_blocks(self, request: Request, mark_cached: bool) -> None:
        """Index the request's full blocks and the complete sub-blocks of its
        partial block.

        Args:
            mark_cached: Whether to assign a synthetic ``block_hash`` to the
                partial block so the upstream LRU preserves it after free.
        """
        assert self.sub_block_size is not None
        state = self._get_or_compute_sub_hashes(request)
        blocks = self.coordinator.get_blocks(request.request_id)[0]
        num_computed_tokens = request.num_computed_tokens
        sbpb = self.sub_blocks_per_block

        # Full blocks: index only hashed (upstream-cached) ones, so every
        # indexed block carries a hash and its eviction fires the pool hook.
        # Re-scanning from block 0 is fine: SubBlockIndex.update is idempotent
        # and O(1) for an already-indexed block.
        num_full_blocks = num_computed_tokens // self.block_size
        for blk_idx in range(min(num_full_blocks, len(blocks))):
            blk = blocks[blk_idx]
            if blk.block_hash is None:
                continue
            sub_start = blk_idx * sbpb
            sub_end = min(sub_start + sbpb, len(state.hashes))
            if sub_end > sub_start:
                self.sub_block_index.update(
                    blk.block_id, state.hashes[sub_start:sub_end]
                )

        # Partial block: index its complete sub-blocks.
        remainder = num_computed_tokens % self.block_size
        num_sub_blocks = remainder // self.sub_block_size
        if num_sub_blocks == 0 or num_full_blocks >= len(blocks):
            return
        blk = blocks[num_full_blocks]
        sub_start = num_full_blocks * sbpb
        sub_end = min(sub_start + num_sub_blocks, len(state.hashes))
        if sub_end <= sub_start:
            return
        partial_sub_hashes = state.hashes[sub_start:sub_end]
        self.sub_block_index.update(blk.block_id, partial_sub_hashes)

        if mark_cached and blk.block_hash is None:
            # Give the block a synthetic block_hash so the upstream block pool
            # keeps it in the LRU cache instead of immediately reusing it, and
            # so its eventual eviction fires the pool hook that pops the
            # index. The value only needs to be unique.
            synthetic_hash = make_block_hash_with_group_id(
                BlockHash(
                    b"partial_block_"
                    + str(blk.block_id).encode("ascii")
                    + b"_"
                    + partial_sub_hashes[-1]
                ),
                0,
            )
            blk.set_block_hash(synthetic_hash, num_tokens=num_computed_tokens)
            self.block_pool.cached_block_hash_to_block.insert(synthetic_hash, blk)

    def _finalize_sub_block_state(self, request: Request) -> None:
        """Index what this request still owes, then drop its per-request
        state. Runs before the blocks leave this manager."""
        self._pending_indexing.pop(request.request_id, None)
        if request.request_id in self._req_sub_hashes or request.num_computed_tokens:
            self._index_request_blocks(request, mark_cached=True)
        self._req_sub_hashes.pop(request.request_id, None)

    def _on_block_evicted(self, block_id: int) -> None:
        self.sub_block_index.pop(block_id)
