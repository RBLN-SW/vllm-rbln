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


import torch
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.optimum_kv_cache_coordinator import RBLNKVCacheCoordinator
from vllm_rbln.v1.core.prefix_cache_manager import RBLNPrefixKVCacheManager

logger = init_logger(__name__)


class RBLNKVCacheManager(KVCacheManager):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        max_num_batched_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        attn_block_size: int | None = None,
        max_num_seqs: int = 1,
        is_encoder_decoder: bool = False,
        prefill_chunk_size: int | None = None,
        image_prefill_chunk_sizes: list[int] | None = None,
        needs_chunked_prefill_pad: bool = False,
    ) -> None:
        """
        RBLNKVCacheManager = KVCacheManager + PrefixKVCacheManager.
        PrefixKVCacheManager manages the mapping
        between inner blocks and outer blocks for prefix caching.
        """
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None
        # NOTE(eunji.lee):
        # max_num_batched_tokens may exceed max_model_len. It only feeds the
        # recycling-aware admission cap for SWA / chunked-local specs, and even
        # there it is clamped by max_model_len. Full/cross-attention block
        # allocation (e.g. Whisper) sizes purely off the request's own tokens.
        assert max_num_batched_tokens is not None, (
            "max_num_batched_tokens must be set in `sync_vllm_and_optimum`."
        )
        self.coordinator = RBLNKVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
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
        self.image_prefill_chunk_sizes = image_prefill_chunk_sizes
        self.attn_block_size = attn_block_size
        if needs_chunked_prefill_pad:
            assert prefill_chunk_size is not None, (
                "prefill_chunk_size is required when needs_chunked_prefill_pad "
                "is set (gemma3/gemma4)."
            )
        if enable_caching:
            assert attn_block_size is not None, (
                "attn_block_size must be specified for prefix caching"
            )
            self.prefix_cache_manager = RBLNPrefixKVCacheManager(
                ob_size=attn_block_size,
                ib_size=block_size,
                max_model_len=self.max_model_len,
                max_num_seqs=max_num_seqs,
                num_inner_blocks=self.block_pool.num_gpu_blocks - 1,
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
        buckets = self.image_prefill_chunk_sizes
        if not buckets:
            assert self.prefill_chunk_size is not None, (
                "prefill_chunk_size must be set when image_prefill_chunk_sizes is empty"
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
        block_size = (
            self.attn_block_size
            if self.attn_block_size is not None
            else self.block_size
        )
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

    def free(self, request: Request, preemption: bool = False) -> None:
        """Free the blocks allocated for the request."""
        if self.enable_caching:
            self.prefix_cache_manager.free_request(
                request.request_id, preemption=preemption
            )
        self.coordinator.free(request.request_id)
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
        assert not delay_cache_blocks
        assert num_encoder_tokens == 0
        # NOTE: They are retrieved after the blocks are allocated
        assert num_new_computed_tokens == 0
        assert new_computed_blocks is None
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        # In prefill,
        # `num_computed_tokens` = 0,
        # `num_new_tokens` = the length of the input prompt.
        # In decode,
        # `num_computed_tokens` = the length of prompt + generated text
        # `num_new_tokens` = 1.
        num_computed_tokens = request.num_computed_tokens
        num_tokens_need_slot = min(request.num_tokens, self.max_model_len)
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
            new_computed_blocks=self.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_tokens_main_model=num_tokens_need_slot,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        if self.enable_caching and not self.prefix_cache_manager.can_allocate(
            num_blocks_to_allocate,
            num_computed_tokens,
        ):
            # Cannot allocate new outer blocks for prefix caching
            return None

        # Generate req_to_blocks, num_cached_block
        # in the coordinator
        # `empty_computed_block_list` is used here to avoid
        # saving the computed blocks to the request state
        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=self.empty_kv_cache_blocks.blocks,
            num_local_computed_tokens=0,
            num_external_computed_tokens=0,
        )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, 0
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching:
            return self.create_kv_cache_blocks(new_blocks)

        # Allocate outer blocks for prefix caching
        # following the inner blocks allocation
        inner_block_ids = [block.block_id for block in new_blocks[0]]
        self.prefix_cache_manager.allocate_blocks(
            request.request_id,
            num_computed_tokens,
            inner_block_ids,
        )

        # Generate hashed values of newly allocated blocks
        # In prefill,
        # `num_new_tokens` = the length of the input prompt.
        # In decode,
        # `num_new_tokens` = 1.
        self.coordinator.cache_blocks(request, num_new_tokens)
        return self.create_kv_cache_blocks(new_blocks)

    def get_prefix_cached_blocks(
        self,
        request: Request,
        new_computed_blocks: KVCacheBlocks,
        num_new_computed_tokens: int,
    ) -> tuple[list[int], list[int]]:
        cached_blocks = new_computed_blocks.get_block_ids()[0]
        cached_block_table, cached_length = (
            self.prefix_cache_manager.get_matched_outer_blocks(
                request.request_id,
                cached_blocks,
                num_new_computed_tokens,
            )
        )

        return cached_block_table, cached_length

    def get_block_table(self, request_id: str) -> torch.Tensor:
        return self.prefix_cache_manager.get_blocks(request_id)

    def get_dummy_block(self) -> int:
        # Encoder-decoder models reserve a dummy block in the block pool;
        # prefix caching uses a separate dummy block managed by the prefix
        # cache manager.
        if self.block_pool.dummy_block is not None:
            # In V1, block ID 0 is the null_block, so scheduler-side block
            # IDs start at 1. The compiler expects valid blocks to start at
            # 0, so shift by -1 to translate into compiler-space.
            return self.block_pool.dummy_block.block_id - 1
        return self.prefix_cache_manager.get_dummy_block()
