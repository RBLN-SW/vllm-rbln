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

    def _chunked_prefill_pad(self, request: Request, query_len: int) -> int:
        chunk_size = self.prefill_chunk_size
        if self.attn_block_size is None:
            block_size = self.block_pool.block_size
        else:
            block_size = self.attn_block_size
        # Image placeholder ranges within the prompt, sorted by start offset.
        # NOTE: a range covers the whole image block incl. special tokens (gemma3:
        # \n\n+boi+256+eoi+\n\n = 260), not just the 256 image soft tokens.
        image_ranges = sorted(
            (f.mm_position.offset, f.mm_position.offset + f.mm_position.length)
            for f in request.mm_features
        )
        # `step`: next prompt token to process (excludes alignment padding).
        # `align_pad`: alignment padding so far; the token sits at cache slot
        #   `step + align_pad`.
        #
        # Example: [text 0..99][image 100..359], chunk_size=256
        #   step=0   text  -> run_end=100 -> run_len=min(100, 256)=100
        #   step=100 image -> run_end=360 -> run_len=min(260, 256)=256
        #   step=356 image -> run_end=360 -> run_len=min(4,   256)=4
        step = 0
        align_pad = 0
        while step < query_len:
            # End of the current run: an image's end if `step` is inside an image,
            # else the next image's start (or `query_len`) for a text run.
            img_end = next((e for s, e in image_ranges if s <= step < e), None)
            if img_end is not None:
                run_end = img_end
            else:
                run_end = min(
                    (s for s, _ in image_ranges if s > step), default=query_len
                )
            run_len = min(run_end - step, chunk_size)

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
            # top of it, so base the pad on the prompt length only.
            pad = self._chunked_prefill_pad(
                request, min(request.num_prompt_tokens, self.max_model_len)
            )
            print("@@@@ pad", pad)
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
