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

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class RBLNBlockPool(BlockPool):
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        is_encoder_decoder: bool = False,
    ):
        super().__init__(
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # Called with the block id whenever a cached block is actually
        # evicted. The KV cache manager registers its sub-block index cleanup
        # here, since eviction is the only point where a hash-carrying block
        # leaves the cache.
        self.evicted_block_hook: Callable[[int], None] | None = None

        # The decoder runtime writes into every block-table slot of a padded
        # decode batch, so padded slots need a scratch block that never holds
        # a real request's KV. Encoder-decoder models (e.g. Whisper) also pad
        # the decoder's block table during prefill. With prefix caching, a
        # free block may hold cached KV that such a write would clobber, so a
        # dedicated block is reserved instead of borrowing a free one.
        # Reserve the last block by removing it from the free queue so it is
        # never handed out to a real request.
        self.dummy_block: KVCacheBlock | None = None
        if is_encoder_decoder or enable_caching:
            assert num_gpu_blocks >= 2, (
                "Reserving a dummy block requires at least 2 blocks "
                "(1 null block + 1 dummy block)."
            )
            self.dummy_block = self.blocks[num_gpu_blocks - 1]
            self.free_block_queue.remove(self.dummy_block)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        evicted = super()._maybe_evict_cached_block(block)
        if evicted and self.evicted_block_hook is not None:
            self.evicted_block_hook(block.block_id)
        return evicted
