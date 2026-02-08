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

from typing import Optional

import torch
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.optimum_kv_cache_coordinator import (
    RBLNKVCacheCoordinator)
from vllm_rbln.v1.core.prefix_cache_manager import RBLNPrefixKVCacheManager

logger = init_logger(__name__)


class RBLNKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        attn_block_size: Optional[int] = None,
        max_num_seqs: int = 1,
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

        self.coordinator = RBLNKVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        block_size = kv_cache_config.kv_cache_groups[
            0].kv_cache_spec.block_size
        if enable_caching:
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
            tuple(() for _ in range(self.num_kv_cache_groups)))

    def free(self, request: Request, preemption: int = False) -> None:
        """Free the blocks allocated for the request.
        """
        if self.enable_caching:
            self.prefix_cache_manager.free_request(request.request_id,
                                                   preemption=preemption)
        self.coordinator.free(request.request_id)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
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
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=self.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        if self.enable_caching and \
            not self.prefix_cache_manager.can_allocate(
                    num_blocks_to_allocate,
                    num_computed_tokens,
            ):
            # Cannot allocate new outer blocks for prefix caching
            return None

        # Generate req_to_blocks, num_cached_block
        # in the coordinator
        # `empty_computed_block_list` is used here to avoid
        # saving the computed blocks to the request state
        self.coordinator.save_new_computed_blocks(
            request.request_id,
            self.empty_kv_cache_blocks.blocks,
        )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, 0)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
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
        cached_block_table, cached_length = \
            self.prefix_cache_manager.get_matched_outer_blocks(
            request.request_id, cached_blocks,
            num_new_computed_tokens,
        )

        return cached_block_table, cached_length

    def get_block_table(self, request_id: str) -> torch.Tensor:
        return self.prefix_cache_manager.get_blocks(request_id)

    def get_dummy_block(self) -> int:
        return self.prefix_cache_manager.get_dummy_block()
