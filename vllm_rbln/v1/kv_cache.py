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

from collections.abc import Sequence
from dataclasses import dataclass

import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import Request


@dataclass(frozen=True)
class RBLNSlidingWindowSpec(SlidingWindowSpec):
    def __post_init__(self):
        # NOTE: The block size here means to be the physical block size. The
        # logical kernel_block_size that the kernel actually uses is equal to
        # sliding_window. The physical block is split into logical blocks.
        assert self.block_size % self.sliding_window == 0

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return self.page_size_bytes


class RBLNSlidingWindowManager(SingleTypeKVCacheManager):
    """
    The RBLN SWA kernel uses a single block and slides the contents in-place.
    To support this, this manager:
    * Allocates a single block per request.
    * Disables prefix caching. This is technically not needed if we do
      vllm_config.cache_config.enable_prefix_caching = False,
      but we keep it here for clarity.
    """

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        return 0 if self.req_to_blocks[request_id] else 1

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        if self.req_to_blocks[request_id]:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        self.req_to_blocks[request_id].extend(new_blocks)
        return new_blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        use_eagle,
        alignment_tokens,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        return tuple([] for _ in kv_cache_group_ids)

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        pass

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        pass

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0


single_type_kv_cache_manager.spec_manager_map.update(
    {
        RBLNSlidingWindowSpec: RBLNSlidingWindowManager,
    }
)
