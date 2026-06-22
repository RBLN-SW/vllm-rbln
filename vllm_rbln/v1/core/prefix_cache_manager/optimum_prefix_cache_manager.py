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

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import torch

from vllm_rbln.logger import init_logger

from .optimum_block_mapping_manager import BlockMappingManager, RBLNBlock
from .optimum_eviction_policy import FIFOEvictionPolicy, LRUEvictionPolicy

logger = init_logger(__name__)
NO_MATCH_FOUND = -1


@dataclass
class BlockConfiguration:
    ob_size: int
    ib_size: int
    max_model_len: int
    max_num_seqs: int
    num_ob: int

    def __post_init__(self):
        assert self.ob_size % self.ib_size == 0, "ob_size must be a multiple of ib_size"

    @property
    def block_ratio(self) -> int:
        return self.ob_size // self.ib_size


@dataclass
class CacheSearchResult:
    cached_outer_blocks: list[int]
    cached_lengths: list[int]

    @property
    def has_cache_hit(self) -> bool:
        return len(self.cached_outer_blocks) > 0


class BlockAllocatorInterface(ABC):
    @abstractmethod
    def allocate(self, count: int) -> list[RBLNBlock]:
        pass

    @abstractmethod
    def deallocate(self, block: RBLNBlock) -> None:
        pass

    @abstractmethod
    def get_free_count(self) -> int:
        pass


class RBLNBlockAllocator(BlockAllocatorInterface):
    def __init__(self, num_blocks: int):
        self._free_blocks = deque([RBLNBlock(i) for i in range(num_blocks)])
        # Keep track of allocated blocks for validation
        self._allocated_blocks: dict[int, RBLNBlock] = {}

    def allocate(self, count: int) -> list[RBLNBlock]:
        if len(self._free_blocks) < count:
            raise RuntimeError(
                f"Insufficient free blocks. Requested: {count}, "
                f"Available: {len(self._free_blocks)}"
            )

        allocated = []
        for _ in range(count):
            block = self._free_blocks.popleft()
            self._allocated_blocks[block.block_id] = block
            allocated.append(block)
        return allocated

    def deallocate(self, block: RBLNBlock) -> None:
        if block.block_id in self._allocated_blocks:
            self._free_blocks.append(block)
            del self._allocated_blocks[block.block_id]
        else:
            logger.warning(
                "[PFX] [DEALLOCATE-WARNING] BLOCK=%d | REASON=unallocated_block",
                block.block_id,
            )

    def get_free_count(self) -> int:
        return len(self._free_blocks)

    def get_allocated_block(self, block_id: int) -> RBLNBlock | None:
        return self._allocated_blocks.get(block_id)

    def peek_dummy_block(self) -> int:
        """
        Peek the free block without allocating it.
        """
        return self._free_blocks[0].block_id


class CacheSearchManager:
    """
    Search for cached blocks that can be reused.
    It maintains a per-request cache of search results
    until the request is freed.
    """

    def __init__(self, config: BlockConfiguration):
        self._config = config

    def find_cached_blocks(
        self,
        request_id: str,
        cached_blocks: list[int],
        num_new_computed_tokens: int,
        mapping_manager: BlockMappingManager,
    ) -> CacheSearchResult:
        """
        Find cached outer blocks that match the given inner blocks.
        """

        best_match = self._try_match_request(cached_blocks, mapping_manager)
        final_num_cached_tokens = sum(best_match.cached_lengths)

        if logger.isEnabledFor(logging.DEBUG):
            if final_num_cached_tokens < num_new_computed_tokens:
                miss_rate = (
                    (
                        (num_new_computed_tokens - final_num_cached_tokens)
                        / num_new_computed_tokens
                        * 100
                    )
                    if num_new_computed_tokens > 0
                    else 0
                )
                logger.debug(
                    "[PFX] [CACHE-PARTIAL] REQUEST=%s | "
                    "REUSED=%d/%d tokens (%.1f%%) | "
                    "MISSED=%d tokens (%.1f%%) | "
                    "REASON=partial_cache_miss",
                    request_id,
                    final_num_cached_tokens,
                    num_new_computed_tokens,
                    (final_num_cached_tokens / num_new_computed_tokens * 100),
                    num_new_computed_tokens - final_num_cached_tokens,
                    miss_rate,
                )

            if best_match.has_cache_hit:
                real_cached_ibs = [
                    cached_blocks[: self._calculate_cached_inner_blocks(length)]
                    for length in best_match.cached_lengths
                ]
                # TODO specify the hit ratio?
                logger.debug(
                    "[PFX] [CACHE-HIT] REQUEST=%s | "
                    "OB_COUNT=%d OB=%s | "
                    "IB_COUNT=%d IB=%s",
                    request_id,
                    len(best_match.cached_outer_blocks),
                    best_match.cached_outer_blocks,
                    len(real_cached_ibs),
                    real_cached_ibs,
                )

        return best_match

    def _calculate_cached_inner_blocks(self, cached_len_tokens: int) -> int:
        """
        Calculate the number of cached inner blocks
        based on the cached length in tokens.
        """
        return cached_len_tokens // self._config.ib_size

    def _try_match_request(
        self, cached_ib: list[int], mapping_manager: BlockMappingManager
    ) -> CacheSearchResult:
        """
        Try to find the best matching outer blocks for the given inner blocks.
        NOTE Currently, we only support exact match of the inner blocks
        """
        cached_obs = []
        cached_lengths = []
        for start_ib_idx in range(0, len(cached_ib), self._config.block_ratio):
            end_ib_idx = min(start_ib_idx + self._config.block_ratio, len(cached_ib))
            cur_ib_segment = cached_ib[start_ib_idx:end_ib_idx]
            cached_ob, num_cached_ibs = mapping_manager.get_longest_matched_block(
                cur_ib_segment
            )
            # FIXME simple stop condition
            # Upgrade the readability later
            if cached_ob == NO_MATCH_FOUND:
                break
            cached_obs.append(cached_ob)
            cached_lengths.append(num_cached_ibs * self._config.ib_size)

        return CacheSearchResult(
            cached_outer_blocks=cached_obs, cached_lengths=cached_lengths
        )


class MemoryPoolManager:
    """
    Manage a memory pool to return a tensor of block IDs.
    """

    def __init__(self, max_model_len: int, ob_size: int):
        self.dtype = torch.int16
        self._pooled_tensor = torch.zeros(max_model_len // ob_size, dtype=self.dtype)

    def get_tensor_for_blocks(self, block_ids: list[int]) -> torch.Tensor:
        self._pooled_tensor.fill_(-1)

        if block_ids:
            value = torch.tensor(block_ids, dtype=self.dtype)
            self._pooled_tensor[: len(value)].copy_(value)

        return self._pooled_tensor.clone()


class RBLNPrefixKVCacheManager:
    def __init__(
        self,
        ob_size: int,
        ib_size: int,
        max_model_len: int,
        max_num_seqs: int,
        num_inner_blocks: int,
    ):
        assert ob_size % ib_size == 0, (
            "Outer block size must be a multiple of inner block size"
        )
        num_ob = math.ceil(num_inner_blocks / (ob_size // ib_size))
        self._config = BlockConfiguration(
            ob_size, ib_size, max_model_len, max_num_seqs, num_ob
        )
        self._allocator = RBLNBlockAllocator(num_ob)
        self._mapping_manager = BlockMappingManager()
        self._cache_search_manager = CacheSearchManager(self._config)
        self._memory_pool_manager = MemoryPoolManager(max_model_len, ob_size)
        self._eviction_policy = FIFOEvictionPolicy()

    def is_full_block_available(self) -> bool:
        blocks_per_seq = math.ceil(self._config.max_model_len / self._config.ob_size)
        ideal_total = self._config.max_num_seqs * blocks_per_seq
        return self._config.num_ob >= ideal_total

    def _handle_decode_allocation(
        self, request_id: str, num_new_ob: int, inner_blocks: list[int]
    ) -> None:
        """
        Allocate new blocks for DECODE phase.
        """
        if num_new_ob > 0:
            self._allocate_new_blocks(request_id, num_new_ob, inner_blocks)
        else:
            # Append to the last outer block
            request_blocks = self.get_block_ids(request_id)
            last_ob_id = request_blocks[-1]
            self._append_to_existing_block(last_ob_id, inner_blocks)

    def _handle_prefill_allocation(
        self, request_id: str, num_new_ob: int, inner_blocks: list[int]
    ) -> None:
        """
        Allocate new blocks for PREFILL phase.
        """
        self._allocate_new_blocks(request_id, num_new_ob, inner_blocks)

    def _allocate_new_blocks(
        self, request_id: str, num_new_ob: int, inner_blocks: list[int]
    ) -> None:
        """
        Allocate new outer blocks and create mappings.
        """
        assert num_new_ob > 0, "One or more new blocks must be allocated"
        # Allocate blocks
        new_blocks = self._allocator.allocate(num_new_ob)

        # Create mappings
        block_ids = []
        for i, block in enumerate(new_blocks):
            start_idx = i * self._config.block_ratio
            end_idx = min((i + 1) * self._config.block_ratio, len(inner_blocks))
            block_inner_blocks = inner_blocks[start_idx:end_idx]

            self._mapping_manager.create_mapping(block, block_inner_blocks, request_id)
            self._eviction_policy.register_block(block.block_id)
            block_ids.append(block.block_id)

        logger.debug(
            "[PFX] [ALLOC] REQUEST=%s | OB_COUNT=%d OB=%s | IB_COUNT=%d IB=%s",
            request_id,
            len(block_ids),
            block_ids,
            len(inner_blocks),
            inner_blocks,
        )

    def _compute_num_blocks_to_allocate(
        self, num_inner_blocks: int, num_allocated_tokens: int = 0
    ) -> int:
        """
        Compute the number of outer blocks to allocate based on inner blocks
        and the number of already allocated tokens.
        """
        if num_inner_blocks == 0:
            return 0

        if num_allocated_tokens == 0:
            # PREFILL
            num_obs_needed = math.ceil(num_inner_blocks / self._config.block_ratio)
        else:
            # DECODE
            num_already_allocated_ibs = num_allocated_tokens // self._config.ib_size

            if num_already_allocated_ibs % self._config.block_ratio == 0:
                num_obs_needed = 1
            else:
                num_obs_needed = 0

        return num_obs_needed

    def _check_free_blocks(self, num_new_blocks: int) -> None:
        """
        Check that there are enough free blocks to allocate.
        """
        free_count = self._allocator.get_free_count()
        assert free_count >= num_new_blocks, (
            f"Free blocks ({free_count}) should be "
            f"greater than or equal to {num_new_blocks}"
        )

    def _evict_block(self, block_id: int) -> None:
        """
        Evict a block and free its resources.
        """
        mapping = self._mapping_manager.remove_mapping(block_id)
        if mapping:
            block = self._allocator.get_allocated_block(block_id)
            if block:
                self._allocator.deallocate(block)
                self._eviction_policy.unregister_block(block_id)
                logger.debug(
                    "[PFX] [EVICTION] OB=%d | "
                    "IB_COUNT=%d | "
                    "FREE_BLOCKS_AFTER=%d | "
                    "INACTIVE_MAPPINGS_AFTER=%s",
                    block_id,
                    len(mapping.inner_block_ids),
                    self._allocator.get_free_count(),
                    [
                        m.outer_block_id
                        for m in self._mapping_manager.get_inactive_mappings()
                    ],
                )
            else:
                logger.error(
                    "[PFX] [EVICTION-ERROR] OB=%d | "
                    "REASON=block_not_found_in_allocator",
                    block_id,
                )

    def _append_to_existing_block(
        self, outer_block_id: int, inner_blocks: list[int]
    ) -> None:
        """
        Add inner blocks to an existing outer block.
        """
        assert len(inner_blocks) == 1, "Can only append one inner block at a time"
        ib_id = inner_blocks[0]

        # Update the outer to inner mapping
        mapping = self._mapping_manager.get_mapping(outer_block_id)
        if not mapping:
            raise RuntimeError(f"Mapping not found for outer block {outer_block_id}")

        mapping.inner_block_ids.append(ib_id)

        # Update the inner to outer mapping
        self._mapping_manager.add_new_inner_to_outer(ib_id, outer_block_id)

    def free_request(self, request_id: str, preemption: bool = False) -> None:
        """
        Called when a request is completed or preempted.
        Free all blocks associated with a given request.
        If it is not preemption, keep the mappings for potential future reuse.
        """
        outer_blocks = self._mapping_manager.remove_request(request_id)
        logger.debug(
            "[PFX] [FREE-REQUEST] REQUEST=%s | PREEMPTION=%s | OUTER_BLOCKS=%s",
            request_id,
            preemption,
            outer_blocks,
        )
        for block_id in outer_blocks:
            mapping = self._mapping_manager.get_mapping(block_id)
            if preemption:
                self._evict_block(block_id)
            else:
                assert mapping is not None, "Mapping not found for block"
                mapping.is_active = False
                mapping.request_id = None

    def get_matched_outer_blocks(
        self, request_id: str, cached_blocks: list[int], num_new_computed_tokens: int
    ) -> tuple[list[int], list[int]]:
        """
        Get the matched outer blocks using inner blocks.
        """
        allocated_blocks = set(self.get_block_ids(request_id))
        result = self._cache_search_manager.find_cached_blocks(
            request_id, cached_blocks, num_new_computed_tokens, self._mapping_manager
        )
        assert len(set(result.cached_outer_blocks) & allocated_blocks) == 0, (
            "Cached outer blocks and allocated outer blocks should be disjoint"
        )

        if result.has_cache_hit and isinstance(
            self._eviction_policy, LRUEvictionPolicy
        ):
            # NOTE(eunji.lee):
            # To increase the hit ratio,
            # we need to touch the blocks in reverse order.
            for ob_id in reversed(result.cached_outer_blocks):
                self._eviction_policy.touch(ob_id)

        return result.cached_outer_blocks, result.cached_lengths

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get the tensor of outer block IDs for a given request.
        """
        block_ids = self.get_block_ids(request_id)
        return self._memory_pool_manager.get_tensor_for_blocks(block_ids)

    def get_block_ids(self, request_id: str) -> list[int]:
        """
        Get the list of outer block IDs for a given request.
        """
        return self._mapping_manager.get_request_blocks(request_id)

    def can_allocate(self, num_new_blocks: int, num_computed_tokens: int) -> bool:
        # 1. Check if the enough outer blocks are free
        required_num_ob = self._compute_num_blocks_to_allocate(
            num_new_blocks, num_computed_tokens
        )
        free_count = self._allocator.get_free_count()
        if free_count >= required_num_ob:
            return True

        # 2. Check if we can evict enough blocks
        evict_count = required_num_ob - free_count
        blocks_to_evict = self._eviction_policy.select_blocks_for_eviction(
            self._mapping_manager, evict_count
        )
        can_evict = len(blocks_to_evict) >= evict_count

        # 3. Evict blocks if possible
        if can_evict:
            # NOTE: In vLLM, the blocks are returned
            # to the free block queue in reversed order.
            # It is for preventing memory fragmentation.
            # But here, we don't need to reverse the order.
            for block_id in blocks_to_evict:
                self._evict_block(block_id)
        return can_evict

    def allocate_blocks(
        self, request_id: str, num_allocated_tokens: int, inner_blocks: list[int]
    ) -> None:
        """
        Allocate blocks for a given request
        based on its phase (PREFILL or DECODE).
        """
        num_new_ib = len(inner_blocks)
        if num_new_ib == 0:
            return

        num_new_ob = self._compute_num_blocks_to_allocate(
            num_new_ib, num_allocated_tokens
        )

        self._check_free_blocks(num_new_ob)

        if self._mapping_manager.is_request_registered(request_id):
            self._handle_decode_allocation(request_id, num_new_ob, inner_blocks)
        else:
            self._handle_prefill_allocation(request_id, num_new_ob, inner_blocks)

    def get_dummy_block(self) -> int:
        if not self.is_full_block_available():
            # NOTE(eunji.lee):
            # If the kv cache is not a full-block,
            # there is an additional block kept for padding.
            # So we utilize it as a dummy block.
            return self._config.num_ob
        else:
            # NOTE(eunji.lee):
            # If the kv cache is full-block,
            # we utilize remaining blocks for padding.
            num_blocks_to_allocate = 1
            can_allocate = self.can_allocate(num_blocks_to_allocate, 0)
            if not can_allocate:
                raise RuntimeError(
                    "[PFX] [GET-DUMMY-BLOCK-ERROR] "
                    "REASON=failed_to_allocate_dummy_block"
                )
            self._check_free_blocks(num_blocks_to_allocate)
            dummy_block = self._allocator.peek_dummy_block()
            return dummy_block
