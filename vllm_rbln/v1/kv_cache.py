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

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import Request


@dataclass(frozen=True)
class RBLNBlockAuxSpec:
    """A per-block cache region beside a scheduled KV layer's cache."""

    owner: str
    block_shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class RBLNSlidingWindowSpec(SlidingWindowSpec):
    def __post_init__(self):
        super().__post_init__()
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
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
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

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not list(new_computed_blocks), (
            "RBLNSlidingWindowManager does not support prefix-cache hits "
            "(find_longest_cache_hit returns empty)"
        )
        assert len(self.req_to_blocks[request_id]) == 0
        self.num_cached_block[request_id] = 0

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """One block per request, matching `allocate_new_blocks`.

        Overrides the base `cdiv(num_total_computed_tokens, block_size)`
        formula — that fits upstream SWA's block-table layout but not RBLN's
        single-block in-place ring buffer. The D-side P/D receive path routes
        through here, so without this override D over-allocates and mismatches
        the P-side single block.
        """
        if num_external_computed_tokens <= 0:
            return
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        req_blocks.extend(self.block_pool.get_new_blocks(1))

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        pass

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        pass

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0


@dataclass(frozen=True)
class RBLNRingSlidingWindowSpec(SlidingWindowSpec):
    pass


class RBLNRingSlidingWindowManager(SlidingWindowManager):
    """
    Once free blocks fall to `free_block_reserve`, a decode step entering a new
    block takes over the block the window has just left instead of freeing it.
    The append kernel writes token p at row p % block_size and masks by
    position, so each live token is read once; the worker's table still names
    the block at its old column. Prefill and multi-token steps take new blocks,
    as they write rows their own queries or padding still cover.
    """

    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.ring_span = cdiv(self.sliding_window, self.block_size)
        self._prefill_end: dict[str, int] = {}
        self._recycled: set[str] = set()
        # Set by the scheduler, which sees the waiting queue and every group.
        self.free_block_reserve = 0
        self.no_recycle_ids: set[str] = set()

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        if not apply_admission_cap:  # a plan for the allocation that follows
            self._recycled.discard(request_id)
            blocks = self.req_to_blocks.get(request_id)
            column = (num_tokens - 1) // self.block_size
            # The coordinator zeroes prefill_end unless retention is 0.
            decode_from = max(prefill_end, self._prefill_end.get(request_id, 0))
            if (
                blocks
                and request_id not in self.no_recycle_ids
                and num_tokens - total_computed_tokens == 1
                and total_computed_tokens >= decode_from
                and column == len(blocks)
                and column >= self.ring_span
                and self.block_pool.get_num_free_blocks() <= self.free_block_reserve
            ):
                departing = blocks[column - self.ring_span]
                # Nulls must stay a prefix; an in-flight step can delay freeing.
                freed = (
                    column == self.ring_span
                    or blocks[column - self.ring_span - 1].is_null
                )
                if not departing.is_null and departing.ref_cnt == 1 and freed:
                    self._recycled.add(request_id)
                    return 0
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
            prefill_end,
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        if request_id not in self._recycled:
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        self._recycled.remove(request_id)
        departing = (num_tokens - 1) // self.block_size - self.ring_span
        blocks = self.req_to_blocks[request_id]
        assert len(blocks) == departing + self.ring_span
        block = blocks[departing]
        self.block_pool.evict_blocks({block.block_id})
        blocks[departing] = self._null_block
        blocks.append(block)
        return [block]

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        self._prefill_end[request.request_id] = max(
            request.num_prompt_tokens, request.num_tokens - 1
        )
        super().cache_blocks(
            request,
            num_tokens,
            retention_interval,
            replay_boundaries=replay_boundaries,
        )

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._recycled.discard(request_id)
        self._prefill_end.pop(request_id, None)
        return super().pop_blocks_for_free(request_id)
