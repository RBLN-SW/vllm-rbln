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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import KVCacheTensor, SlidingWindowSpec
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec


class RBLNFixedKVCacheTensor(KVCacheTensor):
    """A cache allocation whose capacity does not grow with the attention pool."""


class RBLNDeepseekV4StateManager(CircularBufferManager):
    kv_cache_spec: "RBLNDeepseekV4StateSpec"

    def __init__(self, kv_cache_spec, *, block_pool, **kwargs):
        kwargs["enable_caching"] &= kv_cache_spec.prefix_cacheable
        self.attention_pool = block_pool
        state_pool = BlockPool(
            num_gpu_blocks=kv_cache_spec.num_state_slots,
            enable_caching=kwargs["enable_caching"],
            hash_block_size=block_pool.hash_block_size,
        )
        super().__init__(kv_cache_spec, block_pool=state_pool, **kwargs)
        # The physical ring capacity is independent of the prefix hit interval.
        if self.enable_caching:
            self.block_size = self.scheduler_block_size
        self._scheduled_ends: dict[str, int] = {}
        self._pending_copies: dict[str, tuple[int, int]] = {}

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
        current = self.req_to_blocks.get(request_id)
        needed = int(not current or current[0].block_hash is not None)
        needed += sum(b.ref_cnt == 0 for b in new_computed_blocks)
        if needed > self.block_pool.get_num_free_blocks():
            return self.attention_pool.get_num_free_blocks() + 1
        return 0

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not num_external_computed_tokens
        assert not self.req_to_blocks[request_id]
        assert len(new_computed_blocks) <= 1
        self.block_pool.touch(new_computed_blocks)
        self.req_to_blocks[request_id].extend(new_computed_blocks)
        self.num_cached_block[request_id] = 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        if self.enable_caching:
            self._scheduled_ends[request_id] = num_tokens_main_model
        blocks = self.req_to_blocks[request_id]
        if not blocks:
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        if blocks[0].block_hash is not None:
            source = blocks[0]
            destination = self.block_pool.get_new_blocks(1)[0]
            blocks[0] = destination
            # Transfer the request's source reference to the pending copy. It
            # survives eviction/preemption until the runner has read the ring.
            assert request_id not in self._pending_copies
            self._pending_copies[request_id] = (source.block_id, destination.block_id)
        # A ring has one table entry. The scheduler sends a replacement, not an
        # append, when copy-on-write changes its physical slot.
        return []

    def drain_pending_copies(
        self, request_ids: Iterable[str]
    ) -> dict[str, tuple[int, int]]:
        return {
            req_id: self._pending_copies.pop(req_id)
            for req_id in request_ids
            if req_id in self._pending_copies
        }

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._scheduled_ends.pop(request_id, None)
        blocks = super().pop_blocks_for_free(request_id)
        if (pending := self._pending_copies.pop(request_id, None)) is not None:
            blocks.append(self.block_pool.blocks[pending[0]])
        return blocks

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        if not self.enable_caching:
            return
        assert retention_interval == 0
        end = self._scheduled_ends[request.request_id]
        if end == 0 or end > request.num_prompt_tokens:
            return
        boundaries = (*replay_boundaries, request.shared_prefix_boundary)
        if not any(
            end == boundary // self.block_size * self.block_size
            for boundary in boundaries
        ):
            return
        block = self.req_to_blocks[request.request_id][0]
        if block.block_hash is not None:
            return
        # An MTP row consumes the next input token. Its hash must include the
        # following paged block, even when HCA imposes a larger hit interval.
        hash_size = self.block_pool.hash_block_size
        hash_end = end + hash_size * self.kv_cache_spec.is_draft
        if hash_end > request.num_prompt_tokens:
            return
        count = hash_end // hash_size
        self.block_pool.cache_full_blocks(
            request,
            [self._null_block] * (count - 1) + [block],
            count - 1,
            count,
            hash_size,
            self.kv_cache_group_id,
        )

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
        dcp_world_size=1,
        pcp_world_size=1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hash_size = block_pool.hash_block_size
        lookahead = hash_size if kv_cache_spec.is_draft else 0
        last = min(
            (max_length - (lookahead if drop_eagle_block else 0)) // alignment_tokens,
            (len(block_hashes) * hash_size - lookahead) // alignment_tokens,
        )
        for count in range(last, 0, -1):
            blocks = block_pool.get_cached_block(
                block_hashes[(count * alignment_tokens + lookahead) // hash_size - 1],
                kv_cache_group_ids,
            )
            if blocks is not None:
                return tuple([b] for b in blocks), count * alignment_tokens
        return tuple([] for _ in kv_cache_group_ids), 0


@dataclass(frozen=True)
class RBLNBlockAuxSpec:
    """A per-block cache region beside a scheduled KV layer's cache.

    `prefix_cacheable=False` means this region's previous contents do not affect
    the result when a prefix resumes at a block boundary, so it need not be saved
    or restored. The buffer must still be allocated and readable: kernels may read
    it physically (the tail ring with two or more MTP drafts, masked draft rows).
    It is per aux region, not vLLM's group-level `KVCacheSpec.prefix_cacheable`.

    `verify_only=True` means only a spec-decode verify step reads this region, and
    only what an earlier verify on the same engine wrote. A strict KV producer never
    verifies, so it does not allocate the region, and a request's first verify on a
    peer reads none of it, so no KV connector registers it on any role. It does not
    follow from `prefix_cacheable=False`: the tail ring resumes mid-block on a peer.
    """

    owner: str
    block_shape: tuple[int, ...]
    dtype: torch.dtype
    prefix_cacheable: bool = True
    verify_only: bool = False
    # The axis of block_shape a TP rank's share of its owner's heads cuts, each
    # index before it one run of them; None where every rank holds the same bytes.
    head_axis: int | None = None

    @property
    def page_size_bytes(self) -> int:
        return prod(self.block_shape) * get_dtype_size(self.dtype)


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
