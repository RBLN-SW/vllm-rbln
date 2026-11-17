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

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import Request


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


@dataclass(frozen=True)
class RBLNRingSlidingWindowSpec(SlidingWindowSpec):
    pass


class RBLNRingSlidingWindowManager(SlidingWindowManager):
    """Keeps a decoding request's window in one block.

    A step whose first token lands in a block new to the request first has the
    rows it has yet to write copied from the block before (`pending_copies`), so
    row r of that block holds the latest position congruent to r. Once the
    request decodes, the blocks before it are freed; they stay cached.
    """

    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        if self.sliding_window != self.block_size:
            raise NotImplementedError(
                "use_swa_ring needs a sliding window as wide as the block."
            )
        self._step_start: dict[str, int] = {}
        self._window_column: dict[str, int] = {}
        self._decode_from: dict[str, int] = {}
        # (source block, destination block, first row), sources held by a ref.
        self.pending_copies: list[tuple[int, int, int]] = []

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        *args,
        **kwargs,
    ) -> int:
        self._step_start[request_id] = total_computed_tokens
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            *args,
            **kwargs,
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        new_blocks = super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model
        )
        start = self._step_start[request_id]
        column = start // self.block_size
        if self._window_column.get(request_id) != column:
            self._window_column[request_id] = column
            blocks = self.req_to_blocks[request_id]
            if column and not blocks[column - 1].is_null:
                self.block_pool.touch([blocks[column - 1]])
                self.pending_copies.append(
                    (
                        blocks[column - 1].block_id,
                        blocks[column].block_id,
                        start % self.block_size,
                    )
                )
        return new_blocks

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
        # A prefill chunk reads the block before its own, and a step still in
        # flight may; a request that has not cached yet may be either.
        column = self._window_column.get(request_id, 0)
        if (
            request_id in self._decode_from
            and processed_computed_tokens > column * self.block_size
            and processed_computed_tokens >= self._decode_from[request_id]
        ):
            self._remove_blocks_in_range(request_id, 0, column)

    def cache_blocks(self, request: Request, *args, **kwargs) -> None:
        # A streaming session prefills again after it decodes.
        if not request.resumable:
            self._decode_from.setdefault(
                request.request_id,
                max(request.num_prompt_tokens, request.num_tokens - 1),
            )
        super().cache_blocks(request, *args, **kwargs)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._step_start.pop(request_id, None)
        self._window_column.pop(request_id, None)
        self._decode_from.pop(request_id, None)
        return super().pop_blocks_for_free(request_id)
