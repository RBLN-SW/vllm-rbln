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
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    get_mamba_prefill_checkpoint_position,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


def register_kv_cache_with_subindex(
    context: dict[str, Any],
    name: str,
    cache: torch.nn.Module,
    *,
    subindex: int,
    already_registered: bool = False,
) -> None:
    """Register a cache whose module path has no numeric slot of its own."""
    if subindex < 1:
        raise ValueError(f"KV cache {name} requires a positive subindex")
    if already_registered:
        if context.get(name) is not cache:
            raise ValueError(f"Expected upstream to register KV cache {name}")
    elif name in context:
        raise ValueError(f"Duplicate layer name: {name}")

    cache.kv_cache_subindex = subindex
    context[name] = cache


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


#: kv_transfer_params key a producer sets to the token count its extra recurrent
#: state block ends at, so a consumer knows to allocate and cache a block for it.
REMOTE_STATE_BOUNDARY = "remote_state_boundary"

#: kv_transfer_params key a producer sets to its prompt length before upstream
#: truncates the prompt to what the consumer receives.
UNTRUNCATED_PROMPT_LENGTH = "rbln_untruncated_prompt_length"


def prompt_boundary_block(
    num_prompt_tokens: int,
    num_transferred_tokens: int,
    block_size: int,
    drop_eagle_block: bool,
) -> int | None:
    """The block holding the state a repeat of the prompt can hit, when the
    transfer covers it and it is not the transferred tokens' last state.

    A repeat looks up from the whole prompt, while the transfer stops short of
    the tokens the consumer recomputes, so the two lengths differ.
    """
    position = get_mamba_prefill_checkpoint_position(
        num_prompt_tokens, block_size, drop_eagle_block
    )
    if position == 0 or position >= num_transferred_tokens:
        return None
    return position // block_size - 1


@dataclass(frozen=True)
class RBLNMambaSpec(MambaSpec):
    """A recurrent state the RBLN KV connector moves with its block-boundary state.

    A subclass because KVCacheSpecRegistry picks a manager by spec class and upstream
    already registers MambaSpec to MambaManager; it also keeps other Mamba models on
    upstream's manager.
    """


class RBLNMambaManager(MambaManager):
    """Carries the block-boundary state a repeat of a prompt can hit across a
    P/D transfer (`prompt_boundary_block`).

    In align mode a transfer moves only the running state. That serves the
    request, but the consumer holds no state at a block boundary, so a repeat of
    the prompt misses the prefix cache. The producer keeps the boundary block
    until the transfer releases it; the consumer allocates a block for it and
    caches it once received, as if it had computed it.
    """

    def __init__(self, kv_cache_spec: MambaSpec, block_pool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self._kept_boundary: dict[str, int] = {}
        self._received_boundary: dict[str, int] = {}

    def track_transfer(
        self,
        request: Request,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Note a request's boundary block before its first allocation."""
        if self.mamba_cache_mode != "align" or not self.enable_caching:
            return
        params = request.kv_transfer_params or {}
        request_id = request.request_id
        if params.get("do_remote_decode"):
            block_idx = prompt_boundary_block(
                params[UNTRUNCATED_PROMPT_LENGTH],
                request.num_prompt_tokens,
                self.block_size,
                self.drop_eagle_checkpoint_block,
            )
            if block_idx is not None:
                self._kept_boundary[request_id] = block_idx
            return
        self._received_boundary.pop(request_id, None)
        boundary = params.get(REMOTE_STATE_BOUNDARY)
        if (
            boundary is None
            or num_external_computed_tokens == 0
            or num_local_computed_tokens >= boundary
        ):
            return
        num_received_tokens = num_local_computed_tokens + num_external_computed_tokens
        block_idx = prompt_boundary_block(
            request.num_prompt_tokens,
            num_received_tokens,
            self.block_size,
            self.drop_eagle_checkpoint_block,
        )
        if block_idx != boundary // self.block_size - 1:
            # The value comes from the peer; the running state alone still
            # serves the request, so a mismatch only costs the cache entry.
            logger.warning(
                "%s: ignoring %s=%d, not a boundary that %d received token(s) "
                "cover and a repeat of the %d-token prompt can hit",
                request_id,
                REMOTE_STATE_BOUNDARY,
                boundary,
                num_received_tokens,
                request.num_prompt_tokens,
            )
            return
        self._received_boundary[request_id] = block_idx

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        # Upstream frees every state block behind the running one; hiding the
        # kept block as null for the call is what both its paths skip.
        block_idx = self._kept_boundary.get(request_id)
        blocks = self.req_to_blocks.get(request_id, [])
        kept = None
        if block_idx is not None and block_idx < len(blocks):
            kept = blocks[block_idx]
            blocks[block_idx] = self._null_block
        try:
            super().remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )
        finally:
            if kept is not None:
                blocks[block_idx] = kept

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
        num_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        if request_id in self._received_boundary:
            num_blocks += 1
        return num_blocks

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        super().allocate_external_computed_blocks(
            request_id, num_local_computed_tokens, num_external_computed_tokens
        )
        block_idx = self._received_boundary.pop(request_id, None)
        if block_idx is None:
            return
        blocks = self.req_to_blocks[request_id]
        assert block_idx < len(blocks) - 1 and blocks[block_idx].is_null, (
            f"{request_id}: no null slot {block_idx} before the running state"
        )
        (block,) = self.block_pool.get_new_blocks(1)
        blocks[block_idx] = block
        if self._record_new_block_ids:
            self.new_block_ids.append(block.block_id)
        # The block counted as cached while it was a null placeholder.
        self.num_cached_block[request_id] = min(
            self.num_cached_block.get(request_id, 0), block_idx
        )

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._kept_boundary.pop(request_id, None)
        self._received_boundary.pop(request_id, None)
        return super().pop_blocks_for_free(request_id)
