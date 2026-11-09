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
from math import prod

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec
from vllm.v1.request import Request


@dataclass(frozen=True)
class RBLNBlockAuxSpec:
    """A per-block cache region beside a scheduled KV layer's cache, addressed
    by the owner's block ids and allocated outside the KV cache config."""

    owner: str
    block_shape: tuple[int, ...]
    dtype: torch.dtype

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
class RBLNGDNStateSpec(MambaSpec):
    """A GatedDeltaNet layer's recurrent state, in a block pool of its own.

    Upstream pads the page to the attention page when it groups the layers, so
    the group forms and the layout checks pass as for any Mamba layer. What
    the worker allocates is `real_page_size_bytes` a block, in a tensor
    sized to the state pool, and `RBLNGDNManager` draws the block ids from
    that pool rather than the attention one.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Upstream multiplies the declared page; the state occupies the real one.
        pages = super().max_memory_usage_bytes(vllm_config) // self.page_size_bytes
        return pages * self.real_page_size_bytes

    def pool_blocks(self, vllm_config: VllmConfig) -> int:
        """Blocks in the group's pool: one a running request (its whole state,
        while `mamba_block_size` is the model length) and the null block the
        pool reserves, which the decode batch pads onto."""
        return vllm_config.scheduler_config.max_num_seqs + 1


class RBLNGDNManager(MambaManager):
    """`MambaManager` on a pool of its own, bound after construction.

    The coordinator builds every manager on the attention pool. `bind_own_pool`
    swaps this group's pool in, and from then on the manager stays out of the
    attention count: it reports 0 while its pool has room and the attention
    pool's refusal sentinel (`num_gpu_blocks + 1`) when it does not, the way
    upstream's HiSparse host tier keeps its private pool out of the device one.
    """

    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self._attention_pool = block_pool

    def bind_own_pool(self, vllm_config: VllmConfig) -> None:
        """Replace the attention pool this group was built with by one of its own."""
        spec = self.kv_cache_spec
        assert isinstance(spec, RBLNGDNStateSpec), spec
        # Prefix caching is off for GatedDeltaNet models (platform), so the
        # pool neither hashes its blocks nor publishes cache events.
        self.block_pool = BlockPool(
            num_gpu_blocks=spec.pool_blocks(vllm_config),
            enable_caching=False,
            hash_block_size=self._attention_pool.hash_block_size,
        )
        self._null_block = self.block_pool.null_block

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
        assert self.block_pool is not self._attention_pool, (
            "RBLNGDNManager schedules on a pool of its own: call bind_own_pool first"
        )
        required = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        if required > self.block_pool.get_num_free_blocks():
            return self._attention_pool.num_gpu_blocks + 1
        return 0


def bind_own_pools(kv_cache_manager: KVCacheManager, vllm_config: VllmConfig) -> None:
    """The coordinator built every manager on the attention pool; a manager
    that defines `bind_own_pool` schedules on one of its own."""
    for manager in kv_cache_manager.coordinator.single_type_managers:
        bind = getattr(manager, "bind_own_pool", None)
        if bind is not None:
            bind(vllm_config)
