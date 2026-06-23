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
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
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


@dataclass(frozen=True)
class RBLNScratchStableSWASpec(SlidingWindowSpec):
    """Scratch + stable SWA spec (new_code_plan/scratch_stable_prefix_cache_design.MD).

    The append-only STABLE blocks (held in ``req_to_blocks``) are managed exactly like the
    upstream sliding-window lifecycle and are prefix-cacheable. In ADDITION, one MUTABLE
    SCRATCH block per request is reserved separately (in the manager's ``scratch_blocks``
    side-table, kept OUT of ``req_to_blocks`` so it is never hashed/cached/freed as a stable
    block). The attention kernel reads only the scratch block; the stable blocks exist purely
    for prefix caching. Requires ``block_size == sliding_window``.
    """

    def __post_init__(self):
        assert self.block_size % self.sliding_window == 0

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # NOTE: Real multi-block SWA footprint (the stable blocks) + 1 page for the per-request
        # scratch block, so the pool is sized to fit one scratch per concurrent request.
        return super().max_memory_usage_bytes(vllm_config) + self.page_size_bytes


class RBLNScratchStableSWAManager(SlidingWindowManager):
    """Inherit the upstream sliding-window lifecycle for ``req_to_blocks`` (= the cacheable
    append-only STABLE blocks) and additionally reserve ONE mutable SCRATCH block per
    request, kept OUT of ``req_to_blocks`` so the cache machinery
    (``cache_blocks``/``find_longest_cache_hit``/``remove_skipped_blocks``, which all index
    ``req_to_blocks`` positionally) never touches it. The scratch block id is plumbed to the
    attention metadata builder via a runner-side channel (see the design doc).

    Freeing policy = STRICT FREE-ON-FILL (``cache_blocks`` override): a stable block is freed
    to the free queue the instant it is full + hashed (it is write-only -- the scratch holds
    the window), keeping only the in-progress stable block live per request. On a prefix-cache
    HIT, ``allocate_new_blocks`` seeds the fresh scratch from the last cached stable block via a
    pending D2D copy (drained by the scheduler into a KVCacheCopyOp).
    """

    def __init__(self, kv_cache_spec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        # req_id -> the request's mutable scratch block (separate from req_to_blocks).
        self.scratch_blocks: dict[str, KVCacheBlock] = {}
        # Pending scratch-seed copies (stable_block_id -> scratch_block_id) emitted on a
        # prefix-cache hit; drained each schedule step into KVCacheCopyOps (see the
        # scheduler). Each entry is (src_block_id, dst_scratch_block_id, num_tokens).
        self.pending_seed_copies: list[tuple[int, int, int]] = []

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        n = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
        )
        # Reserve one scratch block on the request's first allocation.
        if request_id not in self.scratch_blocks:
            n += 1
        return n

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        if request_id not in self.scratch_blocks:
            scratch = self.block_pool.get_new_blocks(1)[0]
            self.scratch_blocks[request_id] = scratch
            # Case: Prefix-cache HIT seed copy. On a hit, the attention window for the
            # first step lives in the LAST cached stable block (block==window, so the hit
            # returns exactly that one block and num_computed = m*W -> cache_seq_len = W).
            # The freshly-allocated scratch is empty, so seed it from that cached block
            # (a whole-block, offset-0 D2D copy) before the forward, else the first step
            # attends to garbage. allocate_new_computed_blocks() ran already this step, so
            # req_to_blocks holds [null]*(m-1) + [cached] and num_cached_block == m.


            # NOTE: this logic handles only the block-aligned case so
            # we can't have a sub block prefix cache hit (handled in apply_sub_block_matching) where the 
            # initial window we copy into our scratch block spans two blocks

            # num_cached_block is set by allocate_new_computed_blocks() 
            # based on the returned blocks from find_longest_cache_hit(), which is based on 
            # req_to_blocks, which is updated by allocate_new_computed_blocks() in a way that the first 
            # num_cached_block entries are the cached blocks (the rest are null).
            num_cached = self.num_cached_block.get(request_id, 0)
            if num_cached > 0:
                req_blocks = self.req_to_blocks.get(request_id, [])

                # pretty much here we reverse from our cached blocks and find
                # the rightmost non-null block to seed from
                seed = next(
                    (
                        blk
                        for blk in reversed(req_blocks[:num_cached])
                        if blk is not self._null_block and not blk.is_null
                    ),
                    None,
                )
                if seed is not None:
                    self.pending_seed_copies.append(
                        (seed.block_id, scratch.block_id, self.block_size)
                    )
        # stable blocks allocated by super man the man above bc stable blocks
        # in req_to_blocks
        return super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model
        )

    def cache_blocks(self, request, num_tokens: int) -> None:
        super().cache_blocks(request, num_tokens)
        if not self.enable_caching:
            return
        # NOTE (sgwon): I implemented a STRICT FREE-ON-FILL policy here in case we run out of memory (also 
        # LRU behavior). Because stable block is write-only, so a stable block is never re-read by this request once
        # full. The instant it is full + hashed, free it (-> ref 0, still hashed so it stays
        # available for LRU prefix-cache reuse until evicted) and null it in req_to_blocks so
        # free()/remove_skipped_blocks never double-free it. This keeps only the in-progress
        # stable block live per request (one fewer than the inherited window-tied freeing).
        num_full_blocks = num_tokens // self.block_size
        req_blocks = self.req_to_blocks[request.request_id]
        to_free: list[KVCacheBlock] = []
        for i in range(min(num_full_blocks, len(req_blocks))):
            blk = req_blocks[i]
            if blk is not self._null_block and not blk.is_null:
                to_free.append(blk)  # forward order -> oldest evicted first (LRU)
                req_blocks[i] = self._null_block
        if to_free:
            self.block_pool.free_blocks(to_free)

    def free(self, request_id: str) -> None:
        scratch = self.scratch_blocks.pop(request_id, None)
        if scratch is not None:
            self.block_pool.free_blocks([scratch])
        super().free(request_id)

    def cancel_pending_seed(self, request_id: str) -> None:
        """Drop the offset-0 full-block seed queued by allocate_new_blocks for
        this request's scratch. We have this because if we have sub block caching
        enabled we can have a prefix hit where the live window spans multiple 
        cached blocks so we do a straddle seed instead (the rolling window is not 
        block-aligned, so the whole-block seed would be wrong)."""
        scratch = self.scratch_blocks.get(request_id)
        if scratch is None:
            return
        self.pending_seed_copies = [
            (s, d, n)
            for (s, d, n) in self.pending_seed_copies
            if d != scratch.block_id
        ]

    def drain_pending_seed_copies(self) -> list[tuple[int, int, int]]:
        """Return and clear pending scratch-seed copies (src, dst, num_tokens)."""
        ops = self.pending_seed_copies
        self.pending_seed_copies = []
        return ops

    def scratch_block_id(self, request_id: str) -> int:
        """Physical id of the request's mutable scratch block (for the metadata builder)."""
        return self.scratch_blocks[request_id].block_id


single_type_kv_cache_manager.spec_manager_map.update(
    {
        RBLNSlidingWindowSpec: RBLNSlidingWindowManager,
        RBLNScratchStableSWASpec: RBLNScratchStableSWAManager,
    }
)
