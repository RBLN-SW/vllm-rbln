# Copyright 2026 Rebellions Inc. All rights reserved.
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

"""Tests for the RBLN sliding-window KV cache manager (v1/kv_cache.py): one
physical block per request and prefix caching disabled, unlike the base."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_rbln.v1.kv_cache import RBLNSlidingWindowManager, RBLNSlidingWindowSpec


def _spec(*, block_size=16, sliding_window=16):
    return RBLNSlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        sliding_window=sliding_window,
    )


def _pool(num_blocks=4):
    return BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=False, hash_block_size=16
    )


def _never_pool():
    # A block_pool whose allocation is a hard error -- used to prove a code path
    # never draws a block.
    def _boom(n):
        raise AssertionError("block_pool.get_new_blocks must not be called")

    return SimpleNamespace(get_new_blocks=_boom)


def _manager(block_pool=None):
    m = object.__new__(RBLNSlidingWindowManager)
    m.req_to_blocks = defaultdict(list)
    m.num_cached_block = {}
    m.block_pool = block_pool
    return m


class TestRBLNSlidingWindowSpec:
    def test_post_init_accepts_divisible_block_size(self):
        # 32 % 16 == 0 -> constructs; head_size_v proves super().__post_init__ ran.
        spec = _spec(block_size=32, sliding_window=16)
        assert isinstance(spec, RBLNSlidingWindowSpec)
        assert spec.head_size_v == spec.head_size

    def test_post_init_rejects_indivisible_block_size(self):
        with pytest.raises(AssertionError):
            _spec(block_size=24, sliding_window=16)  # 24 % 16 != 0

    def test_max_memory_usage_bytes_is_one_page(self):
        # The base would dereference vllm_config; the RBLN override pins one
        # physical block and ignores it, so None is safe.
        spec = _spec()
        assert spec.max_memory_usage_bytes(None) == spec.page_size_bytes


class TestRBLNSlidingWindowManager:
    def test_num_blocks_to_allocate_is_one_when_empty(self):
        assert _manager().get_num_blocks_to_allocate("r", 10, [], 0, 10) == 1

    def test_num_blocks_to_allocate_is_zero_when_present(self):
        m = _manager()
        m.req_to_blocks["r"] = [object()]
        assert m.get_num_blocks_to_allocate("r", 10, [], 0, 10) == 0

    def test_allocate_new_blocks_first_call_allocates_one(self):
        pool = _pool()
        free0 = pool.get_num_free_blocks()
        m = _manager(pool)
        blocks = m.allocate_new_blocks("r", 10, 10)
        assert len(blocks) == 1
        assert m.req_to_blocks["r"] == blocks
        assert pool.get_num_free_blocks() == free0 - 1

    def test_allocate_new_blocks_noop_when_present(self):
        m = _manager(_never_pool())
        m.req_to_blocks["r"] = [object()]
        assert m.allocate_new_blocks("r", 10, 10) == []

    def test_add_local_computed_fresh_marks_cached_without_blocks(self):
        m = _manager(_never_pool())
        m.add_local_computed_blocks("r", [], 0, 5)
        assert m.num_cached_block["r"] == 0
        assert m.req_to_blocks["r"] == []

    def test_add_local_computed_rejects_prefix_hit_blocks(self):
        # A prefix-cache hit (non-empty new_computed_blocks) is unsupported.
        m = _manager(_never_pool())
        with pytest.raises(AssertionError):
            m.add_local_computed_blocks("r", [object()], 0, 0)

    def test_allocate_external_no_external_allocates_nothing(self):
        m = _manager(_never_pool())
        m.allocate_external_computed_blocks("r", 0, 0)
        assert m.req_to_blocks["r"] == []

    def test_allocate_external_allocates_one_regardless_of_token_count(self):
        # cdiv(6077, 16) would be 380 blocks upstream; RBLN keeps one.
        pool = _pool()
        free0 = pool.get_num_free_blocks()
        m = _manager(pool)
        m.allocate_external_computed_blocks("r", 0, 6077)
        assert len(m.req_to_blocks["r"]) == 1
        assert pool.get_num_free_blocks() == free0 - 1

    def test_allocate_external_rejects_preexisting_blocks(self):
        # add_local_computed_blocks runs first and leaves the list empty; a
        # block already present means the coordinator sequence broke.
        m = _manager(_never_pool())
        m.req_to_blocks["r"] = [object()]
        with pytest.raises(AssertionError):
            m.allocate_external_computed_blocks("r", 0, 5)

    def test_find_longest_cache_hit_returns_empty_per_group(self):
        # Prefix caching disabled: one empty list per kv_cache_group_id.
        hits = RBLNSlidingWindowManager.find_longest_cache_hit(
            block_hashes=None,
            max_length=0,
            kv_cache_group_ids=[0, 1, 2],
            block_pool=None,
            kv_cache_spec=None,
            drop_eagle_block=False,
            alignment_tokens=None,
        )
        assert hits == ([], [], [])

    def test_get_num_common_prefix_blocks_is_zero(self):
        assert _manager().get_num_common_prefix_blocks("r") == 0


class TestRegistration:
    def test_platform_registers_rbln_sliding_window_manager(self):
        # RblnPlatform registers the sliding-window spec -> manager mapping in
        # KVCacheSpecRegistry; get_manager_class resolves it via the spec's MRO.
        from vllm_rbln.platform import RblnPlatform

        RblnPlatform.register_custom_kv_cache_specs(None)
        assert (
            KVCacheSpecRegistry.get_manager_class(_spec()) is RBLNSlidingWindowManager
        )


class TestCoordinatorExternalTokens:
    """Drive the real KVCacheCoordinator so the override is exercised through
    the hook names the pinned vllm actually calls (the KV-connector D-side
    receive path), not by direct method call."""

    def _coordinator(self, block_size, sliding_window, num_blocks=64):
        from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )

        from vllm_rbln.platform import RblnPlatform

        # Built-in registration is lazy and skipped once any spec is present.
        register_all_kvcache_specs(None)
        RblnPlatform.register_custom_kv_cache_specs(None)
        common = dict(num_kv_heads=2, head_size=8, dtype=torch.float16)
        swa = RBLNSlidingWindowSpec(
            block_size=block_size, sliding_window=sliding_window, **common
        )
        full = FullAttentionSpec(block_size=block_size, **common)
        config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(size=swa.page_size_bytes * num_blocks, shared_by=["l0"]),
                KVCacheTensor(size=full.page_size_bytes * num_blocks, shared_by=["l1"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["l0"], swa),
                KVCacheGroupSpec(["l1"], full),
            ],
        )
        return get_kv_cache_coordinator(
            kv_cache_config=config,
            max_model_len=block_size * 32,
            max_num_batched_tokens=512,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
        )

    def test_external_tokens_over_one_block_keep_swa_group_at_one_block(self):
        # ICR-15: gpt-oss P/D with 6077 external tokens and block_size 4096.
        # P holds one SWA block; D must not allocate cdiv(6077, 4096) == 2.
        block_size, ext = 4096, 6077
        coord = self._coordinator(block_size=block_size, sliding_window=128)
        assert isinstance(coord.single_type_managers[0], RBLNSlidingWindowManager)

        coord.allocate_new_computed_blocks(
            "r",
            new_computed_blocks=([], []),
            num_local_computed_tokens=0,
            num_external_computed_tokens=ext,
        )
        swa_blocks, full_blocks = (
            m.req_to_blocks["r"] for m in coord.single_type_managers
        )
        assert len(swa_blocks) == 1
        assert len(full_blocks) == -(-ext // block_size) == 2

        # Nothing further for the (one) new token of the remote-prefilled request.
        new = coord.allocate_new_blocks(
            "r", num_tokens=ext + 1, num_tokens_main_model=ext + 1
        )
        assert new[0] == []
        assert len(coord.single_type_managers[0].req_to_blocks["r"]) == 1
