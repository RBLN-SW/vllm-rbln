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

    def test_allocate_computed_fresh_no_external_allocates_nothing(self):
        m = _manager(_never_pool())
        m.allocate_new_computed_blocks("r", [], 0, 0)
        assert m.num_cached_block["r"] == 0
        assert m.req_to_blocks["r"] == []

    def test_allocate_computed_fresh_with_external_allocates_one(self):
        pool = _pool()
        free0 = pool.get_num_free_blocks()
        m = _manager(pool)
        m.allocate_new_computed_blocks("r", [], 0, 5)
        assert len(m.req_to_blocks["r"]) == 1
        assert pool.get_num_free_blocks() == free0 - 1

    def test_allocate_computed_already_cached_is_noop(self):
        # Idempotent fast path: a request already in num_cached_block draws no
        # block (external tokens are ignored once cached).
        m = _manager(_never_pool())
        m.num_cached_block["r"] = 0
        m.allocate_new_computed_blocks("r", [], 0, 5)
        assert m.req_to_blocks["r"] == []

    def test_allocate_computed_rejects_prefix_hit_blocks(self):
        # A prefix-cache hit (non-empty new_computed_blocks) is unsupported.
        m = _manager(_never_pool())
        with pytest.raises(AssertionError):
            m.allocate_new_computed_blocks("r", [object()], 0, 0)

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
