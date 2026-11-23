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


class TestDeepseekV4PrefixState:
    def _manager(self, *, draft=False, slots=5):
        from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec
        from vllm_rbln.v1.kv_cache import RBLNDeepseekV4StateManager

        return RBLNDeepseekV4StateManager(
            RBLNDeepseekV4StateSpec(
                num_state_slots=slots,
                block_size=4,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float32,
                prefix_caching=True,
                is_draft=draft,
            ),
            block_pool=BlockPool(64, True, 8),
            enable_caching=True,
            kv_cache_group_id=0,
            scheduler_block_size=8,
        )

    def _request(self, request_id="a", tokens=None):
        from tests.vllm.v1.core.utils import make_request

        return make_request(
            request_id, list(range(25)) if tokens is None else tokens, block_size=8
        )

    def _hit(self, manager, request, length, *, drop=False):
        return manager.find_longest_cache_hit(
            request.block_hashes,
            length,
            [0],
            manager.block_pool,
            manager.kv_cache_spec,
            drop,
            8,
        )

    def test_checkpoint_is_immutable_until_copy_completes(self):
        manager = self._manager()
        request = self._request()
        original = manager.allocate_new_blocks("a", 16, 16)[0]
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        assert self._hit(manager, request, 24) == (([original],), 16)

        assert manager.allocate_new_blocks("a", 20, 20) == []
        current = manager.req_to_blocks["a"][0]
        assert current is not original
        copies = manager.drain_pending_copies({"a"})
        assert copies == {"a": (original.block_id, current.block_id)}
        assert original.ref_cnt == 1
        manager.free("a")
        assert original.ref_cnt == 1
        manager.block_pool.free_blocks([original])
        assert self._hit(manager, request, 24) == (([original],), 16)

    def test_two_hits_get_private_rings_and_do_not_consume_attention_blocks(self):
        manager = self._manager()
        request = self._request()
        checkpoint = manager.allocate_new_blocks("a", 16, 16)[0]
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        manager.free("a")
        for rid in ("b", "c"):
            manager.add_local_computed_blocks(rid, [checkpoint], 16, 0)
            manager.allocate_new_blocks(rid, 20, 20)
        assert manager.req_to_blocks["b"] != manager.req_to_blocks["c"]
        assert manager.attention_pool.get_num_free_blocks() == 63
        assert checkpoint.ref_cnt == 2
        manager.free("b")  # A canceled, not-yet-submitted copy releases its source.
        assert checkpoint.ref_cnt == 1
        manager.free("c")
        assert manager.block_pool.get_num_free_blocks() == 4

    @pytest.mark.parametrize("draft", [False, True])
    def test_only_exact_replay_boundary_is_cached(self, draft):
        manager = self._manager(draft=draft)
        request = self._request()
        manager.allocate_new_blocks("a", 17, 17)
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        assert self._hit(manager, request, 24, drop=draft)[1] == 0

    def test_draft_checkpoint_requires_matching_lookahead_block(self):
        manager = self._manager(draft=True)
        request = self._request()
        block = manager.allocate_new_blocks("a", 16, 16)[0]
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        assert self._hit(manager, request, 24, drop=True) == (([block],), 16)
        assert self._hit(manager, request, 16, drop=False) == (([block],), 16)
        sibling = self._request("b", list(range(16)) + [99] * 9)
        assert sibling.block_hashes[:2] == request.block_hashes[:2]
        assert self._hit(manager, sibling, 24, drop=True)[1] == 0

    def test_draft_lookahead_uses_hash_blocks_not_the_larger_hca_interval(self):
        manager = self._manager(draft=True)
        manager.block_size = manager.scheduler_block_size = 16
        request = self._request()
        checkpoint = manager.allocate_new_blocks("a", 16, 16)[0]
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        hits, length = manager.find_longest_cache_hit(
            request.block_hashes,
            24,
            [0],
            manager.block_pool,
            manager.kv_cache_spec,
            True,
            16,
        )
        assert hits == ([checkpoint],)
        assert length == 16

    def test_decode_reuses_writable_ring_without_copies(self):
        manager = self._manager()
        original = manager.allocate_new_blocks("a", 25, 25)[0]
        for end in range(26, 40):
            assert manager.allocate_new_blocks("a", end, end) == []
        assert manager.req_to_blocks["a"] == [original]
        assert manager.drain_pending_copies({"a"}) == {}

    def test_admission_reserves_both_the_hit_and_its_private_copy(self):
        manager = self._manager(slots=3)
        request = self._request()
        checkpoint = manager.allocate_new_blocks("a", 16, 16)[0]
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        manager.free("a")
        manager.allocate_new_blocks("busy", 1, 1)
        assert manager.block_pool.get_num_free_blocks() == 1
        assert (
            manager.get_num_blocks_to_allocate("hit", 20, [checkpoint], 16, 16, 20)
            > manager.attention_pool.get_num_free_blocks()
        )
        manager.free("busy")
        assert (
            manager.get_num_blocks_to_allocate("hit", 20, [checkpoint], 16, 16, 20) == 0
        )

    def test_unscheduled_copy_stays_pending_until_resume_or_preemption(self):
        manager = self._manager()
        request = self._request()
        checkpoint = manager.allocate_new_blocks("a", 16, 16)[0]
        manager.cache_blocks(request, 16, 0, replay_boundaries=(16,))
        manager.allocate_new_blocks("a", 20, 20)
        assert manager.drain_pending_copies({"other"}) == {}
        assert checkpoint.ref_cnt == 1
        manager.free("a")
        assert checkpoint.ref_cnt == 0
        assert manager.drain_pending_copies({"a"}) == {}


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
        assert _manager().get_num_blocks_to_allocate("r", 10, [], 0, 0, 10) == 1

    def test_num_blocks_to_allocate_is_zero_when_present(self):
        m = _manager()
        m.req_to_blocks["r"] = [object()]
        assert m.get_num_blocks_to_allocate("r", 10, [], 0, 0, 10) == 0

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
        # Prefix caching disabled: one empty list per kv_cache_group_id, and a
        # hit length of zero to go with it.
        hits = RBLNSlidingWindowManager.find_longest_cache_hit(
            block_hashes=None,
            max_length=0,
            kv_cache_group_ids=[0, 1, 2],
            block_pool=None,
            kv_cache_spec=None,
            drop_eagle_block=False,
            alignment_tokens=None,
        )
        assert hits == (([], [], []), 0)

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
                # Groups overlay from byte 0, so the pool is sized by the widest.
                KVCacheTensor(
                    size=max(swa.page_size_bytes, full.page_size_bytes) * num_blocks,
                    layers=["l0"],
                    layer_stride=swa.page_size_bytes * num_blocks,
                    block_stride=swa.page_size_bytes,
                ),
                KVCacheTensor(
                    size=max(swa.page_size_bytes, full.page_size_bytes) * num_blocks,
                    layers=["l1"],
                    layer_stride=full.page_size_bytes * num_blocks,
                    block_stride=full.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["l0"], swa),
                KVCacheGroupSpec(["l1"], full),
            ],
        )
        return get_kv_cache_coordinator(
            kv_cache_config=config,
            max_model_len=block_size * 32,
            max_in_flight_tokens=512,
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


class TestCoordinatorCallsTheOverrides:
    """Drive the coordinator so every override runs with the arguments the
    pinned vllm passes and answers in the shape it unpacks. Prefix caching
    stays on in the config -- this manager is what makes a sliding-window group
    report no hit -- so the lookup really does reach us."""

    def _coordinator(self, block_size=16, sliding_window=16, num_blocks=8):
        from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )
        from vllm.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )

        from vllm_rbln.platform import RblnPlatform

        register_all_kvcache_specs(None)
        RblnPlatform.register_custom_kv_cache_specs(None)
        swa = _spec(block_size=block_size, sliding_window=sliding_window)
        config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=swa.page_size_bytes * num_blocks,
                    layers=["l0"],
                    layer_stride=swa.page_size_bytes * num_blocks,
                    block_stride=swa.page_size_bytes,
                )
            ],
            kv_cache_groups=[KVCacheGroupSpec(["l0"], swa)],
        )
        return get_kv_cache_coordinator(
            kv_cache_config=config,
            max_model_len=block_size * 32,
            max_in_flight_tokens=512,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
        )

    def test_lookup_through_the_coordinator_reports_no_hit(self):
        coord = self._coordinator()
        assert isinstance(coord.single_type_managers[0], RBLNSlidingWindowManager)
        # (per-group blocks, hit length, uncached shared prefix)
        assert coord.find_longest_cache_hit([], 64) == (([],), 0, 0)

    def test_blocks_to_allocate_through_the_coordinator_is_one(self):
        coord = self._coordinator()
        assert (
            coord.get_num_blocks_to_allocate(
                request_id="r",
                num_tokens=1024,
                new_computed_blocks=([],),
                num_encoder_tokens=0,
                total_computed_tokens=0,
                num_local_computed_tokens=0,
                num_tokens_main_model=1024,
            )
            == 1
        )

    def test_remove_skipped_blocks_through_the_coordinator_is_a_noop(self):
        # The in-place ring buffer never skips a block; what matters is that
        # the call lands.
        coord = self._coordinator()
        coord.allocate_new_blocks("r", num_tokens=1024, num_tokens_main_model=1024)
        coord.remove_skipped_blocks("r", 512, 1024)
        assert len(coord.single_type_managers[0].req_to_blocks["r"]) == 1
