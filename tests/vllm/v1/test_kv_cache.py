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

"""Tests for the RBLN KV cache managers (v1/kv_cache.py): the sliding-window one
holds one physical block per request with prefix caching disabled, and the Mamba
one carries a prompt's block-boundary state across a P/D transfer."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

import vllm_rbln.v1.kv_cache as kv_cache
from tests.vllm.v1.core.utils import make_request
from vllm_rbln.v1.kv_cache import (
    REMOTE_STATE_BOUNDARY,
    UNTRUNCATED_PROMPT_LENGTH,
    RBLNMambaManager,
    RBLNMambaSpec,
    RBLNSlidingWindowManager,
    RBLNSlidingWindowSpec,
)


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


class TestRBLNMambaManager:
    """An align-mode hybrid driven through the coordinator: block 16, a 40-token
    prompt, so the last boundary state sits in block 1 and the running one in 2."""

    BLOCK = 16

    def _coordinator(self, num_blocks=32, use_eagle=False):
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

        register_all_kvcache_specs(None)
        RblnPlatform.register_custom_kv_cache_specs(None)
        full = FullAttentionSpec(
            block_size=self.BLOCK, num_kv_heads=1, head_size=4, dtype=torch.float16
        )
        state = RBLNMambaSpec(
            block_size=self.BLOCK,
            shapes=((16, 16),),
            dtypes=(torch.uint8,),
            mamba_cache_mode="align",
        )
        assert full.page_size_bytes == state.page_size_bytes
        page = full.page_size_bytes
        config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=page * num_blocks,
                    layers=[layer],
                    layer_stride=page * num_blocks,
                    block_stride=page,
                )
                for layer in ("attn", "state")
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["attn"], full),
                KVCacheGroupSpec(["state"], state),
            ],
            # CacheConfig's default, which keeps only the reachable boundaries.
            prefix_cache_retention_interval=0,
        )
        coord = get_kv_cache_coordinator(
            kv_cache_config=config,
            max_model_len=self.BLOCK * 32,
            max_in_flight_tokens=512,
            use_eagle=use_eagle,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=self.BLOCK,
            hash_block_size=self.BLOCK,
        )
        assert isinstance(coord.single_type_managers[1], RBLNMambaManager)
        return coord

    def _request(self, request_id, params, num_tokens=40):
        request = make_request(request_id, list(range(num_tokens)), self.BLOCK)
        request.kv_transfer_params = params
        return request

    def _producer_params(self, num_prompt_tokens=40):
        return {"do_remote_decode": True, UNTRUNCATED_PROMPT_LENGTH: num_prompt_tokens}

    @pytest.mark.parametrize("producer", [True, False])
    def test_producer_keeps_the_last_boundary_state(self, producer):
        coord = self._coordinator()
        state = coord.single_type_managers[1]
        request = self._request("p", self._producer_params() if producer else None)
        state.track_transfer(request, 0, 0)
        for computed, end in ((0, 16), (16, 32), (32, 36), (36, 40)):
            coord.remove_skipped_blocks("p", computed, 40)
            coord.allocate_new_blocks("p", end, end)

        blocks = state.req_to_blocks["p"]
        assert blocks[0].is_null and not blocks[2].is_null
        assert blocks[1].is_null is not producer

    @pytest.mark.parametrize(
        ("boundary", "cached"), [(32, True), (None, False), (16, False)]
    )
    def test_consumer_receives_and_caches_the_boundary_state(self, boundary, cached):
        coord = self._coordinator()
        state = coord.single_type_managers[1]
        params = {"do_remote_prefill": True}
        if boundary is not None:
            params[REMOTE_STATE_BOUNDARY] = boundary
        request = self._request("d", params)
        num_external = 39
        state.track_transfer(request, 0, num_external)

        num_free = coord.block_pool.get_num_free_blocks()
        num_to_allocate = coord.get_num_blocks_to_allocate(
            request_id="d",
            num_tokens=num_external,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=num_external,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_external,
        )
        coord.allocate_new_computed_blocks("d", ([], []), 0, num_external)
        coord.allocate_new_blocks("d", num_external, num_external)
        assert num_free - coord.block_pool.get_num_free_blocks() == num_to_allocate

        blocks = state.req_to_blocks["d"]
        assert [b.is_null for b in blocks] == [True, not cached, False]
        coord.cache_blocks(request, num_external)
        _, hit_length, _ = coord.find_longest_cache_hit(request.block_hashes, 39)
        assert hit_length == (32 if cached else 0)

    @pytest.mark.parametrize(
        ("num_local", "num_external"),
        [
            # Re-admitted once its KV has arrived: nothing more is received.
            (0, 0),
            # A local prefix hit already reaches the boundary.
            (32, 7),
        ],
    )
    def test_consumer_notes_no_block_it_will_not_receive(
        self, monkeypatch, num_local, num_external
    ):
        state = self._coordinator().single_type_managers[1]
        warnings = []
        monkeypatch.setattr(kv_cache.logger, "warning", lambda *a: warnings.append(a))
        request = self._request(
            "d", {"do_remote_prefill": True, REMOTE_STATE_BOUNDARY: 32}
        )
        # Left by an earlier admission attempt whose allocation failed.
        state._received_boundary["d"] = 1

        state.track_transfer(request, num_local, num_external)

        assert "d" not in state._received_boundary
        assert not warnings

    def test_freeing_a_producer_request_drops_its_kept_boundary(self):
        coord = self._coordinator()
        state = coord.single_type_managers[1]
        state.track_transfer(self._request("p", self._producer_params()), 0, 0)
        coord.allocate_new_blocks("p", 16, 16)
        assert state._kept_boundary == {"p": 1}

        coord.free("p")

        assert state._kept_boundary == {}

    # The producer holds the prompt less its last token, while a repeat looks
    # up from the whole prompt: a 49-token one hits 32, not 16.
    @pytest.mark.parametrize(("num_prompt_tokens", "kept"), [(40, 0), (49, 1)])
    def test_an_eagle_producer_keeps_the_state_before_the_drop(
        self, num_prompt_tokens, kept
    ):
        coord = self._coordinator(use_eagle=True)
        state = coord.single_type_managers[1]
        num_tokens = num_prompt_tokens - 1
        request = self._request(
            "p", self._producer_params(num_prompt_tokens), num_tokens
        )
        state.track_transfer(request, 0, 0)
        for computed in range(0, num_tokens, self.BLOCK):
            end = min(computed + self.BLOCK, num_tokens)
            coord.remove_skipped_blocks("p", computed, num_tokens)
            coord.allocate_new_blocks("p", end, end)

        assert state._kept_boundary == {"p": kept}
        assert not state.req_to_blocks["p"][kept].is_null

    @pytest.mark.parametrize(
        ("num_prompt_tokens", "boundary", "cached"),
        [(40, 16, True), (40, 32, False), (49, 32, True), (49, 16, False)],
    )
    def test_an_eagle_consumer_caches_the_state_before_the_drop(
        self, num_prompt_tokens, boundary, cached
    ):
        coord = self._coordinator(use_eagle=True)
        state = coord.single_type_managers[1]
        request = self._request(
            "d",
            {"do_remote_prefill": True, REMOTE_STATE_BOUNDARY: boundary},
            num_prompt_tokens,
        )
        num_external = num_prompt_tokens - 1
        state.track_transfer(request, 0, num_external)
        coord.allocate_new_computed_blocks("d", ([], []), 0, num_external)
        coord.allocate_new_blocks("d", num_external, num_external)

        blocks = state.req_to_blocks["d"]
        boundary_slot = boundary // self.BLOCK - 1
        assert blocks[boundary_slot].is_null is not cached
        coord.cache_blocks(request, num_external)
        _, hit_length, _ = coord.find_longest_cache_hit(
            request.block_hashes, num_external
        )
        assert hit_length == (boundary if cached else 0)


class TestRBLNSingleBlockSlidingWindowManager:
    def _manager(self, num_blocks=6, sliding_window=16):
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )

        from vllm_rbln.platform import RblnPlatform
        from vllm_rbln.v1.kv_cache import RBLNSingleBlockSlidingWindowSpec

        register_all_kvcache_specs(None)
        RblnPlatform.register_custom_kv_cache_specs(None)
        spec = RBLNSingleBlockSlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.float16,
            sliding_window=sliding_window,
        )
        pool = BlockPool(
            num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16
        )
        return KVCacheSpecRegistry.get_manager_class(spec)(
            spec,
            block_pool=pool,
            enable_caching=True,
            kv_cache_group_id=0,
            scheduler_block_size=16,
        )

    def _step(self, m, num_computed, num_new, num_prompt=20, resumable=False):
        # allocate_slots' order: free what left the window, count, allocate, cache.
        from tests.vllm.v1.core.utils import make_request

        m.remove_skipped_blocks("r", num_computed)
        n = num_computed + num_new
        m.get_num_blocks_to_allocate("r", n, [], num_computed, num_computed, n)
        m.allocate_new_blocks("r", n, n)
        request = make_request("r", list(range(max(n, num_prompt))), 16)
        request.num_prompt_tokens = num_prompt
        request.resumable = resumable
        m.cache_blocks(request, n, replay_boundaries=[])
        copies, m.pending_copies = m.pending_copies, []
        return copies

    def test_a_step_entering_a_block_copies_the_rows_it_has_yet_to_write(self):
        m = self._manager()
        # A prefill chunk that stays in the first block has nothing to copy.
        assert self._step(m, 0, 16) == []
        first = m.req_to_blocks["r"][0]
        # The last chunk enters the second block at its row 0, the first decode
        # step stays in it, and a later one enters the third.
        assert self._step(m, 16, 4) == [
            (first.block_id, m.req_to_blocks["r"][1].block_id, 0)
        ]
        assert first.ref_cnt == 2
        assert self._step(m, 20, 1) == []
        for pos in range(21, 32):
            self._step(m, pos, 1)
        second = m.req_to_blocks["r"][1]
        assert self._step(m, 32, 1) == [
            (second.block_id, m.req_to_blocks["r"][2].block_id, 0)
        ]

    def test_a_first_step_at_the_start_of_a_block_copies_too(self):
        # A resumed request: cache hits up to a block boundary, then one token,
        # before the manager has cached anything for it.
        m = self._manager()
        m.get_num_blocks_to_allocate("r", 33, [], 32, 32, 33)
        m.allocate_new_blocks("r", 33, 33)
        blocks = m.req_to_blocks["r"]
        assert m.pending_copies == [(blocks[1].block_id, blocks[2].block_id, 0)]

    def test_the_blocks_before_the_window_are_freed_once_the_request_decodes(self):
        m = self._manager()
        self._step(m, 0, 16, num_prompt=24)
        first = m.req_to_blocks["r"][0]
        # Prefill enters the second block and goes on reading the first.
        self._step(m, 16, 4, num_prompt=24)
        m.block_pool.free_blocks([first])  # the copy's ref
        self._step(m, 20, 4, num_prompt=24)
        assert m.req_to_blocks["r"][0] is first
        self._step(m, 24, 1, num_prompt=24)
        assert m.req_to_blocks["r"][0].is_null
        assert first.ref_cnt == 0

    def test_a_streaming_session_keeps_the_blocks_it_prefills_over_again(self):
        m = self._manager()
        self._step(m, 0, 16, resumable=True)
        first = m.req_to_blocks["r"][0]
        for pos, num_new in ((16, 4), (20, 1), (21, 1)):
            self._step(m, pos, num_new, resumable=True)
        assert m.req_to_blocks["r"][0] is first

    def test_a_window_wider_than_the_block_is_refused(self):
        with pytest.raises(NotImplementedError, match="use_swa_single_block"):
            self._manager(sliding_window=17)
