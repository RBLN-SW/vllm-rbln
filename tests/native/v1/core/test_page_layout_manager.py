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

"""The manager driven through upstream's own allocation path.

What is being pinned is the identity map: every page a request holds must sit at
the slot its own id names, and every group of `pages_per_kernel_block` pages must
sit in one kernel block. Break either and the worker silently reads the wrong
memory, so these tests assert on the layout rather than on the decisions that
produced it.
"""

import pytest
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

from vllm_rbln.v1.core.kernel_block_pool import RBLNKVCacheBlock
from vllm_rbln.v1.core.page_layout import PageLayout, PageLayoutConfig, resolve_config
from vllm_rbln.v1.core.rbln_page_layout_kv_cache_manager import (
    RBLNKVCacheBlocks,
    RBLNPageLayoutKVCacheManager,
)
from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler

from .utils import full_attention_spec, make_kv_cache_config, make_request

PAGE = 4
PPE = 4  # pages per kernel block
KERNEL_BLOCK = PAGE * PPE


def make_manager(num_pages: int = 32, max_model_len: int = 256):
    config = resolve_config(
        page_size=PAGE, kernel_block_size=KERNEL_BLOCK, num_pages=num_pages
    )
    return RBLNPageLayoutKVCacheManager(
        kv_cache_config=make_kv_cache_config([full_attention_spec(PAGE)], num_pages),
        page_layout_config=config,
        max_model_len=max_model_len,
        scheduler_block_size=PAGE,
        hash_block_size=PAGE,
        enable_caching=True,
    )


def admit(manager, request_id: str, tokens: list[int]):
    """Run a request through match -> allocate, as the scheduler does."""
    request = make_request(request_id, tokens, PAGE)
    blocks, num_computed = manager.get_computed_blocks(request)
    request.num_computed_tokens = num_computed
    result = manager.allocate_slots(
        request, len(tokens) - num_computed, num_computed, blocks
    )
    return request, num_computed, result


def finish(manager, request, num_tokens: int):
    """Cache what the forward pass wrote, then release the request."""
    manager.coordinator.cache_blocks(request, num_tokens)
    manager.free(request)


def page_ids(manager, request_id):
    return [b.block_id for b in manager.coordinator.get_blocks(request_id)[0]]


def assert_identity(manager, request_id):
    ids = page_ids(manager, request_id)
    for index, page_id in enumerate(ids):
        assert page_id % PPE == index % PPE, (index, page_id, ids)
    for start in range(0, len(ids), PPE):
        group = ids[start : start + PPE]
        assert len({page // PPE for page in group}) == 1, (start, ids)


class TestLayout:
    def test_a_fresh_request_is_laid_out_as_runs(self):
        manager = make_manager()
        request, _, _ = admit(manager, "r1", list(range(40)))
        assert_identity(manager, "r1")
        assert manager.block_table("r1") == [
            page // PPE for page in page_ids(manager, "r1")[::PPE]
        ]

    def test_allocate_slots_returns_rbln_pages(self):
        manager = make_manager()
        _, _, result = admit(manager, "r1", list(range(40)))
        assert isinstance(result, RBLNKVCacheBlocks)
        assert result.pages
        assert all(isinstance(page, RBLNKVCacheBlock) for page in result.pages)
        assert all(
            page.kernel_block.index == page.block_id // PPE for page in result.pages
        )

    def test_growth_stays_in_the_same_kernel_block(self):
        manager = make_manager()
        request, _, _ = admit(manager, "r1", list(range(PAGE)))
        first = manager.block_table("r1")
        request.num_computed_tokens = PAGE
        manager.allocate_slots(request, PAGE)
        assert manager.block_table("r1") == first
        assert_identity(manager, "r1")

    def test_two_requests_never_share_an_open_kernel_block(self):
        manager = make_manager()
        admit(manager, "r1", list(range(PAGE)))
        admit(manager, "r2", list(range(100, 100 + PAGE)))
        assert manager.block_table("r1") != manager.block_table("r2")


class TestReuse:
    def test_a_whole_kernel_block_match_is_shared_without_copying(self):
        manager = make_manager()
        tokens = list(range(KERNEL_BLOCK + PAGE))
        request, _, _ = admit(manager, "r1", tokens)
        first = manager.block_table("r1")
        finish(manager, request, KERNEL_BLOCK)

        _, num_computed, _ = admit(manager, "r2", tokens)
        assert num_computed >= KERNEL_BLOCK
        assert manager.block_table("r2")[0] == first[0]
        assert manager.pending_copy_ops == []

    def test_a_partial_match_is_appended_to_when_nobody_holds_it(self):
        # The multi-turn shape: the previous turn stopped mid kernel block and
        # this one resumes it. Appending in place is what keeps the copy at zero.
        manager = make_manager()
        first_turn = list(range(2 * PAGE))
        request, _, _ = admit(manager, "r1", first_turn)
        held = manager.block_table("r1")
        finish(manager, request, len(first_turn))

        _, num_computed, _ = admit(manager, "r2", first_turn + list(range(50, 58)))
        assert num_computed == len(first_turn)
        assert manager.block_table("r2")[0] == held[0]
        assert manager.pending_copy_ops == []
        assert_identity(manager, "r2")

    def test_a_partial_match_is_copied_out_when_the_producer_is_live(self):
        manager = make_manager()
        prefix = list(range(2 * PAGE))
        request, _, _ = admit(manager, "r1", prefix + list(range(60, 64)))
        manager.coordinator.cache_blocks(request, len(prefix))  # live, not freed
        matched_block = manager.block_table("r1")[0]

        _, num_computed, _ = admit(manager, "r2", prefix + list(range(70, 78)))
        assert num_computed == len(prefix)
        table = manager.block_table("r2")
        assert table[0] != matched_block
        assert_identity(manager, "r2")

        (op,) = manager.pending_copy_ops
        assert op.src_block_id == matched_block
        assert op.dst_block_id == table[0]
        assert (op.src_start, op.dst_start) == (0, 0)
        assert op.num_tokens == len(prefix)

    def test_the_copy_source_stays_alive_until_the_worker_has_read_it(self):
        manager = make_manager()
        prefix = list(range(2 * PAGE))
        request, _, _ = admit(manager, "r1", prefix + list(range(60, 64)))
        manager.coordinator.cache_blocks(request, len(prefix))
        source = page_ids(manager, "r1")[: len(prefix) // PAGE]

        admit(manager, "r2", prefix + list(range(70, 78)))
        manager.free(request)
        assert all(manager.pool.blocks[page].ref_cnt > 0 for page in source)

        ops = manager.drain_pending_copy_ops()
        manager.release_copy_ops(ops)
        assert all(manager.pool.blocks[page].ref_cnt == 0 for page in source)


def publish(manager, page_ids, hashes):
    """Publish `hashes[i]` on page `page_ids[i]`, as `cache_full_blocks` would."""
    for page_id, block_hash in zip(page_ids, hashes):
        key = make_block_hash_with_group_id(block_hash, 0)
        block = manager.pool.blocks[page_id]
        if block.block_hash is None:
            block.set_block_hash(key)
        manager.pool.cached_block_hash_to_block.insert(key, block)


def hashes(n, salt=b""):
    return [BlockHash(bytes([i]) + salt) for i in range(n)]


class TestLookup:
    """R2: the longest addressable prefix, group at a time."""

    def test_a_whole_group_in_one_block_matches(self):
        manager = make_manager()
        keys = hashes(PPE)
        publish(manager, [4, 5, 6, 7], keys)
        assert [b.block_id for b in manager._match(keys, PPE)] == [4, 5, 6, 7]

    def test_it_prefers_the_block_that_continues_the_run(self):
        # A copy publishes a duplicate of one page; upstream would return either.
        manager = make_manager()
        keys = hashes(PPE)
        publish(manager, [4, 5, 6, 7], keys)
        publish(manager, [12], keys[:1])
        assert [b.block_id for b in manager._match(keys, PPE)] == [4, 5, 6, 7]

    def test_a_partial_group_ends_the_match(self):
        manager = make_manager()
        keys = hashes(2 * PPE)
        publish(manager, [4, 5], keys[:2])
        publish(manager, [10, 11], keys[2:4])  # right hashes, wrong slots
        assert [b.block_id for b in manager._match(keys, 2 * PPE)] == [4, 5]

    def test_pages_at_the_wrong_slot_do_not_match(self):
        manager = make_manager()
        keys = hashes(PPE)
        publish(manager, [5, 6, 7], keys[:3])  # group starts at slot 1
        assert manager._match(keys, PPE) == []

    def test_the_match_never_covers_the_last_token(self):
        # Upstream needs the last token recomputed to produce logits.
        manager = make_manager()
        request = make_request("r1", list(range(PPE * PAGE)), PAGE)
        publish(manager, [4, 5, 6, 7], request.block_hashes)
        _, num_computed = manager.get_computed_blocks(request)
        assert num_computed == (PPE - 1) * PAGE


class TestAdmission:
    def test_the_pool_refuses_what_it_cannot_back(self):
        # Two kernel blocks' worth of pages, but block 0 is unusable (page 0 is
        # upstream's null block), so only one request can be backed at a time.
        manager = make_manager(num_pages=2 * PPE, max_model_len=KERNEL_BLOCK)
        _, _, first = admit(manager, "r1", list(range(KERNEL_BLOCK)))
        assert first is not None
        _, _, second = admit(manager, "r2", list(range(100, 100 + KERNEL_BLOCK)))
        assert second is None

    def test_a_freed_request_returns_its_kernel_block(self):
        manager = make_manager(num_pages=2 * PPE, max_model_len=KERNEL_BLOCK)
        request, _, _ = admit(manager, "r1", list(range(KERNEL_BLOCK)))
        manager.free(request)
        _, _, second = admit(manager, "r2", list(range(100, 100 + KERNEL_BLOCK)))
        assert second is not None


class TestInflightReservation:
    """`RBLNScheduler._inflight_prefill_reserved_blocks` over a real pool.

    Only three attributes of the scheduler are in play, so the override is
    exercised directly rather than through a whole engine.
    """

    class _Prefill:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id

    def _reserved(self, manager, remaining: dict[str, int]) -> int:
        prefills = [self._Prefill(request_id) for request_id in remaining]

        class _Sched:
            kv_cache_manager = manager
            _inflight_prefills = prefills

            def _request_remaining_blocks(self, prefill):
                return remaining[prefill.request_id]

        return RBLNScheduler._inflight_prefill_reserved_blocks(_Sched())

    def test_a_page_short_each_reserves_a_whole_group_each(self):
        manager = make_manager()
        admit(manager, "r1", list(range(KERNEL_BLOCK)))
        admit(manager, "r2", list(range(100, 100 + KERNEL_BLOCK)))
        # Both filled their group exactly, so one more page apiece opens a new
        # one. Upstream's page sum would be 2 -- under page layout that is 2
        # whole groups.
        reserved = self._reserved(manager, {"r1": 1, "r2": 1})
        assert reserved == 2 * PPE
        assert manager.pool.kernel_blocks_needed(reserved) == 2

    def test_what_fits_in_a_prefills_own_group_reserves_nothing(self):
        manager = make_manager()
        admit(manager, "r1", list(range(PAGE)))  # one page of a fresh group
        assert self._reserved(manager, {"r1": PPE - 1}) == 0
        assert self._reserved(manager, {"r1": PPE}) == PPE


class TestConfig:
    def test_degenerate_geometry_is_rejected(self):
        config = PageLayoutConfig(PageLayout(PAGE, PAGE), num_kernel_blocks=8)
        assert not RBLNPageLayoutKVCacheManager.can_use_page_layout(
            make_kv_cache_config([full_attention_spec(PAGE)], 32), config
        )

    def test_a_pool_smaller_than_one_kernel_block_is_rejected(self):
        with pytest.raises(ValueError, match="less than one kernel block"):
            make_manager(num_pages=PPE - 1)
