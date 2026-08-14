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

from vllm_rbln.v1.core.page_layout import PageLayout, PageLayoutConfig, resolve_config
from vllm_rbln.v1.core.rbln_page_layout_kv_cache_manager import (
    RBLNPageLayoutKVCacheManager,
)

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
        producer = manager.block_table("r1")[0]

        _, num_computed, _ = admit(manager, "r2", prefix + list(range(70, 78)))
        assert num_computed == len(prefix)
        table = manager.block_table("r2")
        assert table[0] != producer
        assert_identity(manager, "r2")

        (op,) = manager.pending_copy_ops
        assert op.src_kernel_block_id == producer
        assert op.dst_kernel_block_id == table[0]
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


class TestMatchGuard:
    def test_a_match_that_is_not_a_run_is_truncated(self, monkeypatch):
        """A hash can name two blocks, so a lookup can straddle kernel blocks."""
        manager = make_manager()
        request = make_request("r1", list(range(4 * PAGE)), PAGE)

        stitched = [manager.pool.blocks[i] for i in (0, 1, 6, 7)]
        monkeypatch.setattr(
            RBLNPageLayoutKVCacheManager.__bases__[0],
            "get_computed_blocks",
            lambda self, req: (self.create_kv_cache_blocks((stitched,)), 4 * PAGE),
        )
        blocks, num_computed = manager.get_computed_blocks(request)
        assert num_computed == 2 * PAGE
        assert [b.block_id for b in blocks.blocks[0]] == [0, 1]

    def test_a_laid_out_match_is_left_alone(self, monkeypatch):
        manager = make_manager()
        request = make_request("r1", list(range(4 * PAGE)), PAGE)

        run = [manager.pool.blocks[i] for i in (4, 5, 6, 7)]
        monkeypatch.setattr(
            RBLNPageLayoutKVCacheManager.__bases__[0],
            "get_computed_blocks",
            lambda self, req: (self.create_kv_cache_blocks((run,)), 4 * PAGE),
        )
        blocks, num_computed = manager.get_computed_blocks(request)
        assert num_computed == 4 * PAGE
        assert [b.block_id for b in blocks.blocks[0]] == [4, 5, 6, 7]


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


class TestConfig:
    def test_degenerate_geometry_is_rejected(self):
        config = PageLayoutConfig(PageLayout(PAGE, PAGE), num_kernel_blocks=8)
        assert not RBLNPageLayoutKVCacheManager.can_use_page_layout(
            make_kv_cache_config([full_attention_spec(PAGE)], 32), config
        )

    def test_a_pool_smaller_than_one_kernel_block_is_rejected(self):
        with pytest.raises(ValueError, match="less than one kernel block"):
            make_manager(num_pages=PPE - 1)
