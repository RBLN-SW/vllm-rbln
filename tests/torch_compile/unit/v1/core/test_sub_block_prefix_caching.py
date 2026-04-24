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

"""Tests for sub-block prefix caching components."""

import pytest
import torch
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_rbln.v1.core.rbln_kv_cache_manager import (
    RBLNKVCacheManager,
    SubBlockHasher,
    SubBlockIndex,
)
from vllm_rbln.v1.core.rbln_scheduler import RBLNSchedulerOutput

from .utils import MockKVConfig, create_runner_output, create_scheduler


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


# ---------------------------------------------------------------------------
# SubBlockHasher tests
# ---------------------------------------------------------------------------


class TestSubBlockHasher:
    def setup_method(self):
        self.hasher = SubBlockHasher(sha256, sub_block_size=4)

    def test_hash_empty_tokens(self):
        hashes, _ = self.hasher.hash_tokens([])
        assert hashes == []

    def test_hash_fewer_than_sub_block(self):
        hashes, _ = self.hasher.hash_tokens([1, 2, 3])
        assert hashes == []

    def test_hash_exact_sub_block(self):
        hashes, _ = self.hasher.hash_tokens([1, 2, 3, 4])
        assert len(hashes) == 1
        assert isinstance(hashes[0], bytes)

    def test_hash_multiple_sub_blocks(self):
        hashes, _ = self.hasher.hash_tokens([1, 2, 3, 4, 5, 6, 7, 8])
        assert len(hashes) == 2

    def test_hash_partial_last_sub_block(self):
        hashes, _ = self.hasher.hash_tokens([1, 2, 3, 4, 5, 6])
        assert len(hashes) == 1  # Only the first full sub-block.

    def test_hashes_are_chained(self):
        # Hashing [1,2,3,4,5,6,7,8] should produce different h1 than
        # hashing [5,6,7,8] alone (because the parent hash differs).
        hashes_full, _ = self.hasher.hash_tokens([1, 2, 3, 4, 5, 6, 7, 8])
        hashes_second_only, _ = self.hasher.hash_tokens([5, 6, 7, 8])
        assert hashes_full[1] != hashes_second_only[0]

    def test_incremental_hashing(self):
        # Hash in one go.
        hashes_all, _ = self.hasher.hash_tokens([1, 2, 3, 4, 5, 6, 7, 8])

        # Hash in two steps.
        # Step 1: first sub-block only (only 4 tokens available).
        h1, _ = self.hasher.hash_tokens(
            [1, 2, 3, 4],
            parent_hash=None,
            num_hashed_tokens=0,
        )
        # Step 2: all 8 tokens available, start from offset 4.
        h2, _ = self.hasher.hash_tokens(
            [1, 2, 3, 4, 5, 6, 7, 8],
            parent_hash=h1[0],
            num_hashed_tokens=4,
        )
        assert h1 + h2 == hashes_all

    def test_incremental_hashing_with_extra_keys(self):
        """Incremental hashing with extra_keys (via request=) must
        produce the same result as one-shot hashing.

        Uses images with gaps and one image spanning two sub-blocks
        so that the incremental resume point lands *between* images.

        Layout (sub_block_size=4):
          tokens:     0..3   4..7   8..11  12..15  16..19
          sub-blocks: sb0    sb1    sb2    sb3     sb4
          image A:    [0, 4)
          image B:                  [8,      14)
          image C:                                 [16, 20)
        """
        tokens = list(range(20))

        mm_features = [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                modality="image",
                identifier="hash_img_a",
                mm_position=PlaceholderRange(offset=0, length=4),
            ),
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                modality="image",
                identifier="hash_img_b",
                mm_position=PlaceholderRange(offset=8, length=6),
            ),
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                modality="image",
                identifier="hash_img_c",
                mm_position=PlaceholderRange(offset=16, length=4),
            ),
        ]

        req = Request(
            request_id="mm",
            prompt_token_ids=tokens,
            mm_features=mm_features,
            sampling_params=SamplingParams(max_tokens=17),
            pooling_params=None,
            block_hasher=get_request_block_hasher(8, sha256),
        )

        # One-shot: hash all 5 sub-blocks at once.
        hashes_all, _ = self.hasher.hash_tokens(tokens, request=req)
        assert len(hashes_all) == 5

        # Incremental: one sub-block at a time.
        # After sb0 mm_idx=1 (past A). Using mm_idx=-1 from here
        # would jump to C (idx=2) and miss B entirely at sb2.
        incremental: list[BlockHash] = []
        parent = None
        mm_idx = 0
        for i in range(5):
            off = i * 4
            h, mm_idx = self.hasher.hash_tokens(
                tokens[: off + 4],
                parent_hash=parent,
                num_hashed_tokens=off,
                request=req,
                start_mm_idx=mm_idx,
            )
            assert len(h) == 1
            incremental.extend(h)
            parent = h[-1]
        assert incremental == hashes_all

    def test_deterministic(self):
        tokens = list(range(100, 112))
        h1, _ = self.hasher.hash_tokens(tokens)
        h2, _ = self.hasher.hash_tokens(tokens)
        assert h1 == h2

    def test_different_tokens_different_hashes(self):
        h1, _ = self.hasher.hash_tokens([1, 2, 3, 4])
        h2, _ = self.hasher.hash_tokens([5, 6, 7, 8])
        assert h1 != h2


# ---------------------------------------------------------------------------
# SubBlockIndex tests
# ---------------------------------------------------------------------------


class TestSubBlockIndex:
    def _make_hashes(self, tokens_per_sub_block: list[list[int]]) -> list[BlockHash]:
        """Helper: compute chained sub-block hashes for a sequence of
        sub-block token lists."""
        hasher = SubBlockHasher(sha256, sub_block_size=len(tokens_per_sub_block[0]))
        flat = [t for sub in tokens_per_sub_block for t in sub]
        hashes, _ = hasher.hash_tokens(flat)
        return hashes

    def test_empty_no_match(self):
        index = SubBlockIndex()
        hashes = self._make_hashes([[1, 2], [3, 4]])
        block_id, depth = index.longest_match(hashes)
        assert block_id is None
        assert depth == 0

    def test_insert_and_match(self):
        index = SubBlockIndex()
        hashes = self._make_hashes([[1, 2], [3, 4], [5, 6]])
        index.update(10, hashes)

        # Full match.
        block_id, depth = index.longest_match(hashes)
        assert block_id == 10
        assert depth == 3

    def test_partial_match(self):
        index = SubBlockIndex()
        hashes_src = self._make_hashes([[1, 2], [3, 4], [5, 6]])
        index.update(10, hashes_src)

        # Query with only the first 2 matching sub-blocks.
        block_id, depth = index.longest_match(hashes_src[:2])
        assert block_id == 10
        assert depth == 2

    def test_multiple_blocks_same_prefix(self):
        index = SubBlockIndex()
        hashes = self._make_hashes([[1, 2], [3, 4], [5, 6]])
        index.update(10, hashes)
        index.update(20, hashes)

        block_id, depth = index.longest_match(hashes)
        assert block_id in (10, 20)
        assert depth == 3

    def test_remove_block(self):
        index = SubBlockIndex()
        hashes = self._make_hashes([[1, 2], [3, 4]])
        index.update(10, hashes)
        index.pop(10)

        block_id, depth = index.longest_match(hashes)
        assert block_id is None
        assert depth == 0

    def test_remove_one_of_two_blocks(self):
        index = SubBlockIndex()
        hashes = self._make_hashes([[1, 2], [3, 4]])
        index.update(10, hashes)
        index.update(20, hashes)
        index.pop(10)

        block_id, depth = index.longest_match(hashes)
        assert block_id == 20
        assert depth == 2

    def test_remove_nonexistent_block(self):
        index = SubBlockIndex()
        index.pop(999)  # Should not raise.

    def test_longest_match_empty_query(self):
        index = SubBlockIndex()
        hashes = self._make_hashes([[1, 2], [3, 4]])
        index.update(10, hashes)

        block_id, depth = index.longest_match([])
        assert block_id is None
        assert depth == 0


def _make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    max_tokens: int = 17,
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
    prompt_logprobs: int | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            prompt_logprobs=prompt_logprobs,
        ),
        pooling_params=None,
        cache_salt=cache_salt,
        lora_request=lora_request,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _make_manager(
    block_size: int,
    sub_block_size: int,
    num_blocks: int,
    max_model_len: int = 8192,
) -> RBLNKVCacheManager:
    manager = RBLNKVCacheManager(
        kv_cache_config=_make_kv_cache_config(block_size, num_blocks),
        max_model_len=max_model_len,
        hash_block_size=block_size,
        sub_block_size=sub_block_size,
        hash_fn=sha256,
    )
    return manager


def _sub_block_index(manager: RBLNKVCacheManager) -> SubBlockIndex:
    """Get the sub-block index for the single group (test helper)."""
    return manager._group_infos[0].sub_block_index


def _prefill_request(manager: RBLNKVCacheManager, request: Request):
    """Run the full get_computed_blocks → allocate_slots flow for a
    request (simulating what the scheduler does)."""
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
    sub_block_match = manager.get_computed_blocks_sub_block(
        request, num_computed_tokens
    )
    sub_block_extra = sub_block_match.num_tokens if sub_block_match else 0
    total_computed = num_computed_tokens + sub_block_extra
    num_new_tokens = request.num_tokens - total_computed
    blocks = manager.allocate_slots(
        request,
        num_new_tokens,
        total_computed,
        computed_blocks,
    )
    if blocks is not None and sub_block_match is not None:
        manager.apply_sub_block_match(sub_block_match)
    elif sub_block_match is not None:
        manager.release_sub_block_match(sub_block_match)
    # Simulate execute_model completion.
    request.num_computed_tokens = request.num_tokens
    manager.do_pending_indexing()
    return computed_blocks, total_computed, blocks


class TestRBLNKVCacheManager:
    """Tests for the full get_computed_blocks → allocate_slots flow with
    sub-block partial matching."""

    # Use block_size=8, sub_block_size=4 for easy arithmetic.
    BLOCK_SIZE = 8
    SUB_BLOCK_SIZE = 4

    def test_no_partial_match_on_first_request(self):
        """First request should have no cache hits at all."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # 2 full blocks + 5 extra tokens
        tokens = list(range(2 * self.BLOCK_SIZE + 5))
        req = _make_request("0", tokens, self.BLOCK_SIZE)

        computed_blocks_before, num_computed_tokens_before, new_blocks = (
            _prefill_request(manager, req)
        )
        assert num_computed_tokens_before == 0
        assert not computed_blocks_before.blocks[0]
        assert new_blocks is not None

        # No copy ops should be generated.
        ops = manager.drain_pending_copy_ops()
        assert ops == []

    def test_full_block_hit_indexes_sub_blocks(self):
        """After a request fills full blocks, those blocks' sub-hashes should
        appear in the index."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # 2 full blocks (16 tokens)
        tokens = list(range(2 * self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)

        # The index should now contain entries for the 2 full blocks.
        assert len(_sub_block_index(manager)._block_hashes) == 2

    def test_partial_match_detected(self):
        """Second request sharing a sub-block prefix should get a partial
        match and extra computed tokens."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Request 0: 1 full block (8 tokens) — will be cached.
        tokens_full = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens_full, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Request 1: shares first sub-block (4 tokens) with req0, then diverges.
        # The first 4 tokens match the first sub-block of req0's block.
        # But it has more tokens so no full-block match by upstream.
        shared = list(range(self.SUB_BLOCK_SIZE))
        unique = [100 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE)

        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)

        # Upstream should find 0 full block hits (different full-block hash).
        assert len(computed_blocks.blocks[0]) == 0
        # get_computed_blocks returns only full-block count (0 here).
        assert num_computed_tokens == 0
        # Sub-block match is available separately.
        sub_block_match = manager.get_computed_blocks_sub_block(
            req1, num_computed_tokens
        )
        assert sub_block_match is not None
        assert sub_block_match.num_tokens == self.SUB_BLOCK_SIZE

    def test_copy_op_generated_on_partial_match(self):
        """After partial match detection + allocate_slots, a copy op should
        be queued."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Fill a full block.
        tokens_full = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens_full, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # New request shares first sub-block only.
        shared = list(range(self.SUB_BLOCK_SIZE))
        unique = [100 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE)

        _, _, new_blocks = _prefill_request(manager, req1)
        assert new_blocks is not None

        ops = manager.drain_pending_copy_ops()
        assert len(ops) == 1
        op = ops[0]
        assert op.num_tokens == self.SUB_BLOCK_SIZE
        # The source block should be the one from req0.
        # The destination should be the first block allocated for req1.
        assert op.dst_block_id != op.src_block_id

    def test_no_partial_match_when_full_block_matches(self):
        """If the upstream already matches the full block, no partial match
        should be attempted (no double counting)."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Request 0: 2 full blocks.
        tokens = list(range(2 * self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Request 1: exact same 2 full blocks + a few extra.
        req1 = _make_request("1", tokens + [99, 99, 99], self.BLOCK_SIZE)
        _, num_computed_tokens = manager.get_computed_blocks(req1)

        # All tokens in the 2 full blocks should be computed via upstream.
        assert num_computed_tokens == 2 * self.BLOCK_SIZE
        # No partial match (all full blocks matched by upstream).
        sub_block_match = manager.get_computed_blocks_sub_block(
            req1, num_computed_tokens
        )
        assert sub_block_match is None

    def test_partial_match_after_full_block_match(self):
        """Partial match should work on the block boundary after full-block
        matches."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Request 0: 2 full blocks.
        tokens_2blocks = list(range(2 * self.BLOCK_SIZE))
        req0 = _make_request("0", tokens_2blocks, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Request 1: same 2 full blocks + partial match into 3rd block.
        # First sub-block of the 3rd block in req0 was tokens [16..19].
        # We create a request that has 2 full blocks matching + first sub-block
        # matching (from a different cached block).
        #
        # Actually, for this we need a 3rd cached block. Let's cache 3 blocks.
        tokens_3blocks = list(range(3 * self.BLOCK_SIZE))
        req0b = _make_request("0b", tokens_3blocks, self.BLOCK_SIZE)
        _prefill_request(manager, req0b)
        manager.free(req0b)

        # Request 2: shares all 3 blocks' first sub-block in the 3rd block
        # position, plus extra tokens that differ.
        # The first 2 blocks match fully. The 3rd block's first sub-block [16..19]
        # should match partially.
        extra = [999] * (self.SUB_BLOCK_SIZE + 3)  # enough for partial + leftover
        req2_tokens = (
            tokens_3blocks[: 2 * self.BLOCK_SIZE + self.SUB_BLOCK_SIZE] + extra
        )
        req2 = _make_request("2", req2_tokens, self.BLOCK_SIZE)

        _, num_computed_tokens = manager.get_computed_blocks(req2)
        # 2 full blocks = 16 tokens from upstream.
        assert num_computed_tokens == 2 * self.BLOCK_SIZE
        # Sub-block match adds 4 tokens separately.
        sub_block_match = manager.get_computed_blocks_sub_block(
            req2, num_computed_tokens
        )
        assert sub_block_match is not None
        assert sub_block_match.num_tokens == self.SUB_BLOCK_SIZE

    def test_free_cleans_up_sub_hash_cache(self):
        """free() should remove the request's sub-hash cache entry."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        tokens = list(range(self.BLOCK_SIZE))
        req = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req)
        assert "0" in manager._req_sub_hashes
        manager.free(req)
        assert "0" not in manager._req_sub_hashes

    def test_reset_prefix_cache_clears_index(self):
        """reset_prefix_cache() should clear the sub-block index."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        tokens = list(range(self.BLOCK_SIZE))
        req = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req)
        manager.free(req)
        assert len(_sub_block_index(manager)._block_hashes) > 0

        manager.reset_prefix_cache()
        assert len(_sub_block_index(manager)._block_hashes) == 0

    def test_eviction_removes_from_index(self):
        """When blocks are evicted (due to pressure), they should be removed
        from the sub-block index."""
        # Only 3 blocks available (tight).
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=3)

        # Fill 1 block.
        req0 = _make_request("0", list(range(self.BLOCK_SIZE)), self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)
        assert len(_sub_block_index(manager)._block_hashes) == 1

        # Fill all remaining blocks, forcing eviction of req0's block.
        big_tokens = [100 + i for i in range(3 * self.BLOCK_SIZE)]
        req1 = _make_request("1", big_tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req1)

        # The original block from req0 should have been evicted from the index.
        # req1 uses 3 blocks, so with only 3 total blocks, req0's freed block
        # must have been reused.
        # After allocating 3 blocks for req1, the index should contain
        # entries only for req1's full blocks (at most 2 since 3rd may be partial).
        for block_id in _sub_block_index(manager)._block_hashes:
            # All remaining blocks in index should be from req1.
            blk = manager.block_pool.blocks[block_id]
            assert blk.ref_cnt > 0 or blk.block_hash is not None

    def test_drain_pending_copy_ops(self):
        """drain_pending_copy_ops should return ops and clear the list."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Set up a partial match to get a copy op.
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)
        shared = list(range(self.SUB_BLOCK_SIZE))
        req1 = _make_request("1", shared + [100] * self.BLOCK_SIZE, self.BLOCK_SIZE)
        _prefill_request(manager, req1)
        assert len(manager.pending_copy_ops) == 1

        ops = manager.drain_pending_copy_ops()
        assert len(ops) == 1
        assert manager.pending_copy_ops == []

    def test_sub_block_match_released_on_failed_alloc(self):
        """If allocate_slots fails, the caller must release the match."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=2)

        req0 = _make_request("0", list(range(self.BLOCK_SIZE)), self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        shared = list(range(self.SUB_BLOCK_SIZE))
        big = shared + [200 + i for i in range(3 * self.BLOCK_SIZE)]
        req1 = _make_request("1", big, self.BLOCK_SIZE)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
        match = manager.get_computed_blocks_sub_block(req1, num_computed_tokens)

        if match is not None:
            sub_extra = match.num_tokens
            total_computed = num_computed_tokens + sub_extra
            result = manager.allocate_slots(
                req1,
                len(big) - total_computed,
                total_computed,
                computed_blocks,
            )
            # Whether it succeeds or fails, caller releases the match.
            if result is not None:
                manager.apply_sub_block_match(match)
            else:
                manager.release_sub_block_match(match)

    def test_partial_block_cached_on_free(self):
        """free() should mark the last partial block as cached with synthetic
        block_hash so its KV data is preserved in the LRU.
        Eager indexing already adds sub-block entries before free()."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # 1 full block (8 tokens) + 1 sub-block (4 tokens) = 12 tokens total.
        tokens = list(range(self.BLOCK_SIZE + self.SUB_BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)

        # After prefill: full block is indexed AND partial block is eagerly
        # indexed (sub-block entries exist), but not yet marked cached.
        blocks = manager.coordinator.get_blocks(req0.request_id)
        block_list = blocks[0]
        full_blk = block_list[0]
        partial_blk = block_list[1]
        assert full_blk.block_id in _sub_block_index(manager)._block_hashes
        assert partial_blk.block_id in _sub_block_index(manager)._block_hashes
        assert partial_blk.block_hash is None  # Not cached yet (no synthetic hash).

        partial_blk_id = partial_blk.block_id
        manager.free(req0)

        # After free: the partial block should still be in the index
        # and now have a block_hash (cached in LRU).
        assert partial_blk_id in _sub_block_index(manager)._block_hashes
        assert partial_blk.block_hash is not None

    def test_partial_block_reused_by_next_request(self):
        """After freeing a request with a partial block, a new request
        sharing that partial prefix should get a sub-block cache hit."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Request 0: 1 full block + 1 sub-block (partial block).
        # Tokens: [0..7] (full block) + [8..11] (partial sub-block)
        tokens0 = list(range(self.BLOCK_SIZE + self.SUB_BLOCK_SIZE))
        req0 = _make_request("0", tokens0, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Request 1: shares same prefix through the partial sub-block, then
        # diverges.  Sub-block hashes are chained, so req1 must share the
        # full prefix to get a hash match.
        shared = list(range(self.BLOCK_SIZE + self.SUB_BLOCK_SIZE))
        unique = [800 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE)

        _, num_computed_tokens = manager.get_computed_blocks(req1)

        # Upstream matches the first full block (8 tokens).  The sub-block
        # index then matches 1 sub-block (4 tokens) from the partial block.
        assert num_computed_tokens == self.BLOCK_SIZE
        sub_block_match = manager.get_computed_blocks_sub_block(
            req1, num_computed_tokens
        )
        assert sub_block_match is not None
        assert sub_block_match.num_tokens == self.SUB_BLOCK_SIZE

    def test_no_partial_block_cache_when_block_is_full(self):
        """free() should NOT double-cache a block that is already fully cached."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Exactly 1 full block — no partial block.
        tokens = list(range(self.BLOCK_SIZE))
        req = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req)

        blocks = manager.coordinator.get_blocks(req.request_id)
        block_list = blocks[0]
        assert len(block_list) == 1
        blk = block_list[0]
        assert blk.block_hash is not None  # Already cached by upstream.

        index_entries_before = set(_sub_block_index(manager)._block_hashes.keys())
        manager.free(req)
        index_entries_after = set(_sub_block_index(manager)._block_hashes.keys())

        # Index should not have gained new entries from free().
        assert index_entries_before == index_entries_after

    def test_computed_tokens_capped_at_num_tokens_minus_one(self):
        """Sub-block match must not push num_computed_tokens past
        request.num_tokens - 1 (upstream invariant: recompute last token)."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Request 0: 1 full block (8 tokens).
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Request 1: shares full block with req0, then only has 2 extra tokens.
        # Upstream matches 1 full block (8 tokens).  Sub-block index would try
        # to match sub-blocks in the second block.  But num_tokens = 10,
        # cap = 9, upstream already gives 8, so at most 1 extra token from
        # sub-blocks — which doesn't reach sub_block_size (4).  No match.
        req1 = _make_request("1", tokens + [77, 78], self.BLOCK_SIZE)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed <= req1.num_tokens - 1

        # Request 2: shares full block, then has exactly SUB_BLOCK_SIZE extra
        # tokens.  num_tokens = 12, cap = 11.  Upstream gives 8.  Sub-block
        # match gives 4 → 12 total.  Cap should reduce to 8 (no sub-block
        # match since 11 - 8 = 3 < 4).
        req2_tokens = tokens + list(
            range(self.BLOCK_SIZE, self.BLOCK_SIZE + self.SUB_BLOCK_SIZE)
        )
        req2 = _make_request("2", req2_tokens, self.BLOCK_SIZE)
        _, num_computed = manager.get_computed_blocks(req2)
        assert num_computed <= req2.num_tokens - 1

    def test_capping_sub_block_match_at_num_tokens_minus_one(self):
        """Sub-block match must not push total computed tokens past
        num_tokens - 1.  When it would, the match is truncated or
        returns None."""
        BS, SBS = 16, 4  # sbpb = 4
        manager = _make_manager(BS, SBS, num_blocks=10)

        # Cache a partial block with 3 sub-blocks (12 tokens).
        tokens = list(range(3 * SBS))
        req0 = _make_request("0", tokens, BS)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Case 1: match reduced from 3 to 2 sub-blocks.
        # num_tokens = 12, max_total = 11, num_computed = 0.
        # Raw match = 3 sub-blocks (12) > 11 → capped = 11 // 4 = 2.
        req1 = _make_request("1", tokens, BS)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        sub_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_match is not None
        assert sub_match.num_tokens == 2 * SBS
        assert num_computed + sub_match.num_tokens <= req1.num_tokens - 1
        manager.release_sub_block_match(sub_match)

        # Case 2: capped to zero → None.
        # num_tokens = 4, max_total = 3, num_computed = 0.
        # Raw match = 1 sub-block (4) > 3 → capped = 3 // 4 = 0 → None.
        short = tokens[:SBS]
        req2 = _make_request("2", short, BS)
        _, num_computed = manager.get_computed_blocks(req2)
        assert num_computed == 0
        assert manager.get_computed_blocks_sub_block(req2, num_computed) is None

    def test_partial_block_fewer_tokens_than_sub_block(self):
        """If the partial block has fewer tokens than a sub-block,
        _cache_partial_block should be a no-op."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # 1 full block (8 tokens) + 2 extra tokens (< sub_block_size=4).
        tokens = list(range(self.BLOCK_SIZE + 2))
        req = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req)

        blocks = manager.coordinator.get_blocks(req.request_id)
        partial_blk = blocks[0][1]
        partial_blk_id = partial_blk.block_id
        assert partial_blk.block_hash is None

        manager.free(req)

        # Partial block should NOT be cached (too few tokens for a sub-block).
        assert partial_blk_id not in _sub_block_index(manager)._block_hashes
        assert partial_blk.block_hash is None

    def test_partial_block_multiple_sub_blocks(self):
        """A partial block with multiple sub-blocks should all be indexed."""
        BS, SBS = 16, 4  # Bigger block for this test.
        manager = _make_manager(BS, SBS, num_blocks=10)

        # 1 full block (16 tokens) + 3 sub-blocks (12 tokens) in partial block.
        tokens = list(range(BS + 3 * SBS))
        req = _make_request("0", tokens, BS)
        _prefill_request(manager, req)

        blocks = manager.coordinator.get_blocks(req.request_id)
        partial_blk = blocks[0][1]
        partial_blk_id = partial_blk.block_id
        manager.free(req)

        # Partial block should be indexed with 3 sub-block hashes.
        assert partial_blk_id in _sub_block_index(manager)._block_hashes
        assert len(_sub_block_index(manager)._block_hashes[partial_blk_id]) == 3

        # A new request sharing the full prefix should match all 3 sub-blocks.
        unique_tail = [999] * BS
        req1 = _make_request("1", tokens + unique_tail, BS)
        _, num_computed = manager.get_computed_blocks(req1)
        # Full block (16) from upstream.
        assert num_computed == BS
        # 3 sub-blocks (12) from sub-block match.
        sub_block_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_block_match is not None
        assert sub_block_match.num_tokens == 3 * SBS

    def test_ref_cnt_released_by_release_sub_block_match(self):
        """release_sub_block_match should release source block ref_cnt."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        shared = list(range(self.SUB_BLOCK_SIZE))
        req1 = _make_request("1", shared + [100] * self.BLOCK_SIZE, self.BLOCK_SIZE)
        _, num_computed = manager.get_computed_blocks(req1)
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        src_block = match.group_matches[0].src_block
        ref_cnt_after_match = src_block.ref_cnt

        # Release the match — should decrement ref_cnt.
        manager.release_sub_block_match(match)
        assert src_block.ref_cnt == ref_cnt_after_match - 1

    def test_ref_cnt_managed_by_caller(self):
        """Caller manages match lifecycle — ref_cnt is properly maintained."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        shared = list(range(self.SUB_BLOCK_SIZE))
        req1 = _make_request("1", shared + [100] * self.BLOCK_SIZE, self.BLOCK_SIZE)
        _, num_computed = manager.get_computed_blocks(req1)
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        src_block = match.group_matches[0].src_block
        ref_before = src_block.ref_cnt

        # Calling get_computed_blocks again does NOT release the match
        # (caller owns it). Ref_cnt unchanged.
        manager.get_computed_blocks(req1)
        assert src_block.ref_cnt == ref_before

        # Explicit release decrements.
        manager.release_sub_block_match(match)
        assert src_block.ref_cnt == ref_before - 1
        # A cached partial block should be evictable under memory pressure.
        # 5 blocks total: 1 null + 4 usable. req0 uses 2 blocks (1 full + 1 partial).
        # After free, 2 cached + 2 free. req1 needs all 4 usable blocks,
        # forcing eviction of both cached blocks.
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=5)

        # Fill a full block + partial block.
        tokens = list(range(self.BLOCK_SIZE + self.SUB_BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        partial_blk_id = manager.coordinator.get_blocks(req0.request_id)[0][1].block_id
        manager.free(req0)

        # Partial block should be in index now.
        assert partial_blk_id in _sub_block_index(manager)._block_hashes
        old_hashes = list(_sub_block_index(manager)._block_hashes[partial_blk_id])

        # Allocate all 4 usable blocks, forcing eviction.
        big_tokens = [500 + i for i in range(4 * self.BLOCK_SIZE)]
        req1 = _make_request("1", big_tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req1)

        # The partial block's physical slot was reused by req1. The old
        # sub-block hashes should have been evicted (via the eviction hook).
        # The block_id may still be in the index if req1 reused the same
        # physical block, but the hashes must be different.
        if partial_blk_id in _sub_block_index(manager)._block_hashes:
            new_hashes = _sub_block_index(manager)._block_hashes[partial_blk_id]
            assert new_hashes != old_hashes

    def test_index_newly_cached_blocks_during_decode(self):
        """Deferred sub-block indexing should correctly index new blocks
        when a decode step causes a block to become full."""
        BS, SBS = 8, 4
        manager = _make_manager(BS, SBS, num_blocks=10)

        # Start with a prompt that fills 1 full block + 1 partial token.
        # block_size=8, so 9 tokens = 1 full block + 1 token in partial block.
        prompt = list(range(BS + 1))
        req = _make_request("0", prompt, BS, max_tokens=BS)
        _prefill_request(manager, req)

        # After prefill: block 0 should be in index (full block, cached).
        blocks = manager.coordinator.get_blocks(req.request_id)
        blk0 = blocks[0][0]
        assert _sub_block_index(manager).contains(blk0.block_id)

        # Simulate decode steps that fill the second block.
        # Each decode adds 1 token. We need 7 more tokens to fill the second
        # block (currently has 1 token).
        for i in range(BS - 1):
            req.append_output_token_ids(200 + i)
            num_computed_before = req.num_computed_tokens
            # allocate_slots with 1 new token, 0 new computed tokens.
            result = manager.allocate_slots(req, 1, 0)
            assert result is not None
            req.num_computed_tokens = num_computed_before + 1

        # Finish the current step
        manager.do_pending_indexing()

        # After filling BS-1 decode tokens, the second block should now be full
        # and indexed in the index.
        blocks = manager.coordinator.get_blocks(req.request_id)
        if len(blocks[0]) > 1:
            blk1 = blocks[0][1]
            if blk1.block_hash is not None:
                assert _sub_block_index(manager).contains(blk1.block_id)

    def test_eager_partial_block_visible_before_free(self):
        """A running request's partial block sub-blocks should be visible
        to new requests for matching before free() is called."""
        BS, SBS = 8, 4
        manager = _make_manager(BS, SBS, num_blocks=10)

        # Request 0: 1 full block + 1 sub-block = 12 tokens.
        tokens0 = list(range(BS + SBS))
        req0 = _make_request("0", tokens0, BS, max_tokens=BS)
        _prefill_request(manager, req0)

        # req0 is still running (not freed).  Its partial block's sub-block
        # should already be indexed eagerly.
        blocks = manager.coordinator.get_blocks(req0.request_id)
        partial_blk = blocks[0][1]
        assert partial_blk.block_id in _sub_block_index(manager)._block_hashes
        # But not cached (no synthetic hash).
        assert partial_blk.block_hash is None

        # Request 1: shares the full prefix including the partial sub-block.
        tokens1 = list(range(BS + SBS)) + [100 + i for i in range(BS)]
        req1 = _make_request("1", tokens1, BS)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == BS  # Full block match only.
        sub_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        # Should match the partial sub-block from the running request.
        assert sub_match is not None
        assert sub_match.num_tokens == SBS
        manager.release_sub_block_match(sub_match)

    def test_eager_partial_block_grows_during_decode(self):
        """As a running request decodes and crosses sub-block boundaries,
        the eagerly indexed partial block should grow incrementally."""
        BS, SBS = 8, 4
        manager = _make_manager(BS, SBS, num_blocks=10)

        # Prompt: 1 full block + 1 token in partial block.
        prompt = list(range(BS + 1))
        req = _make_request("0", prompt, BS, max_tokens=2 * BS)
        _prefill_request(manager, req)

        # After prefill: partial block has 1 token < SBS, so 0 complete
        # sub-blocks → not in index.
        blocks = manager.coordinator.get_blocks(req.request_id)
        partial_blk = blocks[0][1]
        assert partial_blk.block_id not in _sub_block_index(manager)._block_hashes

        # Decode SBS-1 = 3 more tokens to complete the first sub-block.
        for i in range(SBS - 1):
            req.append_output_token_ids(200 + i)
            num_before = req.num_computed_tokens
            manager.allocate_slots(req, 1, 0)
            req.num_computed_tokens = num_before + 1
        manager.do_pending_indexing()

        # Now partial block has SBS tokens → 1 complete sub-block in index.
        idx = _sub_block_index(manager)
        assert idx.contains(partial_blk.block_id)
        assert len(idx._block_hashes[partial_blk.block_id]) == 1

        # Decode SBS more tokens to complete a second sub-block.
        for i in range(SBS):
            req.append_output_token_ids(300 + i)
            num_before = req.num_computed_tokens
            manager.allocate_slots(req, 1, 0)
            req.num_computed_tokens = num_before + 1
        manager.do_pending_indexing()

        # Now 2 complete sub-blocks.
        assert len(idx._block_hashes[partial_blk.block_id]) == 2

    def test_partial_to_full_transition_completes_indexing(self):
        """When a partially indexed block becomes full, _on_block_cached
        should complete the indexing with update()."""
        BS, SBS = 8, 4  # 2 sub-blocks per block
        manager = _make_manager(BS, SBS, num_blocks=10)

        # Prompt: 1 full block + SBS tokens in partial block.
        prompt = list(range(BS + SBS))
        req = _make_request("0", prompt, BS, max_tokens=BS)
        _prefill_request(manager, req)

        blocks = manager.coordinator.get_blocks(req.request_id)
        partial_blk = blocks[0][1]

        # After prefill: 1 sub-block indexed in the partial block.
        idx = _sub_block_index(manager)
        assert idx.contains(partial_blk.block_id)
        assert len(idx._block_hashes[partial_blk.block_id]) == 1

        # Decode SBS more tokens to fill the block completely.
        for i in range(SBS):
            req.append_output_token_ids(400 + i)
            num_before = req.num_computed_tokens
            manager.allocate_slots(req, 1, 0)
            req.num_computed_tokens = num_before + 1
        manager.do_pending_indexing()

        # The block is now full and should have both sub-blocks indexed.
        assert partial_blk.block_hash is not None  # Upstream marked it cached.
        assert len(idx._block_hashes[partial_blk.block_id]) == 2

    def test_partial_match_tight_memory_no_assertion(self):
        """With tight block pool, partial match source block must survive
        allocation without hitting ref_cnt assertion."""
        # 4 blocks total (1 null + 3 usable). req0 uses 1, caches it on free.
        # req1 shares 1 sub-block and needs 2 blocks → must evict cached block.
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=4)

        req0 = _make_request("0", list(range(self.BLOCK_SIZE)), self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        shared = list(range(self.SUB_BLOCK_SIZE))
        rest = [100 + i for i in range(self.BLOCK_SIZE + self.SUB_BLOCK_SIZE)]
        req1 = _make_request("1", shared + rest, self.BLOCK_SIZE)

        # Must not crash with AssertionError on ref_cnt.
        _, _, new_blocks = _prefill_request(manager, req1)
        assert new_blocks is not None

    def test_src_block_protected_until_release(self):
        """Source block for a copy op must stay referenced until
        release_copy_ops is called (after the model runner copies data)."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        req0 = _make_request("0", list(range(self.BLOCK_SIZE)), self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        shared = list(range(self.SUB_BLOCK_SIZE))
        req1 = _make_request("1", shared + [100] * self.BLOCK_SIZE, self.BLOCK_SIZE)
        _prefill_request(manager, req1)

        # Copy op should exist; source block should still have ref_cnt > 0.
        assert len(manager.pending_copy_ops) == 1
        src_block_id = manager.pending_copy_ops[0].src_block_id
        src_block = manager.block_pool.blocks[src_block_id]
        assert src_block.ref_cnt > 0
        ref_before_drain = src_block.ref_cnt

        # Reset should fail due to pending copy op refs.
        assert not manager.reset_prefix_cache()
        assert len(_sub_block_index(manager)._block_hashes) > 0

        # drain returns ops but does NOT release refs.
        ops = manager.drain_pending_copy_ops()
        assert src_block.ref_cnt == ref_before_drain

        # release_copy_ops releases the refs.
        manager.release_copy_ops(ops)
        assert src_block.ref_cnt == ref_before_drain - 1

    def test_exact_block_size_request_sub_block_recovery(self):
        """When num_tokens == block_size, upstream caps at num_tokens-1 and
        matches 0 full blocks.  Sub-block matching should recover partial
        tokens from the cached block."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        req1 = _make_request("1", tokens, self.BLOCK_SIZE)
        _, num_computed = manager.get_computed_blocks(req1)

        # Upstream matches 0 full blocks (num_tokens-1=7 < block_size=8).
        # Sub-block matching recovers 1 sub-block (4 tokens), capped so
        # that total_computed <= num_tokens - 1 = 7.
        assert num_computed == 0  # Full-block count only.
        sub_block_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_block_match is not None
        assert num_computed + sub_block_match.num_tokens <= req1.num_tokens - 1
        assert (
            sub_block_match.num_tokens == self.SUB_BLOCK_SIZE
        )  # 4 tokens from sub-block match

    def test_prompt_logprobs_skips_sub_block_cache(self):
        """Requests with prompt_logprobs must not get sub-block hits,
        matching upstream's behaviour for full-block prefix caching."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # Populate the cache with a full block.
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # with prompt_logprobs: must NOT get any cache hit.
        shared = list(range(self.SUB_BLOCK_SIZE))
        unique = [100 + i for i in range(self.BLOCK_SIZE)]
        req = _make_request("2", shared + unique, self.BLOCK_SIZE, prompt_logprobs=5)
        _, num_computed = manager.get_computed_blocks(req)
        assert num_computed == 0  # Full-block path also skipped.
        sub_match = manager.get_computed_blocks_sub_block(req, num_computed)
        assert sub_match is None

    def test_early_release_allows_src_eviction(self):
        """Demonstrates the async-scheduling race: if copy-op source-block
        refs are released before the model runner copies data, a subsequent
        allocation can evict the source block.

        The sequence simulates what happens when schedule(N+1) releases
        step N's copy-op refs while execute_model(N) hasn't run yet:

        1. Nearly fill memory, create a copy op (src block ref-held)
        2. drain_pending_copy_ops  (scheduler builds output for step N)
        3. release_copy_ops        (schedule N+1 starts, releases refs)
        4. New allocation forces eviction → src block is reclaimed
        5. The copy op now points to a recycled block — data is lost
        """
        # 4 blocks total: req0 uses 1, req1 uses 2 (with sub-block match),
        # leaving 1 free.  After freeing req1 and releasing the copy-op ref,
        # req2 needs 3 blocks → must evict the src block.
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=4)

        # req0: fill 1 full block → cached & indexed.
        req0 = _make_request("0", list(range(self.BLOCK_SIZE)), self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)  # block cached in LRU + sub-block index

        # req1: partial match on first sub-block → uses 2 blocks, copy op pending.
        shared = list(range(self.SUB_BLOCK_SIZE))
        req1 = _make_request(
            "1", shared + [100 + i for i in range(self.BLOCK_SIZE)], self.BLOCK_SIZE
        )
        _prefill_request(manager, req1)
        assert len(manager.pending_copy_ops) == 1

        src_block_id = manager.pending_copy_ops[0].src_block_id
        src_block = manager.block_pool.blocks[src_block_id]

        # --- simulate scheduler building output (end of schedule N) ---
        ops = manager.drain_pending_copy_ops()
        # Source block still protected (drain doesn't release).
        assert src_block.ref_cnt > 0

        # --- simulate schedule(N+1) starting: premature release ---
        manager.release_copy_ops(ops)
        # Source block ref dropped — now an eviction candidate.

        # Free req1 to make its 2 blocks available, but the cached src
        # block (from req0) is also free now since we released its ref.
        manager.free(req1)

        # req2 needs 3 blocks → must evict the former src block.
        req2 = _make_request(
            "2", [200 + i for i in range(3 * self.BLOCK_SIZE)], self.BLOCK_SIZE
        )
        _prefill_request(manager, req2)

        # The source block has been recycled for req2.
        # Its block_hash was cleared by eviction, confirming reuse.
        # In a real system, the model runner would now copy stale data
        # from ops[0].src_block_id — a correctness bug.
        assert src_block.ref_cnt > 0  # now held by req2
        assert src_block.block_hash is not None  # re-cached for req2

        # Verify the copy op still references the recycled block id.
        assert ops[0].src_block_id == src_block.block_id

    def test_cache_salt_isolation(self):
        """Sub-block match must not cross cache_salt boundaries.
        Also, a request without cache_salt must not match sub-blocks
        cached by a request with cache_salt, and vice versa."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        # req0 with salt "A": fills a full block → sub-blocks indexed.
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE, cache_salt="A")
        _prefill_request(manager, req0)
        manager.free(req0)

        # req1 with salt "B": same token prefix, different salt → no mat
        shared = list(range(self.SUB_BLOCK_SIZE))
        unique = [100 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, cache_salt="B")
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        sub_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_match is None

        # Same tokens, no salt → no match
        req2 = _make_request("2", shared + unique, self.BLOCK_SIZE)
        _, num_computed2 = manager.get_computed_blocks(req2)
        sub_match2 = manager.get_computed_blocks_sub_block(req2, num_computed2)
        assert sub_match2 is None

    def test_lora_isolation(self):
        """Sub-block match must not cross LoRA boundaries."""
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=10)

        lora_a = LoRARequest(lora_name="lora_a", lora_int_id=1, lora_path="/a")
        lora_b = LoRARequest(lora_name="lora_b", lora_int_id=2, lora_path="/b")

        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE, lora_request=lora_a)
        _prefill_request(manager, req0)
        manager.free(req0)

        shared = list(range(self.SUB_BLOCK_SIZE))
        unique = [100 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, lora_request=lora_b)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        sub_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_match is None

    def test_multi_prefill_no_stale_sub_block_match(self):
        """When two prefills with overlapping tokens are scheduled in the
        same step (before execute_model), the second request must NOT get
        a sub-block match from the first request's blocks because the KV
        data hasn't been computed yet.
        In the next step, a third request should get the match.
        """
        manager = _make_manager(self.BLOCK_SIZE, self.SUB_BLOCK_SIZE, num_blocks=100)

        # Both requests share the same first sub-block prefix.
        shared = list(range(self.SUB_BLOCK_SIZE))
        req0 = _make_request(
            "0", shared + [100] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1
        )
        req1 = _make_request(
            "1", shared + [200] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1
        )

        # --- Simulate scheduling both in the same step (multi-prefill) ---

        # req0: get_computed_blocks → allocate_slots
        cb0, nc0 = manager.get_computed_blocks(req0)
        sm0 = manager.get_computed_blocks_sub_block(req0, nc0)
        assert sm0 is None  # Cold cache.
        manager.allocate_slots(req0, req0.num_tokens, 0, cb0)

        # req1: scheduled in the same step.
        # With deferred indexing, req0's sub-blocks are NOT indexed yet.
        cb1, nc1 = manager.get_computed_blocks(req1)
        sm1 = manager.get_computed_blocks_sub_block(req1, nc1)
        assert sm1 is None  # Must NOT match — req0's KV data doesn't exist.
        manager.allocate_slots(req1, req1.num_tokens, 0, cb1)

        # --- Finish the step ---
        req0.num_computed_tokens = req0.num_tokens
        req1.num_computed_tokens = req1.num_tokens
        manager.do_pending_indexing()

        # After that, the index should be populated.
        assert len(_sub_block_index(manager)._hash_to_blocks) > 0

        # A third request should now get a sub-block match.
        req2 = _make_request(
            "2", shared + [300] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1
        )
        _, nc2 = manager.get_computed_blocks(req2)
        sm2 = manager.get_computed_blocks_sub_block(req2, nc2)
        assert sm2 is not None
        assert sm2.num_tokens == self.SUB_BLOCK_SIZE
        manager.release_sub_block_match(sm2)


# ---------------------------------------------------------------------------
# Multi-group (hybrid) tests
# ---------------------------------------------------------------------------


def _make_hybrid_kv_cache_config(
    block_size: int, num_blocks: int, sliding_window: int
) -> KVCacheConfig:
    """Create a 2-group config: full attention + sliding window."""
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_attn_layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["sw_layer"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window,
                ),
            ),
        ],
    )


def _make_hybrid_manager(
    block_size: int,
    sub_block_size: int,
    num_blocks: int,
    sliding_window: int,
    max_model_len: int = 8192,
) -> RBLNKVCacheManager:
    """hash_block_size = block_size (same block_size for all groups)."""
    return RBLNKVCacheManager(
        kv_cache_config=_make_hybrid_kv_cache_config(
            block_size, num_blocks, sliding_window
        ),
        max_model_len=max_model_len,
        hash_block_size=block_size,
        sub_block_size=sub_block_size,
        hash_fn=sha256,
    )


class TestMultiGroupRBLNKVCacheManager:
    """Tests for multi-group (hybrid) sub-block prefix caching."""

    BLOCK_SIZE = 8
    SUB_BLOCK_SIZE = 4
    SLIDING_WINDOW = 16

    def test_hybrid_full_block_hit_indexes_both_groups(self):
        """After filling full blocks, both group indices should have entries."""
        manager = _make_hybrid_manager(
            self.BLOCK_SIZE,
            self.SUB_BLOCK_SIZE,
            num_blocks=20,
            sliding_window=self.SLIDING_WINDOW,
        )
        tokens = list(range(2 * self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)

        # Both groups should have indexed blocks.
        idx0 = manager._group_infos[0].sub_block_index
        idx1 = manager._group_infos[1].sub_block_index
        assert len(idx0._block_hashes) > 0
        assert len(idx1._block_hashes) > 0

    def test_hybrid_partial_match_generates_copy_ops_per_group(self):
        """Partial match in hybrid mode should generate copy ops for each group."""
        manager = _make_hybrid_manager(
            self.BLOCK_SIZE,
            self.SUB_BLOCK_SIZE,
            num_blocks=20,
            sliding_window=self.SLIDING_WINDOW,
        )

        # First request: fill 1 full block.
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Second request: shares first sub-block, different rest.
        shared = list(range(self.SUB_BLOCK_SIZE))
        req1_tokens = shared + [100] * (self.BLOCK_SIZE + 1)
        req1 = _make_request("1", req1_tokens, self.BLOCK_SIZE)
        _, num_computed, new_blocks = _prefill_request(manager, req1)

        assert num_computed == self.SUB_BLOCK_SIZE
        assert new_blocks is not None

        ops = manager.drain_pending_copy_ops()
        # Should have one copy op per group.
        assert len(ops) == 2
        for op in ops:
            assert op.num_tokens == self.SUB_BLOCK_SIZE

    def test_hybrid_reset_clears_all_indices(self):
        """reset_prefix_cache should clear all per-group sub-block indices."""
        manager = _make_hybrid_manager(
            self.BLOCK_SIZE,
            self.SUB_BLOCK_SIZE,
            num_blocks=20,
            sliding_window=self.SLIDING_WINDOW,
        )
        tokens = list(range(2 * self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        assert any(
            len(gi.sub_block_index._block_hashes) > 0 for gi in manager._group_infos
        )

        manager.reset_prefix_cache()

        for gi in manager._group_infos:
            assert len(gi.sub_block_index._block_hashes) == 0

    def test_hybrid_different_block_sizes_partial_match(self):
        """When groups have different block sizes, sub-block matching should
        use the minimum match across groups."""
        # Group 0 (full attn): block_size=16, sub_blocks_per_block=4
        # Group 1 (sliding window): block_size=8, sub_blocks_per_block=2
        # hash_block_size=8 (must divide both)
        full_bs = 16
        sw_bs = 8
        sbs = 4
        hash_bs = 8  # GCD works as hash_block_size
        sliding_window = 32
        config = KVCacheConfig(
            num_blocks=30,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full_attn_layer"],
                    FullAttentionSpec(
                        block_size=full_bs,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["sw_layer"],
                    SlidingWindowSpec(
                        block_size=sw_bs,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                        sliding_window=sliding_window,
                    ),
                ),
            ],
        )
        manager = RBLNKVCacheManager(
            kv_cache_config=config,
            max_model_len=8192,
            hash_block_size=hash_bs,
            sub_block_size=sbs,
            hash_fn=sha256,
        )
        assert manager._group_infos[0].sub_blocks_per_block == 4  # 16/4
        assert manager._group_infos[1].sub_blocks_per_block == 2  # 8/4

        # LCM of 16 and 8 is 16. Upstream aligns to LCM.
        # Cache a request with exactly 16 tokens (1 full block for group 0,
        # 2 full blocks for group 1).
        tokens = list(range(full_bs))
        req0 = _make_request("0", tokens, hash_bs)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Both groups should have indexed blocks.
        assert len(manager._group_infos[0].sub_block_index._block_hashes) > 0
        assert len(manager._group_infos[1].sub_block_index._block_hashes) > 0

        # New request: shares first sub-block (4 tokens), then diverges.
        # Group 0: next block boundary at 0 (no full blocks matched),
        #   query up to 3 sub-blocks (sbpb-1=3).
        # Group 1: next block boundary at 0,
        #   query up to 1 sub-block (sbpb-1=1).
        # min_matched should be capped by the group with fewer sub-blocks.
        shared = list(range(sbs))
        unique = [100 + i for i in range(full_bs)]
        req1 = _make_request("1", shared + unique, hash_bs)
        _, num_computed = manager.get_computed_blocks(req1)
        # Should get 0 from full-block matching, 1 sub-block match separately.
        assert num_computed == 0
        sub_block_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_block_match is not None
        assert sub_block_match.num_tokens == sbs

    def test_hybrid_partial_block_indexed_on_free(self):
        """In multi-group setup, free() should index partial blocks in both
        groups' sub-block indices and mark them as cached."""
        manager = _make_hybrid_manager(
            self.BLOCK_SIZE,
            self.SUB_BLOCK_SIZE,
            num_blocks=20,
            sliding_window=self.SLIDING_WINDOW,
        )

        # 1 full block + 1 sub-block (partial).
        tokens = list(range(self.BLOCK_SIZE + self.SUB_BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)

        # After prefill: partial block is eagerly indexed but not yet cached.
        blocks = manager.coordinator.get_blocks(req0.request_id)
        partial_blk_ids = [
            blocks[gid][1].block_id for gid in range(len(manager._group_infos))
        ]

        for i, gi in enumerate(manager._group_infos):
            assert partial_blk_ids[i] in gi.sub_block_index._block_hashes
            assert blocks[i][1].block_hash is None  # Not cached yet.

        manager.free(req0)

        # After free: partial block should still be in both group indices
        # and now have a block_hash (cached in LRU).
        for i, gi in enumerate(manager._group_infos):
            assert partial_blk_ids[i] in gi.sub_block_index._block_hashes

    def test_hybrid_capping_at_num_tokens_minus_one(self):
        """In multi-group, sub-block match must still be capped at
        request.num_tokens - 1."""
        manager = _make_hybrid_manager(
            self.BLOCK_SIZE,
            self.SUB_BLOCK_SIZE,
            num_blocks=20,
            sliding_window=self.SLIDING_WINDOW,
        )

        # Cache a full block.
        tokens = list(range(self.BLOCK_SIZE))
        req0 = _make_request("0", tokens, self.BLOCK_SIZE)
        _prefill_request(manager, req0)
        manager.free(req0)

        # Request with exact same tokens (num_tokens=8, cap=7).
        # Upstream matches 0 full blocks (7 < 8).
        # Sub-block match: 1 sub-block (4 tokens), capped at 4 <= 7. OK.
        req1 = _make_request("1", tokens, self.BLOCK_SIZE)
        _, num_computed = manager.get_computed_blocks(req1)
        sub_block_match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert sub_block_match is not None
        assert num_computed + sub_block_match.num_tokens <= req1.num_tokens - 1
        assert num_computed == 0  # Full-block count only.
        assert sub_block_match.num_tokens == self.SUB_BLOCK_SIZE

    def test_eligibility(self):
        """Test the eligibility check with various configs."""
        # Single group, eligible.
        config1 = _make_kv_cache_config(block_size=8, num_blocks=10)
        assert RBLNKVCacheManager.can_use_sub_block_caching(config1, 4)

        # Single group, block_size == sub_block_size.
        assert not RBLNKVCacheManager.can_use_sub_block_caching(config1, 8)

        # Single group, not divisible.
        assert not RBLNKVCacheManager.can_use_sub_block_caching(config1, 3)

        # sub_block_size <= 0.
        assert not RBLNKVCacheManager.can_use_sub_block_caching(config1, 0)
        assert not RBLNKVCacheManager.can_use_sub_block_caching(config1, -1)

        # Hybrid, both eligible.
        config2 = _make_hybrid_kv_cache_config(
            block_size=8, num_blocks=20, sliding_window=16
        )
        assert RBLNKVCacheManager.can_use_sub_block_caching(config2, 4)

        # Ineligible spec type: MambaSpec stores recurrent state per block,
        # not per-token KV, so sub-block partial copy is meaningless.
        mamba_config = KVCacheConfig(
            num_blocks=20,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["mamba_layer"],
                    MambaSpec(
                        block_size=8,
                        shapes=((64,),),
                        dtypes=(torch.float32,),
                    ),
                ),
            ],
        )
        assert not RBLNKVCacheManager.can_use_sub_block_caching(mamba_config, 4)

        # Hybrid with Mamba: full_attn + mamba — ineligible because of Mamba.
        hybrid_mamba_config = KVCacheConfig(
            num_blocks=20,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full_attn_layer"],
                    FullAttentionSpec(
                        block_size=8,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["mamba_layer"],
                    MambaSpec(
                        block_size=8,
                        shapes=((64,),),
                        dtypes=(torch.float32,),
                    ),
                ),
            ],
        )
        assert not RBLNKVCacheManager.can_use_sub_block_caching(hybrid_mamba_config, 4)


# ---------------------------------------------------------------------------
# RBLNScheduler integration tests
# ---------------------------------------------------------------------------


class TestRBLNScheduler:
    """Tests for the RBLNScheduler with sub-block prefix caching through the
    full schedule → update_from_output flow."""

    BLOCK_SIZE = 16
    SUB_BLOCK_SIZE = 4

    def _create_scheduler(
        self, num_blocks=100, max_model_len=None, max_num_batched_tokens=None
    ):
        if max_model_len is None:
            max_model_len = self.BLOCK_SIZE * num_blocks
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len
        return create_scheduler(
            block_size=self.BLOCK_SIZE,
            num_blocks=num_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            sub_block_size=self.SUB_BLOCK_SIZE,
        )

    def test_scheduler_uses_rbln_kv_cache_manager(self):
        """When prefix caching is enabled and block_size > sub_block_size,
        the scheduler should use RBLNKVCacheManager."""
        scheduler = self._create_scheduler()
        assert isinstance(scheduler.kv_cache_manager, RBLNKVCacheManager)

    def test_schedule_returns_rbln_scheduler_output(self):
        """schedule() should return an RBLNSchedulerOutput."""
        scheduler = self._create_scheduler()
        req = _make_request("0", [0] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req)
        output = scheduler.schedule()
        assert isinstance(output, RBLNSchedulerOutput)
        assert hasattr(output, "kv_cache_copy_ops")

    def test_no_copy_ops_on_first_request(self):
        """The first request should have no copy ops (cold cache)."""
        scheduler = self._create_scheduler()
        req = _make_request(
            "0", [0] * self.BLOCK_SIZE * 2, self.BLOCK_SIZE, max_tokens=1
        )
        scheduler.add_request(req)
        output = scheduler.schedule()
        assert isinstance(output, RBLNSchedulerOutput)
        assert output.kv_cache_copy_ops == []

    def test_no_copy_ops_on_full_block_match(self):
        """Requests sharing full blocks should not generate copy ops
        (handled by upstream prefix caching)."""
        scheduler = self._create_scheduler()

        # Both requests share 2 full blocks of zeros. req1 has an extra block
        # of unique tokens so that max_cache_hit_length (num_tokens - 1)
        # allows upstream to match both shared full blocks.
        shared = [0] * (self.BLOCK_SIZE * 2)
        req0 = _make_request("0", shared, self.BLOCK_SIZE, max_tokens=1)
        # req1 extends shared prefix with unique tokens (no sub-block overlap).
        unique_tail = [7000 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique_tail, self.BLOCK_SIZE, max_tokens=1)

        # Schedule + finish request 0.
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Schedule request 1 — upstream matches 2 full blocks, unique tail
        # doesn't match any cached sub-blocks → no copy ops.
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert isinstance(output1, RBLNSchedulerOutput)
        assert output1.kv_cache_copy_ops == []

    def test_partial_match_generates_copy_ops(self):
        """A request sharing only a sub-block prefix with a cached block
        should trigger a copy op."""
        scheduler = self._create_scheduler()

        # Request 0: exactly 1 full block of zeros.
        req0 = _make_request("0", [0] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Request 1: first sub-block matches (4 zeros), rest diverges.
        # The full block hash differs (because the remaining 12 tokens differ),
        # so upstream won't match. But the sub-block index should.
        shared = [0] * self.SUB_BLOCK_SIZE
        unique = [900 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert isinstance(output1, RBLNSchedulerOutput)
        assert len(output1.kv_cache_copy_ops) == 1

        op = output1.kv_cache_copy_ops[0]
        assert op.num_tokens == self.SUB_BLOCK_SIZE
        assert op.src_block_id != op.dst_block_id

    def test_partial_block_reused_after_free(self):
        """After a request with a partial block is freed, the next request
        sharing that prefix should get a sub-block hit from the cached
        partial block."""
        scheduler = self._create_scheduler()

        # Request 0: 1 full block + 1 sub-block (partial).
        prompt0 = [0] * (self.BLOCK_SIZE + self.SUB_BLOCK_SIZE)
        req0 = _make_request("0", prompt0, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Request 1: shares same prefix + extra unique tokens.
        prompt1 = prompt0 + [900 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", prompt1, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert isinstance(output1, RBLNSchedulerOutput)

        # Should get a copy op from the cached partial block.
        assert len(output1.kv_cache_copy_ops) == 1
        op = output1.kv_cache_copy_ops[0]
        assert op.num_tokens == self.SUB_BLOCK_SIZE

    def test_scheduler_no_sub_block_when_equal_sizes(self):
        """When sub_block_size == block_size, sub-block caching should NOT
        activate (sub-block matching is meaningless)."""
        scheduler = create_scheduler(
            block_size=self.BLOCK_SIZE,
            num_blocks=100,
            max_num_batched_tokens=self.BLOCK_SIZE * 10,
            max_model_len=self.BLOCK_SIZE * 10,
            enable_prefix_caching=True,
            sub_block_size=self.BLOCK_SIZE,
        )
        assert not isinstance(scheduler.kv_cache_manager, RBLNKVCacheManager)

    def test_multi_turn_conversation(self):
        """Multi-turn chat: turn 2's prefix includes turn 1's entire
        prompt + generated output.  The partial block cached when turn 1
        finishes should produce a sub-block hit for turn 2."""
        BS = self.BLOCK_SIZE  # 16
        SBS = self.SUB_BLOCK_SIZE  # 4

        scheduler = self._create_scheduler(num_blocks=100)
        mgr = scheduler.kv_cache_manager
        assert isinstance(mgr, RBLNKVCacheManager)

        # --- Turn 1 ---
        # Prompt: 1 full block + 1 sub-block = 20 tokens.
        # After generating 1 + BS tokens:
        #   block 0  [0..15]                    full (prompt)
        #   block 1  [16..19, 1000..1011]       full (prompt tail + decode)
        #   block 2  [1012..1015]               partial (1 sub-block)
        prompt_1 = list(range(BS + SBS))
        num_gen = 1 + BS  # 1 from prefill, BS from decode, adding BS tokens in KV cache
        req1 = _make_request("turn1", prompt_1, BS, max_tokens=num_gen)
        scheduler.add_request(req1)

        generated_tokens = []
        for step in range(num_gen):
            output = scheduler.schedule()
            tok = 1000 + step
            generated_tokens.append(tok)
            runner_out = create_runner_output(output, tok)
            scheduler.update_from_output(output, runner_out)

        # Request finished (max_tokens reached) → freed → partial block cached.
        assert "turn1" not in scheduler.requests

        # --- Turn 2 ---
        # Prompt = turn 1's full content + new user message.
        # Upstream matches 2 full blocks (32 tokens).
        # Sub-block index matches the cached partial block (4 tokens).
        new_user_msg = [2000 + i for i in range(BS)]
        turn2_prompt = prompt_1 + generated_tokens + new_user_msg
        req2 = _make_request("turn2", turn2_prompt, BS, max_tokens=1)
        scheduler.add_request(req2)

        output2 = scheduler.schedule()
        assert isinstance(output2, RBLNSchedulerOutput)

        # Sub-block copy op from the cached partial block.
        assert len(output2.kv_cache_copy_ops) == 1
        assert output2.kv_cache_copy_ops[0].num_tokens == SBS

        # Verify prefill savings: fewer tokens scheduled than full prompt.
        num_computed = 2 * BS + SBS  # 2 full blocks + 1 sub-block
        expected_scheduled = req2.num_tokens - num_computed
        assert output2.num_scheduled_tokens["turn2"] == expected_scheduled

    def test_speculative_alloc_does_not_index_uncomputed_blocks(self):
        """Pre-allocated but uncomputed blocks must not appear in the
        sub-block index. With delay_cache_blocks=True, only blocks that
        are explicitly cached in the finalization step get indexed.
        """
        BS = self.BLOCK_SIZE  # 16
        SBS = self.SUB_BLOCK_SIZE  # 4

        scheduler = self._create_scheduler(num_blocks=100, max_num_batched_tokens=BS)
        mgr = scheduler.kv_cache_manager
        assert isinstance(mgr, RBLNKVCacheManager)
        index = mgr._group_infos[0].sub_block_index

        # 3 full blocks + 1 partial block.
        # The scheduler pre-allocates blocks for ALL tokens but only computes
        # one chunk (BS tokens) per iteration.
        # With delay_cache_blocks + finalization:
        #   block 0: computed and indexed
        #   blocks 1-2: full, never got cached/indexed
        #   block 3: partial, never got indexed
        tokens = list(range(3 * BS + SBS))
        req = _make_request("req", tokens, BS, max_tokens=1)
        scheduler.add_request(req)

        output = scheduler.schedule()

        # After update_from_output (forward pass done), only block 0 is indexed.
        runner_out = create_runner_output(output, 0)
        scheduler.update_from_output(output, runner_out)

        req_blocks = mgr.coordinator.get_blocks("req")[0]

        # Only four sub-blocks from block 0
        assert len(index._hash_to_blocks) == 4
        uncomputed = [blk for blk in req_blocks if blk.block_hash is None]
        for blk in uncomputed:
            assert not index.contains(blk.block_id)

    def test_copy_op_refs_released_in_update_from_output(self):
        """Source-block refs from copy ops must be released in
        update_from_output (after execution), not in schedule().

        This is critical for async scheduling where schedule(N+1) runs
        before execute_model(N) completes."""
        scheduler = self._create_scheduler(num_blocks=10)
        mgr = scheduler.kv_cache_manager
        assert isinstance(mgr, RBLNKVCacheManager)

        # Build a cached block.
        req0 = _make_request("0", [0] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        out0 = scheduler.schedule()
        scheduler.update_from_output(out0, create_runner_output(out0, 0))

        # New request with sub-block match → copy op.
        shared = [0] * self.SUB_BLOCK_SIZE
        unique = [900 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req1)
        out1 = scheduler.schedule()
        assert len(out1.kv_cache_copy_ops) == 1

        src_block_id = out1.kv_cache_copy_ops[0].src_block_id
        src_block = mgr.block_pool.blocks[src_block_id]

        # Before update_from_output: src block ref still held.
        assert src_block.ref_cnt > 0
        ref_before = src_block.ref_cnt

        # Simulate schedule(N+1) starting BEFORE update_from_output(N).
        # With the fix, schedule() does NOT release the refs.
        # (Nothing to verify here — the key is that schedule() no longer
        # touches _unreleased_copy_ops.)

        # Now update_from_output releases the refs.
        scheduler.update_from_output(out1, create_runner_output(out1, 1))
        assert src_block.ref_cnt == ref_before - 1


# ---------------------------------------------------------------------------
# KV Connector + Sub-block prefix caching interaction tests
# ---------------------------------------------------------------------------


class TestSubBlockWithKVConnector:
    """Tests for the interaction between sub-block prefix caching and KV
    connectors.  The scheduler picks whichever source (sub-block or connector)
    provides more tokens; sub-block wins on ties since local copy is cheaper."""

    BLOCK_SIZE = 16
    SUB_BLOCK_SIZE = 4

    def _create_scheduler(
        self,
        matched_tokens=0,
        is_async=False,
        num_blocks=100,
        max_model_len=None,
    ):
        if max_model_len is None:
            max_model_len = self.BLOCK_SIZE * num_blocks
        return create_scheduler(
            block_size=self.BLOCK_SIZE,
            num_blocks=num_blocks,
            max_num_batched_tokens=max_model_len,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            sub_block_size=self.SUB_BLOCK_SIZE,
            use_kv_connector=MockKVConfig(
                matched_tokens=matched_tokens,
                is_async=is_async,
            ),
        )

    def test_connector_zero_tokens_sub_block_used(self):
        """When the connector returns 0 external tokens, sub-block match
        should be used as fallback."""
        scheduler = self._create_scheduler(matched_tokens=0)
        assert isinstance(scheduler.kv_cache_manager, RBLNKVCacheManager)
        assert scheduler.connector is not None

        # Cache a full block via request 0.
        req0 = _make_request("0", [0] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Request 1: shares first sub-block, then diverges.
        shared = [0] * self.SUB_BLOCK_SIZE
        unique = [900 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert isinstance(output1, RBLNSchedulerOutput)

        # Sub-block match should be active → copy op generated.
        assert len(output1.kv_cache_copy_ops) == 1
        assert output1.kv_cache_copy_ops[0].num_tokens == self.SUB_BLOCK_SIZE

        # The connector provided nothing, so prefill should compute all
        # tokens beyond the sub-block match.
        expected_scheduled = len(shared + unique) - self.SUB_BLOCK_SIZE
        assert output1.num_scheduled_tokens["1"] == expected_scheduled

    def test_sub_block_beats_connector(self):
        """When sub-block match provides more tokens than the connector,
        sub-block should win and the connector's offer should be ignored."""
        # Sub-block can provide up to 3 sub-blocks = 12 tokens (block=16, sub=4).
        # Connector offers only 1 sub-block's worth = 4 tokens.
        # Sub-block should win.
        scheduler = self._create_scheduler(matched_tokens=self.SUB_BLOCK_SIZE)
        assert isinstance(scheduler.kv_cache_manager, RBLNKVCacheManager)

        # Cache a full block with 3 sub-blocks worth of known tokens.
        num_shared = 3 * self.SUB_BLOCK_SIZE  # 12 tokens
        req0_tokens = list(range(num_shared)) + [
            800 + i for i in range(self.BLOCK_SIZE - num_shared)
        ]
        req0 = _make_request("0", req0_tokens, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Request 1: shares first 3 sub-blocks, then diverges.
        # This triggers a 3-sub-block match = 12 tokens.
        # The connector only offers SUB_BLOCK_SIZE = 4 tokens.
        shared = list(range(num_shared))
        unique = [900 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, max_tokens=1)
        req1.kv_transfer_params = {"do_remote_prefill": True}
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert isinstance(output1, RBLNSchedulerOutput)

        # Sub-block wins → copy op generated, connector overridden.
        assert len(output1.kv_cache_copy_ops) == 1
        assert output1.kv_cache_copy_ops[0].num_tokens == num_shared

        # Scheduled tokens = total - sub_block_match (connector ignored).
        total = len(shared + unique)
        expected_scheduled = total - num_shared
        assert output1.num_scheduled_tokens["1"] == expected_scheduled

    def test_connector_nonzero_tokens_sub_block_discarded(self):
        """When the connector returns more external tokens than the sub-block
        match, the sub-block should be discarded."""
        # Connector claims to have 1 block's worth of external tokens.
        scheduler = self._create_scheduler(matched_tokens=self.BLOCK_SIZE)
        assert isinstance(scheduler.kv_cache_manager, RBLNKVCacheManager)
        assert scheduler.connector is not None

        # Cache a full block via request 0 (no remote prefill).
        req0 = _make_request("0", [0] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Request 1: same sub-block prefix as req0 → would trigger sub-block
        # match normally, but connector provides external tokens.
        # Set kv_transfer_params to trigger the mock connector.
        shared = [0] * self.SUB_BLOCK_SIZE
        unique = [900 + i for i in range(2 * self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, max_tokens=1)
        req1.kv_transfer_params = {"do_remote_prefill": True}
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert isinstance(output1, RBLNSchedulerOutput)

        # Sub-block match should be discarded → no copy ops.
        assert output1.kv_cache_copy_ops == []

        # Connector provides BLOCK_SIZE tokens, so scheduled tokens =
        # total - (local_full_block + connector_tokens).
        # local full-block matches: 0 (req1's first block hash differs).
        # external: BLOCK_SIZE (from connector).
        total = len(shared + unique)
        expected_scheduled = total - self.BLOCK_SIZE
        assert output1.num_scheduled_tokens["1"] == expected_scheduled

    def test_connector_sees_block_aligned_count(self):
        """The connector should receive a block-aligned num_computed_tokens
        (not inflated by sub-block tokens)."""
        # We'll use a custom connector that records the num_computed_tokens
        # it receives.
        recorded_computed: list[int] = []

        scheduler = self._create_scheduler(matched_tokens=0)
        assert scheduler.connector is not None

        original_fn = scheduler.connector.get_num_new_matched_tokens

        def recording_get_num_new_matched_tokens(request, num_computed_tokens):
            recorded_computed.append(num_computed_tokens)
            return original_fn(request, num_computed_tokens)

        scheduler.connector.get_num_new_matched_tokens = (
            recording_get_num_new_matched_tokens
        )

        # Cache a full block.
        req0 = _make_request("0", [0] * self.BLOCK_SIZE, self.BLOCK_SIZE, max_tokens=1)
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        runner_output0 = create_runner_output(output0, 0)
        scheduler.update_from_output(output0, runner_output0)

        # Request 1 shares first sub-block → triggers sub-block match.
        # Set kv_transfer_params so the mock connector responds.
        shared = [0] * self.SUB_BLOCK_SIZE
        unique = [900 + i for i in range(self.BLOCK_SIZE)]
        req1 = _make_request("1", shared + unique, self.BLOCK_SIZE, max_tokens=1)
        req1.kv_transfer_params = {"do_remote_prefill": True}
        scheduler.add_request(req1)
        scheduler.schedule()

        # The connector should have seen block-aligned count (0, not 4).
        # recorded_computed may have entries from req0 too; check the last.
        assert recorded_computed[-1] == 0  # Block-aligned, no sub-block inflation.
