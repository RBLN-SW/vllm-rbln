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

# Sub-block prefix caching: single vs multi-group, extra-key isolation, the
# partial-block lifecycle (create -> grow -> promote -> reuse), refcounts and
# eviction. Pure pieces need no manager; the rest drive one on CPU.

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    hash_block_tokens,
    init_none_hash,
    maybe_convert_block_hash,
)

from tests.native.v1.core.utils import (
    full_attention_spec,
    make_hybrid_manager,
    make_kv_cache_config,
    make_manager,
    make_request,
    prefill_request,
    sliding_window_spec,
    sub_block_index,
)
from vllm_rbln.v1.core.rbln_kv_cache_manager import (
    RBLNKVCacheManager,
    SubBlockHasher,
    SubBlockIndex,
)

BLOCK_SIZE = 8
SUB_BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _init_none_hash():
    # hash_block_tokens reads a module-global NONE_HASH seeded from the hash fn.
    init_none_hash(sha256)


def _hasher(sub_block_size: int = SUB_BLOCK_SIZE) -> SubBlockHasher:
    return SubBlockHasher(sha256, sub_block_size)


def _hashes(num_sub_blocks: int, sub_block_size: int = SUB_BLOCK_SIZE) -> list:
    """A chain of ``num_sub_blocks`` distinct sub-block hashes."""
    hashes, _, _ = _hasher(sub_block_size).hash_tokens(
        list(range(num_sub_blocks * sub_block_size))
    )
    return hashes


class TestSubBlockHasher:
    def test_empty_tokens_yields_no_hashes(self):
        # No tokens -> empty hash/extra-keys lists, mm_idx unchanged.
        hashes, extras, mm_idx = _hasher().hash_tokens([])
        assert hashes == []
        assert extras == []
        assert mm_idx == 0

    def test_fewer_than_sub_block_yields_no_hashes(self):
        # Fewer than sub_block_size tokens -> no complete sub-block -> no hashes.
        hashes, _, _ = _hasher().hash_tokens([1, 2, 3])
        assert hashes == []

    def test_exact_sub_block_yields_one_hash(self):
        # Exactly sub_block_size tokens -> a single hash (boundary).
        hashes, _, _ = _hasher().hash_tokens(list(range(SUB_BLOCK_SIZE)))
        assert len(hashes) == 1

    def test_hashes_only_full_sub_blocks(self):
        # Several sub-blocks plus a trailing partial (< sub_block_size) that is
        # ignored: 10 tokens at sbs=4 -> 2 hashes.
        hashes, _, _ = _hasher().hash_tokens(list(range(10)))
        assert len(hashes) == 2

    def test_chains_via_parent_hash(self):
        # Each hash chains off its parent, so the second sub-block's hash depends
        # on the first and differs from hashing it alone.
        tokens = list(range(2 * SUB_BLOCK_SIZE))
        full, _, _ = _hasher().hash_tokens(tokens)
        first, _, _ = _hasher().hash_tokens(tokens[:SUB_BLOCK_SIZE])
        chained, _, _ = _hasher().hash_tokens(
            tokens, parent_hash=first[0], num_hashed_tokens=SUB_BLOCK_SIZE
        )
        assert chained[0] == full[1]
        lone, _, _ = _hasher().hash_tokens(tokens[SUB_BLOCK_SIZE:])
        assert chained[0] != lone[0]

    def test_num_hashed_tokens_offsets_start(self):
        # Hashing starts num_hashed_tokens into the sequence: with an offset of
        # one sub-block only the second sub-block is hashed.
        tokens = list(range(2 * SUB_BLOCK_SIZE))
        hashes, _, _ = _hasher().hash_tokens(tokens, num_hashed_tokens=SUB_BLOCK_SIZE)
        assert len(hashes) == 1

    def test_incremental_hashing_continues_from_previous(self):
        # Re-hashing a grown sequence from where it left off (offset + parent)
        # reproduces the one-shot chain exactly.
        tokens = list(range(3 * SUB_BLOCK_SIZE))
        one_shot, _, _ = _hasher().hash_tokens(tokens)
        part1, _, _ = _hasher().hash_tokens(tokens[: 2 * SUB_BLOCK_SIZE])
        part2, _, _ = _hasher().hash_tokens(
            tokens, parent_hash=part1[-1], num_hashed_tokens=2 * SUB_BLOCK_SIZE
        )
        assert part1 + part2 == one_shot

    def test_parent_hash_changes_result(self):
        # A different parent_hash changes the result for identical tokens.
        other, _, _ = _hasher().hash_tokens(list(range(100, 100 + SUB_BLOCK_SIZE)))
        without_parent, _, _ = _hasher().hash_tokens(list(range(SUB_BLOCK_SIZE)))
        with_parent, _, _ = _hasher().hash_tokens(
            list(range(SUB_BLOCK_SIZE)), parent_hash=other[0]
        )
        assert without_parent[0] != with_parent[0]

    def test_different_tokens_different_hashes(self):
        # Different token content -> different hash.
        a, _, _ = _hasher().hash_tokens([0, 1, 2, 3])
        b, _, _ = _hasher().hash_tokens([0, 1, 2, 9])
        assert a[0] != b[0]

    def test_no_request_yields_no_extra_keys(self):
        # Without a request the extra-keys list is all None.
        _, extras, _ = _hasher().hash_tokens(list(range(2 * SUB_BLOCK_SIZE)))
        assert extras == [None, None]

    def test_request_extra_keys_mixed_and_recorded(self):
        # A request carrying an extra key (cache_salt) mixes it into the hash and
        # records it in the extra-keys list, changing the result vs no request.
        req = make_request(
            "r", list(range(2 * SUB_BLOCK_SIZE)), block_size=8, cache_salt="salt"
        )
        with_req, extras, _ = _hasher().hash_tokens(req.all_token_ids, request=req)
        without, _, _ = _hasher().hash_tokens(req.all_token_ids)
        assert with_req[0] != without[0]
        assert extras[0] is not None


class TestSubBlockIndex:
    def test_empty_index_no_match(self):
        # longest_match on an empty index -> (None, 0).
        assert SubBlockIndex().longest_match(_hashes(2)) == (None, 0)

    def test_update_returns_first_fresh_idx(self):
        # Inserting all-new hashes returns the first globally-fresh index (0).
        idx = SubBlockIndex()
        assert idx.update(1, _hashes(3)) == 0

    def test_update_idempotent(self):
        # Re-updating the same (block, hashes) reports nothing fresh (== len).
        idx = SubBlockIndex()
        h = _hashes(2)
        idx.update(1, h)
        assert idx.update(1, h) == len(h)

    def test_update_partial_extend(self):
        # Extending a block with new trailing hashes: first_fresh_idx is the
        # extension start.
        idx = SubBlockIndex()
        h = _hashes(3)
        idx.update(1, h[:2])
        assert idx.update(1, h) == 2

    def test_update_hash_shared_across_blocks_not_fresh(self):
        # A hash already held by another block is not globally fresh, which is
        # what KV-event dedup rests on.
        idx = SubBlockIndex()
        h = _hashes(2)
        idx.update(1, h)
        assert idx.update(2, h) == len(h)

    def test_multiple_blocks_same_prefix(self):
        # Two blocks sharing a prefix are both indexed; longest_match returns one.
        idx = SubBlockIndex()
        h = _hashes(2)
        idx.update(1, h)
        idx.update(2, h)
        block_id, depth = idx.longest_match(h)
        assert block_id in (1, 2)
        assert depth == 2

    def test_pop_single_holder_returns_fully_removed(self):
        # Popping the sole holder returns its hashes as fully-removed.
        idx = SubBlockIndex()
        h = _hashes(2)
        idx.update(1, h)
        assert set(idx.pop(1)) == set(h)
        assert idx.longest_match(h) == (None, 0)

    def test_pop_shared_holder_not_removed(self):
        # When another block still holds the hashes, pop removes nothing.
        idx = SubBlockIndex()
        h = _hashes(2)
        idx.update(1, h)
        idx.update(2, h)
        assert idx.pop(1) == []
        assert idx.longest_match(h) == (2, 2)

    def test_pop_unknown_block_returns_empty(self):
        # Popping a block that was never indexed -> [].
        assert SubBlockIndex().pop(99) == []

    def test_longest_match_full_prefix(self):
        # A full prefix match returns (block_id, len).
        idx = SubBlockIndex()
        h = _hashes(3)
        idx.update(1, h)
        assert idx.longest_match(h) == (1, 3)

    def test_longest_match_partial_breaks_on_first_miss(self):
        # Matching stops at the first hash not in the index, returning the
        # partial depth.
        idx = SubBlockIndex()
        h = _hashes(3)
        idx.update(1, h[:2])
        assert idx.longest_match(h) == (1, 2)

    def test_longest_match_empty_query(self):
        # An empty query -> (None, 0).
        assert SubBlockIndex().longest_match([]) == (None, 0)

    def test_contains_and_all_hashes(self):
        # contains() is exact; all_hashes() returns every currently-indexed hash.
        idx = SubBlockIndex()
        h = _hashes(2)
        idx.update(1, h)
        assert idx.contains(1)
        assert not idx.contains(2)
        assert set(idx.all_hashes()) == set(h)


class TestCanUseSubBlockCaching:
    @staticmethod
    def _ineligible_config():
        # Rejected at the isinstance gate before block_size is read, so a
        # duck-typed stand-in exercises exactly that branch.
        fake = SimpleNamespace(block_size=16)
        return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=fake)])

    def test_rejects_nonpositive_sub_block_size(self):
        # sub_block_size <= 0 -> False.
        cfg = make_kv_cache_config([full_attention_spec(8)])
        assert not RBLNKVCacheManager.can_use_sub_block_caching(cfg, 0)
        assert not RBLNKVCacheManager.can_use_sub_block_caching(cfg, -1)

    def test_rejects_ineligible_spec_type(self):
        # A spec type outside {Full, SlidingWindow, ChunkedLocal} -> False.
        assert not RBLNKVCacheManager.can_use_sub_block_caching(
            self._ineligible_config(), 4
        )

    def test_rejects_block_size_not_multiple_of_sub(self):
        # block_size % sub_block_size != 0 -> False.
        cfg = make_kv_cache_config([full_attention_spec(8)])
        assert not RBLNKVCacheManager.can_use_sub_block_caching(cfg, 3)

    def test_rejects_block_size_le_sub(self):
        # block_size <= sub_block_size -> False (both equal and greater).
        cfg = make_kv_cache_config([full_attention_spec(4)])
        assert not RBLNKVCacheManager.can_use_sub_block_caching(cfg, 4)
        assert not RBLNKVCacheManager.can_use_sub_block_caching(cfg, 8)

    def test_accepts_valid_config(self):
        # All groups eligible and size constraints satisfied -> True.
        cfg = make_kv_cache_config([full_attention_spec(8)])
        assert RBLNKVCacheManager.can_use_sub_block_caching(cfg, 4)

    def test_multi_group_all_must_be_eligible(self):
        # Every group must be eligible; one ineligible group flips it to False.
        cfg_ok = make_kv_cache_config(
            [full_attention_spec(8), sliding_window_spec(8, 16)]
        )
        assert RBLNKVCacheManager.can_use_sub_block_caching(cfg_ok, 4)
        cfg_bad = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=full_attention_spec(8)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8)),
            ]
        )
        assert not RBLNKVCacheManager.can_use_sub_block_caching(cfg_bad, 4)


class TestManagerInit:
    def test_asserts_on_invalid_config(self):
        # A can_use-False config (block_size not a multiple of sub_block_size)
        # trips the __init__ assert.
        with pytest.raises(AssertionError):
            make_manager(block_size=8, sub_block_size=3, num_blocks=10)

    def test_group_infos_sub_blocks_per_block(self):
        # Per group: block_size and sub_blocks_per_block == block_size // sub.
        manager = make_manager(8, 4, 10)
        assert len(manager._group_infos) == 1
        gi = manager._group_infos[0]
        assert gi.block_size == 8
        assert gi.sub_blocks_per_block == 2

    def test_block_pool_events_suppressed(self):
        # The manager takes over event emission; the pool's own emission is off.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        assert manager.enable_kv_cache_events is True
        assert manager.block_pool.enable_kv_cache_events is False

    def test_multi_group_with_events_raises(self):
        # Multi-group + KV events -> ValueError (hybrid does not support events).
        with pytest.raises(ValueError):
            make_hybrid_manager(
                8, 4, 10, sliding_window=16, enable_kv_cache_events=True
            )


class TestGetComputedBlocksSubBlock:
    def test_no_partial_match_on_first_request(self):
        # The first request finds an empty index -> no hits, no copy ops.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(2 * 8 + 5)), 8)
        computed, total, blocks = prefill_request(manager, req)
        assert total == 0
        assert not computed.blocks[0]
        assert blocks is not None
        assert manager.drain_pending_copy_ops() == []

    def test_partial_match_after_full_block_boundary(self):
        # After two full blocks match upstream, a sub-block match is found in the
        # third block right past the boundary.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(3 * 8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        # Two full blocks + the third block's first sub-block, then divergence.
        req1_tokens = list(range(2 * 8 + 4)) + [999] * 7
        req1 = make_request("1", req1_tokens, 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 2 * 8
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == 4
        manager.release_sub_block_match(match)

    def test_no_partial_match_when_full_block_matches(self):
        # When upstream already matched the full block there is no partial match
        # (no double counting).
        manager = make_manager(8, 4, 10)
        tokens = list(range(2 * 8))
        req0 = make_request("0", tokens, 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", tokens + [99, 99, 99], 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 2 * 8
        assert manager.get_computed_blocks_sub_block(req1, num_computed) is None

    def test_no_match_returns_none(self):
        # A populated index that shares no prefix with this request -> None.
        manager = make_manager(8, 4, 10)
        prefill_request(manager, make_request("0", list(range(8)), 8))
        req1 = make_request("1", [500 + i for i in range(2 * 8)], 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        assert manager.get_computed_blocks_sub_block(req1, num_computed) is None

    def test_empty_query_returns_none(self):
        # A request with fewer than sub_block_size tokens yields no sub-block
        # hashes -> the query is empty -> None (the `if not query` branch).
        manager = make_manager(8, 4, 10)
        prefill_request(manager, make_request("0", list(range(8)), 8))
        req1 = make_request("1", [0, 1], 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert manager.get_computed_blocks_sub_block(req1, num_computed) is None

    def test_computed_tokens_capped_at_num_tokens_minus_one(self):
        # A raw 3-sub-block match is capped so total computed <= num_tokens - 1:
        # num_tokens=12, cap=11, 11 // 4 = 2 sub-blocks.
        manager = make_manager(16, 4, 10)
        tokens = list(range(3 * 4))
        req0 = make_request("0", tokens, 16)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", tokens, 16)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == 2 * 4
        assert num_computed + match.num_tokens <= req1.num_tokens - 1
        manager.release_sub_block_match(match)

    def test_capping_to_zero_returns_none(self):
        # When the cap leaves zero sub-blocks -> None: num_tokens=4, cap=3,
        # 3 // 4 = 0.
        manager = make_manager(16, 4, 10)
        req0 = make_request("0", list(range(3 * 4)), 16)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)), 16)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        assert manager.get_computed_blocks_sub_block(req1, num_computed) is None

    def test_skip_reading_prefix_cache_returns_none(self):
        # request.skip_reading_prefix_cache short-circuits to None.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * 8, 8)
        req1.skip_reading_prefix_cache = True
        assert manager.get_computed_blocks_sub_block(req1, 0) is None

    def test_prompt_logprobs_skips_sub_block_cache(self):
        # A prompt_logprobs request skips prefix-cache reads (both full and
        # sub-block), matching upstream.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req = make_request(
            "2", list(range(4)) + [100 + i for i in range(8)], 8, prompt_logprobs=5
        )
        _, num_computed = manager.get_computed_blocks(req)
        assert num_computed == 0
        assert manager.get_computed_blocks_sub_block(req, num_computed) is None

    def test_exact_block_size_request_recovery(self):
        # num_tokens == block_size: upstream matches 0 full blocks (caps at
        # num_tokens-1 < block_size), sub-block matching recovers one sub-block.
        manager = make_manager(8, 4, 10)
        tokens = list(range(8))
        req0 = make_request("0", tokens, 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", tokens, 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == 4
        assert num_computed + match.num_tokens <= req1.num_tokens - 1
        manager.release_sub_block_match(match)

    def test_touches_source_blocks_to_prevent_eviction(self):
        # A successful match touches (ref-holds) the source block; releasing the
        # match drops that ref.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * 8, 8)
        _, num_computed = manager.get_computed_blocks(req1)
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        src = match.group_matches[0].src_block
        assert src.ref_cnt == 1
        manager.release_sub_block_match(match)
        assert src.ref_cnt == 0


class TestApplyReleaseSubBlockMatch:
    def test_apply_creates_copy_op_per_group(self):
        # apply -> one KVCacheCopyOp(group_id/src/dst/num_tokens) per group.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100 + i for i in range(8)], 8)
        _, _, new_blocks = prefill_request(manager, req1)
        assert new_blocks is not None
        ops = manager.drain_pending_copy_ops()
        assert len(ops) == 1
        op = ops[0]
        assert op.group_id == 0
        assert op.num_tokens == 4
        assert op.dst_block_id != op.src_block_id
        manager.release_copy_ops(ops)

    def test_apply_records_prefix_cache_stats(self):
        # With log_stats, apply records the sub-block hits.
        manager = make_manager(8, 4, 10, log_stats=True)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * 8, 8)
        prefill_request(manager, req1)
        assert manager.prefix_cache_stats is not None
        assert manager.prefix_cache_stats.hits >= 4
        manager.release_copy_ops(manager.drain_pending_copy_ops())

    def test_release_frees_source_refs(self):
        # release frees the source-block ref the match held.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * 8, 8)
        _, num_computed = manager.get_computed_blocks(req1)
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        src = match.group_matches[0].src_block
        ref = src.ref_cnt
        manager.release_sub_block_match(match)
        assert src.ref_cnt == ref - 1

    def test_src_block_protected_until_release(self):
        # A pending copy op keeps its source block referenced: reset is refused
        # and drain does not release; only release_copy_ops drops the ref.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * 8, 8)
        prefill_request(manager, req1)
        assert len(manager.pending_copy_ops) == 1
        src = manager.block_pool.blocks[manager.pending_copy_ops[0].src_block_id]
        assert src.ref_cnt > 0
        ref_before = src.ref_cnt
        assert not manager.reset_prefix_cache()
        assert len(sub_block_index(manager)._block_hashes) > 0
        ops = manager.drain_pending_copy_ops()
        assert src.ref_cnt == ref_before
        manager.release_copy_ops(ops)
        assert src.ref_cnt == ref_before - 1

    def test_early_release_allows_src_eviction(self):
        # Releasing copy-op refs early (the async-scheduling race) lets a later
        # allocation evict and recycle the source block.
        manager = make_manager(8, 4, 4)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100 + i for i in range(8)], 8)
        prefill_request(manager, req1)
        assert len(manager.pending_copy_ops) == 1
        src = manager.block_pool.blocks[manager.pending_copy_ops[0].src_block_id]
        ops = manager.drain_pending_copy_ops()
        assert src.ref_cnt > 0
        manager.release_copy_ops(ops)
        manager.free(req1)
        req2 = make_request("2", [200 + i for i in range(3 * 8)], 8)
        prefill_request(manager, req2)
        # The source block was recycled for req2.
        assert src.ref_cnt > 0
        assert src.block_hash is not None
        assert ops[0].src_block_id == src.block_id


class TestSubBlockIndexingFlow:
    def test_allocate_slots_schedules_indexing(self):
        # allocate_slots(delay=False) auto-registers the request for indexing.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8)), 8)
        cb, n = manager.get_computed_blocks(req)
        manager.allocate_slots(req, req.num_tokens - n, n, cb)
        assert req.request_id in manager._pending_indexing

    def test_allocate_slots_delay_skips_indexing(self):
        # delay_cache_blocks=True skips auto-registration (caller's job).
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8)), 8)
        cb, n = manager.get_computed_blocks(req)
        manager.allocate_slots(req, req.num_tokens - n, n, cb, delay_cache_blocks=True)
        assert req.request_id not in manager._pending_indexing

    def test_do_pending_indexing_indexes_full_and_partial(self):
        # do_pending_indexing (run inside prefill_request) indexes both the full
        # and the partial block.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 4)), 8)
        prefill_request(manager, req)
        full_blk, partial_blk = manager.coordinator.get_blocks(req.request_id)[0][:2]
        idx = sub_block_index(manager)
        assert idx.contains(full_blk.block_id)
        assert idx.contains(partial_blk.block_id)

    def test_index_newly_cached_blocks_during_decode(self):
        # Deferred indexing picks up a block that becomes full during decode.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 1)), 8, max_tokens=8)
        prefill_request(manager, req)
        blk0 = manager.coordinator.get_blocks(req.request_id)[0][0]
        assert sub_block_index(manager).contains(blk0.block_id)
        for i in range(8 - 1):
            req.append_output_token_ids(200 + i)
            before = req.num_computed_tokens
            assert manager.allocate_slots(req, 1, 0) is not None
            req.num_computed_tokens = before + 1
        manager.do_pending_indexing()
        blocks = manager.coordinator.get_blocks(req.request_id)[0]
        if len(blocks) > 1 and blocks[1].block_hash is not None:
            assert sub_block_index(manager).contains(blocks[1].block_id)

    def test_free_indexes_before_release(self):
        # The partial block is eagerly indexed before free and stays indexed
        # after free consumes its pending entry.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8 + 4)), 8)
        prefill_request(manager, req0)
        partial_blk = manager.coordinator.get_blocks(req0.request_id)[0][1]
        idx = sub_block_index(manager)
        assert idx.contains(partial_blk.block_id)
        partial_id = partial_blk.block_id
        manager.free(req0)
        assert partial_id in idx._block_hashes

    def test_free_cleans_up_sub_hash_cache(self):
        # free removes the request's state from _req_sub_hashes.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8)), 8)
        prefill_request(manager, req)
        assert "0" in manager._req_sub_hashes
        manager.free(req)
        assert "0" not in manager._req_sub_hashes

    def test_pop_blocks_for_free_settles_state_like_free(self):
        # The scheduler's other way out: it takes this one when an in-flight step
        # may still write the blocks, so it keeps them and returns them to the pool
        # later. The blocks leave either way, so the sub-block state has to be
        # settled here as free() settles it.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 4)), 8)
        prefill_request(manager, req)
        partial_id = manager.coordinator.get_blocks(req.request_id)[0][1].block_id

        blocks = manager.pop_blocks_for_free(req)

        assert blocks, "the caller frees these later"
        assert "0" not in manager._req_sub_hashes
        assert "0" not in manager._pending_indexing
        assert partial_id in sub_block_index(manager)._block_hashes

    def test_partial_block_fewer_than_sub_block_skipped(self):
        # A partial block with fewer than sub_block_size tokens is not indexed
        # and not cached.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 2)), 8)
        prefill_request(manager, req)
        partial_blk = manager.coordinator.get_blocks(req.request_id)[0][1]
        partial_id = partial_blk.block_id
        assert partial_blk.block_hash is None
        manager.free(req)
        assert partial_id not in sub_block_index(manager)._block_hashes
        assert partial_blk.block_hash is None

    def test_partial_block_multiple_sub_blocks(self):
        # A partial block spanning several sub-blocks indexes all of them, and a
        # later request matches them all.
        manager = make_manager(16, 4, 10)
        tokens = list(range(16 + 3 * 4))
        req = make_request("0", tokens, 16)
        prefill_request(manager, req)
        partial_blk = manager.coordinator.get_blocks(req.request_id)[0][1]
        partial_id = partial_blk.block_id
        manager.free(req)
        idx = sub_block_index(manager)
        assert partial_id in idx._block_hashes
        assert len(idx._block_hashes[partial_id]) == 3
        req1 = make_request("1", tokens + [999] * 16, 16)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 16
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == 3 * 4
        manager.release_sub_block_match(match)

    def test_no_partial_block_when_remainder_zero(self):
        # An exact full block leaves no partial block; free adds no new index
        # entries.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8)), 8)
        prefill_request(manager, req)
        block_list = manager.coordinator.get_blocks(req.request_id)[0]
        assert len(block_list) == 1
        assert block_list[0].block_hash is not None
        before = set(sub_block_index(manager)._block_hashes)
        manager.free(req)
        assert set(sub_block_index(manager)._block_hashes) == before

    def test_index_partial_block_assigns_synthetic_hash(self):
        # free assigns the partial block a synthetic block_hash so the LRU keeps
        # it (None before, set after).
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 4)), 8)
        prefill_request(manager, req)
        partial_blk = manager.coordinator.get_blocks(req.request_id)[0][1]
        assert partial_blk.block_hash is None
        manager.free(req)
        assert partial_blk.block_hash is not None

    def test_partial_to_full_transition_completes_indexing(self):
        # A partially-indexed block gets the rest of its sub-blocks indexed once
        # it fills up.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 4)), 8, max_tokens=8)
        prefill_request(manager, req)
        partial_blk = manager.coordinator.get_blocks(req.request_id)[0][1]
        idx = sub_block_index(manager)
        assert len(idx._block_hashes[partial_blk.block_id]) == 1
        for i in range(4):
            req.append_output_token_ids(400 + i)
            before = req.num_computed_tokens
            manager.allocate_slots(req, 1, 0)
            req.num_computed_tokens = before + 1
        manager.do_pending_indexing()
        assert partial_blk.block_hash is not None
        assert len(idx._block_hashes[partial_blk.block_id]) == 2

    def test_eager_partial_block_visible_before_free(self):
        # A running request's partial sub-block is matchable by a new request
        # before free is called.
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8 + 4)), 8, max_tokens=8)
        prefill_request(manager, req0)
        partial_blk = manager.coordinator.get_blocks(req0.request_id)[0][1]
        assert partial_blk.block_id in sub_block_index(manager)._block_hashes
        assert partial_blk.block_hash is None
        req1 = make_request("1", list(range(8 + 4)) + [100 + i for i in range(8)], 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 8
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == 4
        manager.release_sub_block_match(match)

    def test_eager_partial_block_grows_during_decode(self):
        # As a running request decodes across sub-block boundaries, its partial
        # block's index grows incrementally.
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8 + 1)), 8, max_tokens=2 * 8)
        prefill_request(manager, req)
        partial_blk = manager.coordinator.get_blocks(req.request_id)[0][1]
        idx = sub_block_index(manager)
        assert partial_blk.block_id not in idx._block_hashes
        for i in range(4 - 1):
            req.append_output_token_ids(200 + i)
            before = req.num_computed_tokens
            manager.allocate_slots(req, 1, 0)
            req.num_computed_tokens = before + 1
        manager.do_pending_indexing()
        assert len(idx._block_hashes[partial_blk.block_id]) == 1
        for i in range(4):
            req.append_output_token_ids(300 + i)
            before = req.num_computed_tokens
            manager.allocate_slots(req, 1, 0)
            req.num_computed_tokens = before + 1
        manager.do_pending_indexing()
        assert len(idx._block_hashes[partial_blk.block_id]) == 2


class TestPartialBlockReuse:
    def test_partial_block_reused_by_next_request(self):
        # After freeing a request with a partial block, a new request sharing the
        # full prefix gets a full-block hit plus a sub-block hit.
        manager = make_manager(8, 4, 10)
        tokens0 = list(range(8 + 4))
        req0 = make_request("0", tokens0, 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", tokens0 + [800 + i for i in range(8)], 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 8
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == 4
        manager.release_sub_block_match(match)

    def test_multi_prefill_no_stale_sub_block_match(self):
        # Two prefills sharing a prefix in the same step must not match each
        # other (KV not computed yet); a third request in the next step does.
        manager = make_manager(8, 4, 100)
        shared = list(range(4))
        req0 = make_request("0", shared + [100] * 8, 8, max_tokens=1)
        req1 = make_request("1", shared + [200] * 8, 8, max_tokens=1)
        cb0, nc0 = manager.get_computed_blocks(req0)
        assert manager.get_computed_blocks_sub_block(req0, nc0) is None
        manager.allocate_slots(req0, req0.num_tokens, 0, cb0)
        cb1, nc1 = manager.get_computed_blocks(req1)
        assert manager.get_computed_blocks_sub_block(req1, nc1) is None
        manager.allocate_slots(req1, req1.num_tokens, 0, cb1)
        req0.num_computed_tokens = req0.num_tokens
        req1.num_computed_tokens = req1.num_tokens
        manager.do_pending_indexing()
        assert len(sub_block_index(manager)._hash_to_blocks) > 0
        req2 = make_request("2", shared + [300] * 8, 8, max_tokens=1)
        _, nc2 = manager.get_computed_blocks(req2)
        match = manager.get_computed_blocks_sub_block(req2, nc2)
        assert match is not None
        assert match.num_tokens == 4
        manager.release_sub_block_match(match)

    def test_partial_match_tight_memory_no_assertion(self):
        # The partial-match path survives a tight block pool without a ref_cnt
        # assertion.
        manager = make_manager(8, 4, 4)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        rest = [100 + i for i in range(8 + 4)]
        req1 = make_request("1", list(range(4)) + rest, 8)
        _, _, new_blocks = prefill_request(manager, req1)
        assert new_blocks is not None


class TestIsolation:
    def test_cache_salt_isolation(self):
        # A different (or absent) cache_salt yields different hashes -> no
        # cross-hit against a salted entry.
        manager = make_manager(8, 4, 10)
        tokens = list(range(8))
        req0 = make_request("0", tokens, 8, cache_salt="A")
        prefill_request(manager, req0)
        manager.free(req0)
        shared = list(range(4))
        unique = [100 + i for i in range(8)]
        req1 = make_request("1", shared + unique, 8, cache_salt="B")
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        assert manager.get_computed_blocks_sub_block(req1, num_computed) is None
        req2 = make_request("2", shared + unique, 8)
        _, nc2 = manager.get_computed_blocks(req2)
        assert manager.get_computed_blocks_sub_block(req2, nc2) is None

    def test_lora_isolation(self):
        # A different LoRA adapter yields different hashes -> no cross-hit.
        manager = make_manager(8, 4, 10)
        lora_a = LoRARequest(lora_name="lora_a", lora_int_id=1, lora_path="/a")
        lora_b = LoRARequest(lora_name="lora_b", lora_int_id=2, lora_path="/b")
        tokens = list(range(8))
        req0 = make_request("0", tokens, 8, lora_request=lora_a)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request(
            "1", list(range(4)) + [100 + i for i in range(8)], 8, lora_request=lora_b
        )
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        assert manager.get_computed_blocks_sub_block(req1, num_computed) is None


class TestEvictionHook:
    def test_evicted_block_removed_from_index(self):
        # _on_block_evicted removes the block from the index structure (event
        # emission is covered by TestKVEvents).
        manager = make_manager(8, 4, 10)
        req = make_request("0", list(range(8)), 8)
        prefill_request(manager, req)
        blk = manager.coordinator.get_blocks(req.request_id)[0][0]
        assert sub_block_index(manager).contains(blk.block_id)
        manager._on_block_evicted(blk.block_id)
        assert not sub_block_index(manager).contains(blk.block_id)


class TestKVEvents:
    # An external router re-hashes the emitted token_ids in sub_block_size chunks
    # from parent_block_hash and matches them against our block_hashes, so the
    # payload must be exact. RBLN re-emits at sub_block granularity.

    R = BLOCK_SIZE // SUB_BLOCK_SIZE  # sub-blocks per block

    @staticmethod
    def _stored(manager):
        return [e for e in manager.take_events() if isinstance(e, BlockStored)]

    def test_events_disabled_returns_empty(self):
        # enable_kv_cache_events=False -> take_events always returns [].
        manager = make_manager(8, 4, 10)
        prefill_request(manager, make_request("0", list(range(8)), 8))
        assert manager.take_events() == []

    def test_take_events_returns_and_clears(self):
        # take_events returns the buffered events then clears (second call []).
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        prefill_request(manager, make_request("0", list(range(8)), 8))
        assert manager.take_events()
        assert manager.take_events() == []

    def test_store_at_sub_block_granularity(self):
        # A cached full block emits BlockStored at block_size == sub_block_size,
        # covering all R sub-blocks.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        prefill_request(manager, make_request("0", list(range(8)), 8))
        stored = self._stored(manager)
        assert stored
        assert all(e.block_size == SUB_BLOCK_SIZE for e in stored)
        assert sum(len(e.block_hashes) for e in stored) == self.R

    def test_store_hash_chain_reconstructible(self):
        # Emitted events chain via parent_block_hash and match the chain a
        # consumer reconstructs by re-hashing the prompt.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        tokens = list(range(2 * 8))
        prefill_request(manager, make_request("0", tokens, 8))
        stored = self._stored(manager)
        expected_hashes, _, _ = SubBlockHasher(sha256, SUB_BLOCK_SIZE).hash_tokens(
            tokens
        )
        expected = [maybe_convert_block_hash(h) for h in expected_hashes]
        flat_chain: list = []
        for e in stored:
            if not flat_chain:
                assert e.parent_block_hash is None
            else:
                assert e.parent_block_hash == flat_chain[-1]
            flat_chain.extend(e.block_hashes)
        assert flat_chain == expected[: len(flat_chain)]

    def test_store_token_ids_and_size_match_sub_block(self):
        # Per event len(token_ids) == len(block_hashes) * block_size, and the
        # concatenated token_ids equal the original prompt tokens.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        tokens = list(range(8))
        prefill_request(manager, make_request("0", tokens, 8))
        flat_tokens: list[int] = []
        for e in self._stored(manager):
            assert len(e.token_ids) == len(e.block_hashes) * e.block_size
            flat_tokens.extend(e.token_ids)
        assert flat_tokens == tokens

    def test_dedup_on_multi_turn_same_prefix(self):
        # A second request caching the same prefix skips already-indexed hashes
        # (first_fresh_idx dedup) -> no fresh BlockStored.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        tokens = list(range(8))
        req0 = make_request("0", tokens, 8)
        prefill_request(manager, req0)
        assert sum(len(e.block_hashes) for e in self._stored(manager)) > 0
        manager.free(req0)
        prefill_request(manager, make_request("1", tokens, 8))
        assert self._stored(manager) == []

    def test_partial_then_full_emits_suffix_only(self):
        # A partial (N<R) block promoted to full later emits only the trailing
        # R-N hashes, chained to the already-stored prefix.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        req0 = make_request(
            "0", list(range(SUB_BLOCK_SIZE)), 8, max_tokens=SUB_BLOCK_SIZE
        )
        prefill_request(manager, req0)
        manager.free(req0)
        assert sum(len(e.block_hashes) for e in self._stored(manager)) == 1

        full_tokens = list(range(8))
        prefill_request(manager, make_request("1", full_tokens, 8))
        stored = self._stored(manager)
        assert sum(len(e.block_hashes) for e in stored) == self.R - 1
        expected_hashes, _, _ = SubBlockHasher(sha256, SUB_BLOCK_SIZE).hash_tokens(
            full_tokens
        )
        assert stored[0].parent_block_hash == maybe_convert_block_hash(
            expected_hashes[0]
        )

    def test_remove_fires_only_on_last_holder(self):
        # Two blocks hold the same prefix hashes: evicting the first fires no
        # BlockRemoved (refcount 1); evicting the second fires it with R hashes.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        tokens = list(range(8))
        hashes, _, _ = SubBlockHasher(sha256, SUB_BLOCK_SIZE).hash_tokens(tokens)
        index = sub_block_index(manager)
        index.update(5, hashes)
        index.update(6, hashes)
        manager.take_events()

        manager._on_block_evicted(5)
        assert not any(isinstance(e, BlockRemoved) for e in manager.take_events())

        manager._on_block_evicted(6)
        removed = [e for e in manager.take_events() if isinstance(e, BlockRemoved)]
        assert len(removed) == 1
        expected = {maybe_convert_block_hash(h) for h in hashes}
        assert set(removed[0].block_hashes) == expected

    def test_eviction_via_real_pool_path_emits_remove(self):
        # BlockRemoved also fires through the real pool eviction path (the
        # _install_eviction_hook monkeypatch), not just a direct _on_block_evicted.
        manager = make_manager(8, 4, 4, enable_kv_cache_events=True)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        assert sub_block_index(manager).all_hashes()
        manager.take_events()

        removed_events: list = []
        for i in range(1, 6):
            req = make_request(str(i), list(range(i * 1000, i * 1000 + 8)), 8)
            prefill_request(manager, req)
            manager.free(req)
            removed_events += [
                e for e in manager.take_events() if isinstance(e, BlockRemoved)
            ]
        assert removed_events
        expected_hashes, _, _ = SubBlockHasher(sha256, SUB_BLOCK_SIZE).hash_tokens(
            list(range(8))
        )
        expected = {maybe_convert_block_hash(h) for h in expected_hashes}
        seen: set = set()
        for ev in removed_events:
            seen.update(ev.block_hashes)
        assert expected & seen

    def test_extra_keys_round_trip_multimodal(self):
        # Re-hashing the emitted token_ids + extra_keys + parent_block_hash must
        # reproduce the emitted block_hashes, with mm_idx advancing mid-chain.
        # Images at [0,4), [8,14), [16,20) over sub_block_size=4, block_size=8.
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
        tokens = list(range(20))
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        req = make_request(
            "mm", tokens, 8, cache_salt="sprinkle", mm_features=mm_features
        )
        prefill_request(manager, req)
        stored = self._stored(manager)
        assert stored

        prev_bare_hash = None
        for e in stored:
            expected_parent = (
                maybe_convert_block_hash(prev_bare_hash)
                if prev_bare_hash is not None
                else None
            )
            assert e.parent_block_hash == expected_parent
            assert e.extra_keys is not None
            assert len(e.extra_keys) == len(e.block_hashes)
            assert len(e.token_ids) == len(e.block_hashes) * e.block_size
            parent = prev_bare_hash
            for i, emitted_hash in enumerate(e.block_hashes):
                sub_tokens = e.token_ids[i * e.block_size : (i + 1) * e.block_size]
                bare = hash_block_tokens(sha256, parent, sub_tokens, e.extra_keys[i])
                assert maybe_convert_block_hash(bare) == emitted_hash
                parent = bare
            prev_bare_hash = parent

        # The reconstructed chain matches what generate_block_hash_extra_keys
        # produces for the same prompt (mm_idx threaded through event boundaries).
        mm_idx = 0
        parent = None
        expected_chain: list = []
        for i in range(len(tokens) // SUB_BLOCK_SIZE):
            extras, mm_idx = generate_block_hash_extra_keys(
                req, i * SUB_BLOCK_SIZE, (i + 1) * SUB_BLOCK_SIZE, mm_idx
            )
            parent = hash_block_tokens(
                sha256,
                parent,
                tokens[i * SUB_BLOCK_SIZE : (i + 1) * SUB_BLOCK_SIZE],
                extras,
            )
            expected_chain.append(maybe_convert_block_hash(parent))
        flat_emitted: list = []
        for e in stored:
            flat_emitted.extend(e.block_hashes)
        assert flat_emitted == expected_chain[: len(flat_emitted)]

    def test_extra_keys_round_trip_lora(self):
        # LoRA + cache_salt: re-hashing the extras reproduces the emitted hash and
        # the lora_id / lora_name fields surface on the event.
        lora = LoRARequest(lora_name="sales_bot", lora_int_id=42, lora_path="/tmp/x")
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        req = make_request(
            "0", list(range(8)), 8, cache_salt="pepper", lora_request=lora
        )
        prefill_request(manager, req)
        stored = self._stored(manager)
        assert stored
        for e in stored:
            assert e.lora_id == 42
            assert e.lora_name == "sales_bot"
            assert e.extra_keys is not None
        parent = None
        for e in stored:
            for i, emitted_hash in enumerate(e.block_hashes):
                sub_tokens = e.token_ids[i * e.block_size : (i + 1) * e.block_size]
                bare = hash_block_tokens(sha256, parent, sub_tokens, e.extra_keys[i])
                assert maybe_convert_block_hash(bare) == emitted_hash
                parent = bare


class TestResetPrefixCache:
    def test_reset_clears_indices_and_emits_all_cleared(self):
        # reset clears the group index and emits AllBlocksCleared.
        manager = make_manager(8, 4, 10, enable_kv_cache_events=True)
        req = make_request("0", list(range(8)), 8)
        prefill_request(manager, req)
        manager.free(req)
        assert sub_block_index(manager).all_hashes()
        manager.take_events()
        assert manager.reset_prefix_cache() is True
        assert len(sub_block_index(manager)._block_hashes) == 0
        assert [type(e) for e in manager.take_events()] == [AllBlocksCleared]


class TestDrainReleaseCopyOps:
    @staticmethod
    def _manager_with_pending_copy_op():
        manager = make_manager(8, 4, 10)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * 8, 8)
        prefill_request(manager, req1)
        return manager

    def test_drain_returns_and_clears_pending(self):
        # drain_pending_copy_ops returns the ops then empties the pending list.
        manager = self._manager_with_pending_copy_op()
        assert len(manager.pending_copy_ops) == 1
        ops = manager.drain_pending_copy_ops()
        assert len(ops) == 1
        assert manager.pending_copy_ops == []
        manager.release_copy_ops(ops)

    def test_release_copy_ops_frees_src_refs(self):
        # release_copy_ops frees the source-block refs of the drained ops.
        manager = self._manager_with_pending_copy_op()
        ops = manager.drain_pending_copy_ops()
        src = manager.block_pool.blocks[ops[0].src_block_id]
        ref = src.ref_cnt
        manager.release_copy_ops(ops)
        assert src.ref_cnt == ref - 1


class TestMultiGroup:
    SLIDING_WINDOW = 16

    def test_full_block_hit_indexes_both_groups(self):
        # Hybrid: a full-block hit sub-block-indexes both groups.
        manager = make_hybrid_manager(8, 4, 20, self.SLIDING_WINDOW)
        prefill_request(manager, make_request("0", list(range(2 * 8)), 8))
        assert len(manager._group_infos[0].sub_block_index._block_hashes) > 0
        assert len(manager._group_infos[1].sub_block_index._block_hashes) > 0

    def test_partial_match_generates_copy_ops_per_group(self):
        # A partial match generates one copy op per group.
        manager = make_hybrid_manager(8, 4, 20, self.SLIDING_WINDOW)
        req0 = make_request("0", list(range(8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", list(range(4)) + [100] * (8 + 1), 8)
        _, total_computed, new_blocks = prefill_request(manager, req1)
        assert total_computed == SUB_BLOCK_SIZE
        assert new_blocks is not None
        ops = manager.drain_pending_copy_ops()
        assert len(ops) == 2
        assert all(op.num_tokens == SUB_BLOCK_SIZE for op in ops)
        manager.release_copy_ops(ops)

    def test_reset_clears_all_group_indices(self):
        # reset clears the sub-block index of every group.
        manager = make_hybrid_manager(8, 4, 20, self.SLIDING_WINDOW)
        req0 = make_request("0", list(range(2 * 8)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        assert any(gi.sub_block_index._block_hashes for gi in manager._group_infos)
        manager.reset_prefix_cache()
        for gi in manager._group_infos:
            assert len(gi.sub_block_index._block_hashes) == 0

    def test_different_block_sizes_partial_match(self):
        # Groups with different block sizes (16 and 8): the sub-block match takes
        # the minimum across groups.
        config = make_kv_cache_config(
            [full_attention_spec(16), sliding_window_spec(8, 32)], num_blocks=30
        )
        manager = RBLNKVCacheManager(
            kv_cache_config=config,
            max_model_len=8192,
            # LCM of the group block sizes (16, 8); must divide by hash_block_size.
            scheduler_block_size=16,
            hash_block_size=8,
            sub_block_size=4,
            hash_fn=sha256,
        )
        assert manager._group_infos[0].sub_blocks_per_block == 4
        assert manager._group_infos[1].sub_blocks_per_block == 2
        req0 = make_request("0", list(range(16)), 8)
        prefill_request(manager, req0)
        manager.free(req0)
        assert len(manager._group_infos[0].sub_block_index._block_hashes) > 0
        assert len(manager._group_infos[1].sub_block_index._block_hashes) > 0
        req1 = make_request("1", list(range(4)) + [100 + i for i in range(16)], 8)
        _, num_computed = manager.get_computed_blocks(req1)
        assert num_computed == 0
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert match.num_tokens == SUB_BLOCK_SIZE
        manager.release_sub_block_match(match)

    def test_partial_block_indexed_on_free_per_group(self):
        # free indexes the partial block of each group.
        manager = make_hybrid_manager(8, 4, 20, self.SLIDING_WINDOW)
        req0 = make_request("0", list(range(8 + 4)), 8)
        prefill_request(manager, req0)
        blocks = manager.coordinator.get_blocks(req0.request_id)
        partial_ids = [
            blocks[gid][1].block_id for gid in range(len(manager._group_infos))
        ]
        for i, gi in enumerate(manager._group_infos):
            assert partial_ids[i] in gi.sub_block_index._block_hashes
            assert blocks[i][1].block_hash is None
        manager.free(req0)
        for i, gi in enumerate(manager._group_infos):
            assert partial_ids[i] in gi.sub_block_index._block_hashes

    def test_capping_at_num_tokens_minus_one(self):
        # The num_tokens-1 cap holds in the multi-group case.
        manager = make_hybrid_manager(8, 4, 20, self.SLIDING_WINDOW)
        tokens = list(range(8))
        req0 = make_request("0", tokens, 8)
        prefill_request(manager, req0)
        manager.free(req0)
        req1 = make_request("1", tokens, 8)
        _, num_computed = manager.get_computed_blocks(req1)
        match = manager.get_computed_blocks_sub_block(req1, num_computed)
        assert match is not None
        assert num_computed == 0
        assert match.num_tokens == SUB_BLOCK_SIZE
        assert num_computed + match.num_tokens <= req1.num_tokens - 1
        manager.release_sub_block_match(match)
