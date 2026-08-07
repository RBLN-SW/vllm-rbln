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

# RBLNScheduler's differences from upstream's schedule(): no mixed batching,
# prefill batch of 1, capped decode batch, spec tokens across a block boundary,
# sub-block copy. schedule() is an 820-line port, so drift is the core risk.

import dataclasses
import inspect

import pytest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.native.v1.core.utils import (
    EOS_TOKEN_ID,
    _drain,
    advance_to_decode,
    create_rbln_scheduler,
    create_requests,
    make_model_runner_output,
    make_request,
    prefill_request,
)
from vllm_rbln.v1.core.rbln_kv_cache_manager import RBLNKVCacheManager
from vllm_rbln.v1.core.rbln_scheduler import (
    RBLNScheduler,
    RBLNSchedulerOutput,
    decode_batch_size,
    is_prefill,
    num_base_tokens,
    resolve_propagated_token_write,
    should_defer_spec_step,
)


class TestNumBaseTokens:
    def test_subtracts_the_drafts(self):
        assert num_base_tokens({"a": 5}, {"a": [1, 2, 3, 4]}, "a") == 1

    def test_no_drafts(self):
        assert num_base_tokens({"a": 3}, {}, "a") == 3

    def test_unscheduled_request(self):
        assert num_base_tokens({}, {}, "a") == 0


class TestResolvePropagatedTokenWrite:
    # The payload reaches back past the tip, so the slice is picked by position.
    def test_plain_decode_writes_the_newest_token(self):
        # Payload covers [16, 21); only position 20 is missing.
        got = resolve_propagated_token_write(20, 20, 1, [16, 17, 18, 19, 20])
        assert got == (21, [20])

    def test_multi_accept_lag_fills_the_gap(self):
        # A prior verify accepted 2, so the cursor sits 2 behind num_computed.
        got = resolve_propagated_token_write(18, 20, 1, [16, 17, 18, 19, 20])
        assert got == (21, [18, 19, 20])

    def test_nothing_to_write_when_cursor_is_at_the_tip(self):
        assert resolve_propagated_token_write(21, 20, 1, [19, 20]) is None

    def test_payload_shorter_than_the_span_asserts(self):
        # Cursor 15 needs [15, 21) but the payload only reaches back to 19.
        with pytest.raises(AssertionError, match="shorter than the write span"):
            resolve_propagated_token_write(15, 20, 1, [19, 20])


class TestDecodeBatchSize:
    def test_splits_across_pp_stages(self):
        assert decode_batch_size(8, 2) == 4

    def test_pp1_is_identity(self):
        assert decode_batch_size(8, 1) == 8

    def test_floors(self):
        assert decode_batch_size(7, 2) == 3


class TestShouldDeferSpecStep:
    # Deferring leaves num_computed_tokens intact; scheduling does not.
    def test_inert_when_spec_is_off(self):
        assert should_defer_spec_step(0, [], -5) is False

    def test_drafts_held_defers_without_an_anchor(self):
        # 2 scheduled, both drafts -> no committed token to verify against.
        assert should_defer_spec_step(4, [1, 2], 2) is True

    def test_drafts_held_admits_with_an_anchor(self):
        # 3 scheduled, 2 drafts -> one committed token leads them.
        assert should_defer_spec_step(4, [1, 2], 3) is False

    def test_no_drafts_defers_only_on_negative(self):
        assert should_defer_spec_step(4, [], -1) is True
        # 0 is the caller's ordinary "nothing to do this step".
        assert should_defer_spec_step(4, [], 0) is False
        assert should_defer_spec_step(4, [], 1) is False


class TestSchedulerInit:
    def test_sub_block_caching_enabled_uses_rbln_manager(self):
        # prefix caching + eligible config + sub_block_size -> RBLNKVCacheManager.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True, block_size=16, sub_block_size=8
        )
        assert isinstance(sched.kv_cache_manager, RBLNKVCacheManager)

    def test_sub_block_size_defaults_to_max_num_batched_tokens(self, monkeypatch):
        # With VLLM_RBLN_SUB_BLOCK_CACHE and no explicit sub_block_size, the
        # scheduler uses max_num_batched_tokens as the sub_block_size.
        import vllm_rbln.envs as envs

        monkeypatch.setattr(envs, "VLLM_RBLN_SUB_BLOCK_CACHE", True)
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=1024,
            max_num_batched_tokens=128,
            max_model_len=2048,
        )
        assert isinstance(sched.kv_cache_manager, RBLNKVCacheManager)
        assert sched.kv_cache_manager.sub_block_size == 128

    def test_disabled_falls_back_to_base_manager(self):
        # prefix caching off -> plain KVCacheManager.
        sched = create_rbln_scheduler(enable_prefix_caching=False)
        assert not isinstance(sched.kv_cache_manager, RBLNKVCacheManager)

    def test_equal_block_and_sub_block_size_disables(self):
        # sub_block_size == block_size is ineligible -> plain manager.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True, block_size=16, sub_block_size=16
        )
        assert not isinstance(sched.kv_cache_manager, RBLNKVCacheManager)


class TestPendingRunnerBlockDeltas:
    @staticmethod
    def _scheduled_delta(sched, req):
        # A real non-empty KVCacheBlocks for a scheduled request.
        sched.add_request(req)
        sched.schedule()
        return sched.kv_cache_manager.get_blocks(req.request_id)

    def test_add_none_or_empty_is_noop(self):
        # None or an all-empty delta is not stored.
        sched = create_rbln_scheduler()
        sched._add_pending_runner_block_delta("x", None)
        sched._add_pending_runner_block_delta(
            "x", sched.kv_cache_manager.empty_kv_cache_blocks
        )
        assert "x" not in sched._pending_runner_block_deltas

    def test_add_stores_nonempty_delta(self):
        # A non-empty delta is stored under the request id.
        sched = create_rbln_scheduler()
        delta = self._scheduled_delta(sched, create_requests(1, num_tokens=10)[0])
        assert any(len(g) > 0 for g in delta.get_block_ids())
        sched._add_pending_runner_block_delta("k", delta)
        assert "k" in sched._pending_runner_block_deltas

    def test_add_accumulates_existing(self):
        # Adding twice accumulates (prev + new).
        sched = create_rbln_scheduler()
        delta = self._scheduled_delta(sched, create_requests(1, num_tokens=10)[0])
        single = sum(len(g) for g in delta.get_block_ids())
        assert single > 0
        sched._add_pending_runner_block_delta("k", delta)
        sched._add_pending_runner_block_delta("k", delta)
        stored = sched._pending_runner_block_deltas["k"]
        assert sum(len(g) for g in stored.get_block_ids()) == 2 * single

    def test_drain_attaches_to_cached_reqs_and_pops(self):
        # drain prepends the pending delta to req_to_new_blocks and pops it.
        sched = create_rbln_scheduler()
        req = create_requests(1, num_tokens=10)[0]
        delta = self._scheduled_delta(sched, req)
        sched._add_pending_runner_block_delta(req.request_id, delta)
        req_to_new_blocks = {
            req.request_id: sched.kv_cache_manager.empty_kv_cache_blocks
        }
        sched._drain_pending_runner_block_deltas([req], req_to_new_blocks)
        assert req.request_id not in sched._pending_runner_block_deltas
        assert any(
            len(g) > 0 for g in req_to_new_blocks[req.request_id].get_block_ids()
        )


class TestUpdateFromOutput:
    def test_asserts_rbln_scheduler_output(self):
        # A non-RBLNSchedulerOutput trips the assert on the first line.
        sched = create_rbln_scheduler()
        with pytest.raises(AssertionError):
            sched.update_from_output(object(), None)

    def test_calls_do_pending_indexing_with_rbln_manager(self):
        # With the RBLN manager, update_from_output runs sub-block indexing so
        # the full block's sub-blocks land in the index.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True, block_size=16, sub_block_size=8
        )
        req = create_requests(1, num_tokens=16, block_size=16)[0]
        sched.add_request(req)
        out = sched.schedule()
        sched.update_from_output(out, make_model_runner_output(out, 1))
        idx = sched.kv_cache_manager._group_infos[0].sub_block_index
        assert len(idx._block_hashes) > 0

    def test_non_rbln_manager_skips_sub_block_steps(self):
        # With a plain manager, update_from_output completes without touching the
        # sub-block machinery.
        sched = create_rbln_scheduler(enable_prefix_caching=False)
        req = create_requests(1, num_tokens=10)[0]
        sched.add_request(req)
        out = sched.schedule()
        sched.update_from_output(out, make_model_runner_output(out, 1))
        assert not isinstance(sched.kv_cache_manager, RBLNKVCacheManager)


class TestTrySubBlockMatch:
    @staticmethod
    def _seeded_scheduler():
        # RBLN manager seeded (via its own prefill flow) with a cached block
        # whose first sub-block a query request will share.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=16,
            sub_block_size=8,
            max_num_batched_tokens=128,
        )
        m = sched.kv_cache_manager
        seed = make_request("seed", list(range(16)), 16)
        prefill_request(m, seed)
        m.free(seed)
        return sched

    def test_no_rbln_manager_returns_none(self):
        sched = create_rbln_scheduler(enable_prefix_caching=False)
        req = create_requests(1)[0]
        assert sched._try_sub_block_match(req, 0, 0) == (None, 0)

    def test_match_wins_on_ge_connector_tokens(self):
        # match.num_tokens >= external -> match wins (ties favor local copy).
        sched = self._seeded_scheduler()
        query = make_request("q", list(range(8)) + [100] * 16, 16)
        _, local = sched.kv_cache_manager.get_computed_blocks(query)
        match, n = sched._try_sub_block_match(query, local, 8)
        assert match is not None
        assert n == 8
        sched.kv_cache_manager.release_sub_block_match(match)

    def test_connector_better_releases_and_returns_none(self):
        # external > match -> the match is released and (None, 0) returned.
        sched = self._seeded_scheduler()
        query = make_request("q", list(range(8)) + [100] * 16, 16)
        _, local = sched.kv_cache_manager.get_computed_blocks(query)
        assert sched._try_sub_block_match(query, local, 12) == (None, 0)

    def test_no_match_returns_none(self):
        # No sub-block match at all -> (None, 0).
        sched = self._seeded_scheduler()
        query = make_request("q", [500] * 16, 16)
        _, local = sched.kv_cache_manager.get_computed_blocks(query)
        assert sched._try_sub_block_match(query, local, 0) == (None, 0)


class TestIsPrefill:
    def test_is_prefill_boundary(self):
        # num_computed < num_tokens - 1 is prefill; the last-token point is not.
        req = create_requests(1, num_tokens=10)[0]
        req.num_computed_tokens = 5
        assert is_prefill(req)
        req.num_computed_tokens = req.num_tokens - 1
        assert not is_prefill(req)


class TestScheduleBasic:
    def test_admits_and_counts_tokens(self):
        # add -> schedule returns RBLNSchedulerOutput scheduling one prefill with
        # the exact prompt length (also smokes the ported schedule() on 0.22.0).
        sched = create_rbln_scheduler()
        reqs = create_requests(3, num_tokens=10)
        for r in reqs:
            sched.add_request(r)
        out = sched.schedule()
        assert isinstance(out, RBLNSchedulerOutput)
        assert len(out.scheduled_new_reqs) == 1
        _, n = next(iter(out.num_scheduled_tokens.items()))
        assert n == 10

    def test_waiting_running_transition(self):
        # The scheduled request moves to running; the rest stay waiting.
        sched = create_rbln_scheduler()
        for r in create_requests(3, num_tokens=10):
            sched.add_request(r)
        sched.schedule()
        assert len(sched.running) == 1
        assert len(sched.waiting) == 2

    def test_chunked_prefill_across_steps(self):
        # A prompt longer than max_num_batched_tokens is chunked across steps.
        sched = create_rbln_scheduler(max_num_batched_tokens=256)
        req = create_requests(1, num_tokens=500)[0]
        sched.add_request(req)
        out = sched.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 256
        sched.update_from_output(out, make_model_runner_output(out))
        out = sched.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 244
        sched.update_from_output(out, make_model_runner_output(out, 0))
        out = sched.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 1


class TestSchedulePrefillBatchLimit:
    def test_only_one_prefill_per_step(self):
        # RBLN difference: at most one prefill scheduled per step.
        sched = create_rbln_scheduler()
        for r in create_requests(5, num_tokens=10):
            sched.add_request(r)
        out = sched.schedule()
        assert len(out.scheduled_new_reqs) == 1
        assert len(out.num_scheduled_tokens) == 1


class TestScheduleNoMixedBatching:
    def test_prefill_evicts_running_decode(self):
        # RBLN difference: a new prefill evicts running decodes (no mixing).
        sched = create_rbln_scheduler(
            max_num_batched_tokens=128, block_size=16, num_blocks=10000
        )
        req_a = create_requests(1, num_tokens=64, req_ids=["A"])[0]
        advance_to_decode(sched, req_a)
        req_b = create_requests(1, num_tokens=64, req_ids=["B"])[0]
        sched.add_request(req_b)
        out = sched.schedule()
        assert len(out.scheduled_new_reqs) == 1
        assert req_a.request_id not in out.num_scheduled_tokens
        assert req_b.request_id in out.num_scheduled_tokens


class TestScheduleDecodeBatchLimit:
    def test_decode_batch_capped_by_pipeline_parallel(self):
        # RBLN difference: decode batch capped at max_num_seqs // pp_size.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            pipeline_parallel_size=2,
            block_size=16,
            num_blocks=10000,
        )
        for r in create_requests(4, num_tokens=10, req_ids=["A", "B", "C", "D"]):
            advance_to_decode(sched, r)
        assert len(sched.running) == 4
        out = sched.schedule()
        # cap = max_num_seqs // pp = 2.
        assert len(out.num_scheduled_tokens) == 2

    def test_waiting_decode_ready_join_respects_pipeline_parallel_cap(self):
        # A full prefix-cache match joins the decode batch straight from the
        # waiting queue, bypassing the running loop -- it must still be capped,
        # or the combined batch overflows the compiled decode bucket.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            pipeline_parallel_size=2,
            enable_prefix_caching=True,
            block_size=16,
            num_blocks=10000,
        )
        for req in (
            make_request("A", list(range(16)), 16, max_tokens=50),
            make_request("B", list(range(100, 116)), 16, max_tokens=50),
        ):
            advance_to_decode(sched, req)
        assert len(sched.running) == 2

        # C matches A's cached 16-token block, so it is decode-ready on arrival.
        c = make_request("C", list(range(16)) + [999], 16, max_tokens=50)
        sched.add_request(c)
        out = sched.schedule()

        assert len(out.num_scheduled_tokens) == 2
        assert c.request_id not in out.num_scheduled_tokens


class TestSchedulePrefillAllocation:
    def test_prefill_not_scheduled_when_full_prompt_cannot_fit(self):
        # KV is reserved for the whole prompt: 500 tokens need 32 blocks and only
        # ~19 are usable, so nothing is scheduled even though one chunk would fit.
        sched = create_rbln_scheduler(
            max_num_batched_tokens=128, block_size=16, num_blocks=20
        )
        req = create_requests(1, num_tokens=500, block_size=16)[0]
        sched.add_request(req)
        out = sched.schedule()
        assert req.request_id not in out.num_scheduled_tokens
        # With ample blocks the same request schedules its first full chunk.
        sched2 = create_rbln_scheduler(
            max_num_batched_tokens=128, block_size=16, num_blocks=10000
        )
        req2 = create_requests(1, num_tokens=500, block_size=16)[0]
        sched2.add_request(req2)
        out2 = sched2.schedule()
        assert out2.num_scheduled_tokens[req2.request_id] == 128

    def test_new_prefill_uses_full_budget_when_decode_running(self):
        # The new prefill evicts the running decode and recovers the full budget,
        # so its first chunk is max_num_batched_tokens, not budget-minus-decode.
        max_num_batched_tokens = 128
        sched = create_rbln_scheduler(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=4,
            block_size=16,
            num_blocks=10000,
        )
        req_a = create_requests(1, num_tokens=64, req_ids=["A"])[0]
        advance_to_decode(sched, req_a)
        req_b = create_requests(1, num_tokens=500, req_ids=["B"])[0]
        sched.add_request(req_b)
        out = sched.schedule()
        assert req_a.request_id not in out.num_scheduled_tokens
        assert out.num_scheduled_tokens[req_b.request_id] == max_num_batched_tokens

    def test_partial_block_is_cached_once_decode_fills_it(self):
        # A prompt that stops mid-block leaves it uncached until decode supplies
        # the missing token.
        sched = create_rbln_scheduler(
            block_size=16,
            max_num_batched_tokens=16,
            max_model_len=64,
            enable_prefix_caching=True,
            num_blocks=100,
        )
        req = make_request("a", list(range(15)), 16, max_tokens=2)
        sched.add_request(req)
        out = sched.schedule()

        block = sched.kv_cache_manager.get_blocks("a").blocks[0][0]
        assert block.block_hash is None

        sched.update_from_output(out, make_model_runner_output(out, 0))
        sched.schedule()

        assert block.block_hash is not None


class TestDelayedBlockCaching:
    def test_each_chunk_caches_exactly_its_own_blocks(self):
        # delay_cache_blocks=True everywhere and caching only after the step is
        # finalized, so the scheduler can still change its mind mid-step.
        num_blocks = 4
        sched = create_rbln_scheduler(
            block_size=16,
            max_num_batched_tokens=16,
            max_model_len=128,
            enable_prefix_caching=True,
            num_blocks=100,
        )
        req = make_request("a", list(range(16 * num_blocks)), 16, max_tokens=1)
        sched.add_request(req)

        for step in range(num_blocks):
            out = sched.schedule()
            blocks = sched.kv_cache_manager.get_blocks("a").blocks[0]
            # The whole prompt is reserved from the first chunk on.
            assert len(blocks) == num_blocks
            assert all(block.ref_cnt == 1 for block in blocks)
            cached = [block for block in blocks if block.block_hash is not None]
            assert len(cached) == step + 1, f"chunk {step}"
            sched.update_from_output(out, make_model_runner_output(out))


class TestScheduleSpecDecodeCap:
    _BS = 1024
    _NUM_BLOCKS = 100
    _MAX_NUM_SEQS = 10

    def _scheduler(self, **kwargs):
        return create_rbln_scheduler(
            block_size=self._BS,
            num_blocks=self._NUM_BLOCKS,
            max_num_seqs=self._MAX_NUM_SEQS,
            num_speculative_tokens=4,
            **kwargs,
        )

    def _request(self, num_tokens, req_id):
        return create_requests(
            1,
            num_tokens=num_tokens,
            block_size=self._BS,
            max_tokens=2048,
            req_ids=[req_id],
        )[0]

    def test_spec_at_block_boundary_keeps_full_spec(self):
        # prompt == block_size -> remaining_in_block == block_size; the 4 spec
        # tokens fit so num_scheduled == 1 (decode) + 4 (spec).
        sched = self._scheduler()
        req = self._request(1024, "A")
        advance_to_decode(sched, req)
        req.spec_token_ids = [1] * 4
        out = sched.schedule()
        rid = req.request_id
        assert out.num_scheduled_tokens[rid] == 5
        assert len(out.scheduled_spec_decode_tokens[rid]) == 4

    def test_spec_clamped_when_crossing_block_boundary(self):
        # A decode + spec window crossing into the next block is clamped to the
        # block remainder: prompt 1020 leaves 4 tokens, so 1 + 4 becomes 1 + 3.
        sched = self._scheduler()
        req = self._request(1020, "A")
        advance_to_decode(sched, req)
        req.spec_token_ids = [1] * 4
        out = sched.schedule()
        rid = req.request_id
        assert out.num_scheduled_tokens[rid] == 4
        assert len(out.scheduled_spec_decode_tokens[rid]) == 3

    def test_no_spec_tokens_no_retroactive_trim(self):
        # Without spec tokens the retroactive trim is skipped even when a decode
        # sits mid-block; each running decode schedules exactly 1 token.
        sched = self._scheduler()
        req_a = self._request(1024, "A")
        req_b = self._request(1023, "B")
        advance_to_decode(sched, req_a)
        advance_to_decode(sched, req_b)
        out = sched.schedule()
        assert out.num_scheduled_tokens[req_a.request_id] == 1
        assert out.num_scheduled_tokens[req_b.request_id] == 1
        assert out.scheduled_spec_decode_tokens == {}


class TestBackfillCrossBlockNoSpec:
    # A decode-ready request entering exactly at a block boundary cannot backfill
    # num_spec past tokens without crossing into the previous block, so the whole
    # decode batch is forced to qlen=1 -- even a running decode that is safe.
    _BS = 16
    _NUM_SPEC = 4

    def _scheduler(self):
        return create_rbln_scheduler(
            num_speculative_tokens=self._NUM_SPEC,
            block_size=self._BS,
            num_blocks=256,
            max_num_seqs=8,
            enable_prefix_caching=True,
        )

    def _decode_ready_at_boundary(self, sched, seed_id, req_id):
        # prompt = block_size + 1 with the first block prefix-matched, so the
        # request joins decode-ready right at the boundary (unsafe backfill).
        seed, req = create_requests(
            2,
            num_tokens=self._BS + 1,
            block_size=self._BS,
            max_tokens=64,
            same_prompt=True,
            req_ids=[seed_id, req_id],
        )
        advance_to_decode(sched, seed)  # caches the shared first block
        sched.add_request(req)
        return req

    def test_unsafe_decode_ready_peer_forces_batch_no_spec(self):
        sched = self._scheduler()

        # A running decode that, on its own, keeps full spec (block_size is far
        # larger than num_spec, so its own backfill always fits in-block).
        running = create_requests(
            1, num_tokens=20, block_size=self._BS, max_tokens=64, req_ids=["R"]
        )[0]
        advance_to_decode(sched, running)
        running.spec_token_ids = [1, 2, 3, 4]
        out = sched.schedule()
        assert out.num_scheduled_tokens["R"] == 1 + self._NUM_SPEC
        assert out.scheduled_spec_decode_tokens["R"] == [1, 2, 3, 4]
        sched.update_from_output(out, make_model_runner_output(out, 1))

        # Introduce an unsafe decode-ready peer at a block boundary.
        self._decode_ready_at_boundary(sched, "seed", "A")
        running.spec_token_ids = [1, 2, 3, 4]
        out = sched.schedule()

        # The unsafe peer forces the whole decode batch to no-spec: the running
        # decode loses its otherwise-valid drafts and drops to a single token.
        assert out.num_scheduled_tokens["R"] == 1
        assert "R" not in out.scheduled_spec_decode_tokens
        assert out.num_scheduled_tokens["A"] == 1


class TestStrandedBlockDelta:
    # A block allocated exactly on the prefill->decode transition and then evicted
    # by the no-mixed-batching path is stashed and re-emitted on resume.
    @staticmethod
    def _stash_evicted_block():
        block_size = 16
        sched = create_rbln_scheduler(
            max_num_batched_tokens=128,
            max_num_seqs=4,
            block_size=block_size,
            num_blocks=10000,
        )
        req_a = create_requests(
            1, num_tokens=block_size, block_size=block_size, req_ids=["A"]
        )[0]
        sched.add_request(req_a)
        out1 = sched.schedule()
        assert out1.num_scheduled_tokens[req_a.request_id] == block_size
        sched.update_from_output(out1, make_model_runner_output(out1, 1))
        assert req_a.num_computed_tokens == block_size
        req_b = create_requests(
            1, num_tokens=block_size, block_size=block_size, req_ids=["B"]
        )[0]
        sched.add_request(req_b)
        out2 = sched.schedule()
        assert req_a.request_id not in out2.num_scheduled_tokens
        assert req_b.request_id in out2.num_scheduled_tokens
        assert req_a.request_id in sched._pending_runner_block_deltas
        return sched, req_a, out2

    def test_delta_reemitted_when_req_dropped_then_resumed(self):
        sched, req_a, out2 = self._stash_evicted_block()
        stashed = sched._pending_runner_block_deltas[req_a.request_id].get_block_ids()
        assert any(len(g) > 0 for g in stashed)
        sched.update_from_output(out2, make_model_runner_output(out2, 1))
        out3 = sched.schedule()
        assert req_a.request_id in out3.num_scheduled_tokens
        cached = out3.scheduled_cached_reqs
        reemitted = cached.new_block_ids[cached.req_ids.index(req_a.request_id)]
        assert reemitted is not None
        assert any(len(g) > 0 for g in reemitted)
        assert req_a.request_id not in sched._pending_runner_block_deltas
        flat_reemitted = [b for g in reemitted for b in g]
        flat_stashed = [b for g in stashed for b in g]
        assert flat_stashed
        assert all(b in flat_reemitted for b in flat_stashed)

    def test_cleaned_up_on_finish(self):
        # A stashed delta is dropped if the request finishes before rescheduling.
        sched, req_a, _ = self._stash_evicted_block()
        sched.finish_requests(req_a.request_id, RequestStatus.FINISHED_ABORTED)
        assert req_a.request_id not in sched._pending_runner_block_deltas

    def test_cleaned_up_on_preempt(self):
        # A stashed delta is dropped when the request is preempted (its blocks,
        # including the stashed one, are freed).
        sched, req_a, _ = self._stash_evicted_block()
        assert req_a in sched.running
        sched.running.remove(req_a)
        sched._preempt_request(req_a, 0.0)
        assert req_a.status == RequestStatus.PREEMPTED
        assert req_a.request_id not in sched._pending_runner_block_deltas


class TestStopping:
    def test_eos_stops_request(self):
        # An EOS sample finishes the request and removes it from running.
        sched = create_rbln_scheduler()
        req = create_requests(1, num_tokens=10, max_tokens=20)[0]
        sched.add_request(req)
        out = sched.schedule()
        sched.update_from_output(out, make_model_runner_output(out, EOS_TOKEN_ID))
        assert req.is_finished()
        assert req not in sched.running

    def test_max_tokens_stops_request(self):
        # Reaching max_tokens finishes the request.
        sched = create_rbln_scheduler()
        req = create_requests(1, num_tokens=10, max_tokens=1, ignore_eos=True)[0]
        sched.add_request(req)
        for _ in range(10):
            out = sched.schedule()
            if not out.num_scheduled_tokens:
                break
            sched.update_from_output(out, make_model_runner_output(out, 0))
            if req.is_finished():
                break
        assert req.is_finished()


class TestPreemption:
    def test_kv_exhaustion_preempts(self):
        # Under KV pressure a running request is preempted; the preempted
        # request still receives its sampled token from the in-flight step.
        sched = create_rbln_scheduler(
            max_num_batched_tokens=100,
            block_size=16,
            num_blocks=11,
            enable_prefix_caching=False,
        )
        reqs = create_requests(2, num_tokens=80, block_size=16)
        sched.add_request(reqs[0])
        out0 = sched.schedule()
        assert len(out0.scheduled_new_reqs[0].block_ids[0]) == 5
        sched.add_request(reqs[1])
        out1 = sched.schedule()
        assert len(out1.scheduled_new_reqs[0].block_ids[0]) == 5
        sched.update_from_output(out0, make_model_runner_output(out0, 0))
        sched.schedule()
        assert len(sched.running) == 1
        assert sched.running[0] == reqs[0]
        assert reqs[1].status == RequestStatus.PREEMPTED
        sched.update_from_output(out1, make_model_runner_output(out1, 42))
        assert reqs[1].output_token_ids[0] == 42


class TestPortDriftConformance:
    # The core value of this file: catch drift between the ported schedule() and
    # the installed vllm.
    def test_full_lifecycle_runs_end_to_end(self):
        # A whole prefill -> decode -> stop lifecycle on the installed vllm: a
        # wider surface than one schedule() call, so drift shows up early.
        sched = create_rbln_scheduler()
        reqs = create_requests(3, num_tokens=10, max_tokens=3)
        for r in reqs:
            sched.add_request(r)
        for _ in range(40):
            if not sched.running and not sched.waiting:
                break
            out = sched.schedule()
            sched.update_from_output(out, make_model_runner_output(out, 0))
        assert not sched.running
        assert not sched.waiting

    def test_override_signatures_match_upstream(self):
        # The overrides stay call-compatible with the upstream Scheduler base.
        for name in ("update_from_output", "_preempt_request", "_free_request"):
            base = inspect.signature(getattr(Scheduler, name))
            override = inspect.signature(getattr(RBLNScheduler, name))
            assert list(override.parameters) == list(base.parameters)

    def test_rbln_scheduler_output_is_valid_subclass(self):
        # RBLNSchedulerOutput subclasses SchedulerOutput and adds the copy-ops
        # field while keeping the base fields.
        assert issubclass(RBLNSchedulerOutput, SchedulerOutput)
        fields = {f.name for f in dataclasses.fields(RBLNSchedulerOutput)}
        assert "kv_cache_copy_ops" in fields
        assert "num_scheduled_tokens" in fields


# The classes below drive complete generation runs rather than single steps, to
# check the invariants hold at every step.


class TestFullRunInvariants:
    def test_multi_request_full_drain(self):
        # Six requests to completion: every step homogeneous, prefill batch <= 1,
        # decode batch within max_num_seqs // pp.
        sched = create_rbln_scheduler(max_num_seqs=4, block_size=16, num_blocks=10000)
        reqs = create_requests(6, num_tokens=10, max_tokens=4)
        for r in reqs:
            sched.add_request(r)

        def check(out):
            scheduled = list(out.num_scheduled_tokens)
            phases = {is_prefill(sched.requests[rid]) for rid in scheduled}
            assert len(phases) <= 1, "a step mixed prefill and decode"
            n_prefill = sum(1 for rid in scheduled if is_prefill(sched.requests[rid]))
            assert n_prefill <= 1, "more than one prefill in a step"
            assert len(scheduled) <= 4, "decode batch exceeded the cap"

        _drain(sched, per_step=check)
        assert all(r.is_finished() for r in reqs)
        assert sched.running == []

    def test_scheduled_token_count_matches_output(self):
        # Across a run, every request in num_scheduled_tokens must appear in the
        # model runner output mapping (no orphaned / dropped requests).
        sched = create_rbln_scheduler(max_num_seqs=4, block_size=16, num_blocks=10000)
        for r in create_requests(4, num_tokens=10, max_tokens=3):
            sched.add_request(r)

        def check(out):
            mro = make_model_runner_output(out, 0)
            assert set(out.num_scheduled_tokens) == set(mro.req_ids)

        _drain(sched, per_step=check)


class TestRunningQueueCapacity:
    # Borrowed from upstream test_async_scheduler.test_running_queue: how
    # concurrency is capped over a run. RBLN caps running at max_num_seqs // pp.
    def test_capacity_limited_by_max_num_seqs(self):
        sched = create_rbln_scheduler(max_num_seqs=2, block_size=16, num_blocks=10000)
        reqs = create_requests(5, num_tokens=10, max_tokens=6)
        for r in reqs:
            sched.add_request(r)
        peak = 0

        def check(out):
            nonlocal peak
            peak = max(peak, len(sched.running))

        _drain(sched, per_step=check)
        assert peak <= 2, "running set exceeded max_num_seqs"
        assert all(r.is_finished() for r in reqs)

    def test_capacity_limited_by_blocks(self):
        # With scarce blocks the scheduler cannot run all requests at once, yet
        # it still drains the whole queue (exact counts are allocator-specific).
        sched = create_rbln_scheduler(
            max_num_seqs=8,
            block_size=16,
            num_blocks=6,
            max_num_batched_tokens=128,
        )
        reqs = create_requests(4, num_tokens=30, max_tokens=3, block_size=16)
        for r in reqs:
            sched.add_request(r)
        peak = 0

        def check(out):
            nonlocal peak
            peak = max(peak, len(sched.running))

        _drain(sched, per_step=check)
        assert peak < len(reqs), "blocks did not throttle concurrency"
        assert all(r.is_finished() for r in reqs)


class TestPreemptResumeCompletion:
    # Extends TestPreemption past the preemption event: the victim must resume
    # and finish once KV frees up.
    def test_preempted_request_resumes_and_completes(self):
        # Three prompts growing through decode with only 7 usable blocks forces a
        # genuine preemption.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            block_size=16,
            num_blocks=8,
            max_num_batched_tokens=128,
            enable_prefix_caching=False,
        )
        reqs = create_requests(
            3, num_tokens=10, max_tokens=30, block_size=16, ignore_eos=True
        )
        for r in reqs:
            sched.add_request(r)
        saw_preemption = False

        def check(out):
            nonlocal saw_preemption
            if any(r.status == RequestStatus.PREEMPTED for r in reqs):
                saw_preemption = True

        _drain(sched, per_step=check)
        assert saw_preemption, "expected a preemption under KV pressure"
        assert all(r.is_finished() for r in reqs)


class TestStoppingVariety:
    # Complements TestStopping (eos, max_tokens) with the remaining stop paths.
    def test_stop_token_id_finishes_request(self):
        sched = create_rbln_scheduler()
        req = create_requests(
            1, num_tokens=10, max_tokens=20, stop_token_ids=[7], ignore_eos=True
        )[0]
        sched.add_request(req)
        out = sched.schedule()  # prefill
        sched.update_from_output(out, make_model_runner_output(out, 0))
        out = sched.schedule()  # first decode; sample the stop token
        sched.update_from_output(out, make_model_runner_output(out, 7))
        assert req.is_finished()

    def test_ignore_eos_does_not_stop_on_eos(self):
        sched = create_rbln_scheduler()
        req = create_requests(1, num_tokens=10, max_tokens=5, ignore_eos=True)[0]
        sched.add_request(req)
        out = sched.schedule()  # prefill
        sched.update_from_output(out, make_model_runner_output(out, EOS_TOKEN_ID))
        out = sched.schedule()  # decode with EOS sampled
        sched.update_from_output(out, make_model_runner_output(out, EOS_TOKEN_ID))
        assert not req.is_finished(), "ignore_eos must not stop on EOS"
        # It still stops on max_tokens.
        _drain(sched, token=EOS_TOKEN_ID)
        assert req.is_finished()


# Behaviours from upstream tests/v1/core/test_scheduler.py that apply to
# RBLNScheduler.


class TestMinTokens:
    def test_min_tokens_suppresses_eos(self):
        # min_tokens=3: an EOS sample must NOT stop the request until it has
        # generated 3 output tokens.
        sched = create_rbln_scheduler()
        req = create_requests(1, num_tokens=10, max_tokens=20, min_tokens=3)[0]
        sched.add_request(req)
        out = sched.schedule()  # prefill -> 1 output token
        sched.update_from_output(out, make_model_runner_output(out, 0))
        out = sched.schedule()  # decode EOS -> 2 tokens, still < min_tokens
        sched.update_from_output(out, make_model_runner_output(out, EOS_TOKEN_ID))
        assert not req.is_finished()
        out = sched.schedule()  # decode EOS -> 3 tokens, min reached -> stop
        sched.update_from_output(out, make_model_runner_output(out, EOS_TOKEN_ID))
        assert req.is_finished()
        assert len(req.output_token_ids) == 3


class TestMemoryFreed:
    def test_all_blocks_freed_after_run(self):
        # After every request finishes, the block pool must return to its initial
        # free count (no block leak) -- upstream test_memory_leak.
        sched = create_rbln_scheduler(num_blocks=100, block_size=16)
        pool = sched.kv_cache_manager.block_pool
        free0 = pool.get_num_free_blocks()
        for r in create_requests(5, num_tokens=10, max_tokens=4):
            sched.add_request(r)
        _drain(sched)
        assert sched.get_num_unfinished_requests() == 0
        assert pool.get_num_free_blocks() == free0


class TestFcfsOrdering:
    def test_admits_in_arrival_order(self):
        # prefill batch = 1 admits requests one per step; under FCFS that must be
        # arrival order regardless of how the waiting queue is drained.
        sched = create_rbln_scheduler(max_num_seqs=4, num_blocks=10000)
        reqs = create_requests(
            4, num_tokens=10, max_tokens=4, req_ids=["A", "B", "C", "D"]
        )
        for r in reqs:
            sched.add_request(r)
        admitted: list[str] = []

        def check(out):
            admitted.extend(nr.req_id for nr in out.scheduled_new_reqs)

        _drain(sched, per_step=check)
        assert admitted == ["A", "B", "C", "D"]


class TestPriorityScheduling:
    # RBLNScheduler.schedule() honors SchedulingPolicy.PRIORITY (lower priority
    # value = higher priority).
    def test_higher_priority_scheduled_first(self):
        sched = create_rbln_scheduler(
            max_num_seqs=1, policy="priority", num_blocks=10000
        )
        low = create_requests(1, req_ids=["low"], priority=10)[0]
        high = create_requests(1, req_ids=["high"], priority=0)[0]
        sched.add_request(low)  # arrives first
        sched.add_request(high)  # arrives second, higher priority
        out = sched.schedule()
        # The higher-priority request wins the single prefill slot despite the
        # later arrival.
        assert [nr.req_id for nr in out.scheduled_new_reqs] == ["high"]

    def test_preemption_victim_is_lowest_priority(self):
        # Under KV pressure only lower-priority requests are preempted; the
        # highest-priority request is never a victim and all complete.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            policy="priority",
            num_blocks=8,
            max_num_batched_tokens=128,
        )
        high = create_requests(
            1, req_ids=["high"], num_tokens=10, max_tokens=30, priority=0
        )[0]
        lows = create_requests(
            2, req_ids=["low0", "low1"], num_tokens=10, max_tokens=30, priority=10
        )
        reqs = [high, *lows]
        for r in reqs:
            sched.add_request(r)
        victims: set[str] = set()

        def check(out):
            victims.update(
                r.request_id for r in reqs if r.status == RequestStatus.PREEMPTED
            )

        _drain(sched, per_step=check)
        assert "high" not in victims, "the highest-priority request was preempted"
        assert victims, "expected a preemption under KV pressure"
        assert all(r.is_finished() for r in reqs)


class TestSpecDecodeRetroactiveTrim:
    # A decode-ready join whose backfill window would cross a block boundary
    # forces the whole decode batch to no-spec. Reachable only via a prefix
    # match, the one way a waiting request reaches decode un-prefilled.
    @staticmethod
    def _running_decoder_with_spec():
        # req0: a running decode carrying 4 spec tokens, positioned mid-block so
        # it would normally keep the full 1 + 4 spec query.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=16,
            sub_block_size=8,
            num_speculative_tokens=4,
            max_num_batched_tokens=128,
            num_blocks=10000,
        )
        req0 = make_request("0", list(range(16)), 16, max_tokens=50)
        sched.add_request(req0)
        for _ in range(4):
            out = sched.schedule()
            sched.update_from_output(out, make_model_runner_output(out, 0))
        req0.spec_token_ids = [1] * 4
        return sched, req0

    def test_unsafe_decode_ready_join_trims_whole_batch(self):
        # req1 matches the full 16-token prefix -> joins at the block start, so
        # its backfill would cross the boundary -> the batch loses its spec.
        sched, req0 = self._running_decoder_with_spec()
        req1 = make_request("1", list(range(16)) + [999], 16, max_tokens=50)
        sched.add_request(req1)
        out = sched.schedule()
        assert out.num_scheduled_tokens[req1.request_id] == 1  # decode-ready join
        assert out.num_scheduled_tokens[req0.request_id] == 1  # retroactively trimmed
        assert not out.scheduled_spec_decode_tokens.get(req0.request_id)

    def test_safe_decode_ready_join_keeps_spec(self):
        # req1 matches only the first sub-block -> joins mid-block where the
        # backfill fits, proving it is the crossing that trims, not the join.
        sched, req0 = self._running_decoder_with_spec()
        req1 = make_request("1", list(range(8)) + [999], 16, max_tokens=50)
        sched.add_request(req1)
        out = sched.schedule()
        assert out.num_scheduled_tokens[req1.request_id] == 1
        assert out.num_scheduled_tokens[req0.request_id] == 5  # spec preserved
        assert len(out.scheduled_spec_decode_tokens[req0.request_id]) == 4


class TestNoSpecDuringPrefill:
    def test_prefill_chunk_schedules_no_spec_tokens(self):
        # A still-prefilling request never gets spec tokens: the spec block is
        # gated on `not is_prefill` even when spec_token_ids are set.
        sched = create_rbln_scheduler(
            num_speculative_tokens=4,
            block_size=16,
            max_num_batched_tokens=32,
            max_model_len=2048,
            num_blocks=10000,
        )
        req = create_requests(
            1, num_tokens=100, block_size=16, max_tokens=50, req_ids=["0"]
        )[0]
        sched.add_request(req)
        out = sched.schedule()  # first prefill chunk
        assert is_prefill(req)
        sched.update_from_output(out, make_model_runner_output(out, 0))

        req.spec_token_ids = [1] * 4  # spec present, but still prefilling
        out2 = sched.schedule()
        assert is_prefill(req)
        assert out2.num_scheduled_tokens[req.request_id] == 32  # a full prefill chunk
        assert req.request_id not in out2.scheduled_spec_decode_tokens
