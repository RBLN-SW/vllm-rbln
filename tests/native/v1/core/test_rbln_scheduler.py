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

# RBLNScheduler differences from upstream schedule(): no mixed batching, prefill
# batch of 1, capped decode batch, block-boundary spec, sub-block copy.

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.native.v1.core.utils import (
    EOS_TOKEN_ID,
    MockKVConfig,
    _drain,
    advance_to_decode,
    create_rbln_scheduler,
    create_requests,
    make_model_runner_output,
    make_request,
    prefill_request,
)
from vllm_rbln.v1.core.rbln_kv_cache_manager import RBLNKVCacheManager
from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler, RBLNSchedulerOutput
from vllm_rbln.v1.core.utils import is_prefill, step_is_prefill


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
        # Per-step decode batch is capped at max_num_seqs // pp_size, and the
        # shared budget also gates waiting-loop admissions. With max_num_seqs=4,
        # pp=2 (cap 2), advancing 4 requests one step at a time: the 4th's prefill
        # is held while two decodes saturate the step, so running settles at 3.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            pipeline_parallel_size=2,
            block_size=16,
            num_blocks=10000,
        )
        for r in create_requests(4, num_tokens=10, req_ids=["A", "B", "C", "D"]):
            advance_to_decode(sched, r)
        assert len(sched.running) == 3
        out = sched.schedule()
        # cap = max_num_seqs // pp = 2.
        assert len(out.num_scheduled_tokens) == 2

    def test_prefill_held_by_cap_is_not_starved(self):
        # A prefill held behind a saturated decode cap still enters within a
        # bounded number of steps: the gate holds it for a step, not forever,
        # because running decodes drain and free slots.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            pipeline_parallel_size=2,
            block_size=16,
            num_blocks=10000,
        )
        # Saturate the per-step decode cap (max_num_seqs // pp = 2).
        for r in create_requests(2, num_tokens=10, req_ids=["A", "B"]):
            advance_to_decode(sched, r)
        # A fresh prefill arrives while the cap is saturated.
        sched.add_request(create_requests(1, num_tokens=10, req_ids=["C"])[0])

        scheduled = set()

        def check(out):
            # Never mixed and never over the cap.
            assert len(out.num_scheduled_tokens) <= 2
            scheduled.update(out.num_scheduled_tokens)

        _drain(sched, per_step=check)
        # The held prefill was admitted and every request drained to completion.
        assert "C" in scheduled
        assert not sched.requests


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


def _check_phase_agrees(sched, out) -> bool:
    """The scheduler's phase for a step against the phase the runner derives from
    it, and that the step carries only one of the two. Returns the phase.

    schedule() advances num_computed_tokens by the scheduled chunk before it
    returns, so back the chunk out to reach the state the scheduler classified.
    """
    phases = set()
    for rid, chunk in out.num_scheduled_tokens.items():
        req = sched.requests[rid]
        phases.add(
            is_prefill(
                SimpleNamespace(
                    num_computed_tokens=req.num_computed_tokens - chunk,
                    num_tokens=req.num_tokens,
                )
            )
        )
    assert len(phases) <= 1, "a step mixed prefill and decode"
    assert step_is_prefill(out) is (phases == {True}), (
        f"phase {phases} but chunks {out.num_scheduled_tokens} "
        f"(spec {out.scheduled_spec_decode_tokens})"
    )
    return phases == {True}


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
            if _check_phase_agrees(sched, out):
                assert len(scheduled) == 1, "more than one request in a prefill step"
            assert len(scheduled) <= 4, "decode batch exceeded the cap"

        _drain(sched, per_step=check)
        assert all(r.is_finished() for r in reqs)
        assert sched.running == []

    @pytest.mark.parametrize(
        ("num_tokens", "max_num_batched_tokens"),
        [
            (10, 8192),  # prompt in one chunk
            (1, 8192),  # single-token prompt: no prefill step at all
            (33, 16),  # chunked, tail chunk of 1 token
            (32, 16),  # chunked, block-aligned
        ],
    )
    def test_step_phase_matches_the_scheduled_chunks(
        self, num_tokens, max_num_batched_tokens
    ):
        # What the runner derives from a step (step_is_prefill) has to be what the
        # scheduler decided (is_prefill), on every step of a run. The chunk sizes
        # here reach the cases where the two could come apart -- above all a
        # prefill whose last chunk is a single token.
        sched = create_rbln_scheduler(
            max_num_seqs=4,
            block_size=16,
            num_blocks=10000,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        for r in create_requests(3, num_tokens=num_tokens, max_tokens=4):
            sched.add_request(r)

        _drain(sched, per_step=lambda out: _check_phase_agrees(sched, out))

    def test_step_phase_matches_the_scheduled_chunks_under_spec_decode(self):
        # A verify step schedules 1 + num_spec tokens; only the base token counts,
        # so the step still reads as decode.
        sched = create_rbln_scheduler(
            max_num_seqs=4, block_size=16, num_blocks=10000, num_speculative_tokens=3
        )
        for r in create_requests(2, num_tokens=10, max_tokens=6):
            sched.add_request(r)

        _drain(sched, per_step=lambda out: _check_phase_agrees(sched, out))

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


class TestDecodeCapMachinery:
    # Covers the per-step decode-batch admission budget in v1/core/utils
    # (DecodeBatchBudget) and its wiring into schedule().

    def test_decode_budget_for_step_spreads_demand(self):
        # for_step: hard cap = max_num_seqs // pp; soft cap = max(1, ceil(demand/pp)).
        from vllm_rbln.v1.core.utils import DecodeBatchBudget

        # max_num_seqs=16, pp=2 -> hard 8; demand 6 -> soft ceil(6/2)=3.
        b = DecodeBatchBudget.for_step(16, 2, 6)
        b.admit(3)  # count == soft
        assert not b.can_admit()  # budgeted gate closes at the soft cap
        assert b.can_admit(apply_soft_cap=False)  # hard (8) still has room
        b.admit(5)  # count == hard
        assert not b.can_admit(apply_soft_cap=False)

        # No demand -> soft floored at 1 (never 0, which would wedge the budget).
        nb = DecodeBatchBudget.for_step(16, 2, 0)
        nb.admit()
        assert not nb.can_admit()

        # pp == 1 -> soft == demand, a no-op against the max_num_seqs hard cap.
        b1 = DecodeBatchBudget.for_step(16, 1, 5)
        b1.admit(5)
        assert not b1.can_admit()  # soft == demand == 5
        assert b1.can_admit(apply_soft_cap=False)  # hard == 16

    def test_decode_budget_hard_vs_soft_cap(self):
        # can_admit: the hard cap (compiled bucket ceiling) always applies; the
        # soft (spreading) cap only when apply_soft_cap. Demand-unbudgeted joins
        # (full local prefix / resumed-after-eviction) fill to the hard cap.
        from vllm_rbln.v1.core.utils import DecodeBatchBudget

        b = DecodeBatchBudget(hard_cap=8, soft_cap=2)
        b.admit(2)  # count == soft
        assert not b.can_admit()  # budgeted: gated at soft (2)
        assert b.can_admit(apply_soft_cap=False)  # unbudgeted: hard (8) has room
        b.admit(6)  # count == hard
        assert not b.can_admit(apply_soft_cap=False)  # hard cap reached
        assert not b.can_admit()

        # soft == hard: apply_soft_cap makes no difference.
        b2 = DecodeBatchBudget(hard_cap=8, soft_cap=8)
        b2.admit(8)
        assert not b2.can_admit()
        assert not b2.can_admit(apply_soft_cap=False)

    def test_decode_budget_discard(self):
        # discard() un-admits decodes dropped from the step so can_admit() is not
        # stopped early on a stale over-count (still-admitted decodes stay counted).
        from vllm_rbln.v1.core.utils import DecodeBatchBudget

        b = DecodeBatchBudget(hard_cap=2, soft_cap=2)
        b.admit(2)  # batch full at the cap
        assert not b.can_admit()  # gate closed
        b.discard()  # a scheduled decode is preempted -> one slot freed
        assert b.count == 1
        assert b.can_admit()  # gate reopens for another admit
        # reset() would instead zero the whole count (whole-batch eviction).
        b.reset()
        assert b.count == 0

    def test_decode_budget_discard_underflow_asserts(self):
        # A discard() without a matching admit would drive the count negative and
        # silently disable both caps -- the guard asserts instead.
        from vllm_rbln.v1.core.utils import DecodeBatchBudget

        b = DecodeBatchBudget(hard_cap=2, soft_cap=2)
        with pytest.raises(AssertionError, match="without a matching admit"):
            b.discard()  # count 0 -> would underflow
        b.admit()
        b.discard()  # matched -> count back to 0
        with pytest.raises(AssertionError):
            b.discard()  # underflow again

    def test_schedule_spreads_decode_across_microbatch(self):
        # Under PP the per-step decode batch is sized to ~ceil(active/pp),
        # spreading active decodes across microbatches instead of packing them
        # into one (which would idle the other PP stages).
        import math

        n = 6
        # max_num_seqs=16, pp=2 -> hard cap 8; n=6 active -> soft ceil(6/2)=3.
        sched = create_rbln_scheduler(
            max_num_seqs=16, pipeline_parallel_size=2, block_size=16
        )
        reqs = create_requests(
            n, num_tokens=32, block_size=16, req_ids=[f"d{i}" for i in range(n)]
        )
        for r in reqs:
            advance_to_decode(sched, r)
        running_decodes = sum(1 for r in sched.running if not is_prefill(r))
        out = sched.schedule()
        # At most ceil(active / pp) decodes admitted this step, and at least one.
        assert 1 <= len(out.num_scheduled_tokens) <= math.ceil(running_decodes / 2)

    def test_priority_preemption_discards_admitted_decode(self, monkeypatch):
        # When PRIORITY preempts an already-scheduled decode to free KV, the
        # scheduler discard()s it from the per-step budget so can_admit() isn't
        # stopped on a stale count. The victim was scheduled this step, so
        # discard() fires exactly once.
        from vllm_rbln.v1.core.utils import DecodeBatchBudget

        discard_calls = []
        orig_discard = DecodeBatchBudget.discard

        def spy(self, n=1):
            discard_calls.append(n)
            return orig_discard(self, n)

        monkeypatch.setattr(DecodeBatchBudget, "discard", spy)

        # num_blocks=4 -> 3 usable (block 0 is the null block): one block per
        # prefill (2) leaves exactly one free for a decode boundary block, so
        # only one of the two decodes can grow this step.
        sched = create_rbln_scheduler(
            max_num_batched_tokens=128,
            max_num_seqs=4,
            block_size=16,
            num_blocks=4,
            enable_prefix_caching=False,
            policy="priority",
        )
        victim, trigger = create_requests(
            2, num_tokens=16, block_size=16, req_ids=["victim", "trigger"]
        )
        # Higher priority VALUE == lower scheduling importance == preempted first.
        victim.priority = 1
        trigger.priority = 0
        for r in (victim, trigger):
            advance_to_decode(sched, r)

        out = sched.schedule()

        assert victim.status == RequestStatus.PREEMPTED
        assert trigger.request_id in out.num_scheduled_tokens
        assert discard_calls == [1], (
            "discard() must fire exactly once for the preempted already-scheduled "
            f"decode, got {discard_calls}"
        )


class TestDeferredBlockFree:
    # The ``defer_block_free`` fence (``sched_step_seq``): blocks of a request
    # aborted mid-step must not return to the pool until that step's output is
    # processed, because with several batches in flight (PP) a connector load can
    # refill blocks the in-flight step is still writing. On only when >1 batch is
    # in flight AND the instance is a KV consumer. The fence lives in the copied
    # schedule() and had no native unit coverage.

    def test_deferred_free_fenced_by_inflight_step(self):
        sched = create_rbln_scheduler(
            pipeline_parallel_size=2, use_kv_connector=MockKVConfig()
        )
        assert sched.defer_block_free

        request = create_requests(1)[0]
        sched.add_request(request)
        output = sched.schedule()
        assert output.total_num_scheduled_tokens > 0
        assert sched.sched_step_seq == 1
        assert request.last_sched_seq == 1

        block_pool = sched.kv_cache_manager.block_pool
        free_before = block_pool.get_num_free_blocks()
        sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        # Fenced: blocks stay out of the pool until the in-flight step drains.
        assert sched.deferred_frees, "blocks must be fenced, not freed"
        assert block_pool.get_num_free_blocks() == free_before

        sched.update_from_output(output, make_model_runner_output(output, 0))
        assert sched.processed_step_seq == 1
        assert not sched.deferred_frees
        assert block_pool.get_num_free_blocks() > free_before

    def test_no_deferred_free_without_multiple_inflight_batches(self):
        # A guard rather than a test of the fence itself: with a single batch in
        # flight (no PP) the whole mechanism has to stay inert even though a KV
        # connector is present, so freeing is immediate and neither counter moves.
        sched = create_rbln_scheduler(use_kv_connector=MockKVConfig())
        assert not sched.defer_block_free

        request = create_requests(1)[0]
        sched.add_request(request)
        output = sched.schedule()
        assert sched.sched_step_seq == 0

        block_pool = sched.kv_cache_manager.block_pool
        free_before = block_pool.get_num_free_blocks()
        sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        assert not sched.deferred_frees
        assert block_pool.get_num_free_blocks() > free_before

        # Processing the step leaves the fence untouched: nothing to drain, and
        # the release counter stays with the scheduling one at 0.
        sched.update_from_output(output, make_model_runner_output(output, 0))
        assert sched.processed_step_seq == 0
        assert not sched.deferred_frees

    def test_empty_step_does_not_advance_the_fence(self):
        # The other half of the condition. update_from_output only advances
        # processed_step_seq for a step that has tokens, so a scheduling counter
        # that moved on an empty step would run ahead for good -- the fence would
        # stop clearing and deferred blocks would never return to the pool. Empty
        # steps are normal here: a KV consumer parks requests that wait on a
        # remote KV load.
        sched = create_rbln_scheduler(
            pipeline_parallel_size=2, use_kv_connector=MockKVConfig()
        )
        assert sched.defer_block_free

        output = sched.schedule()  # nothing queued

        assert output.total_num_scheduled_tokens == 0
        assert sched.sched_step_seq == 0

    def test_deferred_free_settles_sub_block_state(self):
        # The fenced path releases blocks through pop_blocks_for_free(), so the
        # sub-block bookkeeping has to be settled there as free() settles it.
        # Otherwise the request's hashes outlive it, and its partial block never
        # gets the synthetic hash that keeps it in the LRU.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=16,
            sub_block_size=8,
            pipeline_parallel_size=2,
            use_kv_connector=MockKVConfig(),
        )
        manager = sched.kv_cache_manager
        # 32 tokens over a 16-token block with 8-token sub-blocks: the second
        # block is partial once the step advances, which is what gets indexed.
        request = create_requests(1, num_tokens=32)[0]
        sched.add_request(request)
        sched.schedule()
        partial_block = manager.coordinator.get_blocks(request.request_id)[0][-1]

        sched.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

        assert sched.deferred_frees, "blocks must be fenced, not freed"
        assert request.request_id not in manager._req_sub_hashes
        assert request.request_id not in manager._pending_indexing
        assert partial_block.block_hash is not None
