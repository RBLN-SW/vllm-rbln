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

from vllm.v1.request import RequestStatus

from .utils import (
    advance_to_decode,
    create_requests,
    create_runner_output,
    create_scheduler,
)


def test_schedule():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    # Test prefill scheduling
    for i in range(len(requests)):
        output = scheduler.schedule()
        req_id, num_tokens = next(iter(output.num_scheduled_tokens.items()))

        assert len(output.scheduled_new_reqs) == 1
        assert output.scheduled_cached_reqs.num_reqs == 0
        assert len(output.finished_req_ids) == 0
        assert len(output.num_scheduled_tokens) == 1
        assert int(req_id) == i
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

        model_runner_output = create_runner_output(output, 0)
        scheduler.update_from_output(output, model_runner_output)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request

    # Test decode scheduling
    output = scheduler.schedule()
    assert output.scheduled_cached_reqs.num_reqs == len(requests)
    assert len(output.num_scheduled_tokens) == len(requests)
    assert all(num_tokens == 1 for num_tokens in output.num_scheduled_tokens.values())
    assert len(output.finished_req_ids) == 0


def test_schedule_chunked_prefill():
    scheduler = create_scheduler(max_num_batched_tokens=256)
    request = create_requests(num_requests=1, num_tokens=500)[0]
    scheduler.add_request(request)

    # first iteration
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[request.request_id] == 256
    model_runner_output = create_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    # second iteration
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[request.request_id] == 244
    model_runner_output = create_runner_output(output, 0)
    scheduler.update_from_output(output, model_runner_output)

    # third iteration
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(output.finished_req_ids) == 0

    assert output.num_scheduled_tokens[request.request_id] == 1


def test_new_prefill_uses_full_budget_when_decode_running():
    """When a decode request is running and a new prefill enters, the RBLN
    scheduler kicks out the decode and gives the full token budget to the
    prefill.  Before the fix, num_new_tokens was clipped to the
    already-reduced token_budget (e.g. 127) instead of the restored
    prefill_token_budget (128), causing an off-by-one in chunk positions.
    """
    max_num_batched_tokens = 128
    scheduler = create_scheduler(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=4,
        block_size=16,
        num_blocks=10000,
    )

    # First request: short prompt so it finishes prefill in one chunk.
    req_a = create_requests(num_requests=1, num_tokens=64, req_ids=["A"])[0]
    scheduler.add_request(req_a)

    # Prefill req_a (64 < 128, fits in one chunk).
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req_a.request_id] == 64
    scheduler.update_from_output(output, create_runner_output(output, 1))

    # req_a is now in decode.  Add req_b (long prompt, needs multiple chunks).
    req_b = create_requests(num_requests=1, num_tokens=500, req_ids=["B"])[0]
    scheduler.add_request(req_b)

    # Schedule: running loop picks req_a (decode, 1 token), then new-request
    # loop picks req_b (prefill) and kicks out req_a.
    output = scheduler.schedule()

    # req_a should have been kicked out (no mixed batching).
    assert req_a.request_id not in output.num_scheduled_tokens
    # req_b should get the FULL budget, not budget-minus-1.
    assert output.num_scheduled_tokens[req_b.request_id] == max_num_batched_tokens


def test_evicted_decode_block_is_restashed_and_reemitted():
    """Regression: a decode request whose next KV block is allocated exactly
    on the prefill->decode transition step, and which is then evicted by the
    "disable mixed batching" path, must not lose that block.

    The block is already committed in the coordinator, so the next step's
    allocate_slots returns an empty delta — without re-emitting the stashed
    delta the runner's block table would keep a stale block-id 0. This is the
    failure mode observed for block-aligned prompts (prompt length an exact
    multiple of block_size), whose second block is needed on the very first
    decode step that the eviction targets.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        block_size=block_size,
        num_blocks=10000,
    )

    # Block-aligned prompt: exactly one block of prompt tokens.
    req_a = create_requests(
        num_requests=1, num_tokens=block_size, block_size=block_size, req_ids=["A"]
    )[0]
    scheduler.add_request(req_a)

    # Prefill req_a (fits in one chunk); after the update it enters decode
    # with exactly one allocated block (num_computed == block_size).
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens[req_a.request_id] == block_size
    scheduler.update_from_output(out1, create_runner_output(out1, 1))
    assert req_a.num_computed_tokens == block_size

    # A new prefill enters: the running loop schedules req_a's first decode
    # (which needs a second block at the boundary), then the waiting loop
    # schedules req_b and evicts req_a.
    req_b = create_requests(
        num_requests=1, num_tokens=block_size, block_size=block_size, req_ids=["B"]
    )[0]
    scheduler.add_request(req_b)
    out2 = scheduler.schedule()

    # req_a was kicked; req_b runs.
    assert req_a.request_id not in out2.num_scheduled_tokens
    assert req_b.request_id in out2.num_scheduled_tokens
    # The fix stashed req_a's just-allocated block delta instead of dropping it.
    assert req_a.request_id in scheduler._stranded_new_blocks
    stashed_ids = scheduler._stranded_new_blocks[req_a.request_id].get_block_ids()
    assert any(len(g) > 0 for g in stashed_ids)

    scheduler.update_from_output(out2, create_runner_output(out2, 1))

    # Next step: no waiting prefill, so req_a runs as decode and the stashed
    # block must be re-emitted in its cached new_block_ids (this step's own
    # allocate_slots returns nothing — the block is already committed).
    out3 = scheduler.schedule()
    assert req_a.request_id in out3.num_scheduled_tokens
    cached = out3.scheduled_cached_reqs
    idx = cached.req_ids.index(req_a.request_id)
    reemitted = cached.new_block_ids[idx]
    assert reemitted is not None
    assert any(len(g) > 0 for g in reemitted)
    # The stash is drained, and the re-emitted ids cover the stashed blocks.
    assert req_a.request_id not in scheduler._stranded_new_blocks
    flat_reemitted = [b for g in reemitted for b in g]
    flat_stashed = [b for g in stashed_ids for b in g]
    assert flat_stashed
    assert all(b in flat_reemitted for b in flat_stashed)


def test_stranded_blocks_cleaned_up_on_finish():
    """A block delta stashed for an evicted request must be dropped if the
    request finishes before it is scheduled again, so the stash never leaks."""
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        block_size=block_size,
        num_blocks=10000,
    )

    req_a = create_requests(
        num_requests=1, num_tokens=block_size, block_size=block_size, req_ids=["A"]
    )[0]
    scheduler.add_request(req_a)
    out1 = scheduler.schedule()
    scheduler.update_from_output(out1, create_runner_output(out1, 1))

    req_b = create_requests(
        num_requests=1, num_tokens=block_size, block_size=block_size, req_ids=["B"]
    )[0]
    scheduler.add_request(req_b)
    scheduler.schedule()
    # req_a evicted at the boundary -> its block delta is stashed.
    assert req_a.request_id in scheduler._stranded_new_blocks

    # Finish req_a before it is rescheduled; the _free_request override must
    # drop the stash entry.
    scheduler.finish_requests(req_a.request_id, RequestStatus.FINISHED_ABORTED)
    assert req_a.request_id not in scheduler._stranded_new_blocks


def test_stranded_blocks_cleaned_up_on_preempt():
    """A stashed block delta must be dropped when the request is preempted.

    Preemption frees ALL of the request's blocks (including the stashed one)
    and returns the request to the waiting queue. If the stash survived, the
    request could resume, re-enter the decode batch, and have a now-freed
    (possibly reused) block id re-emitted into its block table.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        block_size=block_size,
        num_blocks=10000,
    )

    req_a = create_requests(
        num_requests=1, num_tokens=block_size, block_size=block_size, req_ids=["A"]
    )[0]
    scheduler.add_request(req_a)
    out1 = scheduler.schedule()
    scheduler.update_from_output(out1, create_runner_output(out1, 1))

    req_b = create_requests(
        num_requests=1, num_tokens=block_size, block_size=block_size, req_ids=["B"]
    )[0]
    scheduler.add_request(req_b)
    scheduler.schedule()
    # req_a evicted at the boundary -> its block delta is stashed. It stays in
    # the running queue (eviction only drops it from this step's output).
    assert req_a.request_id in scheduler._stranded_new_blocks
    assert req_a in scheduler.running

    # Preempt req_a (frees its blocks). The override must drop the stash so the
    # freed block id cannot be re-emitted after the request resumes.
    scheduler.running.remove(req_a)
    scheduler._preempt_request(req_a, 0.0)
    assert req_a.status == RequestStatus.PREEMPTED
    assert req_a.request_id not in scheduler._stranded_new_blocks


def test_preempt_during_execution():
    # Test copied from https://github.com/vllm-project/vllm/blob/4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e/tests/v1/core/test_scheduler.py#L672-L728

    # NOTE(woosuk): The actual number of available blocks is 10 instead of 11
    # because block 0 is reserved as the null block.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.num_scheduled_tokens) == 1
    assert len(scheduler_output0.scheduled_new_reqs[0].block_ids[0]) == 5

    # Schedule the second request while the first request is still running.
    # This scenario can occur in certain cases, when max_concurrent_batches > 1
    # (e.g., when pipeline parallelism is used).
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.num_scheduled_tokens) == 1
    assert len(scheduler_output1.scheduled_new_reqs[0].block_ids[0]) == 5

    # Get the output of the first request.
    model_runner_output0 = create_runner_output(scheduler_output0, 0)
    scheduler.update_from_output(scheduler_output0, model_runner_output0)

    # Schedule the first request again. This will cause the preemption
    # of the second request because the KV cache is full.
    _ = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert scheduler.running[0] == requests[0]
    assert requests[1].status == RequestStatus.PREEMPTED

    model_runner_output1 = create_runner_output(scheduler_output1, 42)
    scheduler.update_from_output(scheduler_output1, model_runner_output1)

    # The second request (that is preempted) should be updated with the
    # sampled token id.
    assert len(requests[1].output_token_ids) == 1
    assert requests[1].output_token_ids[0] == 42


# ---------------------------------------------------------------------------
# Helpers for spec_decode_cap tests
# ---------------------------------------------------------------------------

_SD_BLOCK_SIZE = 1024
_SD_NUM_BLOCKS = 100
_SD_MAX_NUM_SEQS = 10


def _sd_scheduler(**kwargs):
    return create_scheduler(
        block_size=_SD_BLOCK_SIZE,
        num_blocks=_SD_NUM_BLOCKS,
        max_num_seqs=_SD_MAX_NUM_SEQS,
        **kwargs,
    )


def _sd_request(num_tokens, req_id):
    return create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        block_size=_SD_BLOCK_SIZE,
        max_tokens=2048,
        req_ids=[req_id],
    )[0]


def _check_invariant(sched_out, req_id):
    """num_scheduled_tokens == 1 (decode token) + len(spec_tokens)."""
    n = sched_out.num_scheduled_tokens[req_id]
    spec = sched_out.scheduled_spec_decode_tokens.get(req_id, [])
    assert n == 1 + len(spec), (
        f"req {req_id}: num_scheduled_tokens={n} but 1+spec={1 + len(spec)}"
    )


# ---------------------------------------------------------------------------
# spec_decode_cap [1/10]:
# block boundary → cap == block_size → no retroactive trim
# ---------------------------------------------------------------------------


def test_spec_decode_cap_at_block_boundary():
    """prompt=1024 → remaining_in_block=1024 == block_size; cap unchanged."""
    scheduler = _sd_scheduler()
    req = _sd_request(1024, "A")
    advance_to_decode(scheduler, req)

    req.spec_token_ids = [1] * 4
    sched_out = scheduler.schedule()

    rid = req.request_id
    assert sched_out.num_scheduled_tokens[rid] == 5
    assert len(sched_out.scheduled_spec_decode_tokens[rid]) == 4
    _check_invariant(sched_out, rid)


# ---------------------------------------------------------------------------
# spec_decode_cap [4/10]:
# no spec tokens → retroactive trim skipped even when cap < block_size
# ---------------------------------------------------------------------------


def test_spec_decode_cap_no_spec_tokens_no_retroactive_trim():
    """cap=1 but scheduled_spec_decode_tokens is empty → trim skipped."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(1023, "B")
    advance_to_decode(scheduler, req_a)
    advance_to_decode(scheduler, req_b)

    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens[req_a.request_id] == 1
    assert sched_out.num_scheduled_tokens[req_b.request_id] == 1
    assert sched_out.scheduled_spec_decode_tokens == {}


# ---------------------------------------------------------------------------
# spec_decode_cap [10/10]:
# new prefill in waiting triggers no-mixed-batching → decode excluded
# ---------------------------------------------------------------------------


def test_spec_decode_cap_prefill_triggers_no_mixed_batching():
    """A(1024,decode,spec=4) running + B(512) waiting → only B scheduled."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(512, "B")
    advance_to_decode(scheduler, req_a)

    req_a.spec_token_ids = [1] * 4
    scheduler.add_request(req_b)
    sched_out = scheduler.schedule()

    assert len(sched_out.scheduled_new_reqs) == 1
    assert req_a.request_id not in sched_out.num_scheduled_tokens
    assert req_b.request_id in sched_out.num_scheduled_tokens


def test_dynamic_decode_cap_policy():
    """DynamicDecodeCapPolicy spreads active decodes across PP stages:
    cap = max(1, min(static_max, ceil(active / pp_size)))."""
    import math

    from vllm_rbln.v1.core.utils import DynamicDecodeCapPolicy

    # static_max = 8 (e.g. max_num_seqs=16, pp=2)
    # Few active -> spread below the static ceiling (avoids collapse).
    assert DynamicDecodeCapPolicy(8, 2, 6).cap() == math.ceil(6 / 2) == 3
    assert DynamicDecodeCapPolicy(8, 2, 1).cap() == 1
    # No active -> floored at 1 (never 0, which would break the budget).
    assert DynamicDecodeCapPolicy(8, 2, 0).cap() == 1
    # ceil(15/2)=8 reaches the ceiling.
    assert DynamicDecodeCapPolicy(8, 2, 15).cap() == 8
    # Clamp: never exceed static_max (the compiled bucket ceiling).
    assert DynamicDecodeCapPolicy(8, 2, 100).cap() == 8
    # Non-divisible max_num_seqs: ceil(33/2)=17 clamps down to static_max 16.
    assert DynamicDecodeCapPolicy(16, 2, 33).cap() == 16
    # pp_size == 1: cap = min(static_max, active) -> no-op vs static.
    assert DynamicDecodeCapPolicy(16, 1, 5).cap() == 5
    assert DynamicDecodeCapPolicy(16, 1, 20).cap() == 16


def test_decode_budget_hard_vs_soft_cap():
    """DecodeBatchBudget.can_admit: the hard cap (compiled bucket ceiling)
    always applies; the soft (spreading) cap applies only when apply_soft_cap.
    Demand-unbudgeted joins (apply_soft_cap=False -- full local prefix match /
    resumed-after-eviction) fill up to the hard cap, not the soft one."""
    from vllm_rbln.v1.core.utils import (
        DecodeBatchBudget,
        DynamicDecodeCapPolicy,
        StaticDecodeCapPolicy,
    )

    # Balance: hard=8, soft=ceil(4/2)=2.
    b = DecodeBatchBudget(DynamicDecodeCapPolicy(8, 2, 4), hard_cap=8)
    b.admit(2)  # count == soft
    assert not b.can_admit()  # budgeted: gated at soft (2)
    assert b.can_admit(apply_soft_cap=False)  # unbudgeted: hard (8) has room
    b.admit(6)  # count == hard
    assert not b.can_admit(apply_soft_cap=False)  # hard cap reached
    assert not b.can_admit()

    # Static: soft == hard, so apply_soft_cap makes no difference.
    b2 = DecodeBatchBudget(StaticDecodeCapPolicy(8), hard_cap=8)
    b2.admit(8)
    assert not b2.can_admit()
    assert not b2.can_admit(apply_soft_cap=False)


def test_decode_budget_discard():
    """discard() un-admits decodes dropped from the step (PRIORITY-policy
    preemption of an already-scheduled decode) so the can_admit() gate is
    not stopped early on a stale over-count. Unlike reset() it only removes
    the dropped ones, leaving the still-admitted decodes counted."""
    from vllm_rbln.v1.core.utils import DecodeBatchBudget, StaticDecodeCapPolicy

    b = DecodeBatchBudget(StaticDecodeCapPolicy(2), hard_cap=2)
    b.admit(2)  # batch full at the cap
    assert not b.can_admit()  # gate closed
    b.discard()  # a scheduled decode is preempted -> one slot freed
    assert b.count == 1
    assert b.can_admit()  # gate reopens for another admit
    # reset() would instead zero the whole count (whole-batch eviction).
    b.reset()
    assert b.count == 0


def test_priority_preemption_discards_admitted_decode(monkeypatch):
    """When the PRIORITY policy preempts an ALREADY-SCHEDULED decode to free KV
    blocks, the scheduler un-admits it from the per-step decode budget via
    `discard()`, keeping the admitted count in step with the batch so the
    `can_admit()` gate is not stopped early on a stale over-count.

    Two decodes each need a fresh block this step but only one block is free,
    so scheduling the trigger preempts the victim (higher priority value =
    lower scheduling importance). The victim was already scheduled this step,
    so `discard()` must fire exactly once.
    """
    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    from vllm_rbln.v1.core.utils import DecodeBatchBudget

    discard_calls = []
    orig_discard = DecodeBatchBudget.discard

    def spy(self, n=1):
        discard_calls.append(n)
        return orig_discard(self, n)

    monkeypatch.setattr(DecodeBatchBudget, "discard", spy)

    block_size = 16
    # num_blocks=4 -> 3 usable (block 0 is the null block): one block per
    # prefill (2) leaves exactly one free for a decode boundary block, so only
    # one of the two decodes can grow this step.
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        block_size=block_size,
        num_blocks=4,
        enable_prefix_caching=False,
    )
    scheduler.policy = SchedulingPolicy.PRIORITY

    victim, trigger = create_requests(
        num_requests=2,
        num_tokens=block_size,
        block_size=block_size,
        req_ids=["victim", "trigger"],
    )
    # Higher priority VALUE == lower scheduling importance == preempted first.
    victim.priority = 1
    trigger.priority = 0
    for r in (victim, trigger):
        advance_to_decode(scheduler, r)

    out = scheduler.schedule()

    assert victim.status == RequestStatus.PREEMPTED
    assert trigger.request_id in out.num_scheduled_tokens
    assert discard_calls == [1], (
        "discard() must fire exactly once for the preempted already-scheduled "
        f"decode, got {discard_calls}"
    )


def test_pp_balance_decode_spreads_microbatch(monkeypatch):
    """With VLLM_RBLN_PP_BALANCE_DECODE_BATCH=1 under PP, one step's
    decode batch is sized to ~ceil(active / pp_size), spreading the active
    decodes across the PP microbatches instead of packing them into one
    (which would idle the other stage). The static default packs more.
    """
    import math

    from vllm_rbln.v1.core.rbln_scheduler import is_prefill

    n = 6

    def run(balance: bool):
        if balance:
            monkeypatch.setenv("VLLM_RBLN_PP_BALANCE_DECODE_BATCH", "1")
        else:
            # Force the static cap explicitly: the default is now on under PP,
            # so unsetting would still give the balanced (dynamic) behavior.
            monkeypatch.setenv("VLLM_RBLN_PP_BALANCE_DECODE_BATCH", "0")
        # max_num_seqs=16, pp=2 -> static cap 8; n=6 active -> ceil(6/2)=3.
        scheduler = create_scheduler(
            max_num_seqs=16, pipeline_parallel_size=2, block_size=16
        )
        reqs = create_requests(
            num_requests=n,
            num_tokens=32,
            block_size=16,
            req_ids=[f"d{i}" for i in range(n)],
        )
        for r in reqs:
            advance_to_decode(scheduler, r)
        running_decodes = sum(1 for r in scheduler.running if not is_prefill(r))
        out = scheduler.schedule()
        return running_decodes, len(out.num_scheduled_tokens)

    run_on, sched_on = run(True)
    run_off, sched_off = run(False)

    # Balanced: at most ceil(active / pp) admitted this step.
    assert sched_on <= math.ceil(run_on / 2)
    assert sched_on >= 1
    # Static packs more decodes into the single microbatch than balanced does.
    assert sched_off > sched_on
