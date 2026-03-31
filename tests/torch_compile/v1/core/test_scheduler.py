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

from tests.torch_compile.v1.core.utils import (
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


def _advance_to_decode(scheduler, request):
    """Run one prefill step + update so the request enters decode state."""
    scheduler.add_request(request)
    sched_out = scheduler.schedule()
    scheduler.update_from_output(sched_out, create_runner_output(sched_out, 1))


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
    _advance_to_decode(scheduler, req)

    req.spec_token_ids = [1] * 4
    sched_out = scheduler.schedule()

    rid = req.request_id
    assert sched_out.num_scheduled_tokens[rid] == 5
    assert len(sched_out.scheduled_spec_decode_tokens[rid]) == 4
    _check_invariant(sched_out, rid)


# ---------------------------------------------------------------------------
# spec_decode_cap [2/10]:
# near block boundary → all spec tokens trimmed
# ---------------------------------------------------------------------------


def test_spec_decode_cap_near_block_boundary_all_trimmed():
    """prompt=1023 → remaining_in_block=1 → cap=1 → all spec removed."""
    scheduler = _sd_scheduler()
    req = _sd_request(1023, "A")
    _advance_to_decode(scheduler, req)

    req.spec_token_ids = [1] * 4
    sched_out = scheduler.schedule()

    rid = req.request_id
    assert sched_out.num_scheduled_tokens[rid] == 1
    assert rid not in sched_out.scheduled_spec_decode_tokens


# ---------------------------------------------------------------------------
# spec_decode_cap [3/10]:
# partial spec tokens fit (remaining=4, spec=6 → 3 spec survive)
# ---------------------------------------------------------------------------


def test_spec_decode_cap_partial_spec_tokens_fit():
    """prompt=1020 → remaining_in_block=4 → cap=4 → 3 spec tokens survive."""
    scheduler = _sd_scheduler()
    req = _sd_request(1020, "A")
    _advance_to_decode(scheduler, req)

    req.spec_token_ids = [1] * 6
    sched_out = scheduler.schedule()

    rid = req.request_id
    assert sched_out.num_scheduled_tokens[rid] == 4
    assert len(sched_out.scheduled_spec_decode_tokens[rid]) == 3
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
    _advance_to_decode(scheduler, req_a)
    _advance_to_decode(scheduler, req_b)

    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens[req_a.request_id] == 1
    assert sched_out.num_scheduled_tokens[req_b.request_id] == 1
    assert sched_out.scheduled_spec_decode_tokens == {}


# ---------------------------------------------------------------------------
# spec_decode_cap [5/10]:
# B tightens cap=1 → both A and B lose all spec tokens
# ---------------------------------------------------------------------------


def test_spec_decode_cap_retroactive_trim_all_spec_removed():
    """A(1024)+B(1023) with spec=4 each; B sets cap=1 → both trimmed to 1."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(1023, "B")
    _advance_to_decode(scheduler, req_a)
    _advance_to_decode(scheduler, req_b)

    req_a.spec_token_ids = [1] * 4
    req_b.spec_token_ids = [1] * 4
    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens[req_a.request_id] == 1
    assert sched_out.num_scheduled_tokens[req_b.request_id] == 1
    assert req_a.request_id not in sched_out.scheduled_spec_decode_tokens
    assert req_b.request_id not in sched_out.scheduled_spec_decode_tokens


# ---------------------------------------------------------------------------
# spec_decode_cap [6/10]:
# B tightens cap=4 → both A and B trimmed to 4 (1+3 spec)
# ---------------------------------------------------------------------------


def test_spec_decode_cap_retroactive_trim_partial_spec_preserved():
    """A(1024)+B(1020) with spec=6 each; B sets cap=4 → 3 spec each."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(1020, "B")
    _advance_to_decode(scheduler, req_a)
    _advance_to_decode(scheduler, req_b)

    req_a.spec_token_ids = [1] * 6
    req_b.spec_token_ids = [1] * 6
    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens[req_a.request_id] == 4
    assert sched_out.num_scheduled_tokens[req_b.request_id] == 4
    assert len(sched_out.scheduled_spec_decode_tokens[req_a.request_id]) == 3
    assert len(sched_out.scheduled_spec_decode_tokens[req_b.request_id]) == 3
    _check_invariant(sched_out, req_a.request_id)
    _check_invariant(sched_out, req_b.request_id)


# ---------------------------------------------------------------------------
# spec_decode_cap [7/10]:
# three requests; C sets cap=2 → all trimmed to 2 (1+1 spec)
# ---------------------------------------------------------------------------


def test_spec_decode_cap_retroactive_trim_three_requests():
    """A(1024)+B(512)+C(1022) with spec=6 each; C sets cap=2 → 1 spec each."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(512, "B")
    req_c = _sd_request(1022, "C")
    _advance_to_decode(scheduler, req_a)
    _advance_to_decode(scheduler, req_b)
    _advance_to_decode(scheduler, req_c)

    req_a.spec_token_ids = [1] * 6
    req_b.spec_token_ids = [1] * 6
    req_c.spec_token_ids = [1] * 6
    sched_out = scheduler.schedule()

    for req in (req_a, req_b, req_c):
        rid = req.request_id
        assert sched_out.num_scheduled_tokens[rid] == 2
        assert len(sched_out.scheduled_spec_decode_tokens[rid]) == 1
        _check_invariant(sched_out, rid)


# ---------------------------------------------------------------------------
# spec_decode_cap [8/10]:
# decode-only B tightens cap → A retroactively trimmed
# ---------------------------------------------------------------------------


def test_spec_decode_cap_decode_only_tightens_cap():
    """A(1024,spec=4)+B(1020,no spec); B sets cap=4 → A trimmed to 4 (1+3)."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(1020, "B")
    _advance_to_decode(scheduler, req_a)
    _advance_to_decode(scheduler, req_b)

    req_a.spec_token_ids = [1] * 4
    # req_b has no spec tokens
    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens[req_a.request_id] == 4
    assert sched_out.num_scheduled_tokens[req_b.request_id] == 1
    assert len(sched_out.scheduled_spec_decode_tokens[req_a.request_id]) == 3
    assert req_b.request_id not in sched_out.scheduled_spec_decode_tokens
    _check_invariant(sched_out, req_a.request_id)


# ---------------------------------------------------------------------------
# spec_decode_cap [9/10]:
# max_model_len constraint tightens cap via remaining_in_maxlen
# ---------------------------------------------------------------------------


def test_spec_decode_cap_maxlen_constraint():
    """A(1024,spec=6)+B(2046,no spec); B's remaining_in_maxlen=2 → cap=2."""
    scheduler = _sd_scheduler(max_model_len=2048, max_num_batched_tokens=2048)
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(2046, "B")
    _advance_to_decode(scheduler, req_a)
    _advance_to_decode(scheduler, req_b)

    req_a.spec_token_ids = [1] * 6
    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens[req_a.request_id] == 2
    assert sched_out.num_scheduled_tokens[req_b.request_id] == 1
    assert len(sched_out.scheduled_spec_decode_tokens[req_a.request_id]) == 1
    assert req_b.request_id not in sched_out.scheduled_spec_decode_tokens
    _check_invariant(sched_out, req_a.request_id)


# ---------------------------------------------------------------------------
# spec_decode_cap [10/10]:
# new prefill in waiting triggers no-mixed-batching → decode excluded
# ---------------------------------------------------------------------------


def test_spec_decode_cap_prefill_triggers_no_mixed_batching():
    """A(1024,decode,spec=4) running + B(512) waiting → only B scheduled."""
    scheduler = _sd_scheduler()
    req_a = _sd_request(1024, "A")
    req_b = _sd_request(512, "B")
    _advance_to_decode(scheduler, req_a)

    req_a.spec_token_ids = [1] * 4
    scheduler.add_request(req_b)
    sched_out = scheduler.schedule()

    assert len(sched_out.scheduled_new_reqs) == 1
    assert req_a.request_id not in sched_out.num_scheduled_tokens
    assert req_b.request_id in sched_out.num_scheduled_tokens
