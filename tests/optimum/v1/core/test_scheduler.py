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


import pytest
from vllm.v1.request import RequestStatus

from .utils import create_model_runner_output, create_requests, create_scheduler


def test_add_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


def test_finish_request():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


def test_schedule_single_seq():
    """Test scheduling.
    Only one case: no prompt logprobs
    """
    scheduler = create_scheduler(max_num_seqs=1)
    requests = create_requests(num_requests=8, num_tokens=16)

    # Add requests to the waiting queue
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # Verify all tokens in the request are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == len(requests) - 1
    assert len(scheduler.running) == 1
    assert requests[0] == scheduler.running[0]


def test_schedule_multi_seq():
    """Test scheduling.
    Only one case: no prompt logprobs
    """
    scheduler = create_scheduler(max_num_seqs=2)
    requests = create_requests(num_requests=2)

    # Add requests to the waiting queue
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    for _ in range(scheduler.max_num_running_reqs):
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        assert output.scheduled_cached_reqs.num_reqs == 0
        assert len(output.finished_req_ids) == 0

    # Verify all tokens in the request are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == len(requests) - scheduler.max_num_running_reqs
    assert len(scheduler.running) == scheduler.max_num_running_reqs

    for i, running_request in enumerate(scheduler.running):
        assert requests[i] == running_request


@pytest.mark.parametrize(
    "max_num_seqs, block_size, max_model_len, "
    "num_blocks, num_tokens_per_batch, "
    "exp_new_req0_blocks, exp_cached0_new, "
    "exp_new_req1_blocks, exp_cached1_new, ",
    [
        # Scenario (block_size=16, 32-token prompt -> 2 full prefill blocks +
        # 1 block on the first decode; block 0 is the null block so ids start
        # at 1; prefix caching off -> blocks stay unhashed):
        #   req0 prefill -> [1, 2], decode -> [3]; req0 finishes and frees
        #   [1, 2, 3]. vLLM 0.24 returns freed *unhashed* blocks to the HEAD of
        #   the free queue in reverse-free order (prepend_n([3, 2, 1])), so they
        #   are reused FIRST -> req1 prefill pops [3, 2], decode pops [1].
        #   (<=0.22 appended freed blocks to the tail -> reused last, so this
        #    case gave [4, 5]/[6].)
        pytest.param(2, 16, 64, 7, 32, [1, 2], [3], [3, 2], [1], id="kv16-len32-blk7"),
        # Same request sequence with a tighter pool (4 usable blocks). Because
        # 0.24 reuses freed blocks first, the outcome no longer depends on pool
        # size and matches blk7. (<=0.22 gave [4, 3]/[2] here.)
        pytest.param(3, 16, 64, 5, 32, [1, 2], [3], [3, 2], [1], id="kv16-len32-blk5"),
    ],
)
def test_schedule_alloc_block_policy(
    max_num_seqs: int,
    block_size: int,
    max_model_len: int,
    num_blocks: int,
    num_tokens_per_batch: int,
    exp_new_req0_blocks: list[int],
    exp_cached0_new: list[int],
    exp_new_req1_blocks: list[int],
    exp_cached1_new: list[int],
):
    scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
    )
    requests = create_requests(
        num_requests=max_num_seqs,
        num_tokens=num_tokens_per_batch,
        block_size=block_size,
    )

    # [Prefill] Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert scheduler_output0.scheduled_new_reqs[0].block_ids[0] == exp_new_req0_blocks

    # Model output of the first request.
    model_runner_output = create_model_runner_output(scheduler_output0)
    # first request status update
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # [Decode] Schedule again the first request.
    scheduler_output1 = scheduler.schedule()
    scheduled_cached_reqs = scheduler_output1.scheduled_cached_reqs
    assert scheduled_cached_reqs.new_block_ids[0][0] == exp_cached0_new

    # finish the first request
    scheduler.finish_requests(requests[0].request_id, RequestStatus.FINISHED_STOPPED)

    # [Prefill] Schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output2 = scheduler.schedule()
    assert scheduler_output2.scheduled_new_reqs[0].block_ids[0] == exp_new_req1_blocks

    # Model output of the second request.
    model_runner_output = create_model_runner_output(scheduler_output2)
    # second request status update
    scheduler.update_from_output(scheduler_output2, model_runner_output)

    # [Decode] Schedule again the first request.
    scheduler_output3 = scheduler.schedule()
    scheduled_cached_reqs = scheduler_output3.scheduled_cached_reqs
    assert scheduled_cached_reqs.new_block_ids[0][0] == exp_cached1_new


def test_local_block_table_id_lifecycle():
    """Each admitted request is pinned to the lowest free local block table id,
    keeps it across decode steps, and releases it on finish for reuse."""
    scheduler = create_scheduler(max_num_seqs=2)
    requests = create_requests(num_requests=3, num_tokens=16)

    scheduler.add_request(requests[0])
    scheduler.add_request(requests[1])
    output = scheduler.schedule()
    assert output.local_block_table_dict == {"0": 0}
    scheduler.update_from_output(output, create_model_runner_output(output))

    output = scheduler.schedule()
    assert output.local_block_table_dict == {"1": 1}
    scheduler.update_from_output(output, create_model_runner_output(output))

    # Decode covers every scheduled request with its pinned id.
    output = scheduler.schedule()
    assert output.local_block_table_dict == {"0": 0, "1": 1}
    scheduler.update_from_output(output, create_model_runner_output(output))

    # Finishing frees the id; the next admission reuses the lowest free one.
    scheduler.finish_requests(requests[0].request_id, RequestStatus.FINISHED_STOPPED)
    scheduler.add_request(requests[2])
    output = scheduler.schedule()
    assert output.local_block_table_dict == {"2": 0}


def test_local_block_table_id_freed_on_preemption():
    """A preempted request releases its local block table id immediately and is
    re-admitted with a freshly allocated one."""
    # 32-token prompts fill 2 blocks each; num_blocks=6 leaves 5 usable blocks
    # (block 0 is the null block), so the first decode step preempts request 1:
    # request 0 takes the fifth block and request 1 finds the pool empty.
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=64,
        num_blocks=6,
    )
    requests = create_requests(num_requests=2, num_tokens=32)

    for request in requests:
        scheduler.add_request(request)
    for _ in range(2):
        output = scheduler.schedule()
        scheduler.update_from_output(output, create_model_runner_output(output))

    output = scheduler.schedule()
    assert output.preempted_req_ids == {"1"}
    assert output.local_block_table_dict == {"0": 0}
    assert scheduler._local_block_table_ids == {"0": 0}
    scheduler.update_from_output(output, create_model_runner_output(output))

    # Once request 0 finishes, the preempted request resumes from prefill with
    # the lowest free id, not its old one.
    scheduler.finish_requests(requests[0].request_id, RequestStatus.FINISHED_STOPPED)
    output = scheduler.schedule()
    assert output.local_block_table_dict == {"1": 0}
