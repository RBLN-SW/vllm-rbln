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
        pytest.param(2, 16, 64, 7, 32, [1, 2], [3], [4, 5], [6], id="kv16-len32-blk7"),
        pytest.param(3, 16, 64, 5, 32, [1, 2], [3], [4, 3], [2], id="kv16-len32-blk5"),
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
