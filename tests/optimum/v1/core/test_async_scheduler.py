# Copyright 2025 Rebellions Inc. All rights reserved.
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


import pytest
from vllm.v1.request import RequestStatus

from .utils import create_model_runner_output, create_requests, create_scheduler


@pytest.mark.parametrize(
    "max_num_seqs, block_size, max_model_len, "
    "num_blocks, num_tokens_per_batch, "
    "exp_new_req0_blocks, exp_new_req1_blocks, "
    "exp_cached0_new, exp_cached1_new",
    [
        pytest.param(
            2, 16, 64, 9, 32, [1, 2], [3, 4], ([5],), ([6],), id="16bsize-32len"
        ),
        pytest.param(
            2, 8, 24, 7, 17, [1, 2, 3], [4, 5, 6], None, None, id="8bsize-17len"
        ),
    ],
)
def test_schedule_alloc_block(
    max_num_seqs: int,
    block_size: int,
    max_model_len: int,
    num_blocks: int,
    num_tokens_per_batch: int,
    exp_new_req0_blocks: list[int],
    exp_new_req1_blocks: list[int],
    exp_cached0_new: list[int] | None,
    exp_cached1_new: list[int] | None,
):
    scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
        async_scheduling=True,
    )

    requests = create_requests(
        num_requests=max_num_seqs,
        num_tokens=num_tokens_per_batch,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert scheduler_output0.scheduled_new_reqs[0].block_ids[0] == exp_new_req0_blocks

    # Schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert scheduler_output1.scheduled_new_reqs[0].block_ids[0] == exp_new_req1_blocks

    # Model output of the first request.
    model_runner_output = create_model_runner_output(scheduler_output0)
    # first request status update
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Model output of the second request.
    model_runner_output = create_model_runner_output(scheduler_output1)
    # second request status update
    scheduler.update_from_output(scheduler_output1, model_runner_output)

    # Schedule the next step again. The first request and second request
    # can be scheduled with decode phase.
    scheduler_output2 = scheduler.schedule()
    scheduled_cached_reqs = scheduler_output2.scheduled_cached_reqs
    assert scheduled_cached_reqs.new_block_ids[0] == exp_cached0_new
    assert scheduled_cached_reqs.new_block_ids[1] == exp_cached1_new


@pytest.mark.parametrize(
    "max_num_seqs, num_requests, num_blocks, exp_running_sz",
    [
        pytest.param(5, 5, 6, [1, 2, 3, 4, 5], id="normal"),
        pytest.param(2, 5, 5, [1, 2, 2, 2, 2], id="limited-max_num_seqs"),
        pytest.param(3, 5, 4, [1, 2, 3, 1, 1], id="limited-blocks"),
    ],
)
def test_running_queue(
    max_num_seqs: int,
    num_requests: int,
    num_blocks: int,
    exp_running_sz: list[int],
):
    assert num_requests == len(exp_running_sz)
    scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        num_blocks=num_blocks,
        block_size=10,
        async_scheduling=True,
    )
    requests = create_requests(num_requests=num_requests, max_tokens=5)

    for req in requests:
        scheduler.add_request(req)

    assert len(scheduler.running) == 0

    for _, sz in zip(requests, exp_running_sz):
        sched_output = scheduler.schedule()
        model_runner_output = create_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        assert len(scheduler.running) == sz


def test_preempt(
    num_requests=10,
    max_num_seqs=3,
):
    MAX_TOKENS = 5
    NUM_TOKENS = 5
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_seqs=max_num_seqs,
        num_blocks=MAX_TOKENS * max_num_seqs + 1,
        block_size=MAX_TOKENS + NUM_TOKENS,
    )
    requests = create_requests(
        num_requests=num_requests,
        max_tokens=MAX_TOKENS,
        num_tokens=NUM_TOKENS,
    )

    for req in requests:
        scheduler.add_request(req)

    abort_order = [requests[i].request_id for i in range(num_requests)]

    # Mark `max_num_seqs` requests as RUNNING
    for idx in range(max_num_seqs):
        sched_output = scheduler.schedule()
        model_runner_output = create_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

    # A request is preempted,
    # and the WAITING request with the highest priority is scheduled.
    for abort_idx, abort_req in enumerate(abort_order[:-max_num_seqs]):
        scheduler.finish_requests(abort_req, RequestStatus.FINISHED_ABORTED)
        next_req = scheduler.requests.get(abort_order[abort_idx + max_num_seqs])
        assert requests[abort_idx].status == RequestStatus.FINISHED_ABORTED
        assert next_req.status == RequestStatus.WAITING

        sched_output = scheduler.schedule()
        model_runner_output = create_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        next_req = scheduler.requests.get(abort_order[abort_idx + max_num_seqs])
        assert next_req.status == RequestStatus.RUNNING
