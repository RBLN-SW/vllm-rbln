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

    # Test prefill scheduling. The scheduler now allows the running
    # decodes to ride along with each new prefill in the same step
    # (prior "new prefill kicks decodes" behavior was removed in
    # the Cycle 1 mixed-batching simplification), so the cached-req
    # count grows with each iteration.
    for i in range(len(requests)):
        output = scheduler.schedule()

        assert len(output.scheduled_new_reqs) == 1
        assert output.scheduled_cached_reqs.num_reqs == i
        assert len(output.finished_req_ids) == 0
        # One new prefill + i previously-running decodes = i + 1 streams.
        assert len(output.num_scheduled_tokens) == i + 1
        new_req = output.scheduled_new_reqs[0]
        assert int(new_req.req_id) == i
        assert (
            output.num_scheduled_tokens[new_req.req_id]
            == len(requests[int(new_req.req_id)].prompt_token_ids)
        )

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

