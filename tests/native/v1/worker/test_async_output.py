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

# What get_output() hands to its two consumers: the returned ModelRunnerOutput,
# which the scheduler mutates in place, and the writeback queue, which the main
# thread drains a step later.

from collections import deque

import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_rbln.v1.worker.async_output import (
    AsyncRBLNModelRunnerOutput,
    PendingTokenWriteback,
)

pytestmark = pytest.mark.maybe_use_device


def _output(req_ids):
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=None,
    )


def _async_output(tokens, *, invalid_req_indices=()):
    req_ids = [f"r{i}" for i in range(len(tokens))]
    queue: PendingTokenWriteback = deque()
    async_out = AsyncRBLNModelRunnerOutput(
        model_runner_output=_output(req_ids),
        sampled_token_ids=torch.tensor(tokens, dtype=torch.int32),
        invalid_req_indices=list(invalid_req_indices),
        pending_token_writeback=queue,
        req_ids=req_ids,
        placeholder_pos={req_id: 3 for req_id in req_ids},
        logprobs_tensors=None,
    )
    return async_out, queue


class TestGetOutput:
    def test_returns_the_sampled_tokens(self):
        async_out, _ = _async_output([[7], [8]])
        assert async_out.get_output().sampled_token_ids == [[7], [8]]

    def test_clears_the_rows_the_sampler_discarded(self):
        async_out, _ = _async_output([[7], [8]], invalid_req_indices=[1])
        assert async_out.get_output().sampled_token_ids == [[7], []]

    def test_queues_the_tokens_for_the_writeback(self):
        async_out, queue = _async_output([[7], [8]])
        async_out.get_output()
        req_ids, tokens, placeholder_pos = queue.popleft()
        assert req_ids == ["r0", "r1"]
        assert tokens == [[7], [8]]
        assert placeholder_pos == {"r0": 3, "r1": 3}

    def test_the_queued_tokens_survive_an_in_place_trim_of_the_output(self):
        # The scheduler trims sampled_token_ids[i] in place once a request stops
        # (Scheduler._update_request_with_output), and under world_size == 1 the
        # output reaches it by reference -- UniProcExecutor does not serialise.
        async_out, queue = _async_output([[7], [8]])
        output = async_out.get_output()
        for ids in output.sampled_token_ids:
            del ids[0:]
        assert output.sampled_token_ids == [[], []]
        assert queue.popleft()[1] == [[7], [8]]
