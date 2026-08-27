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

"""The ModelRunnerOutput async scheduling hands back before its tokens exist.

The runner returns one of these instead of a ModelRunnerOutput; vLLM calls
get_output() on the worker's async output thread, once the next forward is
already in flight.
"""

from collections import deque

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput

# Queued by get_output() on the output thread, drained by the main thread in
# RBLNModelRunner._apply_pending_token_writeback: the step's request ids, its
# sampled token ids per request, and where each request's placeholder landed.
PendingTokenWriteback = deque[tuple[list[str], list[list[int]], dict[str, int]]]


class AsyncRBLNModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        pending_token_writeback: PendingTokenWriteback,
        req_ids: list[str],
        placeholder_pos: dict[str, int],
        logprobs_tensors: LogprobsTensors | None,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        # For the token_ids_cpu write-back, applied by the main thread.
        self._pending_token_writeback = pending_token_writeback
        self._req_ids = req_ids
        self._placeholder_pos = placeholder_pos

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self._sampled_token_ids_cpu = torch.empty(
            sampled_token_ids.shape,
            dtype=sampled_token_ids.dtype,
            device="cpu",
        )
        # Logprobs ride the same deferral. Only the dense form is free of the
        # tokens: the topk form indexes by the sampled ids, so building it pulls
        # them to the host mid-step and serialises what async just decoupled.
        self._logprobs_tensors = logprobs_tensors

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        Blocks until the copy finishes. Runs on the worker's async output thread.
        InferenceMode is thread-local, hence off here, and updating
        _sampled_token_ids_cpu - an inference tensor allocated
        under sample_tokens - in place with it off is a hard error.
        """
        # Blocking copy, not non_blocking + a device synchronize: synchronizing
        # waits on every pending transfer, so once forward(N+1) is dispatched
        # this thread would wait out a whole forward. This waits on the sampler.
        with torch.inference_mode():
            self._sampled_token_ids_cpu.copy_(self._sampled_token_ids)

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        # The -1 placeholders the async path left in token_ids_cpu still have to be
        # replaced by the real tokens, but not from this thread: the main thread
        # reads token_ids_cpu in _preprocess. Queue the tokens instead and let the
        # main thread apply them at the top of its next step.
        # Copied: the scheduler trims these lists in place when a request stops.
        self._pending_token_writeback.append(
            (
                self._req_ids,
                [list(ids) for ids in valid_sampled_token_ids],
                self._placeholder_pos,
            )
        )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        if self._logprobs_tensors is not None:
            # tolists() is where the logprobs D2H actually happens - on this
            # thread, off the step's critical path.
            output.logprobs = self._logprobs_tensors.tolists()
        return output
