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

The runner returns one of these instead of a ModelRunnerOutput. MultiprocExecutor
and its subclasses call get_output() on the worker's async output thread, once
the next forward is already in flight.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput

from vllm_rbln.v1.worker.utils import worker_fail_fast

# Queued by get_output() on the output thread, drained by the main thread in
# RBLNModelRunner._apply_pending_token_writeback: the step's request ids, its
# sampled token ids per request, and where each request's placeholder landed.
PendingTokenWriteback = deque[tuple[list[str], list[list[int]], dict[str, int]]]

# Non-output ranks drop their output without get_output(), and torch-rbln frees a
# pinned buffer at once, pending D2H or not. Hold each copy's tensors here until
# its event completes instead of blocking on it.
# TODO(torch-rbln): drop once its pinned allocator defers frees like CUDA's.
_inflight_d2h: deque[tuple[torch.Event, torch.Tensor, torch.Tensor]] = deque()


@dataclass(frozen=True)
class SampledTokenState:
    """One step's tokens, host buffer, event, and request-row snapshot.

    CPU sources fill the host buffer in get_output(); device sources use D2H.
    Request rows are separate from InputBatch's mutable valid-request view.
    """

    source_token_ids: torch.Tensor
    host_buffer: torch.Tensor
    copy_ready_event: torch.Event | None
    request_id_to_index: Mapping[str, int]


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
        fail_fast: bool,
        rank: int | None = None,
        dp_rank: int | None = None,
    ):
        self.fail_fast = fail_fast
        # Named in the fatal event when get_output() fails on the output thread.
        self.rank = rank
        self.dp_rank = dp_rank
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        # For the token_ids_cpu write-back, applied by the main thread.
        self._pending_token_writeback = pending_token_writeback
        self._placeholder_pos = placeholder_pos

        host_buffer = torch.empty(
            sampled_token_ids.shape,
            dtype=sampled_token_ids.dtype,
            device="cpu",
            # Pinned, or torch-rbln turns the non_blocking copy below into a sync one.
            pin_memory=not sampled_token_ids.is_cpu,
        )
        # Start the D2H right behind the sampler, as AsyncGPUModelRunnerOutput
        # does. From get_output() it would queue behind the next forward.
        copy_ready_event: torch.Event | None = None
        while _inflight_d2h and _inflight_d2h[0][0].query():
            _inflight_d2h.popleft()
        if not sampled_token_ids.is_cpu:
            host_buffer.copy_(sampled_token_ids, non_blocking=True)
            copy_ready_event = torch.Event(device=sampled_token_ids.device)
            copy_ready_event.record()
            _inflight_d2h.append((copy_ready_event, sampled_token_ids, host_buffer))
        self.sampled_token_state = SampledTokenState(
            source_token_ids=sampled_token_ids,
            host_buffer=host_buffer,
            copy_ready_event=copy_ready_event,
            request_id_to_index=MappingProxyType(
                {req_id: i for i, req_id in enumerate(req_ids)}
            ),
        )
        # Logprobs ride the same deferral. Only the dense form is free of the
        # tokens: the topk form indexes by the sampled ids, so building it pulls
        # them to the host mid-step and serialises what async just decoupled.
        self._logprobs_tensors = logprobs_tensors

    @worker_fail_fast
    def get_output(self) -> ModelRunnerOutput:
        """Wait for the sampled tokens on the host and return a ModelRunnerOutput.

        Blocks until the copy finishes; the executor decides which thread calls it.
        """
        token_state = self.sampled_token_state
        if token_state.copy_ready_event is not None:
            token_state.copy_ready_event.synchronize()
        else:
            # InferenceMode is thread-local, hence off here, and updating
            # host_buffer - an inference tensor allocated under
            # sample_tokens - in place with it off is a hard error.
            with torch.inference_mode():
                token_state.host_buffer.copy_(token_state.source_token_ids)

        valid_sampled_token_ids = token_state.host_buffer.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        # The -1 placeholders the async path left in token_ids_cpu still have to be
        # replaced by the real tokens, but not from this thread: the main thread
        # reads token_ids_cpu in _preprocess. Queue the tokens instead and let the
        # main thread apply them at the top of its next step.
        # Copied: the scheduler trims these lists in place when a request stops.
        self._pending_token_writeback.append(
            (
                list(token_state.request_id_to_index),
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
