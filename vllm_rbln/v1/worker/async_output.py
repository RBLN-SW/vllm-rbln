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

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput

from vllm_rbln.v1.worker.utils import worker_fail_fast


class AsyncRBLNModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        req_ids: list[str],
        placeholder_pos: dict[str, int],
        logprobs_tensors: LogprobsTensors | None,
        fail_fast: bool,
        rank: int | None = None,
        dp_rank: int | None = None,
        device_input_ids: torch.Tensor | None = None,
    ):
        self.fail_fast = fail_fast
        # Named in the fatal event when get_output() fails on the output thread.
        self.rank = rank
        self.dp_rank = dp_rank
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        self.req_ids = req_ids
        self.placeholder_pos = placeholder_pos

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids: torch.Tensor | None = sampled_token_ids
        # Aliased feedback belongs to this forward's completion, not its producer's.
        self._device_input_ids = device_input_ids
        # Start the D2H right behind the sampler, as AsyncGPUModelRunnerOutput
        # does. From get_output() it would queue behind the next forward.
        self._copy_ready_event: torch.Event | None = None
        if sampled_token_ids.is_cpu:
            self._sampled_token_ids_cpu = sampled_token_ids.clone()
        else:
            self._sampled_token_ids_cpu = torch.empty(
                sampled_token_ids.shape,
                dtype=sampled_token_ids.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._sampled_token_ids_cpu.copy_(sampled_token_ids, non_blocking=True)
            self._copy_ready_event = torch.Event(device=sampled_token_ids.device)
            self._copy_ready_event.record()
        # Logprobs ride the same deferral. Only the dense form is free of the
        # tokens: the topk form indexes by the sampled ids, so building it pulls
        # them to the host mid-step and serialises what async just decoupled.
        self._logprobs_tensors = logprobs_tensors

    def get_sampled_token_ids_cpu(self) -> torch.Tensor:
        """Wait for the snapshot shared by output handling and token feedback."""
        if self._copy_ready_event is not None:
            self._copy_ready_event.synchronize()
        # Output handling may retain this object after D2H completes.
        self._sampled_token_ids = None
        self._device_input_ids = None
        return self._sampled_token_ids_cpu

    @worker_fail_fast
    def get_output(self) -> ModelRunnerOutput:
        """Wait for the sampled tokens on the host and return a ModelRunnerOutput.

        Blocks until the copy finishes; the executor decides which thread calls it.
        """
        valid_sampled_token_ids = self.get_sampled_token_ids_cpu().tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        if self._logprobs_tensors is not None:
            # tolists() is where the logprobs D2H actually happens - on this
            # thread, off the step's critical path.
            output.logprobs = self._logprobs_tensors.tolists()
        return output
