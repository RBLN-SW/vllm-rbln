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
"""Wall-clock instrumentation for the vllm model path.

The shared context records execute_model through sample_tokens as E2E and
sums the model call, sampler, and blocking bookkeeping as MODEL + SAMPLE.
Spec decode's postprocess gather also holds a blocking read, so its profiler
region is included in that sum. Drafter time stays in the E2E residual.
Non-last PP ranks close their pass after the hand-off waits on compute.
"""

import contextlib
import functools
import time

from vllm.sequence import IntermediateTensors
from vllm.v1.utils import record_function_or_nullcontext

from vllm_rbln.patches import register_patch
from vllm_rbln.v1.worker.metrics import (
    _CTX_ATTR,
    PerformanceContext,
    Phase,
    get_performance_context,
    metrics_enabled,
)
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

_execute_model = RBLNModelRunner.execute_model
_sample_tokens = RBLNModelRunner.sample_tokens
_sample = RBLNModelRunner._sample
_load_model = RBLNModelRunner.load_model
_determine_batch_execution_and_padding = (
    RBLNModelRunner._determine_batch_execution_and_padding
)
_bookkeeping_sync = RBLNModelRunner._bookkeeping_sync
_shutdown = RBLNWorker.shutdown


@functools.wraps(_execute_model)
def execute_model(self, *args, **kwargs):
    global _ACTIVE_CTX
    ctx = get_performance_context(self)
    _ACTIVE_CTX = ctx
    ctx.start_pass()
    output = _execute_model(self, *args, **kwargs)
    # A None pass is closed by sample_tokens, an IntermediateTensors pass by
    # send_handoff below; anything else nobody would end.
    if output is not None and not isinstance(output, IntermediateTensors):
        ctx.end_pass()
    return output


@functools.wraps(_sample_tokens)
def sample_tokens(self, *args, **kwargs):
    output = _sample_tokens(self, *args, **kwargs)
    get_performance_context(self).end_pass()
    return output


@functools.wraps(_bookkeeping_sync)
def bookkeeping_sync(self, *args, **kwargs):
    start = time.perf_counter()
    try:
        return _bookkeeping_sync(self, *args, **kwargs)
    finally:
        get_performance_context(self).add_graph_time(time.perf_counter() - start)


@functools.wraps(_sample)
def sample(self, *args, **kwargs):
    if self.is_intermediate_chunked_prefill:
        # Returns a placeholder without running the sampler, so there is no sampler
        # time to attribute -- only the cost of building that placeholder.
        return _sample(self, *args, **kwargs)
    start = time.perf_counter()
    output = _sample(self, *args, **kwargs)
    get_performance_context(self).add_graph_time(time.perf_counter() - start)
    return output


@functools.wraps(_load_model)
def load_model(self, *args, **kwargs):
    _load_model(self, *args, **kwargs)
    executable = self.model_executable

    def timed_executable(*call_args, **call_kwargs):
        ctx = get_performance_context(self)
        if not ctx.in_pass:
            # Warm-up and dummy runs: skip the clock reads as well as the record.
            return executable(*call_args, **call_kwargs)
        start = time.perf_counter()
        output = executable(*call_args, **call_kwargs)
        ctx.add_graph_time(time.perf_counter() - start)
        return output

    self.model_executable = timed_executable


@functools.wraps(_determine_batch_execution_and_padding)
def determine_batch_execution_and_padding(self, *args, **kwargs):
    result = _determine_batch_execution_and_padding(self, *args, **kwargs)
    batch_desc, _route, _num_tokens_across_dp = result
    if batch_desc is None:
        # A drained group runs nothing, so no pass opens and there is no phase to
        # attribute this step to.
        return result
    num_tokens_padded = batch_desc.num_tokens_padded
    if self.is_prefill:
        phase = Phase.PREFILL
    elif num_tokens_padded == self.max_num_tokens:
        # num_tokens_padded is the MoE pad dimension, so matching the prefill dim means
        # this decode ran the prefill-sized graph. Only reachable under DP, where the
        # padding is computed at all. Changing how the runner pads means revisiting
        # this line.
        phase = Phase.PADDED_DECODE
    else:
        phase = Phase.DECODE
    # A no-op on the dummy run, which reaches this with no pass open.
    get_performance_context(self).mark_phase(phase)
    return result


_send_handoff = RBLNWorker._send_handoff


@functools.wraps(_send_handoff)
def send_handoff(self, *args, **kwargs):
    # The send waits on this stage's compute, so its wall carries the stage's time.
    # No try/finally: a raising hand-off leaves the pass open, like any raising pass.
    ctx = get_performance_context(self.model_runner)
    start = time.perf_counter()
    output = _send_handoff(self, *args, **kwargs)
    ctx.add_graph_time(time.perf_counter() - start)
    ctx.end_pass()
    return output


@functools.wraps(_shutdown)
def shutdown(self):
    # The runner is built in init_device(), not __init__, so a worker that failed
    # before that has none and must still tear down.
    runner = getattr(self, "model_runner", None)
    ctx = getattr(runner, _CTX_ATTR, None)
    if ctx is not None:
        ctx.print_stats()
    _shutdown(self)


_ACTIVE_CTX: PerformanceContext | None = None

# The region that holds the spec-only logits gather (see the module docstring); it
# belongs to the graph sum, and is empty without spec decode.
_GRAPH_REGION = "rbln_model_runner: postprocess"


@contextlib.contextmanager
def timed_region(name: str):
    ctx = _ACTIVE_CTX
    if name != _GRAPH_REGION or ctx is None or not ctx.in_pass:
        with record_function_or_nullcontext(name):
            yield
        return
    start = time.perf_counter()
    try:
        with record_function_or_nullcontext(name):
            yield
    finally:
        ctx.add_graph_time(time.perf_counter() - start)


_RUNNER = "vllm_rbln.v1.worker.rbln_model_runner.RBLNModelRunner"
_WORKER = "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"


def _register_patches() -> None:
    for target, replacement, reason in (
        (
            f"{_RUNNER}.execute_model",
            execute_model,
            "Opens the measured pass. The runner cannot mark its own boundary without "
            "carrying metrics code on the serving path.",
        ),
        (
            f"{_RUNNER}.sample_tokens",
            sample_tokens,
            "Closes the measured pass, which spans two inbound calls and so has no "
            "single method to wrap.",
        ),
        (
            f"{_RUNNER}._sample",
            sample,
            "Times the sampler. The sampler call is not a method of its own, so the "
            "whole _sample is the narrowest wrappable range.",
        ),
        (
            f"{_RUNNER}.load_model",
            load_model,
            "Wraps the compiled model_executable once it exists; it is an instance "
            "attribute, so it cannot be replaced through the registry.",
        ),
        (
            f"{_RUNNER}._bookkeeping_sync",
            bookkeeping_sync,
            "Holds the step's first blocking read of the sampler output, which the "
            "graph slice would otherwise miss.",
        ),
        (
            f"{_RUNNER}._determine_batch_execution_and_padding",
            determine_batch_execution_and_padding,
            "Reads the pass phase where it is decided. The padded token dimension is a "
            "local of execute_model and is not observable from any wrapped callable.",
        ),
        (
            f"{_WORKER}._send_handoff",
            send_handoff,
            "Times the hand-off and closes its pass; left outside the pass, "
            "the send hides a non-last rank's stage time.",
        ),
        (
            f"{_WORKER}.shutdown",
            shutdown,
            "Reports the collected metrics before teardown; the worker holds the only "
            "shutdown hook that still has the runner.",
        ),
        (
            "vllm_rbln.v1.worker.rbln_model_runner.record_function_or_nullcontext",
            timed_region,
            "Spec decode syncs on the forward in the spec-only logits gather, "
            "an inline block of execute_model with no method to wrap.",
        ),
    ):
        register_patch(
            target=target,
            reason=reason,
            key=f"vllm_rbln.patches.metrics.{target.rsplit('.', 1)[-1]}",
            owner_module="vllm_rbln.patches.metrics",
            condition=metrics_enabled,
        )(replacement)


_register_patches()
