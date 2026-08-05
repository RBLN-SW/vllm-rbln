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

"""Fill the engine-step latency attributes upstream reserves but never sets.

``vllm.tracing.utils.SpanAttributes`` declares three constants that nothing in
vLLM writes:

    GEN_AI_LATENCY_TIME_IN_SCHEDULER      gen_ai.latency.time_in_scheduler
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD  gen_ai.latency.time_in_model_forward
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE  gen_ai.latency.time_in_model_execute

They are the natural names for the three engine-step boundaries this plugin
already delineates for the torch profiler (``schedule: …`` in the scheduler,
``rbln_model_runner: forward`` in the runner). This module measures those
boundaries and emits them as spans carrying the reserved attribute, so the names
downstream analysis expects are the names that appear.

## These are step spans, not request children

``schedule()`` and ``execute_model()`` run **once per engine step** and a step
serves a batch of requests (continuous batching). A step span therefore has no
single parent request and is *not* nested under ``llm_request`` — its parent is
whatever context the worker process carries (``_get_smart_context()`` falls back
to the ``traceparent`` injected into ``os.environ`` at worker spawn).

That is the honest shape of the operation. Per-request attribution of a batched
step would require splitting each step's cost across the requests in it, which
is a different (and much larger) change; the per-request view is already covered
by ``tracing_request_phases``.
"""

from __future__ import annotations

import time
from typing import Any

from vllm.logger import init_logger
from vllm.tracing import SpanAttributes, is_tracing_available

from vllm_rbln.patches import register_patch
from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

logger = init_logger(__name__)

_original_schedule = RBLNScheduler.schedule
_original_execute_model = RBLNModelRunner.execute_model
_original_load_model = RBLNModelRunner.load_model


def _timed_span(span_name: str, attribute: str, fn, *args, **kwargs):
    """Run ``fn`` inside ``span_name``, recording its duration on ``attribute``.

    The duration is also set as an attribute (not just implied by the span
    bounds) because that is the form the reserved ``SpanAttributes`` constant
    describes, and TraceQL can filter on it directly.

    Falls through to a plain call when OTel is missing so this never becomes a
    hard dependency of the serving path.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:  # pragma: no cover - guarded by `condition`
        return fn(*args, **kwargs)

    tracer = otel_trace.get_tracer(__name__)
    started = time.perf_counter()
    with tracer.start_as_current_span(span_name) as span:
        try:
            return fn(*args, **kwargs)
        finally:
            span.set_attribute(attribute, time.perf_counter() - started)


@register_patch(
    target="vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler.schedule",
    reason=(
        "gen_ai.latency.time_in_scheduler is declared in vLLM's SpanAttributes "
        "but never written, so scheduling cost is invisible in traces and gets "
        "charged to whatever span encloses the step. Measure schedule() and "
        "emit it under that reserved name."
    ),
    condition=is_tracing_available,
)
def patched_schedule(self: RBLNScheduler, *args: Any, **kwargs: Any):
    return _timed_span(
        "schedule",
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_SCHEDULER,
        _original_schedule,
        self,
        *args,
        **kwargs,
    )


@register_patch(
    target="vllm_rbln.v1.worker.rbln_model_runner.RBLNModelRunner.execute_model",
    reason=(
        "gen_ai.latency.time_in_model_execute is declared in vLLM's "
        "SpanAttributes but never written. execute_model is the step boundary "
        "the runner already marks for the torch profiler, so measure it there."
    ),
    condition=is_tracing_available,
)
def patched_execute_model(self: RBLNModelRunner, *args: Any, **kwargs: Any):
    return _timed_span(
        "model_execute",
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE,
        _original_execute_model,
        self,
        *args,
        **kwargs,
    )


@register_patch(
    target="vllm_rbln.v1.worker.rbln_model_runner.RBLNModelRunner.load_model",
    reason=(
        "gen_ai.latency.time_in_model_forward is declared in vLLM's "
        "SpanAttributes but never written. The forward is `self.model_executable`, "
        "an instance attribute assigned in load_model (a compiled/wrapped "
        "callable), so there is no method to patch — wrap the callable once, "
        "right after load_model produces it."
    ),
    condition=is_tracing_available,
)
def patched_load_model(self: RBLNModelRunner, *args: Any, **kwargs: Any):
    result = _original_load_model(self, *args, **kwargs)
    executable = getattr(self, "model_executable", None)
    if executable is None or getattr(executable, "_rbln_traced", False):
        return result

    def traced_executable(*call_args: Any, **call_kwargs: Any):
        return _timed_span(
            "model_forward",
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD,
            executable,
            *call_args,
            **call_kwargs,
        )

    # Marked so a second load_model (warmup re-entry) does not stack wrappers,
    # which would double-count the forward.
    traced_executable._rbln_traced = True
    self.model_executable = traced_executable
    return result
