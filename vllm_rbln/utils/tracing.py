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

"""Tracing helpers for RBLN-owned code paths.

:func:`vllm.tracing.instrument` opens a span but has no way to record how long
the wrapped call took as an *attribute* — its ``attributes`` argument is fixed at
decoration time. Three constants in ``vllm.tracing.SpanAttributes`` are declared
for exactly that purpose and nothing in vLLM writes them:

    gen_ai.latency.time_in_scheduler
    gen_ai.latency.time_in_model_execute
    gen_ai.latency.time_in_model_forward

:func:`instrument_with_latency` fills them. The duration is redundant with the
span's own bounds, but the attribute is what TraceQL filters on and what the
reserved constant describes, so both are emitted.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

#: Set by :func:`pinned_span_id` and consumed by the patched OTel id generator
#: (``vllm_rbln.patches.tracing_span_ids``). Thread-local because the frontend
#: finishes several requests concurrently and each pins its own id.
_PINNED_SPAN_ID = threading.local()


def take_pinned_span_id() -> int | None:
    """Consume the pinned span id for this thread, if one is waiting.

    Consumed rather than read so a pin can only ever apply to the single span it
    was set for; anything else generated on this thread gets a random id.
    """
    pinned = getattr(_PINNED_SPAN_ID, "value", None)
    if pinned is not None:
        _PINNED_SPAN_ID.value = None
    return pinned


@contextlib.contextmanager
def pinned_span_id(span_id: int | None):
    """Force the next span created on this thread to use ``span_id``.

    OTel has no per-call span id override — ``Tracer.start_span`` asks its id
    generator and that is that. Pinning is how ``llm_request`` gets an id that a
    *different process* can predict; see :func:`derive_request_span_id`.
    """
    if span_id is None:
        yield
        return
    _PINNED_SPAN_ID.value = span_id
    try:
        yield
    finally:
        _PINNED_SPAN_ID.value = None


def derive_request_span_id(trace_id: int, parent_span_id: int) -> int:
    """The span id ``llm_request`` will have, computed from its trace context.

    ``llm_request`` is created in the API-server process when the request
    finishes, with a span id nothing else can know. The engine-core and worker
    processes need it *while the request runs* — a KV-transfer or cache span
    that cannot name its parent ends up a sibling of the request it belongs to,
    which reads in a waterfall as if it ran alongside the request rather than
    inside it.

    So both sides compute the id instead of communicating it. The inputs are the
    two things every side already has from the inbound ``traceparent``: the trace
    id, and the span id of the caller (the sidecar leg). Including the caller
    matters — prefill and decode share a trace but are different legs with
    different ``llm_request`` spans, and the trace id alone would collide them.

    SHA-256 rather than anything cheaper because the value must agree across
    processes; ``hash()`` is per-process salted.
    """
    digest = hashlib.sha256(
        trace_id.to_bytes(16, "big")
        + parent_span_id.to_bytes(8, "big")
        + b"vllm.llm_request"
    ).digest()
    # 0 is INVALID_SPAN_ID; the odds are astronomical but a trace silently
    # losing its parentage is not worth leaving to chance.
    return int.from_bytes(digest[:8], "big") or 1


def request_span_context(trace_headers: Mapping[str, str] | None):
    """Context whose current span is the request's ``llm_request``.

    Returns ``None`` when the request carries no usable trace context, which
    callers treat as "leave OTel's default parent alone".

    The parent is a non-recording span: this process is not the one that emits
    ``llm_request``, it only needs to point at it.
    """
    if not trace_headers:
        return None
    try:
        from opentelemetry import trace as otel_trace
        from vllm.tracing import extract_trace_context
    except ImportError:
        return None

    context = extract_trace_context(trace_headers)
    if context is None:
        return None
    caller = otel_trace.get_current_span(context).get_span_context()
    if not caller.is_valid:
        return None

    request_span = otel_trace.SpanContext(
        trace_id=caller.trace_id,
        span_id=derive_request_span_id(caller.trace_id, caller.span_id),
        is_remote=True,
        trace_flags=caller.trace_flags,
        trace_state=caller.trace_state,
    )
    return otel_trace.set_span_in_context(otel_trace.NonRecordingSpan(request_span))


def instrument_with_latency(*, span_name: str, attribute: str) -> Callable[[F], F]:
    """Open ``span_name`` around the call and record its duration on ``attribute``.

    Degrades to the undecorated function when OpenTelemetry is absent, so this
    never becomes a hard dependency of the serving path. The span is opened
    lazily per call (not at decoration time) so a tracer configured later still
    picks it up — the same reason ``instrument_otel`` resolves its tracer inside
    the wrapper.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                from opentelemetry import trace as otel_trace
            except ImportError:
                return func(*args, **kwargs)

            tracer = otel_trace.get_tracer(func.__module__)
            started = time.perf_counter()
            with tracer.start_as_current_span(span_name) as span:
                try:
                    return func(*args, **kwargs)
                finally:
                    span.set_attribute(attribute, time.perf_counter() - started)

        return wrapper  # type: ignore[return-value]

    return decorator


def traced_callable(callable_: Callable[..., Any], *, span_name: str, attribute: str):
    """Wrap a *callable object* (not a method) the same way.

    Needed for ``RBLNModelRunner.model_executable``, which is an instance
    attribute holding a compiled/wrapped model rather than a method that could
    carry a decorator.

    Returns ``callable_`` unchanged if it is already wrapped, so a second
    ``load_model`` (warmup re-entry) cannot stack wrappers and double-count the
    forward.
    """
    if getattr(callable_, "_rbln_latency_traced", False):
        return callable_

    @functools.wraps(callable_)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            from opentelemetry import trace as otel_trace
        except ImportError:
            return callable_(*args, **kwargs)

        tracer = otel_trace.get_tracer(__name__)
        started = time.perf_counter()
        with tracer.start_as_current_span(span_name) as span:
            try:
                return callable_(*args, **kwargs)
            finally:
                span.set_attribute(attribute, time.perf_counter() - started)

    wrapper._rbln_latency_traced = True  # type: ignore[attr-defined]
    return wrapper
