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

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


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


def traced_remote_fetch(func: F) -> F:
    """Open ``remote_fetch`` around a per-request NIXL read.

    ``NixlConnector.start_load_kv`` issues the reads for every request scheduled
    in the step, so the step-level ``nixl.kv_transfer`` span cannot say which
    request's transfer was slow. This wraps the per-request read and tags it with
    ``ca.request.id`` — the key the rest of the stack joins on (gateway span
    attribute, harness correlation log, drill-down panels).

    The join is by attribute rather than parentage on purpose: the worker has no
    per-request trace context (``_get_smart_context()`` falls back to the
    ``traceparent`` injected into ``os.environ`` at worker spawn), so these spans
    cannot nest under ``llm_request``.
    """

    @functools.wraps(func)
    def wrapper(self, req_id: str, meta, *args: Any, **kwargs: Any):
        try:
            from opentelemetry import trace as otel_trace
        except ImportError:
            return func(self, req_id, meta, *args, **kwargs)

        tracer = otel_trace.get_tracer(func.__module__)
        with tracer.start_as_current_span("remote_fetch") as span:
            span.set_attribute("ca.request.id", req_id)
            remote = getattr(meta, "remote", None)
            block_ids = getattr(remote, "block_ids", None)
            if block_ids is not None:
                span.set_attribute("nixl.block_count", len(block_ids))
            engine_id = getattr(remote, "engine_id", None)
            if engine_id is not None:
                span.set_attribute("nixl.remote_engine_id", str(engine_id))
            return func(self, req_id, meta, *args, **kwargs)

    return wrapper  # type: ignore[return-value]
