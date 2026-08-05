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

"""Emit per-request phase spans as children of ``llm_request``.

Upstream records the phase breakdown as *attributes* on a single
``llm_request`` span (``gen_ai.latency.time_in_queue`` /
``…time_in_model_prefill`` / ``…time_in_model_decode``). A trace viewer then
shows one flat span, so "which phase was slow" has to be read off numbers
instead of the waterfall that every other layer of the stack renders as nested
spans (gateway → sidecar → vLLM).

This patch keeps every upstream attribute and additionally emits the three
phases as child spans:

    llm_request (SERVER)   [arrival_time      → iteration_timestamp]
    ├─ queue               [queued_ts         → scheduled_ts]
    ├─ prefill             [scheduled_ts      → first_token_ts]
    └─ decode              [first_token_ts    → last_token_ts]

No new accounting is introduced — ``RequestStateStats`` already carries the
absolute timestamps for every boundary, and upstream computes the same
durations from them one line above. We only draw what is already measured.

## Why the whole method is replaced

``instrument_manual`` does not return the span it creates, so a caller cannot
use it as a parent. To nest anything under ``llm_request`` the span object has
to be held, which means owning its creation. The attribute set below is a
faithful copy of upstream's; when upstream adds an attribute this patch has to
follow (``test_tracing_request_phases`` pins the copied set).

## Per-step spans are deliberately out of scope

``decode.step × N`` would need per-step timestamps, which ``metrics`` does not
carry, and one span per generated token turns a 500-token response into 500
spans — Tempo lookup and waterfall readability both collapse. The aggregate
decode span plus upstream's per-step metrics cover the same question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
    is_tracing_available,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine.output_processor import OutputProcessor

from vllm_rbln.patches import register_patch

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)

#: Child spans are drawn from these boundaries. ``(name, start_attr, end_attr)``
#: where the attrs are absolute epoch seconds on ``RequestStateStats``.
_PHASES: tuple[tuple[str, str, str], ...] = (
    ("queue", "queued_ts", "scheduled_ts"),
    ("prefill", "scheduled_ts", "first_token_ts"),
    ("decode", "first_token_ts", "last_token_ts"),
)


def _to_ns(seconds: float) -> int:
    return int(seconds * 1e9)


@register_patch(
    target="vllm.v1.engine.output_processor.OutputProcessor.do_tracing",
    reason=(
        "Upstream emits the request phase breakdown as attributes on a single "
        "llm_request span, so a trace viewer shows one flat span while every "
        "other layer (gateway/sidecar/vLLM init) renders as nested spans. "
        "Emit queue/prefill/decode as children of llm_request using the "
        "absolute timestamps RequestStateStats already carries. Replacing the "
        "whole method is required because instrument_manual does not return "
        "its span, so it cannot be used as a parent."
    ),
    condition=is_tracing_available,
)
def patched_do_tracing(
    self: OutputProcessor,
    engine_core_output: EngineCoreOutput,
    req_state: RequestState,
    iteration_stats: IterationStats | None,
) -> None:
    assert req_state.stats is not None
    assert iteration_stats is not None

    metrics = req_state.stats
    arrival_time_ns = _to_ns(metrics.arrival_time)
    trace_context = extract_trace_context(engine_core_output.trace_headers)
    prompt_length = length_from_prompt_token_ids_or_embeds(
        req_state.prompt_token_ids, req_state.prompt_embeds
    )

    # Calculate timing metrics — identical to upstream.
    e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
    queued_time = metrics.scheduled_ts - metrics.queued_ts
    prefill_time = metrics.first_token_ts - metrics.scheduled_ts
    decode_time = metrics.last_token_ts - metrics.first_token_ts
    inference_time = metrics.last_token_ts - metrics.scheduled_ts

    attributes: dict[str, Any] = {
        SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
            metrics.first_token_latency
        ),
        SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
        SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
        SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (metrics.num_generation_tokens),
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
        SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
    }

    if req_state.top_p:
        attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
    if req_state.max_tokens_param:
        attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
            req_state.max_tokens_param
        )
    if req_state.temperature:
        attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = req_state.temperature
    if req_state.n:
        attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n

    request_span = _start_request_span(
        start_time=arrival_time_ns,
        attributes=attributes,
        context=trace_context,
    )
    if request_span is None:
        # OTel went away between the condition check and here — fall back to
        # upstream's single-span behaviour so tracing degrades, not breaks.
        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )
        return

    try:
        _emit_phase_spans(request_span, metrics)
    finally:
        request_span.end(end_time=_to_ns(iteration_stats.iteration_timestamp))


def _start_request_span(
    *,
    start_time: int,
    attributes: dict[str, Any],
    context: Any,
):
    """Create ``llm_request`` and return the span so children can nest under it.

    Returns ``None`` when OTel is unavailable, which the caller treats as
    "behave like upstream".
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return None

    tracer = otel_trace.get_tracer(__name__)
    span = tracer.start_span(
        name="llm_request",
        context=context,
        start_time=start_time,
        kind=SpanKind.SERVER,
    )
    span.set_attributes(attributes)
    return span


def _emit_phase_spans(request_span: Any, metrics: Any) -> None:
    """Draw queue/prefill/decode under ``request_span``.

    A phase whose boundaries are missing or non-advancing is skipped rather
    than drawn as a zero/negative-width span — a request that finished during
    prefill has no decode phase, and inventing one would misreport it.
    """
    try:
        from opentelemetry import trace as otel_trace

        parent_context = otel_trace.set_span_in_context(request_span)
    except ImportError:  # pragma: no cover - guarded by _start_request_span
        return

    for name, start_attr, end_attr in _PHASES:
        start = getattr(metrics, start_attr, None)
        end = getattr(metrics, end_attr, None)
        if start is None or end is None or end <= start:
            continue
        instrument_manual(
            span_name=name,
            start_time=_to_ns(start),
            end_time=_to_ns(end),
            context=parent_context,
        )
