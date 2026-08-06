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
phases as child spans, laid end-to-end from the request's wall-clock arrival:

    llm_request (SERVER)   [arrival_time → iteration_timestamp]
    ├─ queue               (scheduled_ts - queued_ts)
    ├─ to_first_token      (first_token_ts - scheduled_ts)
    └─ generate            (last_token_ts - first_token_ts)

No new accounting is introduced — ``RequestStateStats`` already carries every
boundary and upstream computes the same durations from them one line above. We
only draw what is already measured.

## The two clocks

``arrival_time`` is wall-clock; ``queued_ts`` / ``scheduled_ts`` /
``first_token_ts`` / ``last_token_ts`` are **monotonic** (engine core). Upstream
only ever subtracts the monotonic ones, which is valid. Passing one to a span as
an absolute start time is *not* — the span would land at monotonic-origin time,
decades from the request. So phase *durations* come from the monotonic fields
and their *position* is laid out from the single wall-clock anchor.

A consequence: the frontend→engine handoff (``arrival_time`` → ``queued_ts``)
cannot be sized across the clock split, so it is absorbed into the start of the
first phase rather than drawn as a gap we cannot measure.

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

#: Phase boundaries as ``(name, start_attr, end_attr)`` on ``RequestStateStats``.
#:
#: ⚠️ These four fields are **monotonic** (engine core clock), while
#: ``arrival_time`` is wall-clock — the dataclass says so explicitly and
#: ``queued_ts`` is filled from ``time.monotonic()`` events. Upstream only ever
#: subtracts them, which is valid across the two clocks; handing a monotonic
#: value to a span as an absolute timestamp is not (it would place the span
#: decades away from the request). So the durations are taken from these fields
#: and the *position* comes from the one wall-clock anchor we have.
#: Names describe what the interval *measures*, not what the instance is assumed
#: to be doing. Upstream calls the middle one ``time_in_model_prefill``, but on a
#: decode instance in PD-disaggregation that window is KV receive + first decode
#: forward — no prefill happens there (measured on ca2: prefill instance did 43
#: tokens in 188ms while the decode instance's same-named window was 1737ms with
#: forwards steady at 23ms). Naming it ``prefill`` made the waterfall read as
#: "decode is prefilling", which is wrong.
_PHASES: tuple[tuple[str, str, str], ...] = (
    ("queue", "queued_ts", "scheduled_ts"),
    ("to_first_token", "scheduled_ts", "first_token_ts"),
    ("generate", "first_token_ts", "last_token_ts"),
)


def _to_ns(seconds: float) -> int:
    return int(seconds * 1e9)


def _served_model_name() -> str | None:
    """Served model name, or ``None`` when it cannot be determined.

    ``served_model_name`` is what the client asked for and may be a list when
    several aliases are served; the first entry is the canonical one. Falls back
    to the model path.
    """
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    except Exception:  # noqa: BLE001 - never let attribute lookup break tracing
        return None
    if config is None:
        return None
    model_config = getattr(config, "model_config", None)
    if model_config is None:
        return None
    served = getattr(model_config, "served_model_name", None)
    if isinstance(served, list):
        served = served[0] if served else None
    return served or getattr(model_config, "model", None)


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
    # n>1 요청은 자식으로 갈라지므로 "실제 몇 개가 생성됐나" 는 요청 n 과 다를 수
    # 있다. parent 가 있을 때만 그 수를 싣는다 — 없으면 sequence 는 1개다.
    parent_req = req_state.parent_req
    if parent_req is not None:
        child_requests = getattr(parent_req, "child_requests", None)
        if child_requests is not None:
            attributes[SpanAttributes.GEN_AI_USAGE_NUM_SEQUENCES] = len(child_requests)

    # OutputProcessor 는 vllm_config 를 들고 있지 않다. `_or_none` 변형을 쓰는 이유는
    # config context 밖에서 호출되면 예외가 아니라 None 이 와야 하기 때문이다 —
    # 모델명을 못 얻는 것이 trace 를 잃을 이유는 아니다.
    model_name = _served_model_name()
    if model_name:
        attributes[SpanAttributes.GEN_AI_RESPONSE_MODEL] = model_name

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
        _emit_phase_spans(request_span, metrics, arrival_time_ns)
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


def _emit_phase_spans(request_span: Any, metrics: Any, anchor_ns: int) -> None:
    """Draw queue/prefill/decode under ``request_span``.

    ``anchor_ns`` is the wall-clock start of the request. Phase *durations* come
    from the monotonic engine-core timestamps (differences are valid across the
    clock split) and are laid end-to-end from the anchor, so the children sit
    inside the parent instead of at a monotonic-clock absolute time.

    The frontend→engine handoff (arrival_time → queued_ts) is not measurable
    across the two clocks, so it is absorbed into the start of the first phase
    rather than reported as a gap we cannot size.

    A phase whose boundaries are missing or non-advancing is skipped rather than
    drawn as a zero/negative-width span — a request that finished during prefill
    has no decode phase, and inventing one would misreport it.
    """
    try:
        from opentelemetry import trace as otel_trace

        parent_context = otel_trace.set_span_in_context(request_span)
    except ImportError:  # pragma: no cover - guarded by _start_request_span
        return

    cursor_ns = anchor_ns
    for name, start_attr, end_attr in _PHASES:
        start = getattr(metrics, start_attr, None)
        end = getattr(metrics, end_attr, None)
        if start is None or end is None or end <= start:
            continue
        duration_ns = _to_ns(end - start)
        instrument_manual(
            span_name=name,
            start_time=cursor_ns,
            end_time=cursor_ns + duration_ns,
            context=parent_context,
        )
        cursor_ns += duration_ns
