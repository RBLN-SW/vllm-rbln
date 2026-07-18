# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import vllm_rbln.tracing as tracing
from vllm_rbln.tracing import (
    batch_span_attributes,
    configure_batch_span_link_limit,
    start_batch_span,
)


class _FakeSpanContext:
    def __init__(self, trace_id: int, span_id: int, *, is_valid: bool = True):
        self.trace_id = trace_id
        self.span_id = span_id
        self.is_valid = is_valid


class _FakeLink:
    def __init__(self, context):
        self.context = context


def _install_fake_otel(monkeypatch, contexts):
    tracer = MagicMock()
    span_context_manager = object()
    tracer.start_as_current_span.return_value = span_context_manager

    fake_trace = SimpleNamespace(
        get_current_span=lambda context: SimpleNamespace(
            get_span_context=lambda: context
        ),
        get_tracer=lambda _: tracer,
    )
    otel_module = ModuleType("opentelemetry")
    otel_module.__path__ = []
    otel_module.trace = fake_trace  # type: ignore[attr-defined]
    otel_trace_module = ModuleType("opentelemetry.trace")
    otel_trace_module.Link = _FakeLink  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", otel_module)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", otel_trace_module)
    monkeypatch.setattr(tracing, "is_tracing_available", lambda: True)
    monkeypatch.setattr(
        tracing,
        "extract_trace_context",
        lambda headers: contexts.get(headers["traceparent"]),
    )
    return tracer, span_context_manager


def test_batch_span_attributes():
    assert batch_span_attributes({"req-1": 4, "req-2": 2}) == {
        "vllm.request_ids": ["req-1", "req-2"],
        "vllm.num_requests": 2,
        "vllm.num_scheduled_tokens": 6,
    }


def test_configure_batch_span_link_limit_uses_max_batch_size(monkeypatch):
    monkeypatch.delenv("OTEL_SPAN_LINK_COUNT_LIMIT", raising=False)

    configure_batch_span_link_limit(256)

    assert tracing.os.environ["OTEL_SPAN_LINK_COUNT_LIMIT"] == "256"


def test_configure_batch_span_link_limit_preserves_user_setting(monkeypatch):
    monkeypatch.setenv("OTEL_SPAN_LINK_COUNT_LIMIT", "512")

    configure_batch_span_link_limit(256)

    assert tracing.os.environ["OTEL_SPAN_LINK_COUNT_LIMIT"] == "512"


def test_start_batch_span_is_noop_when_tracing_is_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "is_tracing_available", lambda: False)

    with start_batch_span(
        "vllm.model_execute",
        enabled=True,
        request_trace_headers={},
    ) as span:
        assert span is None


def test_start_batch_span_links_unique_request_contexts(monkeypatch):
    first = _FakeSpanContext(1, 10)
    second = _FakeSpanContext(2, 20)
    invalid = _FakeSpanContext(0, 0, is_valid=False)
    tracer, span_context_manager = _install_fake_otel(
        monkeypatch,
        {
            "first": first,
            "second": second,
            "invalid": invalid,
        },
    )
    attributes = {"vllm.num_requests": 4}

    result = start_batch_span(
        "vllm.model_execute",
        enabled=True,
        request_trace_headers={
            "req-1": {"traceparent": "first"},
            "req-2": {"traceparent": "second"},
            "req-3": {"traceparent": "first"},
            "req-4": {"traceparent": "invalid"},
        },
        attributes=attributes,
    )

    assert result is span_context_manager
    call = tracer.start_as_current_span.call_args
    assert call.args == ("vllm.model_execute",)
    assert call.kwargs["attributes"] == attributes
    assert [link.context for link in call.kwargs["links"]] == [first, second]
    assert "context" not in call.kwargs
