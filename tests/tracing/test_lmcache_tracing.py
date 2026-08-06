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

"""Guards for :mod:`vllm_rbln.patches.tracing_lmcache`.

Each patch is a wrapper around a third-party call, so the two things that can
break silently are the wrapper swallowing the original's return value and the
span landing outside the request's trace. Both are pinned here.

The originals are replaced per test: exercising the real LMCache calls would
need a live cache engine, and what is under test is the wrapping, not LMCache.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip("lmcache", reason="LMCache tracing patches need LMCache installed")

from vllm_rbln.patches import tracing_lmcache  # noqa: E402

_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_HEADERS = {"traceparent": f"00-{_TRACE_ID}-00f067aa0ba902b7-01"}
_REQ_ID = "chatcmpl-abc"


@pytest.fixture
def spans():
    """Collect finished spans, with a provider scoped to the test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = otel_trace.get_tracer_provider()
    otel_trace._TRACER_PROVIDER = provider  # set_tracer_provider() is once-only
    yield exporter
    otel_trace._TRACER_PROVIDER = previous


def _named(exporter, name):
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _trace_id_of(span) -> str:
    return format(span.context.trace_id, "032x")


def test_lookup_reports_the_hit_in_the_requests_trace(spans, monkeypatch):
    monkeypatch.setattr(
        tracing_lmcache, "_ORIGINAL_LOOKUP", lambda self, request, computed: 512
    )
    impl = SimpleNamespace()
    request = SimpleNamespace(request_id=_REQ_ID, trace_headers=_HEADERS)

    matched = tracing_lmcache.get_num_new_matched_tokens(impl, request, 128)

    assert matched == 512, "the wrapper must not swallow the lookup result"
    (span,) = _named(spans, "lmcache.lookup")
    assert _trace_id_of(span) == _TRACE_ID
    assert span.attributes["ca.request.id"] == _REQ_ID
    assert span.attributes["lmcache.num_computed_tokens"] == 128
    assert span.attributes["lmcache.num_matched_tokens"] == 512
    assert span.attributes["lmcache.hit"] is True


def test_a_miss_is_recorded_as_a_miss(spans, monkeypatch):
    monkeypatch.setattr(
        tracing_lmcache, "_ORIGINAL_LOOKUP", lambda self, request, computed: None
    )

    tracing_lmcache.get_num_new_matched_tokens(
        SimpleNamespace(), SimpleNamespace(request_id=_REQ_ID, trace_headers=None), 0
    )

    (span,) = _named(spans, "lmcache.lookup")
    assert span.attributes["lmcache.num_matched_tokens"] == 0
    assert span.attributes["lmcache.hit"] is False


def test_the_carrier_only_ships_requests_in_this_metadata(monkeypatch):
    """Headers for requests the step does not mention must not cross the wire."""
    monkeypatch.setattr(
        tracing_lmcache,
        "_ORIGINAL_BUILD_META",
        lambda self, scheduler_output: SimpleNamespace(
            requests=[SimpleNamespace(req_id=_REQ_ID)]
        ),
    )
    impl = SimpleNamespace(
        _rbln_trace_headers={_REQ_ID: _HEADERS, "some-other-request": _HEADERS}
    )

    metadata = tracing_lmcache.build_connector_meta(impl, None)

    assert metadata.rbln_trace_headers == {_REQ_ID: _HEADERS}


def test_the_worker_picks_the_headers_up_off_the_metadata(monkeypatch):
    monkeypatch.setattr(tracing_lmcache, "_WORKER_TRACE_HEADERS", {})
    monkeypatch.setattr(
        tracing_lmcache, "_ORIGINAL_START_LOAD_KV", lambda self, ctx, **kwargs: "loaded"
    )
    metadata = SimpleNamespace(rbln_trace_headers={_REQ_ID: _HEADERS})
    impl = SimpleNamespace(
        _parent=SimpleNamespace(_get_connector_metadata=lambda: metadata)
    )

    assert tracing_lmcache.start_load_kv(impl, None) == "loaded"
    assert tracing_lmcache._WORKER_TRACE_HEADERS == {_REQ_ID: _HEADERS}


def test_retrieve_lands_in_the_requests_trace_with_its_token_counts(spans, monkeypatch):
    monkeypatch.setattr(tracing_lmcache, "_WORKER_TRACE_HEADERS", {_REQ_ID: _HEADERS})
    retrieved = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 384))
    monkeypatch.setattr(
        tracing_lmcache, "_ORIGINAL_RETRIEVE", lambda self, *a, **kw: retrieved
    )

    result = tracing_lmcache.retrieve(None, list(range(512)), req_id=_REQ_ID)

    assert result is retrieved
    (span,) = _named(spans, "lmcache.retrieve")
    assert _trace_id_of(span) == _TRACE_ID
    assert span.attributes["ca.request.id"] == _REQ_ID
    assert span.attributes["lmcache.num_requested_tokens"] == 512
    assert span.attributes["lmcache.num_retrieved_tokens"] == 384


def test_store_lands_in_the_requests_trace(spans, monkeypatch):
    monkeypatch.setattr(tracing_lmcache, "_WORKER_TRACE_HEADERS", {_REQ_ID: _HEADERS})
    monkeypatch.setattr(tracing_lmcache, "_ORIGINAL_STORE", lambda self, *a, **kw: None)

    tracing_lmcache.store(None, list(range(1024)), req_id=_REQ_ID)

    (span,) = _named(spans, "lmcache.store")
    assert _trace_id_of(span) == _TRACE_ID
    assert span.attributes["lmcache.num_tokens"] == 1024


def test_an_untraced_request_does_not_borrow_another_requests_trace(spans, monkeypatch):
    monkeypatch.setattr(tracing_lmcache, "_WORKER_TRACE_HEADERS", {_REQ_ID: _HEADERS})
    retrieved = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 0))
    monkeypatch.setattr(
        tracing_lmcache, "_ORIGINAL_RETRIEVE", lambda self, *a, **kw: retrieved
    )

    tracing_lmcache.retrieve(None, [1, 2, 3], req_id="a-request-nobody-traced")

    (span,) = _named(spans, "lmcache.retrieve")
    assert _trace_id_of(span) != _TRACE_ID


def test_header_map_is_bounded(monkeypatch):
    """A lookup that never leads to a scheduled request leaves an entry behind."""
    monkeypatch.setattr(tracing_lmcache, "_TRACE_HEADER_LIMIT", 3)
    store: dict = {}

    for index in range(5):
        tracing_lmcache._remember(store, f"req-{index}", _HEADERS)

    assert list(store) == ["req-2", "req-3", "req-4"]
