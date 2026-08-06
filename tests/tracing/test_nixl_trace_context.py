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

"""Guards for the scheduler→worker trace-context carrier.

The KV-transfer spans are emitted in the worker, which has no per-request trace
context of its own. It gets one because the scheduler puts each request's
``traceparent`` into the connector metadata. These tests pin that channel: the
metadata subclass upstream must still accept, the transport that carries it, and
the scheduler actually filling it in.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnectorMetadata

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.scheduler import (
    RblnNixlConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.worker import (
    _step_span_links,
)

_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
_KV_PARAMS = {
    "remote_block_ids": [9],
    "remote_engine_id": "engine-p",
    "remote_request_id": "remote-1",
    "remote_host": "10.0.0.1",
    "remote_port": 5659,
    "tp_size": 1,
}


def _request(trace_headers):
    """The two ``Request`` fields our ``build_connector_meta`` reads."""
    return SimpleNamespace(
        kv_transfer_params=dict(_KV_PARAMS), trace_headers=trace_headers
    )


def _scheduler_with(reqs_need_recv):
    """A scheduler carrying only the state ``build_connector_meta`` touches.

    Constructing the real one needs a VllmConfig and a KV cache config; the
    override under test reads six attributes and none of that.
    """
    scheduler = object.__new__(RblnNixlConnectorScheduler)
    scheduler._reqs_need_recv = reqs_need_recv
    scheduler._reqs_need_save = {}
    scheduler._block_ids_need_save = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_in_batch = set()
    scheduler._reqs_not_processed = set()
    return scheduler


def test_metadata_is_accepted_where_upstream_expects_its_own():
    """``NixlConnector.start_load_kv`` asserts isinstance of the upstream class."""
    assert isinstance(RblnNixlConnectorMetadata(), NixlConnectorMetadata)


def test_trace_headers_survive_the_transport_to_the_worker():
    """Multiproc executor moves connector metadata by pickle."""
    meta = RblnNixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="req-1", local_block_ids=[1, 2], kv_transfer_params=dict(_KV_PARAMS)
    )
    meta.trace_headers["req-1"] = {"traceparent": _TRACEPARENT}

    restored = pickle.loads(pickle.dumps(meta))

    assert restored.trace_headers == {"req-1": {"traceparent": _TRACEPARENT}}
    assert list(restored.reqs_to_recv) == ["req-1"]


def test_build_connector_meta_carries_the_request_trace_context():
    scheduler = _scheduler_with(
        {"req-1": (_request({"traceparent": _TRACEPARENT}), [1, 2])}
    )

    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert isinstance(meta, RblnNixlConnectorMetadata)
    assert meta.trace_headers == {"req-1": {"traceparent": _TRACEPARENT}}
    assert list(meta.reqs_to_recv) == ["req-1"]


def test_untraced_requests_add_no_entry():
    """vLLM only fills ``trace_headers`` when tracing is configured."""
    scheduler = _scheduler_with(
        {"req-1": (_request(None), [1]), "req-2": (_request({}), [2])}
    )

    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert meta.trace_headers == {}
    assert sorted(meta.reqs_to_recv) == ["req-1", "req-2"]


def test_step_span_link_points_at_the_enclosing_step():
    """Reparenting into the request's trace must not lose the step relationship."""
    tracer = TracerProvider().get_tracer(__name__)
    with tracer.start_as_current_span("nixl.kv_transfer") as step:
        links = _step_span_links()

        assert links is not None
        assert len(links) == 1
        assert links[0].context.span_id == step.get_span_context().span_id


def test_no_link_is_emitted_outside_a_step():
    """``get_current_span()`` outside a span is INVALID_SPAN — not a link target."""
    with otel_trace.use_span(otel_trace.INVALID_SPAN, end_on_exit=False):
        assert _step_span_links() is None
