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

"""Trace the NIXL KV-transfer hop between prefill and decode.

In a PD-disaggregated trace the prefill→decode handoff is currently invisible:
the prefill span ends, the decode span starts, and the gap between them carries
no span at all. Every layer around it is instrumented (gateway, routing
sidecar, vLLM phases), so the one hop that moves gigabytes over RDMA is the one
hop nobody can see. Analysis downstream has to report it as an *unexplained*
gap.

This patch draws that hop:

    nixl.kv_transfer        NixlConnector.start_load_kv   (decode side, per step)
    └─ remote_fetch         NixlConnectorWorker._read_blocks_for_req (per request)
    nixl.save_to_host       NixlConnector.wait_for_save   (host-buffer path only)

``start_load_kv`` is called once per engine step and issues the reads for every
request scheduled to receive KV in that step, so ``nixl.kv_transfer`` is a
step-level span and ``remote_fetch`` is its per-request child. That is the real
shape of the operation — one batched issue, N request transfers — and drawing it
as such is why the per-request child carries ``ca.request.id``.

Both spans are created with :func:`vllm.tracing.instrument`, which returns the
original function untouched when no tracing backend is available, so a build
without OTel pays nothing.
"""

from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.logger import init_logger
from vllm.tracing import instrument, is_tracing_available

from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

# Captured at import so the patched callables can delegate. Same convention as
# vllm_rbln/patches/mla.py.
_original_start_load_kv = NixlConnector.start_load_kv
_original_wait_for_save = NixlConnector.wait_for_save
_original_read_blocks_for_req = NixlConnectorWorker._read_blocks_for_req


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector"
        ".NixlConnector.start_load_kv"
    ),
    reason=(
        "The prefill->decode KV hop is the only step in a PD-disaggregated "
        "trace with no span, so downstream analysis can only report it as an "
        "unexplained gap between the prefill and decode spans. Wrap the "
        "decode-side read issue as nixl.kv_transfer."
    ),
    condition=is_tracing_available,
)
@instrument(span_name="nixl.kv_transfer")
def patched_start_load_kv(self: NixlConnector, forward_context, **kwargs) -> None:
    return _original_start_load_kv(self, forward_context, **kwargs)


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector"
        ".NixlConnector.wait_for_save"
    ),
    reason=(
        "The host-buffer path copies KV through host memory before the remote "
        "agent can read it; without a span that copy is charged to the "
        "surrounding forward pass."
    ),
    condition=is_tracing_available,
)
@instrument(span_name="nixl.save_to_host")
def patched_wait_for_save(self: NixlConnector):
    return _original_wait_for_save(self)


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker"
        ".NixlConnectorWorker._read_blocks_for_req"
    ),
    reason=(
        "start_load_kv issues reads for every request scheduled in the step, "
        "so a step-level span alone cannot say which request's transfer was "
        "slow. Emit the per-request read as a child carrying its request id."
    ),
    condition=is_tracing_available,
)
def patched_read_blocks_for_req(self: NixlConnectorWorker, req_id: str, meta):
    # The request id is only known per call, so the span is opened here rather
    # than via @instrument (whose attributes are fixed at decoration time).
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:  # pragma: no cover - guarded by `condition`
        return _original_read_blocks_for_req(self, req_id, meta)

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("remote_fetch") as span:
        # ca.request.id is the correlation key the rest of the stack joins on
        # (gateway span attribute, harness correlation log, drill-down panels).
        span.set_attribute("ca.request.id", req_id)
        remote = getattr(meta, "remote", None)
        block_ids = getattr(remote, "block_ids", None)
        if block_ids is not None:
            span.set_attribute("nixl.block_count", len(block_ids))
        engine_id = getattr(remote, "engine_id", None)
        if engine_id is not None:
            span.set_attribute("nixl.remote_engine_id", str(engine_id))
        return _original_read_blocks_for_req(self, req_id, meta)
