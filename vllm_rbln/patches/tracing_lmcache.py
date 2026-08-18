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

"""Trace LMCache's cache operations: lookup, retrieve, store.

The deployment runs ``MultiConnector`` with ``RblnNixlConnector`` (PD transfer)
and ``RBLNLMCacheConnectorV1`` (the cache) side by side. NIXL's side is
instrumented in ``vllm_rbln.distributed.kv_transfer…rbln_nixl``; LMCache emits
nothing, so a slow request could not be attributed to a cache miss, a slow
restore, or a slow store — only inferred from aggregate metrics.

Three spans, at the narrowest place each question can be answered:

    lmcache.lookup     did this request hit, and by how many tokens
    lmcache.retrieve   how long restoring those tokens took
    lmcache.store      how long persisting this request's KV took

``lookup`` runs scheduler-side and is handed the vLLM ``Request``, so it can be
parented straight into the request's trace. ``retrieve`` / ``store`` run in the
worker, which has no per-request context — the same problem the NIXL spans had —
so the request's ``traceparent`` is carried across on LMCache's own
scheduler→worker channel, its connector metadata.

## Why these live in patches/

Everything patched here belongs to ``lmcache`` / ``lmcache_rbln``, which are
separate packages from this plugin's perspective — upstream, whatever the
vendor. Every patch is a *wrapper*: it captures the original at import time and
calls it, so none of LMCache's logic is copied and none of it can drift.

## Failure policy

A tracing failure must cost a span, not a cache operation. Every wrapper reaches
the original call on the exception path, and the attribute lookups use
``getattr`` so a shape change upstream degrades to a thinner span.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.tracing import is_tracing_available

from vllm_rbln.patches import register_patch
from vllm_rbln.utils.tracing import request_span_context

logger = init_logger(__name__)

try:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache_rbln.integration.vllm.connector import RBLNLMCacheConnectorV1Impl

    _ORIGINAL_LOOKUP = RBLNLMCacheConnectorV1Impl.get_num_new_matched_tokens
    _ORIGINAL_BUILD_META = LMCacheConnectorV1Impl.build_connector_meta
    _ORIGINAL_START_LOAD_KV = LMCacheConnectorV1Impl.start_load_kv
    _ORIGINAL_WAIT_FOR_SAVE = LMCacheConnectorV1Impl.wait_for_save
    _ORIGINAL_RETRIEVE = LMCacheEngine.retrieve
    _ORIGINAL_STORE = LMCacheEngine.store
    _LMCACHE_PRESENT = True
except ImportError:
    _LMCACHE_PRESENT = False


def _lmcache_tracing_enabled() -> bool:
    return _LMCACHE_PRESENT and is_tracing_available()


#: How many request→headers entries either side keeps. A lookup that never
#: leads to a scheduled request leaves an entry behind, so the map is capped
#: rather than tied to a request lifecycle that may not complete. Evicting the
#: oldest costs a span's parent, nothing more.
_TRACE_HEADER_LIMIT = 4096

#: Worker-side req_id → trace headers. A module global rather than instance
#: state because the two halves that need it — the connector impl that receives
#: the metadata and the cache engine that runs the operation — are different
#: objects, and there is one LMCache engine per worker process.
_WORKER_TRACE_HEADERS: dict[str, Mapping[str, str]] = {}


def _remember(store: dict[str, Mapping[str, str]], req_id, headers) -> None:
    if not req_id or not headers:
        return
    store[req_id] = headers
    while len(store) > _TRACE_HEADER_LIMIT:
        store.pop(next(iter(store)))


def _tracer(name: str):
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return None
    return otel_trace.get_tracer(name)


@register_patch(
    target=(
        "lmcache_rbln.integration.vllm.connector."
        "RBLNLMCacheConnectorV1Impl.get_num_new_matched_tokens"
    ),
    reason=(
        "LMCache emits no spans, so a cache miss and a slow restore look the "
        "same in a trace. This is the lookup — it runs scheduler-side with the "
        "vLLM Request in hand, so the span carries the hit/miss answer and "
        "nests directly in the request's trace. The RBLN subclass is the "
        "outermost override; wrapping the base as well would double-count."
    ),
    condition=_lmcache_tracing_enabled,
)
def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int):
    req_id = getattr(request, "request_id", None)
    trace_headers = getattr(request, "trace_headers", None)
    # Stashed here rather than in update_state_after_alloc because lookup is the
    # first hook that sees the Request, and a request that is looked up but
    # never allocated still runs a retrieve on a later attempt.
    _remember(_scheduler_trace_headers(self), req_id, trace_headers)

    tracer = _tracer(__name__)
    if tracer is None:
        return _ORIGINAL_LOOKUP(self, request, num_computed_tokens)

    with tracer.start_as_current_span(
        "lmcache.lookup", context=request_span_context(trace_headers)
    ) as span:
        matched = _ORIGINAL_LOOKUP(self, request, num_computed_tokens)
        if req_id:
            span.set_attribute("ca.request.id", req_id)
        span.set_attribute("lmcache.num_computed_tokens", num_computed_tokens)
        span.set_attribute("lmcache.num_matched_tokens", matched or 0)
        span.set_attribute("lmcache.hit", bool(matched))
        return matched


def _scheduler_trace_headers(impl) -> dict[str, Mapping[str, str]]:
    headers = getattr(impl, "_rbln_trace_headers", None)
    if headers is None:
        headers = {}
        impl._rbln_trace_headers = headers
    return headers


@register_patch(
    target=(
        "lmcache.integration.vllm.vllm_v1_adapter."
        "LMCacheConnectorV1Impl.build_connector_meta"
    ),
    reason=(
        "Carry each request's traceparent to the worker on LMCache's own "
        "scheduler→worker channel, so retrieve/store spans join the request's "
        "trace instead of starting their own. Only requests actually in this "
        "metadata are included, which also bounds what crosses the wire."
    ),
    condition=_lmcache_tracing_enabled,
)
def build_connector_meta(self, scheduler_output):
    metadata = _ORIGINAL_BUILD_META(self, scheduler_output)
    known = getattr(self, "_rbln_trace_headers", None)
    if not known:
        return metadata
    try:
        metadata.rbln_trace_headers = {
            req.req_id: known[req.req_id]
            for req in metadata.requests
            if req.req_id in known
        }
    except Exception:  # noqa: BLE001 - tracing must not break cache metadata
        logger.debug("lmcache trace header attach failed", exc_info=True)
    return metadata


def _publish_worker_trace_headers(impl) -> None:
    """Move the headers off the step's metadata into the worker-wide map.

    Read from the metadata rather than passed in: ``start_load_kv`` and
    ``wait_for_save`` both fetch it themselves, and the engine call that needs
    the headers is several frames below with no way to receive them.
    """
    try:
        metadata = impl._parent._get_connector_metadata()
    except Exception:  # noqa: BLE001 - no metadata is not a cache failure
        return
    carried = getattr(metadata, "rbln_trace_headers", None) or {}
    for req_id, headers in carried.items():
        _remember(_WORKER_TRACE_HEADERS, req_id, headers)


@register_patch(
    target=(
        "lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.start_load_kv"
    ),
    reason=(
        "Publish this step's trace headers before the retrieves run. The engine "
        "call that opens the span is several frames below and cannot be handed "
        "them directly."
    ),
    condition=_lmcache_tracing_enabled,
)
def start_load_kv(self, forward_context, **kwargs):
    _publish_worker_trace_headers(self)
    return _ORIGINAL_START_LOAD_KV(self, forward_context, **kwargs)


@register_patch(
    target=(
        "lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.wait_for_save"
    ),
    reason=(
        "Same publish for the store side: a request may be saved on a step that "
        "never ran a retrieve, so start_load_kv alone would miss it."
    ),
    condition=_lmcache_tracing_enabled,
)
def wait_for_save(self):
    _publish_worker_trace_headers(self)
    return _ORIGINAL_WAIT_FOR_SAVE(self)


def _tokens_arg(args: tuple, kwargs: dict):
    return args[0] if args else kwargs.get("tokens")


def _token_count(tokens) -> int | None:
    try:
        return len(tokens)
    except TypeError:
        return None


@register_patch(
    target="lmcache.v1.cache_engine.LMCacheEngine.retrieve",
    reason=(
        "The restore itself — how long it took to put the cached tokens back "
        "into the paged KV buffer. Wrapped at the engine rather than at the "
        "connector's step loop because only this call is per-request; the "
        "adapter passes req_id through, which is what ties the span to a "
        "request and to its trace."
    ),
    condition=_lmcache_tracing_enabled,
)
def retrieve(self, *args: Any, **kwargs: Any):
    tracer = _tracer(__name__)
    if tracer is None:
        return _ORIGINAL_RETRIEVE(self, *args, **kwargs)

    req_id = kwargs.get("req_id")
    with tracer.start_as_current_span(
        "lmcache.retrieve",
        context=request_span_context(_WORKER_TRACE_HEADERS.get(req_id)),
    ) as span:
        retrieved_mask = _ORIGINAL_RETRIEVE(self, *args, **kwargs)
        if req_id:
            span.set_attribute("ca.request.id", req_id)
        requested = _token_count(_tokens_arg(args, kwargs))
        if requested is not None:
            span.set_attribute("lmcache.num_requested_tokens", requested)
        # The adapter computes this same sum on the next line to check the
        # restore; the mask is a CPU bool tensor, so it is not a device sync.
        try:
            span.set_attribute(
                "lmcache.num_retrieved_tokens", int(retrieved_mask.sum().item())
            )
        except Exception:  # noqa: BLE001 - shape change must not break retrieve
            logger.debug("lmcache.retrieve token count failed", exc_info=True)
        return retrieved_mask


@register_patch(
    target="lmcache.v1.cache_engine.LMCacheEngine.store",
    reason=(
        "The store side of the same question. Per-request for the same reason "
        "as retrieve — the adapter passes req_id into this call."
    ),
    condition=_lmcache_tracing_enabled,
)
def store(self, *args: Any, **kwargs: Any):
    tracer = _tracer(__name__)
    if tracer is None:
        return _ORIGINAL_STORE(self, *args, **kwargs)

    req_id = kwargs.get("req_id")
    with tracer.start_as_current_span(
        "lmcache.store",
        context=request_span_context(_WORKER_TRACE_HEADERS.get(req_id)),
    ) as span:
        if req_id:
            span.set_attribute("ca.request.id", req_id)
        stored = _token_count(_tokens_arg(args, kwargs))
        if stored is not None:
            span.set_attribute("lmcache.num_tokens", stored)
        return _ORIGINAL_STORE(self, *args, **kwargs)
