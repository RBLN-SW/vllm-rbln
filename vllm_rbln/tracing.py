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

import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from vllm.tracing import extract_trace_context, is_tracing_available

SpanAttribute = str | bool | int | float | Sequence[str]


def configure_batch_span_link_limit(max_num_seqs: int) -> None:
    # The OTel SDK defaults to 128 links. RBLN batches can be larger, so size
    # the worker tracer before vLLM initializes its TracerProvider.
    os.environ.setdefault("OTEL_SPAN_LINK_COUNT_LIMIT", str(max_num_seqs))


def batch_span_attributes(
    num_scheduled_tokens: Mapping[str, int],
) -> dict[str, SpanAttribute]:
    return {
        "vllm.request_ids": list(num_scheduled_tokens),
        "vllm.num_requests": len(num_scheduled_tokens),
        "vllm.num_scheduled_tokens": sum(num_scheduled_tokens.values()),
    }


def start_batch_span(
    span_name: str,
    *,
    enabled: bool,
    request_trace_headers: Mapping[str, Mapping[str, str]],
    attributes: Mapping[str, SpanAttribute] | None = None,
) -> AbstractContextManager[Any]:
    if not enabled or not is_tracing_available():
        return nullcontext()

    from opentelemetry import trace
    from opentelemetry.trace import Link

    links = []
    seen_span_contexts: set[tuple[int, int]] = set()
    for headers in request_trace_headers.values():
        context = extract_trace_context(headers)
        if context is None:
            continue

        span_context = trace.get_current_span(context).get_span_context()
        span_key = (span_context.trace_id, span_context.span_id)
        if not span_context.is_valid or span_key in seen_span_contexts:
            continue

        seen_span_contexts.add(span_key)
        links.append(Link(span_context))

    return trace.get_tracer(__name__).start_as_current_span(
        span_name,
        attributes=attributes,
        links=links,
    )
