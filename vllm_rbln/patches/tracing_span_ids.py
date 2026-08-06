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

"""Make OpenTelemetry span/trace IDs independent of the seeded global RNG.

OTel's default ``RandomIdGenerator`` draws IDs from the **global** ``random``
module::

    def generate_span_id(self) -> int:
        span_id = random.getrandbits(64)

vLLM seeds that same global RNG for reproducibility —
``Worker.init_device`` and again after warmup call
``set_random_seed(self.model_config.seed)``, which runs ``random.seed(seed)``.
Every worker process is handed the *same* seed, so every worker walks the *same*
deterministic ID sequence: the Nth span created in any process gets the Nth
value of that sequence, and so does the Nth span in every other process.

The result is ID collisions across processes rather than unique identifiers.
Measured on a 1P1D deployment (ca2, 2h window): 257 emitted spans carried only
73 distinct span IDs and 50 distinct trace IDs. A single ID
(``d480865f9b38fe80``) was reused by 9 different worker processes spanning both
the prefill and the decode pod. Because root spans draw their *trace* ID from
the same sequence, unrelated spans from different pods were stitched into one
trace — a waterfall showing the same operation repeated N times with different
durations, and no way to tell which request a span belonged to.

The fix restores what the OTel spec requires of these IDs — that they be
randomly generated — by drawing them from ``random.SystemRandom`` (``os.urandom``
backed), which ``random.seed()`` cannot influence. vLLM's reproducibility
guarantee is untouched: the global RNG is still seeded and still drives sampling;
tracing simply stops sharing its stream.

Patching the generator rather than passing ``id_generator=`` to
``TracerProvider`` is deliberate. The provider is constructed inside
``vllm.tracing.otel.init_otel_tracer``, so injecting a generator means replacing
that whole function and keeping the copy in step with upstream. Patching the two
methods that hold the defect copies nothing, and covers every span in the
process — including the ones vLLM emits itself.
"""

from __future__ import annotations

import random

from vllm.logger import init_logger
from vllm.tracing import is_tracing_available

from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

#: ``os.urandom``-backed. ``random.seed()`` has no effect on this instance,
#: which is the entire point — it is what decouples IDs from the seeded stream.
_SYSTEM_RANDOM = random.SystemRandom()

#: ``opentelemetry.trace.INVALID_SPAN_ID`` / ``INVALID_TRACE_ID``. Inlined so the
#: per-span path does no import; ``test_invalid_id_sentinel_matches_otel`` pins
#: the value against the real constants.
_INVALID_ID = 0


@register_patch(
    target="opentelemetry.sdk.trace.id_generator.RandomIdGenerator.generate_span_id",
    reason=(
        "OTel's default generator draws span IDs from the global random module, "
        "which vLLM seeds identically in every worker via set_random_seed(). "
        "All workers then emit the same ID sequence, so span IDs collide across "
        "processes (measured: 257 spans, 73 distinct IDs, one ID shared by 9 "
        "processes). Draw from SystemRandom so seeding cannot reach them."
    ),
    condition=is_tracing_available,
)
def generate_span_id(self) -> int:
    span_id = _SYSTEM_RANDOM.getrandbits(64)
    while span_id == _INVALID_ID:
        span_id = _SYSTEM_RANDOM.getrandbits(64)
    return span_id


@register_patch(
    target="opentelemetry.sdk.trace.id_generator.RandomIdGenerator.generate_trace_id",
    reason=(
        "Same defect as generate_span_id: root spans in separately seeded worker "
        "processes drew identical trace IDs, stitching unrelated spans from the "
        "prefill and decode pods into one trace."
    ),
    condition=is_tracing_available,
)
def generate_trace_id(self) -> int:
    trace_id = _SYSTEM_RANDOM.getrandbits(128)
    while trace_id == _INVALID_ID:
        trace_id = _SYSTEM_RANDOM.getrandbits(128)
    return trace_id
