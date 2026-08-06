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

"""Guards for :mod:`vllm_rbln.patches.tracing_span_ids`.

The patch exists because OTel's default ID generator and vLLM's reproducibility
seeding share the global ``random`` module. These tests pin both halves of that
statement: the defect in the unpatched generator, and the patch's immunity to it.
"""

from __future__ import annotations

import contextlib
import random

from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import INVALID_SPAN_ID, INVALID_TRACE_ID

from vllm_rbln.patches import tracing_span_ids
from vllm_rbln.patches.registry import get_registered_patch_descriptors
from vllm_rbln.utils.tracing import derive_request_span_id, pinned_span_id

_SEED = 0
_DRAWS = 8


def _drawn_twice_under_same_seed(generate):
    """Return the two ID sequences a re-seeded RNG produces.

    Stands in for two worker processes that each ran
    ``set_random_seed(model_config.seed)`` with the same seed.
    """
    random.seed(_SEED)
    first = [generate() for _ in range(_DRAWS)]
    random.seed(_SEED)
    second = [generate() for _ in range(_DRAWS)]
    return first, second


def test_unpatched_otel_generator_repeats_ids_across_seeded_processes():
    """Pins the upstream defect the patch works around.

    If OTel ever stops drawing from the global ``random`` module this fails,
    which is the signal that the patch can be dropped.
    """
    generator = RandomIdGenerator()

    span_first, span_second = _drawn_twice_under_same_seed(generator.generate_span_id)
    trace_first, trace_second = _drawn_twice_under_same_seed(
        generator.generate_trace_id
    )

    assert span_first == span_second
    assert trace_first == trace_second


def test_span_ids_are_independent_of_the_seeded_rng():
    generator = RandomIdGenerator()

    first, second = _drawn_twice_under_same_seed(
        lambda: tracing_span_ids.generate_span_id(generator)
    )

    assert first != second
    assert len(set(first + second)) == 2 * _DRAWS


def test_trace_ids_are_independent_of_the_seeded_rng():
    generator = RandomIdGenerator()

    first, second = _drawn_twice_under_same_seed(
        lambda: tracing_span_ids.generate_trace_id(generator)
    )

    assert first != second
    assert len(set(first + second)) == 2 * _DRAWS


def test_generated_ids_have_the_widths_the_wire_format_requires():
    generator = RandomIdGenerator()

    span_id = tracing_span_ids.generate_span_id(generator)
    trace_id = tracing_span_ids.generate_trace_id(generator)

    assert 0 < span_id < 2**64
    assert 0 < trace_id < 2**128


def test_invalid_id_sentinel_matches_otel():
    """The sentinel is inlined to keep the per-span path import-free."""
    assert tracing_span_ids._INVALID_ID == INVALID_SPAN_ID
    assert tracing_span_ids._INVALID_ID == INVALID_TRACE_ID


def test_both_generator_methods_are_patched():
    """Patching only ``generate_span_id`` would still collide trace IDs."""
    targets = {descriptor.target for descriptor in get_registered_patch_descriptors()}

    prefix = "opentelemetry.sdk.trace.id_generator.RandomIdGenerator"
    assert f"{prefix}.generate_span_id" in targets
    assert f"{prefix}.generate_trace_id" in targets


def test_a_pinned_id_wins_and_is_consumed_once():
    """``llm_request`` pins the id other processes derive for it."""
    generator = RandomIdGenerator()

    with pinned_span_id(0xABCDEF0123456789):
        assert tracing_span_ids.generate_span_id(generator) == 0xABCDEF0123456789
        # Only the span the pin was set for gets it.
        assert tracing_span_ids.generate_span_id(generator) != 0xABCDEF0123456789


def test_the_pin_is_cleared_even_if_the_span_raises():
    generator = RandomIdGenerator()

    with contextlib.suppress(RuntimeError), pinned_span_id(0x1234):
        raise RuntimeError("span creation failed")

    assert tracing_span_ids.generate_span_id(generator) != 0x1234


def test_pinning_nothing_leaves_generation_alone():
    generator = RandomIdGenerator()

    with pinned_span_id(None):
        first, second = (
            tracing_span_ids.generate_span_id(generator),
            tracing_span_ids.generate_span_id(generator),
        )

    assert first != second


def test_derived_ids_agree_across_processes_and_differ_across_legs():
    """The whole point: two processes reach the same id without talking.

    Prefill and decode share a trace but are different sidecar legs, so the
    caller's span id has to be part of the input or their ``llm_request`` spans
    would collide.
    """
    trace_id, prefill_leg, decode_leg = 0x4BF92F3577B34DA6A3CE929D0E0E4736, 0xAA, 0xBB

    assert derive_request_span_id(trace_id, prefill_leg) == derive_request_span_id(
        trace_id, prefill_leg
    )
    assert derive_request_span_id(trace_id, prefill_leg) != derive_request_span_id(
        trace_id, decode_leg
    )
    assert 0 < derive_request_span_id(trace_id, prefill_leg) < 2**64
