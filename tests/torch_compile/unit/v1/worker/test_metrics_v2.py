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

"""Tests for vllm_rbln.v1.worker.metrics_v2.

The invariants that matter here are structural rather than numeric:

* every pass counted in ``MODEL + SAMPLE`` must also be counted in ``E2E``, so
  that subtracting their means gives the host overhead.
* ``E2E`` must never fall below ``MODEL + SAMPLE``, since it encloses it.
* neither level may absorb the engine time between two passes.
"""

import pytest

import vllm_rbln.v1.worker.metrics_v2 as mv2
from vllm_rbln.v1.worker.metrics_v2 import (
    _e2e_ends,
    _e2e_starts,
    _NoopPerformanceContext,
    _PerformanceContext,
)

MS = 0.001  # tests work in seconds; use milliseconds for readability


class FakeClock:
    """Stand-in for the ``time`` module: only ``perf_counter`` is used."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(mv2, "time", fake)
    # Keep the spans on nullcontext even if a real rebel runtime is installed.
    monkeypatch.setattr(mv2, "_REBEL_HAS_CAPTURE", False)
    return fake


@pytest.fixture
def ctx(monkeypatch):
    # _rank_tag() imports vllm.distributed; the tag is irrelevant here.
    monkeypatch.setattr(mv2, "_rank_tag", lambda: "")
    return _PerformanceContext("test")


def report_lines(ctx, monkeypatch):
    """Lines emitted by print_stats()."""
    monkeypatch.setattr(mv2.envs, "VLLM_RBLN_METRICS_DIR", "", raising=False)
    captured: list[str] = []
    monkeypatch.setattr(mv2.logger, "info", lambda fmt, msg: captured.append(msg))
    ctx.print_stats()
    return captured[0].split("\n") if captured else []


def run_pass(
    ctx,
    clock,
    *,
    is_prefill,
    prepare=0.0,
    model=0.0,
    mid=0.0,
    sample=None,
    tail=0.0,
):
    """Drive one execute_model -> sample_tokens pass the way the runner does.

    ``sample=None`` reproduces a pass that never reaches the sampler: an
    intermediate chunked prefill, or a non-last PP rank.
    """
    ctx.start_e2e()
    clock.advance(prepare)
    with ctx.profile_model(is_prefill):
        clock.advance(model)
    clock.advance(mid)
    if sample is not None:
        with ctx.profile_sampler():
            clock.advance(sample)
    clock.advance(tail)
    ctx.end_e2e()


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
class TestInvariants:
    def test_counts_are_in_parity(self, ctx, clock):
        """Their means get subtracted, so no pass may land in only one."""
        for _ in range(4):
            clock.advance(2 * MS)
            run_pass(ctx, clock, is_prefill=False, model=8 * MS, sample=1 * MS)
        run_pass(ctx, clock, is_prefill=True, model=10 * MS)  # sampler-less chunk
        run_pass(ctx, clock, is_prefill=True, model=10 * MS, sample=2 * MS)

        for phase, count in ((False, 4), (True, 2)):
            assert ctx._metrics[phase].call_count == count
            assert ctx._e2e[phase].call_count == count

    def test_e2e_is_never_below_model_plus_sample(self, ctx, clock):
        run_pass(
            ctx,
            clock,
            is_prefill=False,
            prepare=1 * MS,
            model=8 * MS,
            mid=1 * MS,
            sample=1 * MS,
            tail=1 * MS,
        )
        # E2E - (MODEL + SAMPLE) is the host overhead the report is read for:
        # 1ms prepare + 1ms between model and sampler + 1ms tail.
        assert ctx._e2e[False].mean_latency_ms() == pytest.approx(12.0)
        assert ctx._metrics[False].mean_latency_ms() == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------
class TestChunkedPrefill:
    def test_every_chunk_gets_its_own_e2e_record(self, ctx, clock):
        """The regression this rework fixes: E2E used to skip every chunk but
        the last, leaving MODEL + SAMPLE at 4 records and E2E at 1."""
        for _ in range(3):
            run_pass(
                ctx, clock, is_prefill=True, prepare=1 * MS, model=10 * MS, tail=1 * MS
            )
            clock.advance(1 * MS)
        run_pass(
            ctx,
            clock,
            is_prefill=True,
            prepare=1 * MS,
            model=10 * MS,
            mid=1 * MS,
            sample=2 * MS,
            tail=1 * MS,
        )

        assert ctx._metrics[True].call_count == 4
        assert ctx._e2e[True].call_count == 4
        assert ctx._e2e[True].mean_latency_ms() == pytest.approx(12.75)
        assert ctx._metrics[True].mean_latency_ms() == pytest.approx(10.5)

    def test_engine_time_between_chunks_is_excluded(self, ctx, clock):
        run_pass(
            ctx, clock, is_prefill=True, prepare=1 * MS, model=10 * MS, tail=2 * MS
        )  # own span: 13ms
        clock.advance(3 * MS)  # engine time between chunks: must not be counted
        run_pass(
            ctx,
            clock,
            is_prefill=True,
            prepare=1 * MS,
            model=10 * MS,
            mid=1 * MS,
            sample=2 * MS,
            tail=2 * MS,
        )  # own span: 16ms

        assert ctx._e2e[True].latencies == pytest.approx([13 * MS, 16 * MS])

    def test_idle_between_passes_is_excluded(self, ctx, clock):
        """Queue/idle wait sits between passes, so it is never measured."""
        run_pass(ctx, clock, is_prefill=False, model=5 * MS, sample=1 * MS)
        clock.advance(500 * MS)  # engine idle, waiting for a new request
        run_pass(ctx, clock, is_prefill=True, model=10 * MS)
        clock.advance(1 * MS)
        run_pass(ctx, clock, is_prefill=True, model=10 * MS, sample=2 * MS)

        assert ctx._e2e[True].latencies == pytest.approx([10 * MS, 12 * MS])
        assert ctx._e2e[False].latencies == pytest.approx([6 * MS])


# ---------------------------------------------------------------------------
# Paths that skip the sampler or the close
# ---------------------------------------------------------------------------
class TestPartialPasses:
    def test_model_only_pass_is_recorded_with_its_own_pass(self, ctx, clock):
        """Non-last PP rank / intermediate chunk: no sampler, so MODEL + SAMPLE is
        the model span alone, closed out when the pass ends rather than carried
        into the next one."""
        run_pass(ctx, clock, is_prefill=False, model=8 * MS, tail=1 * MS)
        assert ctx._e2e[False].mean_latency_ms() == pytest.approx(9.0)
        assert ctx._metrics[False].call_count == 1
        assert ctx._pending is None

        clock.advance(1 * MS)
        run_pass(ctx, clock, is_prefill=False, model=8 * MS, sample=1 * MS)
        assert ctx._metrics[False].call_count == 2

    def test_pass_without_forward_is_dropped(self, ctx, clock):
        """No scheduled tokens / KV-connector only: nothing to attribute."""
        ctx.start_e2e()
        clock.advance(1 * MS)
        ctx.end_e2e()
        assert not ctx._e2e

    def test_unended_measurement_does_not_leak_into_the_next(self, ctx, clock):
        """A pass whose execute_model raised has no end boundary; drop it."""
        ctx.start_e2e()  # never ended
        clock.advance(50 * MS)
        run_pass(ctx, clock, is_prefill=False, model=8 * MS, sample=1 * MS)

        assert ctx._e2e[False].call_count == 1
        assert ctx._e2e[False].mean_latency_ms() == pytest.approx(9.0)

    def test_ended_by_execute_model_when_no_sampler_follows(self, ctx, clock):
        """execute_model returns the output itself, so it ends the pass; the
        sample_tokens early-return must then not record a second time."""
        ctx.start_e2e()
        clock.advance(1 * MS)
        with ctx.profile_model(False):
            clock.advance(8 * MS)
        clock.advance(1 * MS)
        ctx.end_e2e()  # driven by execute_model returning non-None
        ctx.end_e2e()  # sample_tokens early-returns: must be a no-op

        assert ctx._e2e[False].call_count == 1
        assert ctx._e2e[False].mean_latency_ms() == pytest.approx(10.0)

    def test_end_without_start_is_a_noop(self, ctx, clock):
        ctx.end_e2e()
        assert not ctx._e2e

    def test_is_prefill_does_not_carry_over_to_a_forwardless_pass(self, ctx, clock):
        """start_e2e must clear is_prefill, or this lands in DECODE."""
        run_pass(ctx, clock, is_prefill=False, model=8 * MS, sample=1 * MS)
        ctx.start_e2e()
        clock.advance(50 * MS)
        ctx.end_e2e()  # no forward pass ran

        assert ctx._e2e[False].call_count == 1
        assert ctx._e2e[False].mean_latency_ms() == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
class TestReporting:
    def test_sections_are_grouped_by_metric(self, ctx, clock, monkeypatch):
        run_pass(ctx, clock, is_prefill=True, model=10 * MS, sample=2 * MS)
        clock.advance(1 * MS)
        run_pass(ctx, clock, is_prefill=False, model=8 * MS, sample=1 * MS)

        lines = report_lines(ctx, monkeypatch)
        assert [ln for ln in lines if ln.endswith("METRICS:")] == [
            "PREFILL + SAMPLE METRICS:",
            "DECODE + SAMPLE METRICS:",
            "PREFILL E2E METRICS:",
            "DECODE E2E METRICS:",
        ]

    def test_absent_phase_is_omitted(self, ctx, clock, monkeypatch):
        run_pass(ctx, clock, is_prefill=False, model=8 * MS, sample=1 * MS)
        assert not any("PREFILL" in ln for ln in report_lines(ctx, monkeypatch))

    def test_print_stats_with_no_data(self, ctx):
        ctx.print_stats()  # must not raise

    def test_print_stats_flushes_the_final_pending_span(self, ctx, clock, monkeypatch):
        monkeypatch.setattr(mv2.envs, "VLLM_RBLN_METRICS_DIR", "", raising=False)
        ctx.start_e2e()
        with ctx.profile_model(True):
            clock.advance(10 * MS)
        ctx.print_stats()
        assert ctx._metrics[True].call_count == 1


# ---------------------------------------------------------------------------
# Decorators and the no-op context
# ---------------------------------------------------------------------------
class TestE2EDecorators:
    @staticmethod
    def _runner(calls, execute_returns):
        class Runner:
            performance_ctx = type(
                "C",
                (),
                {
                    "start_e2e": lambda self: calls.append("start"),
                    "end_e2e": lambda self: calls.append("end"),
                },
            )()

            @_e2e_starts
            def execute_model(self):
                calls.append("execute")
                return execute_returns

            @_e2e_ends
            def sample_tokens(self):
                calls.append("sample")
                return "out"

        return Runner()

    def test_sampler_path_ends_in_sample_tokens(self):
        """execute_model returns None -> sample_tokens owns the end."""
        calls: list[str] = []
        runner = self._runner(calls, None)
        assert runner.execute_model() is None
        assert runner.sample_tokens() == "out"
        assert calls == ["start", "execute", "sample", "end"]

    def test_output_path_ends_in_execute_model(self):
        """execute_model returns an output -> it must end the measurement."""
        calls: list[str] = []
        runner = self._runner(calls, "early-output")
        assert runner.execute_model() == "early-output"
        assert calls == ["start", "execute", "end"]

    def test_measurement_is_not_ended_twice(self):
        """sample_tokens still runs after an early output; end must no-op."""
        calls: list[str] = []
        runner = self._runner(calls, "early-output")
        runner.execute_model()
        runner.sample_tokens()
        # Two end calls reach the context; end_e2e() no-ops on the second.
        assert calls == ["start", "execute", "end", "sample", "end"]

    def test_a_raising_body_aborts_instead_of_recording(self):
        """The pass owns rebel's capture slot; it must be released, but not recorded."""
        calls: list[str] = []
        ctx = type(
            "C",
            (),
            {
                "start_e2e": lambda self: calls.append("start"),
                "end_e2e": lambda self: calls.append("end"),
                "abort_e2e": lambda self: calls.append("abort"),
            },
        )()

        class Runner:
            performance_ctx = ctx

            @_e2e_ends
            def sample_tokens(self):
                raise RuntimeError("boom")

            @_e2e_starts
            def execute_model(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            Runner().sample_tokens()
        assert calls == ["abort"]

        calls.clear()
        with pytest.raises(RuntimeError, match="boom"):
            Runner().execute_model()
        assert calls == ["start", "abort"]

    def test_noop_context_matches_the_real_api(self):
        noop = _NoopPerformanceContext()
        noop.start_e2e()
        noop.end_e2e()
        noop.abort_e2e()
        noop.print_stats()
        with noop.profile_model(True):
            pass
        with noop.profile_sampler():
            pass
