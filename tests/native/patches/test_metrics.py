# Copyright 2026 Rebellions Inc. All rights reserved.
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

import json
import types
from typing import Any

import pytest

import vllm_rbln.envs as envs
import vllm_rbln.patches.metrics as pm
from vllm_rbln.patches import registry

MS = 0.001


class _Clock:
    # Stands in for the time module; only perf_counter is used.
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Runner:
    """Stands in for the runner attributes the wrappers read."""

    def __init__(self, **attrs) -> None:
        self.is_prefill = False
        self.is_intermediate_chunked_prefill = False
        self.max_num_tokens = 128
        self.model_executable: Any = None
        self.__dict__.update(attrs)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(pm, "time", fake)
    return fake


@pytest.fixture
def ctx(monkeypatch):
    # The rank tag reads the parallelism groups; it is irrelevant here.
    monkeypatch.setattr(pm, "_rank_tag", lambda: "")
    return pm._PerformanceContext("runner")


def run_pass(
    ctx, clock, *, phase, before=0.0, model=0.0, sample=None, after=0.0, dp=None
):
    """Drive one pass the way the patched runner methods do.

    ``sample=None`` is a pass whose sampler never ran: an intermediate chunked prefill,
    or a non-last PP rank. ``dp=None`` is a pass with no DP group, so no all-reduce.
    """
    ctx.start_pass()
    clock.advance(before)
    if dp is not None:
        clock.advance(dp)
        ctx.add_dp_wait(dp)
    ctx.mark_phase(phase)
    clock.advance(model)
    ctx.add_graph_time(model)
    if sample is not None:
        clock.advance(sample)
        ctx.add_graph_time(sample)
    clock.advance(after)
    ctx.end_pass()


class TestMetrics:
    def test_record_appends_latency(self):
        m = pm._Metrics()
        m.record(0.01)
        m.record(0.02)
        assert m.latencies == [0.01, 0.02]
        assert m.call_count == 2

    def test_mean_latency_ms_scales_to_ms(self):
        m = pm._Metrics()
        m.record(0.01)
        m.record(0.03)
        assert m.mean_latency_ms() == 20.0  # mean 0.02 s -> 20 ms

    def test_empty_metrics_report_zero_and_no_percentiles(self):
        m = pm._Metrics()
        assert m.mean_latency_ms() == 0.0
        assert m.latency_percentiles_ms() == {}

    def test_percentile_keys_and_ms_scaling(self):
        m = pm._Metrics()
        for v in (0.01, 0.02, 0.03):
            m.record(v)
        pct = m.latency_percentiles_ms()
        assert set(pct) == {"p50", "p90", "p99"}
        assert pct["p50"] == 20.0  # median of [10,20,30] ms

    def test_to_dict_carries_count_and_latencies_only(self):
        m = pm._Metrics()
        m.record(0.01)
        assert set(m.to_dict()) == {
            "call_count",
            "mean_latency_ms",
            "latency_percentiles_ms",
        }


class TestPassAccounting:
    def test_e2e_covers_the_whole_pass_and_graph_covers_the_calls(self, ctx, clock):
        run_pass(
            ctx,
            clock,
            phase=pm._Phase.PREFILL,
            before=1 * MS,
            model=8 * MS,
            sample=2 * MS,
            after=1 * MS,
        )
        assert ctx._e2e[pm._Phase.PREFILL].mean_latency_ms() == pytest.approx(12.0)
        # MODEL + SAMPLE is the sum of the two calls; the 2ms outside them is the host
        # overhead the difference of the two means is read for.
        assert ctx._graph[pm._Phase.PREFILL].mean_latency_ms() == pytest.approx(10.0)

    def test_counts_stay_equal_across_the_two_sections(self, ctx, clock):
        for _ in range(3):
            run_pass(ctx, clock, phase=pm._Phase.DECODE, model=8 * MS, sample=1 * MS)
            clock.advance(5 * MS)
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=8 * MS)  # sampler-less

        assert ctx._graph[pm._Phase.DECODE].call_count == 4
        assert ctx._e2e[pm._Phase.DECODE].call_count == 4

    def test_time_between_passes_is_excluded(self, ctx, clock):
        for _ in range(2):
            run_pass(ctx, clock, phase=pm._Phase.DECODE, model=1 * MS)
            clock.advance(10 * MS)  # engine and idle time between passes
        assert ctx._e2e[pm._Phase.DECODE].latencies == pytest.approx([1 * MS, 1 * MS])

    def test_pass_without_a_phase_is_dropped(self, ctx, clock):
        # Empty batch, or KV send/recv with no forward work.
        ctx.start_pass()
        clock.advance(1 * MS)
        ctx.add_graph_time(1 * MS)
        ctx.end_pass()
        assert ctx._e2e == {} and ctx._graph == {}

    def test_pass_without_graph_time_is_dropped(self, ctx, clock):
        # A pass that set its phase but raised before the forward.
        ctx.start_pass()
        ctx.mark_phase(pm._Phase.DECODE)
        clock.advance(1 * MS)
        ctx.end_pass()
        assert ctx._e2e == {} and ctx._graph == {}

    def test_end_without_start_is_a_noop(self, ctx, clock):
        ctx.end_pass()
        assert ctx._e2e == {} and ctx._graph == {}

    def test_second_end_does_not_record_again(self, ctx, clock):
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=1 * MS)
        clock.advance(5 * MS)
        ctx.end_pass()
        assert ctx._e2e[pm._Phase.DECODE].call_count == 1

    def test_phase_and_graph_time_do_not_carry_over(self, ctx, clock):
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=1 * MS)
        ctx.start_pass()
        clock.advance(5 * MS)
        ctx.end_pass()  # no phase, no graph time this time
        assert ctx._e2e[pm._Phase.DECODE].call_count == 1
        assert ctx._graph[pm._Phase.DECODE].latencies == pytest.approx([1 * MS])

    def test_marks_outside_a_pass_are_ignored(self, ctx, clock):
        # Warm-up and dummy runs reach the same code with no pass open.
        ctx.mark_phase(pm._Phase.PREFILL)
        ctx.add_graph_time(9 * MS)
        assert not ctx.in_pass
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=1 * MS)
        assert ctx._graph[pm._Phase.DECODE].latencies == pytest.approx([1 * MS])


class TestReporting:
    def _labels(self, ctx, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", "")
        captured: list[str] = []
        monkeypatch.setattr(pm.logger, "info", lambda fmt, msg: captured.append(msg))
        ctx.print_stats()
        return [
            line.removesuffix(" METRICS:")
            for line in captured[0].split("\n")
            if line.endswith("METRICS:")
        ]

    def test_sections_are_grouped_by_range_then_phase_order(
        self, ctx, clock, monkeypatch
    ):
        run_pass(ctx, clock, phase=pm._Phase.PADDED_DECODE, model=3 * MS)
        run_pass(ctx, clock, phase=pm._Phase.PREFILL, model=10 * MS, sample=2 * MS)
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=8 * MS, sample=1 * MS)

        # Phase order follows the enum, not the order the phases were seen.
        assert self._labels(ctx, monkeypatch) == [
            "MODEL + SAMPLE (PREFILL)",
            "MODEL + SAMPLE (DECODE)",
            "MODEL + SAMPLE (PADDED DECODE)",
            "E2E (PREFILL)",
            "E2E (DECODE)",
            "E2E (PADDED DECODE)",
        ]

    def test_dp_wait_sits_between_the_graph_and_the_whole_pass(
        self, ctx, clock, monkeypatch
    ):
        run_pass(ctx, clock, phase=pm._Phase.PREFILL, dp=5 * MS, model=10 * MS)
        assert self._labels(ctx, monkeypatch) == [
            "MODEL + SAMPLE (PREFILL)",
            "DP WAIT (PREFILL)",
            "E2E (PREFILL)",
        ]

    def test_dp_wait_is_omitted_without_a_dp_group(self, ctx, clock, monkeypatch):
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=8 * MS, sample=1 * MS)
        assert "DP WAIT (DECODE)" not in self._labels(ctx, monkeypatch)

    def test_dp_wait_is_counted_out_of_the_graph_time(self, ctx, clock):
        run_pass(ctx, clock, phase=pm._Phase.PREFILL, dp=5 * MS, model=10 * MS)
        assert ctx._dp[pm._Phase.PREFILL].latencies == pytest.approx([5 * MS])
        assert ctx._graph[pm._Phase.PREFILL].latencies == pytest.approx([10 * MS])
        # The wait is inside E2E, which is why it needs reporting separately.
        assert ctx._e2e[pm._Phase.PREFILL].latencies == pytest.approx([15 * MS])

    def test_unseen_phase_is_omitted(self, ctx, clock, monkeypatch):
        run_pass(ctx, clock, phase=pm._Phase.DECODE, model=8 * MS, sample=1 * MS)
        assert self._labels(ctx, monkeypatch) == [
            "MODEL + SAMPLE (DECODE)",
            "E2E (DECODE)",
        ]

    def test_print_stats_with_no_data(self, ctx, monkeypatch):
        assert self._labels(ctx, monkeypatch) == []


class TestWriteMetricsJson:
    def test_no_metrics_dir_skips_write(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", "")
        pm._write_metrics_json("runner", "", {})  # must not raise
        assert list(tmp_path.iterdir()) == []

    def test_writes_payload_with_section_dicts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        m = pm._Metrics()
        m.record(0.01)
        pm._write_metrics_json("runner", "", {"E2E (DECODE)": m})
        data = json.loads((tmp_path / "metrics.json").read_text())
        assert data["name"] == "runner" and data["rank"] == ""
        assert data["sections"]["E2E (DECODE)"]["call_count"] == 1

    def test_rank_tag_lowercased_into_filename_suffix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        pm._write_metrics_json("runner", "TP1 DP0", {})
        assert (tmp_path / "metrics_tp1_dp0.json").exists()


class TestRankTag:
    @staticmethod
    def _grp(world, rank):
        return types.SimpleNamespace(world_size=world, rank_in_group=rank)

    def _patch_groups(self, monkeypatch, tp, pp, dp, ep):
        import vllm.distributed as d

        monkeypatch.setattr(d, "get_tp_group", lambda: tp)
        monkeypatch.setattr(d, "get_pp_group", lambda: pp)
        monkeypatch.setattr(d, "get_dp_group", lambda: dp)
        monkeypatch.setattr(d, "get_ep_group", lambda: ep)

    def test_degree_one_axes_are_excluded(self, monkeypatch):
        one = self._grp(1, 0)
        self._patch_groups(monkeypatch, one, one, one, one)
        assert pm._rank_tag() == ""

    def test_only_multi_degree_axes_in_fixed_order(self, monkeypatch):
        self._patch_groups(
            monkeypatch,
            tp=self._grp(2, 1),
            pp=self._grp(1, 0),
            dp=self._grp(4, 3),
            ep=self._grp(1, 0),
        )
        assert pm._rank_tag() == "TP1 DP3"

    def test_group_lookup_failure_yields_empty(self, monkeypatch):
        import vllm.distributed as d

        def boom():
            raise RuntimeError("group not initialized")

        for name in ("get_tp_group", "get_pp_group", "get_dp_group", "get_ep_group"):
            monkeypatch.setattr(d, name, boom)
        assert pm._rank_tag() == ""


class TestRenderMetrics:
    def test_no_data_branch(self):
        assert pm._render_metrics("E2E (DECODE)", pm._Metrics()) == [
            "E2E (DECODE) METRICS: No data recorded"
        ]

    def test_p50_is_labelled_median(self):
        m = pm._Metrics()
        m.record(0.01)
        text = "\n".join(pm._render_metrics("E2E (PREFILL)", m))
        assert "Median latency (ms)" in text
        assert "P90 latency (ms)" in text

    def test_report_wraps_sections_with_named_header(self):
        m = pm._Metrics()
        m.record(0.01)
        text = "\n".join(pm._render_metrics_report("runner", {"E2E (DECODE)": m}))
        assert "PERFORMANCE STATISTICS [runner]" in text
        assert "E2E (DECODE) METRICS:" in text


class TestPhase:
    @pytest.mark.parametrize(
        "is_prefill, num_tokens_padded, expected",
        [
            (True, None, pm._Phase.PREFILL),
            (True, 128, pm._Phase.PREFILL),
            # num_tokens_padded is the MoE pad dimension: matching the prefill dim
            # means this decode ran the prefill-sized graph.
            (False, 128, pm._Phase.PADDED_DECODE),
            (False, 64, pm._Phase.DECODE),
            # Without DP the runner computes no padding at all.
            (False, None, pm._Phase.DECODE),
        ],
    )
    def test_phase_follows_the_padded_token_count(
        self, monkeypatch, is_prefill, num_tokens_padded, expected
    ):
        padding = (4, num_tokens_padded, None)
        monkeypatch.setattr(pm, "_determine_batch_padding", lambda self, *a: padding)
        runner = _Runner(is_prefill=is_prefill, max_num_tokens=128)
        ctx = pm._ctx(runner)
        ctx.start_pass()

        assert pm.determine_batch_padding(runner, 1, 1) == padding
        assert ctx._phase is expected

    def test_phase_is_not_marked_outside_a_pass(self, monkeypatch):
        # The dummy run reaches the same method with no pass open.
        monkeypatch.setattr(
            pm, "_determine_batch_padding", lambda self, *a: (4, 128, None)
        )
        runner = _Runner()
        ctx = pm._ctx(runner)

        pm.determine_batch_padding(runner, 1, 1)
        assert ctx._phase is None


class TestPassBoundary:
    def test_deferred_output_leaves_the_pass_open(self, monkeypatch):
        monkeypatch.setattr(pm, "_execute_model", lambda self, *a, **k: None)
        runner = _Runner()

        assert pm.execute_model(runner) is None
        assert pm._ctx(runner).in_pass

    def test_early_output_ends_the_pass(self, monkeypatch):
        # The engine calls sample_tokens only after a None, so nobody else would end it.
        monkeypatch.setattr(pm, "_execute_model", lambda self, *a, **k: "output")
        runner = _Runner()

        assert pm.execute_model(runner) == "output"
        assert not pm._ctx(runner).in_pass

    def test_whole_pass_records_one_sample_per_section(self, monkeypatch):
        monkeypatch.setattr(pm, "_execute_model", lambda self, *a, **k: None)
        monkeypatch.setattr(pm, "_sample_tokens", lambda self, *a, **k: "out")
        monkeypatch.setattr(pm, "_sample", lambda self, *a, **k: "sampled")
        monkeypatch.setattr(
            pm, "_determine_batch_padding", lambda self, *a: (4, 64, None)
        )
        runner = _Runner()

        pm.execute_model(runner)
        pm.determine_batch_padding(runner, 1, 1)
        pm.sample(runner)
        assert pm.sample_tokens(runner) == "out"

        ctx = pm._ctx(runner)
        assert not ctx.in_pass
        assert ctx._e2e[pm._Phase.DECODE].call_count == 1
        assert ctx._graph[pm._Phase.DECODE].call_count == 1

    def test_a_raising_pass_does_not_corrupt_the_next_one(self, monkeypatch, clock):
        def boom(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pm, "_execute_model", boom)
        runner = _Runner()
        with pytest.raises(RuntimeError, match="boom"):
            pm.execute_model(runner)
        ctx = pm._ctx(runner)
        assert ctx.in_pass  # left open: there is no abort path by design

        clock.advance(50 * MS)  # time the failed pass must not claim
        monkeypatch.setattr(pm, "_execute_model", lambda self, *a, **k: None)
        monkeypatch.setattr(
            pm, "_determine_batch_padding", lambda self, *a: (4, 64, None)
        )
        monkeypatch.setattr(pm, "_sample", lambda self, *a, **k: "sampled")

        def sample_tokens(self, *args, **kwargs):
            clock.advance(2 * MS)
            return "out"

        monkeypatch.setattr(pm, "_sample_tokens", sample_tokens)

        pm.execute_model(runner)
        pm.determine_batch_padding(runner, 1, 1)
        pm.sample(runner)
        pm.sample_tokens(runner)

        # One record, timed from the second start_pass -- none of the 50ms before it.
        recorded = ctx._e2e[pm._Phase.DECODE]
        assert recorded.call_count == 1
        assert recorded.latencies == pytest.approx([2 * MS])

    def test_graph_time_from_a_failed_pass_does_not_leak(self, monkeypatch, clock):
        """A pass that raised after its forward must not lend its time to the next."""
        monkeypatch.setattr(pm, "_execute_model", lambda self, *a, **k: None)
        monkeypatch.setattr(
            pm, "_determine_batch_padding", lambda self, *a: (4, 64, None)
        )

        def sample_for(ms):
            def _sample(self, *args, **kwargs):
                clock.advance(ms)
                return "sampled"

            return _sample

        def boom(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pm, "_sample", sample_for(7 * MS))
        monkeypatch.setattr(pm, "_sample_tokens", boom)
        runner = _Runner()
        pm.execute_model(runner)
        pm.determine_batch_padding(runner, 1, 1)
        pm.sample(runner)  # 7ms of graph time on a pass that is about to fail
        with pytest.raises(RuntimeError, match="boom"):
            pm.sample_tokens(runner)

        monkeypatch.setattr(pm, "_sample", sample_for(1 * MS))
        monkeypatch.setattr(pm, "_sample_tokens", lambda self, *a, **k: "out")
        pm.execute_model(runner)
        pm.determine_batch_padding(runner, 1, 1)
        pm.sample(runner)
        pm.sample_tokens(runner)

        graph = pm._ctx(runner)._graph[pm._Phase.DECODE]
        assert graph.latencies == pytest.approx([1 * MS])


class TestSamplerTiming:
    def test_sampler_call_adds_graph_time(self, monkeypatch):
        monkeypatch.setattr(pm, "_sample", lambda self, *a, **k: "sampled")
        runner = _Runner()
        ctx = pm._ctx(runner)
        ctx.start_pass()

        assert pm.sample(runner) == "sampled"
        assert ctx._graph_latency is not None

    def test_intermediate_chunked_prefill_is_not_timed(self, monkeypatch):
        # Returns a placeholder without running the sampler.
        monkeypatch.setattr(pm, "_sample", lambda self, *a, **k: "placeholder")
        runner = _Runner(is_intermediate_chunked_prefill=True)
        ctx = pm._ctx(runner)
        ctx.start_pass()

        assert pm.sample(runner) == "placeholder"
        assert ctx._graph_latency is None


class TestDpWaitProbe:
    def test_the_all_reduce_is_timed_only_inside_a_pass(self, monkeypatch):
        monkeypatch.setattr(
            pm, "_num_tokens_across_dp", lambda tokens, size, rank: "across"
        )
        ctx = pm._PerformanceContext("runner")

        assert pm.num_tokens_across_dp(4, 2, 0) == "across"  # warm-up, no pass open
        assert ctx._dp_latency is None

        ctx.start_pass()
        assert pm.num_tokens_across_dp(4, 2, 0) == "across"
        assert ctx._dp_latency is not None

    def test_a_closed_pass_stops_collecting(self, monkeypatch):
        monkeypatch.setattr(
            pm, "_num_tokens_across_dp", lambda tokens, size, rank: "across"
        )
        ctx = pm._PerformanceContext("runner")
        ctx.start_pass()
        ctx.end_pass()

        pm.num_tokens_across_dp(4, 2, 0)
        assert ctx._dp_latency is None


class TestModelExecutable:
    def test_the_executable_is_timed_only_inside_a_pass(self, monkeypatch):
        calls: list[dict] = []

        def fake_load_model(self, *args, **kwargs):
            def executable(**kwargs):
                calls.append(kwargs)
                return "logits"

            self.model_executable = executable

        monkeypatch.setattr(pm, "_load_model", fake_load_model)
        runner = _Runner()
        pm.load_model(runner)
        ctx = pm._ctx(runner)

        assert runner.model_executable(step=1) == "logits"  # warm-up
        assert ctx._graph_latency is None

        ctx.start_pass()
        runner.model_executable(step=2)
        assert ctx._graph_latency is not None
        assert calls == [{"step": 1}, {"step": 2}]


class TestShutdown:
    def test_stats_are_reported_before_teardown(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(pm, "_shutdown", lambda self: order.append("shutdown"))
        runner = _Runner()
        ctx = pm._ctx(runner)
        monkeypatch.setattr(ctx, "print_stats", lambda: order.append("stats"))

        pm.shutdown(types.SimpleNamespace(model_runner=runner))
        assert order == ["stats", "shutdown"]

    def test_shutdown_before_init_device_still_tears_down(self, monkeypatch):
        # model_runner is built in init_device(), so a worker that failed earlier has
        # none; reporting must not mask that failure.
        order: list[str] = []
        monkeypatch.setattr(pm, "_shutdown", lambda self: order.append("shutdown"))

        pm.shutdown(types.SimpleNamespace())
        assert order == ["shutdown"]

    def test_shutdown_without_a_context_still_tears_down(self, monkeypatch):
        # Nothing ever ran, so no context was built.
        order: list[str] = []
        monkeypatch.setattr(pm, "_shutdown", lambda self: order.append("shutdown"))

        pm.shutdown(types.SimpleNamespace(model_runner=_Runner()))
        assert order == ["shutdown"]


class TestDescriptors:
    def test_every_target_resolves_to_an_existing_attribute(self):
        ours = [
            d
            for d in registry.get_registered_patch_descriptors()
            if d.owner_module == "vllm_rbln.patches.metrics"
        ]
        assert len(ours) == 7
        for descriptor in ours:
            owner, attr = registry._resolve_patch_target_owner(descriptor.target)
            assert hasattr(owner, attr), descriptor.target
