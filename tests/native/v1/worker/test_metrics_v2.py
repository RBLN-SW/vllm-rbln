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

import pytest

import vllm_rbln.envs as envs
import vllm_rbln.v1.worker.metrics_v2 as mm


class TestMetrics:
    def test_record_appends_latency_and_skips_none_timings(self):
        m = mm.Metrics()
        m.record(0.01, host_time=None, device_time=5)
        m.record(0.02, host_time=7)
        assert m.latencies == [0.01, 0.02]
        assert m.host_times == [7]  # the None call contributed nothing
        assert m.device_times == [5]
        assert m.ccl_times == [] and m.prepare_times == []

    def test_call_count_tracks_latencies(self):
        m = mm.Metrics()
        assert m.call_count == 0
        m.record(0.01)
        m.record(0.02)
        assert m.call_count == 2

    def test_mean_latency_ms_scales_to_ms(self):
        m = mm.Metrics()
        m.record(0.01)
        m.record(0.03)
        assert m.mean_latency_ms() == 20.0  # mean 0.02 s -> 20 ms

    def test_mean_latency_ms_empty_is_zero(self):
        assert mm.Metrics().mean_latency_ms() == 0.0

    def test_percentiles_empty_is_empty_dict(self):
        assert mm.Metrics().latency_percentiles_ms() == {}

    def test_percentiles_keys_and_ms_scaling(self):
        m = mm.Metrics()
        for v in (0.01, 0.02, 0.03):
            m.record(v)
        pct = m.latency_percentiles_ms()
        assert set(pct) == {"p50", "p90", "p99"}
        assert pct["p50"] == 20.0  # median of [10,20,30] ms

    def test_to_dict_omits_empty_timing_series(self):
        m = mm.Metrics()
        m.record(0.01, host_time=100, device_time=None)
        d = m.to_dict()
        assert d["call_count"] == 1
        assert d["mean_latency_ms"] == 10.0
        assert d["mean_host_time_us"] == 100.0
        # device/ccl/prepare never recorded -> keys absent, not zero.
        assert "mean_device_time_us" not in d
        assert "mean_ccl_time_us" not in d
        assert "mean_prepare_time_us" not in d


class TestSampleMerged:
    def test_latency_sums_and_model_phase_is_kept(self):
        model = mm._Sample(latency=1.0, phase=True)
        sampler = mm._Sample(latency=0.5, phase=None)
        merged = model.merged(sampler)
        assert merged.latency == 1.5
        assert merged.phase is True  # keeps the model span's phase, not other's

    def test_add_is_none_only_when_both_none_else_treats_none_as_zero(self):
        a = mm._Sample(latency=1.0, host=0, device=None, ccl=5, prepare=None)
        b = mm._Sample(latency=2.0, host=None, device=7, ccl=3, prepare=None)
        merged = a.merged(b)
        assert merged.host == 0  # 0 + None -> 0 (not None)
        assert merged.device == 7  # None + 7 -> 7
        assert merged.ccl == 8  # 5 + 3
        assert merged.prepare is None  # None + None -> None


def _timer(host=0, device=0, ccl=0):
    return {
        "type": "timer",
        "total_host": host,
        "total_device": device,
        "total_ccl": ccl,
    }


def _prep(prepare_in=0, prepare_out=0):
    return {
        "type": "prep",
        "prepare_input_us": prepare_in,
        "prepare_output_us": prepare_out,
    }


class TestParseReports:
    def test_none_or_empty_yields_all_none(self):
        assert mm._parse_reports(None) == (None, None, None, None)
        assert mm._parse_reports([]) == (None, None, None, None)

    def test_timer_only_leaves_prepare_none(self):
        out = mm._parse_reports([_timer(host=10, device=20, ccl=30)])
        assert out == (10, 20, 30, None)

    def test_prep_only_leaves_timings_none(self):
        out = mm._parse_reports([_prep(prepare_in=4, prepare_out=6)])
        assert out == (None, None, None, 10)

    def test_one_graph_run(self):
        out = mm._parse_reports([_timer(host=10), _prep(prepare_in=4, prepare_out=6)])
        assert out == (10, 0, 0, 10)

    def test_every_graph_run_in_the_window_is_summed(self):
        # Two runtime.run() calls -> two timer/prep pairs, interleaved as they arrive.
        out = mm._parse_reports(
            [
                _timer(host=10, device=100, ccl=1),
                _prep(prepare_in=4, prepare_out=6),
                _timer(host=20, device=200, ccl=2),
                _prep(prepare_in=40, prepare_out=60),
            ]
        )
        assert out == (30, 300, 3, 110)

    def test_missing_keys_count_as_zero(self):
        out = mm._parse_reports([{"type": "timer"}, {"type": "prep"}])
        assert out == (0, 0, 0, 0)

    def test_unknown_report_kinds_are_ignored(self):
        # RBLN_APPLY_TIMER pushes "buffer_transform" onto the same queue.
        out = mm._parse_reports(
            [
                {"type": "buffer_transform", "wall_us": 999.0},
                _timer(host=10),
            ]
        )
        assert out == (10, 0, 0, None)

    def test_untyped_reports_contribute_nothing(self):
        assert mm._parse_reports([{"total_host": 10}]) == (None, None, None, None)


class TestMean:
    def test_empty_is_zero(self):
        assert mm._mean([]) == 0.0

    def test_average(self):
        assert mm._mean([2, 4, 6]) == 4.0


def _ctx(monkeypatch, **kwargs):
    # Pin rank_tag so construction never depends on a live distributed group,
    # and keep spans on nullcontext even if a real rebel runtime is installed.
    monkeypatch.setattr(mm, "_rank_tag", lambda: "")
    monkeypatch.setattr(mm, "_REBEL_HAS_CAPTURE", False)
    return mm._PerformanceContext(name="runner", **kwargs)


class _Clock:
    # Stands in for the time module; only perf_counter is used.
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def _fake_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(mm, "time", clock)
    return clock


def _run_pass(ctx, is_prefill, with_sampler=True):
    """Drive one execute_model pass the way the runner decorators do."""
    ctx.start_e2e()
    with ctx.profile_model(is_prefill=is_prefill):
        pass
    if with_sampler:
        with ctx.profile_sampler():
            pass
    ctx.end_e2e()


class TestPerformanceContextStateMachine:
    def test_model_then_sampler_merges_and_records(self, monkeypatch):
        ctx = _ctx(monkeypatch)
        ctx._submit(
            mm._Sample(1.0, host=10, device=20, ccl=30, prepare=5, phase=True),
            is_sampler=False,
        )
        # Model span alone stays pending -- nothing recorded yet.
        assert ctx._metrics[True].call_count == 0
        assert ctx._pending is not None

        ctx._submit(
            mm._Sample(0.5, host=1, device=2, ccl=3, prepare=4, phase=None),
            is_sampler=True,
        )
        m = ctx._metrics[True]
        assert m.latencies == [1.5]
        assert (m.host_times, m.device_times) == ([11], [22])
        assert (m.ccl_times, m.prepare_times) == ([33], [9])
        assert ctx._pending is None

    def test_sampler_without_pending_is_ignored(self, monkeypatch):
        ctx = _ctx(monkeypatch)
        ctx._submit(mm._Sample(0.5, phase=None), is_sampler=True)
        assert ctx._metrics[True].call_count == 0
        assert ctx._metrics[False].call_count == 0

    def test_flush_pending_records_model_only(self, monkeypatch):
        # A model step with no following sampler is recorded model-only on flush.
        ctx = _ctx(monkeypatch)
        ctx._submit(mm._Sample(2.0, host=7, phase=False), is_sampler=False)
        ctx._flush_pending()
        assert ctx._metrics[False].latencies == [2.0]
        assert ctx._metrics[False].host_times == [7]
        assert ctx._pending is None

    def test_prefill_and_decode_go_to_separate_buckets(self, monkeypatch):
        ctx = _ctx(monkeypatch)
        ctx._submit(mm._Sample(1.0, phase=True), is_sampler=False)
        ctx._submit(mm._Sample(0.5, phase=None), is_sampler=True)
        ctx._submit(mm._Sample(0.2, phase=False), is_sampler=False)
        ctx._submit(mm._Sample(0.1, phase=None), is_sampler=True)
        assert ctx._metrics[True].latencies == [1.5]
        assert ctx._metrics[False].latencies == pytest.approx([0.3])

    def test_profile_model_flushes_prior_pending(self, monkeypatch):
        # Back-to-back model steps: the earlier pending one is flushed (recorded
        # model-only) when the next profile_model starts.
        ctx = _ctx(monkeypatch)
        ctx._submit(mm._Sample(1.0, phase=True), is_sampler=False)
        ctx.profile_model(is_prefill=False)  # span not entered; only flushes
        assert ctx._metrics[True].call_count == 1
        assert ctx._pending is None

    def test_drain_backlog_flushes_runtimes_once(self, monkeypatch):
        calls = []
        with_flush = types.SimpleNamespace(flush_reports=lambda: calls.append(1))
        without_flush = types.SimpleNamespace()  # no flush_reports attr -> skipped
        ctx = _ctx(monkeypatch, runtimes=[with_flush, without_flush])
        ctx._drain_report_backlog()
        ctx._drain_report_backlog()  # idempotent
        assert calls == [1]


class TestE2EWindow:
    # e2e is bounded by start_e2e/end_e2e -- one execute_model pass -- not by the
    # spans inside it, and holds whether or not the pass reached a sampler.
    def test_window_covers_the_pass_not_just_the_spans(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.start_e2e()
        clock.now += 1.0  # input preparation, before the model span opens
        with ctx.profile_model(is_prefill=True):
            clock.now += 4.0
        clock.now += 1.0
        with ctx.profile_sampler():
            clock.now += 2.0
        clock.now += 1.0  # output copy, after the sampler span closed
        ctx.end_e2e()
        assert ctx._metrics[True].latencies == [6.0]  # the two spans only
        assert ctx._e2e[True].latencies == [9.0]  # the whole pass
        assert ctx._e2e_start is None

    def test_time_between_passes_is_excluded(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        for _ in range(2):
            ctx.start_e2e()
            with ctx.profile_model(is_prefill=False):
                clock.now += 1.0
            ctx.end_e2e()
            clock.now += 10.0  # engine and idle time between passes
        assert ctx._e2e[False].latencies == [1.0, 1.0]

    def test_sampler_less_pass_is_still_timed(self, monkeypatch):
        # Intermediate chunked prefill / non-last PP rank.
        clock = _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.start_e2e()
        with ctx.profile_model(is_prefill=True):
            clock.now += 3.0
        ctx.end_e2e()
        assert ctx._e2e[True].latencies == [3.0]

    def test_pass_without_a_model_span_is_dropped(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.start_e2e()
        clock.now += 1.0
        ctx.end_e2e()
        assert ctx._e2e == {}

    def test_end_without_start_is_a_noop(self, monkeypatch):
        _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.end_e2e()
        assert ctx._e2e == {}

    def test_second_end_does_not_record_again(self, monkeypatch):
        # execute_model closes the window when it returns an output itself;
        # sample_tokens still runs its own end.
        clock = _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.start_e2e()
        with ctx.profile_model(is_prefill=False):
            clock.now += 1.0
        ctx.end_e2e()
        clock.now += 5.0
        ctx.end_e2e()
        assert ctx._e2e[False].latencies == [1.0]

    def test_phase_does_not_carry_over_to_the_next_window(self, monkeypatch):
        clock = _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.start_e2e()
        with ctx.profile_model(is_prefill=False):
            clock.now += 1.0
        ctx.end_e2e()
        ctx.start_e2e()
        clock.now += 5.0
        ctx.end_e2e()  # no model span: must not land in the decode bucket again
        assert ctx._e2e[False].latencies == [1.0]

    def test_model_span_outside_a_window_is_not_attributed(self, monkeypatch):
        _fake_clock(monkeypatch)
        ctx = _ctx(monkeypatch)
        ctx.profile_model(is_prefill=True)
        assert ctx._e2e_is_prefill is None


class TestTimingSpanWiring:
    def test_model_and_sampler_spans_record_one_merged_step(self, monkeypatch):
        ctx = _ctx(monkeypatch)
        _run_pass(ctx, is_prefill=True)
        m = ctx._metrics[True]
        assert m.call_count == 1
        assert m.host_times == []  # no reports captured -> no timing series
        assert ctx._e2e[True].call_count == 1
        assert ctx._pending is None


class TestReporting:
    def test_sections_are_labelled_by_phase_and_metric(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        ctx = _ctx(monkeypatch)
        _run_pass(ctx, is_prefill=True)
        _run_pass(ctx, is_prefill=False)
        ctx.print_stats()

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert list(data["sections"]) == [
            "PREFILL + SAMPLE",
            "DECODE + SAMPLE",
            "PREFILL E2E",
            "DECODE E2E",
        ]

    def test_unseen_phase_is_omitted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        ctx = _ctx(monkeypatch)
        _run_pass(ctx, is_prefill=False)
        ctx.print_stats()

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert list(data["sections"]) == ["DECODE + SAMPLE", "DECODE E2E"]

    def test_print_stats_flushes_the_final_pending_span(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        ctx = _ctx(monkeypatch)
        _run_pass(ctx, is_prefill=True, with_sampler=False)
        assert ctx._metrics[True].call_count == 0  # waiting for a sampler

        ctx.print_stats()
        assert ctx._metrics[True].call_count == 1


class TestE2EDecorators:
    @staticmethod
    def _runner(calls, execute_output):
        class Runner:
            performance_ctx = types.SimpleNamespace(
                start_e2e=lambda: calls.append("start"),
                end_e2e=lambda: calls.append("end"),
            )

            @mm._e2e_starts
            def execute_model(self):
                calls.append("execute")
                return execute_output

            @mm._e2e_ends
            def sample_tokens(self):
                calls.append("sample")
                return "out"

        return Runner()

    def test_window_closes_in_sample_tokens_when_execute_returns_none(self):
        calls: list = []
        runner = self._runner(calls, None)
        assert runner.execute_model() is None
        assert runner.sample_tokens() == "out"
        assert calls == ["start", "execute", "sample", "end"]

    def test_window_closes_in_execute_model_when_it_returns_an_output(self):
        calls: list = []
        runner = self._runner(calls, "early-output")
        assert runner.execute_model() == "early-output"
        assert calls == ["start", "execute", "end"]

    def test_end_runs_even_when_the_body_raises(self):
        calls = []

        class Runner:
            performance_ctx = types.SimpleNamespace(
                end_e2e=lambda: calls.append("end"),
            )

            @mm._e2e_ends
            def sample_tokens(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            Runner().sample_tokens()
        assert calls == ["end"]

    def test_identity_hands_back_the_undecorated_function(self):
        def fn():
            return 1

        assert mm._identity(fn) is fn

    def test_module_level_hooks_follow_the_metrics_env(self):
        expected = (
            (mm._e2e_starts, mm._e2e_ends)
            if envs.VLLM_RBLN_METRICS
            else (mm._identity, mm._identity)
        )
        assert (mm.e2e_starts, mm.e2e_ends) == expected


class TestWriteMetricsJson:
    def test_no_metrics_dir_skips_write(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", "")
        mm._write_metrics_json("runner", "", {})  # must not raise
        assert list(tmp_path.iterdir()) == []

    def test_writes_payload_with_section_dicts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        m = mm.Metrics()
        m.record(0.01, host_time=100)
        mm._write_metrics_json("runner", "", {"DECODE + SAMPLE": m})
        data = json.loads((tmp_path / "metrics.json").read_text())
        assert data["name"] == "runner"
        assert data["rank"] == ""
        assert data["sections"]["DECODE + SAMPLE"]["call_count"] == 1

    def test_rank_tag_lowercased_into_filename_suffix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(envs, "VLLM_RBLN_METRICS_DIR", str(tmp_path))
        mm._write_metrics_json("runner", "TP1 DP0", {})
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
        assert mm._rank_tag() == ""

    def test_only_multi_degree_axes_in_fixed_order(self, monkeypatch):
        self._patch_groups(
            monkeypatch,
            tp=self._grp(2, 1),
            pp=self._grp(1, 0),
            dp=self._grp(4, 3),
            ep=self._grp(1, 0),
        )
        assert mm._rank_tag() == "TP1 DP3"

    def test_group_lookup_failure_yields_empty(self, monkeypatch):
        import vllm.distributed as d

        def boom():
            raise RuntimeError("group not initialized")

        for name in ("get_tp_group", "get_pp_group", "get_dp_group", "get_ep_group"):
            monkeypatch.setattr(d, name, boom)
        assert mm._rank_tag() == ""


class TestRenderMetrics:
    def test_no_data_branch(self):
        assert mm._render_metrics("DECODE", mm.Metrics()) == [
            "DECODE METRICS: No data recorded"
        ]

    def test_p50_labelled_median_and_only_present_timings_shown(self):
        m = mm.Metrics()
        m.record(0.01, host_time=100, device_time=200)
        text = "\n".join(mm._render_metrics("PREFILL", m))
        assert "Median latency (ms)" in text  # p50 -> Median
        assert "P90 latency (ms)" in text  # others upper-cased
        assert "Mean host time (us)" in text
        assert "Mean device time (us)" in text
        assert "Mean ccl time (us)" not in text  # empty series omitted

    def test_report_wraps_sections_with_named_header(self):
        m = mm.Metrics()
        m.record(0.01)
        text = "\n".join(mm._render_metrics_report("runner", {"DECODE + SAMPLE": m}))
        assert "PERFORMANCE STATISTICS [runner]" in text
        assert "DECODE + SAMPLE METRICS:" in text


class TestNoopVariants:
    def test_noop_span_is_a_context_manager(self):
        with mm._NoopSpan() as span:
            assert isinstance(span, mm._NoopSpan)

    def test_noop_context_methods_are_inert(self):
        ctx = mm._NoopPerformanceContext("runner", runtimes=[])
        ctx.start_e2e()
        with ctx.profile_model(True):
            pass
        with ctx.profile_sampler():
            pass
        ctx.end_e2e()
        ctx.print_stats()  # must not raise
