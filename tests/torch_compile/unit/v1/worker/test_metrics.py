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

"""Tests for vllm_rbln.v1.worker.metrics module.

Covers StepMetrics, PrefillMetricsByRequestID, and PerformanceTracker
with focus on edge cases, statistical correctness, and error paths.
"""

import os
from unittest.mock import patch

import pytest

import vllm_rbln.v1.worker.metrics as metrics_module
from vllm_rbln.v1.worker.metrics import (
    PerformanceTracker,
    PrefillMetricsByRequestID,
    StepMetrics,
    StepReport,
)


# ---------------------------------------------------------------------------
# StepMetrics
# ---------------------------------------------------------------------------
class TestStepMetricsAddMeasurement:
    """Verify measurement recording, including optional timing fields."""

    def test_basic_add(self):
        m = StepMetrics()
        m.add_measurement(0.5, 10)
        assert m.latencies == [0.5]
        assert m.token_counts == [10]
        assert m.host_times == []
        assert m.device_times == []
        assert m.ccl_times == []

    def test_add_with_all_timings(self):
        m = StepMetrics()
        m.add_measurement(1.0, 20, host_time=100, device_time=200, ccl_time=50)
        assert m.host_times == [100]
        assert m.device_times == [200]
        assert m.ccl_times == [50]

    def test_none_timings_are_skipped(self):
        """Explicitly passing None should NOT append to timing lists."""
        m = StepMetrics()
        m.add_measurement(1.0, 5, host_time=None, device_time=None, ccl_time=None)
        assert m.host_times == []
        assert m.device_times == []
        assert m.ccl_times == []

    def test_zero_values_are_recorded(self):
        """Zero is a valid measurement, distinct from None."""
        m = StepMetrics()
        m.add_measurement(0.0, 0, host_time=0, device_time=0, ccl_time=0)
        assert m.host_times == [0]
        assert m.device_times == [0]
        assert m.ccl_times == [0]
        assert m.latencies == [0.0]
        assert m.token_counts == [0]


class TestLatencyPercentiles:
    """get_latency_percentiles reports the distribution (no data-point removal)."""

    def test_empty(self):
        assert StepMetrics().get_latency_percentiles() == {}

    def test_keys_and_max(self):
        m = StepMetrics()
        for lat in [0.010, 0.020, 0.030, 0.040]:
            m.add_measurement(lat, 1)
        pct = m.get_latency_percentiles()
        assert set(pct) == {"p50", "p90", "p99", "max"}
        assert pct["max"] == pytest.approx(40.0)  # 0.040 s -> ms

    def test_spike_shows_in_tail_not_p50(self):
        """A one-off spike inflates max/p99 but leaves p50 (median) robust."""
        m = StepMetrics()
        for lat in [0.010, 0.010, 0.010, 0.010, 5.0]:  # 5 s spike
            m.add_measurement(lat, 1)
        pct = m.get_latency_percentiles()
        assert pct["p50"] == pytest.approx(10.0, abs=1.0)  # ~10 ms, spike-robust
        assert pct["max"] == pytest.approx(5000.0)

    def test_custom_percentiles(self):
        m = StepMetrics()
        for lat in [0.01, 0.02, 0.03]:
            m.add_measurement(lat, 1)
        pct = m.get_latency_percentiles(percentiles=(25.0, 75.0))
        assert set(pct) == {"p25", "p75", "max"}


class TestStepMetricsAverages:
    """Averages are plain means (no outlier removal); throughput is sum-based."""

    def test_avg_latency_empty(self):
        assert StepMetrics().get_avg_latency() == 0.0

    def test_avg_latency_converts_to_ms(self):
        m = StepMetrics()
        m.add_measurement(1.0, 10)  # 1 second
        assert m.get_avg_latency() == 1000.0

    def test_avg_latency_is_plain_mean(self):
        """No outlier removal: a spike is included in the mean (use p50 instead)."""
        m = StepMetrics()
        for lat in [0.01, 0.01, 0.01, 0.01, 5.0]:
            m.add_measurement(lat, 1)
        # mean = (0.04 + 5.0) / 5 = 1.008 s -> 1008 ms
        assert m.get_avg_latency() == pytest.approx(1008.0, abs=1.0)

    def test_avg_throughput_empty(self):
        assert StepMetrics().get_avg_throughput() == 0.0

    def test_avg_throughput_sum_based(self):
        m = StepMetrics()
        m.add_measurement(1.0, 100)
        m.add_measurement(1.0, 100)
        # 200 tokens / 2.0 s = 100 tok/s
        assert m.get_avg_throughput() == pytest.approx(100.0)

    def test_avg_throughput_bimodal_is_stable(self):
        """Sum-based throughput is unaffected by a bimodal per-step split."""
        m = StepMetrics()
        m.add_measurement(0.1, 500)  # fast step
        m.add_measurement(1.9, 500)  # slow step
        # 1000 tokens / 2.0 s = 500 tok/s regardless of the split
        assert m.get_avg_throughput() == pytest.approx(500.0)

    def test_avg_throughput_zero_latency(self):
        """If all latencies are 0, total_time=0 -> returns 0.0."""
        m = StepMetrics()
        m.add_measurement(0.0, 10)
        m.add_measurement(0.0, 10)
        assert m.get_avg_throughput() == 0.0

    def test_avg_throughput_no_tokens(self):
        """Latencies present but no tokens -> 0.0."""
        m = StepMetrics()
        m.latencies = [1.0]
        assert m.get_avg_throughput() == 0.0

    def test_avg_host_time_empty(self):
        assert StepMetrics().get_avg_host_time() == 0.0

    def test_avg_device_time_is_mean(self):
        m = StepMetrics()
        m.add_measurement(1.0, 1, device_time=100)
        m.add_measurement(1.0, 1, device_time=200)
        assert m.get_avg_device_time() == 150.0

    def test_avg_ccl_time_is_mean(self):
        m = StepMetrics()
        for _ in range(3):
            m.add_measurement(1.0, 1, ccl_time=300)
        assert m.get_avg_ccl_time() == 300.0

    def test_get_call_counts(self):
        m = StepMetrics()
        assert m.get_call_counts() == 0
        m.add_measurement(1.0, 5)
        m.add_measurement(2.0, 10)
        assert m.get_call_counts() == 2


class TestShowStats:
    """Verify show_stats logs correctly and handles zero-data case."""

    def test_show_stats_no_data(self):
        m = StepMetrics()
        with patch.object(m, "get_call_counts", return_value=0):
            # Should not raise
            m.show_stats("TEST")

    def test_show_stats_with_data(self):
        m = StepMetrics()
        m.add_measurement(0.5, 100, host_time=50, device_time=80, ccl_time=20)
        m.add_measurement(0.5, 100, host_time=50, device_time=80, ccl_time=20)
        # Should not raise
        m.show_stats("DECODE")

    def test_show_stats_zero_tokens(self):
        """If token_counts sum to 0, throughput line should not be logged."""
        m = StepMetrics()
        m.add_measurement(1.0, 0)
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            m.show_stats("PREFILL")
            # "throughput" should NOT appear in any info call
            for call in mock_logger.info.call_args_list:
                assert "throughput" not in str(call).lower()


# ---------------------------------------------------------------------------
# PrefillMetricsByRequestID
# ---------------------------------------------------------------------------
class TestPrefillMetricsByRequestID:
    def test_separate_request_tracking(self):
        pm = PrefillMetricsByRequestID()
        pm.add_measurement("req-1", 1.0, 50)
        pm.add_measurement("req-1", 0.5, 30)
        pm.add_measurement("req-2", 2.0, 100)

        assert pm.get_num_request_ids() == 2
        latencies = pm.get_avg_latency_per_request()
        assert "req-1" in latencies
        assert "req-2" in latencies

    def test_empty_metrics(self):
        pm = PrefillMetricsByRequestID()
        assert pm.get_num_request_ids() == 0
        assert pm.get_avg_latency_per_request() == {}

    def test_single_measurement_per_request(self):
        pm = PrefillMetricsByRequestID()
        pm.add_measurement("req-1", 0.5, 10)
        latencies = pm.get_avg_latency_per_request()
        # Single measurement: avg = 0.5 * 1000 = 500ms
        assert latencies["req-1"] == pytest.approx(500.0)

    def test_total_latency_per_request_sums_chunks(self):
        """Per-request total = sum of the request's chunk latencies (ms)."""
        pm = PrefillMetricsByRequestID()
        pm.add_measurement("req-1", 0.5, 30)  # chunk 1
        pm.add_measurement("req-1", 0.3, 20)  # chunk 2
        totals = pm.get_total_latency_per_request()
        assert totals["req-1"] == pytest.approx(800.0)  # (0.5 + 0.3) * 1000

    def test_timing_fields_forwarded(self):
        """Ensure host/device/ccl times are forwarded to inner StepMetrics."""
        pm = PrefillMetricsByRequestID()
        pm.add_measurement(
            "req-1", 1.0, 10, host_time=100, device_time=200, ccl_time=50
        )
        inner = pm.metrics["req-1"]
        assert inner.host_times == [100]
        assert inner.device_times == [200]
        assert inner.ccl_times == [50]


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------
class TestPerformanceTrackerInit:
    def test_default_name_is_none(self):
        pt = PerformanceTracker()
        assert pt.name is None

    def test_custom_name(self):
        pt = PerformanceTracker(name="worker-0")
        assert pt.name == "worker-0"


class TestCheckDummyRequest:
    def test_dummy_request_detected(self):
        pt = PerformanceTracker()
        assert pt.check_dummy_request(["dummy_request_0"]) is True
        assert pt.check_dummy_request(["dummy_request_warmup"]) is True

    def test_normal_request_not_dummy(self):
        pt = PerformanceTracker()
        assert pt.check_dummy_request(["real-request-123"]) is False

    def test_none_request_ids(self):
        pt = PerformanceTracker()
        assert pt.check_dummy_request(None) is False

    def test_empty_list(self):
        pt = PerformanceTracker()
        assert pt.check_dummy_request([]) is False

    def test_only_checks_first_element(self):
        """If first element is not dummy, returns False even if others are."""
        pt = PerformanceTracker()
        assert pt.check_dummy_request(["real", "dummy_request_1"]) is False

    def test_dummy_not_at_start(self):
        """A request_id containing 'dummy_request_' but not starting with it."""
        pt = PerformanceTracker()
        assert pt.check_dummy_request(["prefix_dummy_request_0"]) is False


class TestRecordPrefill:
    def test_basic_prefill(self):
        pt = PerformanceTracker()
        pt.record_prefill(0.5, 100, request_ids=["req-1"])
        assert pt.prefill_metrics.get_call_counts() == 1
        assert pt.prefill_metrics_by_request_id.get_num_request_ids() == 1

    def test_prefill_without_request_ids(self):
        pt = PerformanceTracker()
        pt.record_prefill(0.5, 100)
        assert pt.prefill_metrics.get_call_counts() == 1
        # No per-request tracking
        assert pt.prefill_metrics_by_request_id.get_num_request_ids() == 0

    def test_prefill_skips_dummy_request(self):
        pt = PerformanceTracker()
        pt.record_prefill(0.5, 100, request_ids=["dummy_request_0"])
        assert pt.prefill_metrics.get_call_counts() == 0

    def test_prefill_multiple_request_ids_raises(self):
        """Prefill must have exactly one request_id when request_ids is not None."""
        pt = PerformanceTracker()
        with pytest.raises(AssertionError, match="Expected exactly one request_id"):
            pt.record_prefill(0.5, 100, request_ids=["req-1", "req-2"])

    def test_prefill_empty_request_ids_list(self):
        """Empty list passes check_dummy_request (returns False),
        but assertion len==1 should fail."""
        pt = PerformanceTracker()
        with pytest.raises(AssertionError, match="Expected exactly one request_id"):
            pt.record_prefill(0.5, 100, request_ids=[])

    def test_prefill_timing_not_forwarded_to_global_metrics(self):
        pt = PerformanceTracker()
        pt.record_prefill(
            0.5, 100, host_time=10, device_time=20, ccl_time=5, request_ids=["req-1"]
        )
        assert pt.prefill_metrics.host_times == [10]
        assert pt.prefill_metrics.device_times == [20]
        assert pt.prefill_metrics.ccl_times == [5]
        inner = pt.prefill_metrics_by_request_id.metrics["req-1"]
        assert inner.host_times == [10]
        assert inner.device_times == [20]
        assert inner.ccl_times == [5]


class TestRecordDecode:
    def test_basic_decode(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1)
        assert pt.decode_metrics.get_call_counts() == 1
        assert pt.padded_decode_metrics.get_call_counts() == 0

    def test_padded_decode(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, padded_decode=True)
        assert pt.padded_decode_metrics.get_call_counts() == 1
        assert pt.decode_metrics.get_call_counts() == 0

    def test_decode_skips_dummy_request(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, request_ids=["dummy_request_warmup"])
        assert pt.decode_metrics.get_call_counts() == 0

    def test_decode_with_timings(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, host_time=50, device_time=80, ccl_time=20)
        assert pt.decode_metrics.host_times == [50]
        assert pt.decode_metrics.device_times == [80]
        assert pt.decode_metrics.ccl_times == [20]

    def test_padded_decode_with_timings(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, host_time=50, device_time=80, padded_decode=True)
        assert pt.padded_decode_metrics.host_times == [50]
        assert pt.decode_metrics.host_times == []


class TestDecodeBucketRouting:
    """Decode steps carrying a bucket size are also tallied per bucket."""

    def test_no_bucket_leaves_by_bucket_empty(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1)
        assert pt.decode_metrics_by_bucket == {}

    def test_bucket_routed_to_its_own_metrics(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, decode_bucket=4)
        pt.record_decode(0.02, 1, decode_bucket=4)
        pt.record_decode(0.03, 1, decode_bucket=8)
        assert set(pt.decode_metrics_by_bucket) == {4, 8}
        assert pt.decode_metrics_by_bucket[4].get_call_counts() == 2
        assert pt.decode_metrics_by_bucket[8].get_call_counts() == 1
        # Aggregate decode_metrics still sees every step.
        assert pt.decode_metrics.get_call_counts() == 3

    def test_padded_decode_not_routed_by_bucket(self):
        """Padded decodes stay out of the per-bucket tables so those tables
        sum to the (non-padded) DECODE line."""
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, padded_decode=True, decode_bucket=16)
        assert pt.padded_decode_metrics.get_call_counts() == 1
        assert pt.decode_metrics_by_bucket == {}

    def test_record_routes_bucket_from_report(self):
        pt = PerformanceTracker()
        pt.record(StepReport(latency=0.01, token_count=1, decode_bucket=4))
        pt.record(StepReport(latency=0.02, token_count=1, decode_bucket=8))
        assert set(pt.decode_metrics_by_bucket) == {4, 8}

    def test_prefill_report_never_routed_to_bucket(self):
        pt = PerformanceTracker()
        pt.record(
            StepReport(
                latency=0.5,
                token_count=100,
                is_prefill=True,
                request_ids=["req-1"],
            )
        )
        assert pt.decode_metrics_by_bucket == {}


class TestColdStartExclusion:
    """Cold-start steps are only counted, never folded into steady metrics."""

    def test_cold_start_excluded_from_decode(self):
        pt = PerformanceTracker()
        pt.record(StepReport(latency=0.01, token_count=1, is_cold_start=True))
        assert pt.cold_start_count == 1
        assert pt.decode_metrics.get_call_counts() == 0
        assert pt.decode_metrics_by_bucket == {}

    def test_cold_start_excluded_from_prefill(self):
        pt = PerformanceTracker()
        pt.record(
            StepReport(
                latency=0.5,
                token_count=100,
                is_prefill=True,
                is_cold_start=True,
                request_ids=["req-1"],
            )
        )
        assert pt.cold_start_count == 1
        assert pt.prefill_metrics.get_call_counts() == 0

    def test_steady_step_not_counted_as_cold_start(self):
        pt = PerformanceTracker()
        pt.record(StepReport(latency=0.01, token_count=1, decode_bucket=4))
        assert pt.cold_start_count == 0
        assert pt.decode_metrics.get_call_counts() == 1

    def test_cold_start_metadata_survives_merge(self):
        """merged_with keeps the model report's cold-start/bucket metadata."""
        model = StepReport(latency=0.01, token_count=1, is_cold_start=True,
                           decode_bucket=8)
        sampler = StepReport(latency=0.002)
        combined = model.merged_with(sampler)
        assert combined.is_cold_start is True
        assert combined.decode_bucket == 8
        assert combined.latency == pytest.approx(0.012)

    def test_sampler_side_cold_start_excluded_after_merge(self):
        """A sampler-only compile is caught by re-checking is_cold_start after
        the merge (merged_with keeps the model's pre-sampler snapshot)."""
        pt = PerformanceTracker()
        model = StepReport(latency=0.01, token_count=1, is_cold_start=False,
                           decode_bucket=4)
        sampler = StepReport(latency=0.002)
        combined = model.merged_with(sampler)
        combined.is_cold_start = True  # counter grew during the sampler
        pt.record(combined)
        assert pt.cold_start_count == 1
        assert pt.decode_metrics.get_call_counts() == 0
        assert pt.decode_metrics_by_bucket == {}


class TestPrintFinalStats:
    def test_print_with_name(self):
        pt = PerformanceTracker(name="gpu-0")
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            pt.print_final_stats()
            # Check name appears in output
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("gpu-0" in c for c in calls)

    def test_print_without_name(self):
        pt = PerformanceTracker()
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            pt.print_final_stats()
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("FINAL PERFORMANCE STATISTICS" in c for c in calls)

    def test_print_with_data(self):
        pt = PerformanceTracker()
        pt.record_prefill(0.5, 100, request_ids=["req-1"])
        pt.record_decode(0.01, 1)
        pt.record_decode(0.01, 1, padded_decode=True)
        # Should not raise
        pt.print_final_stats()

    def test_per_bucket_table_only_when_multiple_buckets(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, decode_bucket=4)
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            pt.print_final_stats()
        calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
        # Single bucket -> no per-bucket breakdown.
        assert "batch bucket" not in calls

    def test_per_bucket_table_shown_for_multiple_buckets(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1, decode_bucket=4)
        pt.record_decode(0.02, 1, decode_bucket=8)
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            pt.print_final_stats()
        calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "batch bucket 4" in calls
        assert "batch bucket 8" in calls

    def test_cold_start_line_shown_when_nonzero(self):
        pt = PerformanceTracker()
        pt.record(StepReport(latency=0.01, token_count=1, is_cold_start=True))
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            pt.print_final_stats()
        calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Cold-start steps excluded" in calls

    def test_cold_start_line_absent_when_zero(self):
        pt = PerformanceTracker()
        pt.record_decode(0.01, 1)
        with patch("vllm_rbln.v1.worker.metrics.logger") as mock_logger:
            pt.print_final_stats()
        calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Cold-start" not in calls


class TestMetricsFileOutput:
    """Verify VLLM_RBLN_METRICS_FILE mirrors the final report to a file."""

    @pytest.fixture(autouse=True)
    def _reset_handler_state(self):
        metrics_module._metrics_file_attached = False
        original = list(metrics_module.logger.handlers)
        yield
        for handler in list(metrics_module.logger.handlers):
            if handler not in original:
                metrics_module.logger.removeHandler(handler)
                handler.close()
        metrics_module._metrics_file_attached = False

    def test_no_file_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_RBLN_METRICS_FILE", raising=False)
        pt = PerformanceTracker("MODEL")
        pt.record_prefill(0.5, 100, request_ids=["req-1"])
        pt.print_final_stats()
        assert list(tmp_path.iterdir()) == []

    def test_writes_pid_suffixed_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_RBLN_METRICS_FILE", str(tmp_path / "metrics.log"))
        pt = PerformanceTracker("MODEL")
        pt.record_prefill(0.5, 100, request_ids=["req-1"])
        pt.print_final_stats()
        expected = tmp_path / f"metrics.{os.getpid()}.log"
        assert expected.exists()
        content = expected.read_text()
        assert "FINAL PERFORMANCE STATISTICS [MODEL]" in content
        assert "PREFILL METRICS" in content

    def test_handler_attached_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_RBLN_METRICS_FILE", str(tmp_path / "metrics.log"))
        before = len(metrics_module.logger.handlers)
        pt = PerformanceTracker("MODEL")
        pt.print_final_stats()
        pt.print_final_stats()
        assert len(metrics_module.logger.handlers) - before == 1
