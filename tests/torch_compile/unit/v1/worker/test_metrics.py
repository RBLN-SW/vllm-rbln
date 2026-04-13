# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for vllm_rbln.v1.worker.metrics module.

Covers StepMetrics, PrefillMetricsByRequestID, and PerformanceTracker
with focus on edge cases, statistical correctness, and error paths.
"""

from unittest.mock import patch

import pytest

from vllm_rbln.v1.worker.metrics import (
    PerformanceTracker,
    PrefillMetricsByRequestID,
    StepMetrics,
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


class TestOutlierRemoval:
    """Test _without_outlier_f and _without_outlier_i edge cases."""

    def test_empty_list(self):
        m = StepMetrics()
        assert m._without_outlier_f([]) == []
        assert m._without_outlier_i([]) == []

    def test_single_element(self):
        m = StepMetrics()
        assert m._without_outlier_f([42.0]) == [42.0]
        assert m._without_outlier_i([42]) == [42]

    def test_two_elements_removes_farthest_from_mean(self):
        """With [1.0, 100.0], mean=50.5. Both deviate 49.5 — first max wins."""
        m = StepMetrics()
        result = m._without_outlier_f([1.0, 100.0])
        # index(max(deviations)) returns 0 when deviations are equal
        assert len(result) == 1
        assert result == [100.0]

    def test_obvious_outlier_removed(self):
        m = StepMetrics()
        values = [10.0, 10.1, 9.9, 10.0, 500.0]
        result = m._without_outlier_f(values)
        assert 500.0 not in result
        assert len(result) == 4

    def test_all_identical_values(self):
        """All deviations are 0 — removes first element (index 0)."""
        m = StepMetrics()
        result = m._without_outlier_f([5.0, 5.0, 5.0])
        assert len(result) == 2
        assert all(v == 5.0 for v in result)

    def test_negative_values(self):
        m = StepMetrics()
        result = m._without_outlier_f([-100.0, 1.0, 2.0, 3.0])
        assert -100.0 not in result
        assert len(result) == 3

    def test_outlier_removal_int(self):
        m = StepMetrics()
        result = m._without_outlier_i([10, 11, 9, 10, 999])
        assert 999 not in result
        assert len(result) == 4


class TestStepMetricsAverages:
    """Test average computations including edge cases and outlier flag."""

    def test_avg_latency_empty(self):
        m = StepMetrics()
        assert m.get_avg_latency() == 0.0
        assert m.get_avg_latency(ignore_outlier=False) == 0.0

    def test_avg_latency_converts_to_ms(self):
        m = StepMetrics()
        m.add_measurement(1.0, 10)  # 1 second
        # single value, outlier removal returns as-is
        assert m.get_avg_latency() == 1000.0

    def test_avg_latency_with_outlier(self):
        m = StepMetrics()
        for lat in [0.01, 0.01, 0.01, 0.01, 5.0]:
            m.add_measurement(lat, 1)
        avg_with = m.get_avg_latency(ignore_outlier=True)
        avg_without = m.get_avg_latency(ignore_outlier=False)
        # Removing the 5.0 outlier should give much lower average
        assert avg_with < avg_without
        assert avg_with == pytest.approx(10.0, abs=1.0)  # ~0.01s * 1000

    def test_avg_throughput_empty(self):
        m = StepMetrics()
        assert m.get_avg_throughput() == 0.0

    def test_avg_throughput_basic(self):
        m = StepMetrics()
        m.add_measurement(1.0, 100)
        m.add_measurement(1.0, 100)
        # total_tokens=200, total_time=2.0 (single element no outlier removal effective)
        # Actually 2 elements: outlier removal removes one. So tokens=100, time=1.0
        throughput = m.get_avg_throughput(ignore_outlier=True)
        assert throughput == pytest.approx(100.0)

    def test_avg_throughput_no_outlier_flag(self):
        m = StepMetrics()
        m.add_measurement(1.0, 100)
        m.add_measurement(1.0, 100)
        throughput = m.get_avg_throughput(ignore_outlier=False)
        assert throughput == pytest.approx(100.0)

    def test_avg_throughput_zero_latency(self):
        """If all latencies are 0, total_time=0 → returns 0.0."""
        m = StepMetrics()
        m.add_measurement(0.0, 10)
        m.add_measurement(0.0, 10)
        assert m.get_avg_throughput() == 0.0

    def test_avg_throughput_independent_outlier_removal(self):
        """Outlier removal on latencies and token_counts is independent.

        This means different indices may be removed from each list.
        Verify the computation still works and doesn't crash.
        """
        m = StepMetrics()
        # latency outlier at index 2, token_count outlier at index 0
        m.add_measurement(1.0, 9999)
        m.add_measurement(1.0, 10)
        m.add_measurement(100.0, 10)
        throughput = m.get_avg_throughput(ignore_outlier=True)
        # latencies after removal: [1.0, 1.0], tokens after removal: [10, 10]
        # throughput = 20 / 2.0 = 10.0
        assert throughput == pytest.approx(10.0)

    def test_avg_throughput_mismatched_empty_tokens(self):
        """Latencies present but no token_counts → still returns 0.0."""
        m = StepMetrics()
        m.latencies = [1.0]
        # token_counts is empty
        assert m.get_avg_throughput() == 0.0

    def test_avg_host_time_empty(self):
        m = StepMetrics()
        assert m.get_avg_host_time() == 0.0

    def test_avg_device_time(self):
        m = StepMetrics()
        m.add_measurement(1.0, 1, device_time=100)
        m.add_measurement(1.0, 1, device_time=200)
        # With outlier removal: one removed, single value remains
        assert m.get_avg_device_time(ignore_outlier=True) in (100.0, 200.0)
        assert m.get_avg_device_time(ignore_outlier=False) == 150.0

    def test_avg_ccl_time(self):
        m = StepMetrics()
        m.add_measurement(1.0, 1, ccl_time=300)
        m.add_measurement(1.0, 1, ccl_time=300)
        m.add_measurement(1.0, 1, ccl_time=300)
        # All same, outlier removal removes one, avg still 300
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
