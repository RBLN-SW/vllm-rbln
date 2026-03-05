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

import atexit
from collections import defaultdict
from dataclasses import dataclass, field

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single execution step."""

    latencies: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    host_times: list[int] = field(default_factory=list)
    device_times: list[int] = field(default_factory=list)
    ccl_times: list[int] = field(default_factory=list)

    def add_measurement(
        self,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
    ):
        """Add a latency, token count, and timing measurements."""
        self.latencies.append(latency)
        self.token_counts.append(token_count)
        if host_time is not None:
            self.host_times.append(host_time)
        if device_time is not None:
            self.device_times.append(device_time)
        if ccl_time is not None:
            self.ccl_times.append(ccl_time)

    def _without_outlier_f(self, values: list[float]) -> list[float]:
        """Return values excluding one outlier (max absolute deviation)."""
        if len(values) <= 1:
            return values
        mean = sum(values) / len(values)
        deviations = [abs(v - mean) for v in values]
        max_idx = deviations.index(max(deviations))
        return [v for i, v in enumerate(values) if i != max_idx]

    def _without_outlier_i(self, values: list[int]) -> list[int]:
        """Return values excluding one outlier (max absolute deviation)."""
        if len(values) <= 1:
            return values
        mean = sum(values) / len(values)
        deviations = [abs(v - mean) for v in values]
        max_idx = deviations.index(max(deviations))
        return [v for i, v in enumerate(values) if i != max_idx]

    def get_avg_latency(self, ignore_outlier: bool = True) -> float:
        """Get average latency in milliseconds,
        optionally ignoring one outlier."""
        values = (
            self._without_outlier_f(self.latencies)
            if ignore_outlier
            else self.latencies
        )
        return sum(values) / len(values) * 1000 if values else 0.0

    def get_avg_throughput(self, ignore_outlier: bool = True) -> float:
        """Get average throughput in tokens/second,
        optionally ignoring one outlier."""
        if not self.latencies or not self.token_counts:
            return 0.0
        latencies = (
            self._without_outlier_f(self.latencies)
            if ignore_outlier
            else self.latencies
        )
        tokens = (
            self._without_outlier_i(self.token_counts)
            if ignore_outlier
            else self.token_counts
        )
        total_time = sum(latencies)
        total_tokens = sum(tokens)
        return total_tokens / total_time if total_time > 0 else 0.0

    def get_avg_host_time(self, ignore_outlier: bool = True) -> float:
        """Get average host time in microseconds,
        optionally ignoring one outlier."""
        values = (
            self._without_outlier_i(self.host_times)
            if ignore_outlier
            else self.host_times
        )
        return sum(values) / len(values) if values else 0.0

    def get_avg_device_time(self, ignore_outlier: bool = True) -> float:
        """Get average device time in microseconds,
        optionally ignoring one outlier."""
        values = (
            self._without_outlier_i(self.device_times)
            if ignore_outlier
            else self.device_times
        )
        return sum(values) / len(values) if values else 0.0

    def get_avg_ccl_time(self, ignore_outlier: bool = True) -> float:
        """Get average ccl time in microseconds,
        optionally ignoring one outlier."""
        values = (
            self._without_outlier_i(self.ccl_times)
            if ignore_outlier
            else self.ccl_times
        )
        return sum(values) / len(values) if values else 0.0

    def get_call_counts(self) -> int:
        """Get total number of requests processed."""
        return len(self.latencies)

    def show_stats(self, stat_type: str):
        if self.get_call_counts() > 0:
            logger.info("%s METRICS:", stat_type)
            logger.info("  Total call counts: %d", self.get_call_counts())
            logger.info("  Average latency: %.2f ms", self.get_avg_latency())
            if sum(self.token_counts) > 0:
                logger.info("  Total tokens processed: %d", sum(self.token_counts))
                logger.info(
                    "  Average throughput: %.2f tokens/sec", self.get_avg_throughput()
                )
            if self.host_times:
                logger.info("  Average host time: %.2f us", self.get_avg_host_time())
            if self.device_times:
                logger.info(
                    "  Average device time: %.2f us", self.get_avg_device_time()
                )
            if self.ccl_times:
                logger.info("  Average ccl time: %.2f us", self.get_avg_ccl_time())
        else:
            logger.info("%s METRICS: No data recorded", stat_type)


class PrefillMetricsByRequestID:
    """Metrics for prefill step by request id."""

    def __init__(self):
        self.metrics = defaultdict(StepMetrics)

    def add_measurement(
        self,
        request_id: str,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
    ):
        """Add a latency and token count measurement."""
        self.metrics[request_id].add_measurement(
            latency, token_count, host_time, device_time, ccl_time
        )

    def get_avg_latency_per_request(self) -> dict[str, float]:
        """Get average latency per request."""
        return {
            request_id: metric.get_avg_latency()
            for request_id, metric in self.metrics.items()
        }

    def get_num_request_ids(self) -> int:
        """Get total number of request ids processed."""
        return len(self.metrics)


class PerformanceTracker:
    """Tracks performance metrics for prefill and decode steps."""

    def __init__(self, name: str | None = None):
        self.name = name
        self.prefill_metrics = StepMetrics()
        self.decode_metrics = StepMetrics()
        self.prefill_metrics_by_request_id = PrefillMetricsByRequestID()
        self.padded_decode_metrics = StepMetrics()
        self._registered_cleanup = False

    def register_cleanup(self):
        """Register cleanup function to print stats on exit."""
        if not self._registered_cleanup:
            atexit.register(self.print_final_stats)
            self._registered_cleanup = True

    def check_dummy_request(self, request_ids: list[str] | None) -> bool:
        if request_ids:
            request_id = request_ids[0]
            if request_id.startswith("dummy_request_"):
                return True
        return False

    def record_prefill(
        self,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
        request_ids: list[str] | None = None,
    ):
        """Record prefill step metrics."""
        if self.check_dummy_request(request_ids):
            return
        request_id = None
        if request_ids is not None:
            assert len(request_ids) == 1, (
                f"Expected exactly one request_id during prefill, "
                f"got {len(request_ids)}: {request_ids}"
            )
            request_id = request_ids[0]
        self.prefill_metrics.add_measurement(latency, token_count)
        if request_id:
            self.prefill_metrics_by_request_id.add_measurement(
                request_id, latency, token_count, host_time, device_time, ccl_time
            )

    def record_decode(
        self,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
        padded_decode: bool = False,
        request_ids: list[str] | None = None,
    ):
        """Record decode step metrics."""
        if self.check_dummy_request(request_ids):
            return
        metrics = self.padded_decode_metrics if padded_decode else self.decode_metrics
        metrics.add_measurement(latency, token_count, host_time, device_time, ccl_time)

    def print_final_stats(self):
        logger.info("=" * 80)
        if self.name:
            logger.info("FINAL PERFORMANCE STATISTICS [%s]", self.name)
        else:
            logger.info("FINAL PERFORMANCE STATISTICS")
        logger.info("=" * 80)

        # Prefill stats
        self.prefill_metrics.show_stats("PREFILL")
        logger.info("-" * 40)

        # Decode stats
        self.decode_metrics.show_stats("DECODE")
        logger.info("-" * 40)

        # Padded decode stats
        self.padded_decode_metrics.show_stats("PADDED DECODE")
        logger.info("=" * 80)
