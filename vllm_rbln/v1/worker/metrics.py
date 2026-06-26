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
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TypeVar

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger, make_file_handler

logger = init_logger(__name__)

T = TypeVar("T", int, float)

_metrics_file_attached = False


def _attach_metrics_file_handler() -> None:
    """Mirror metrics output to VLLM_RBLN_METRICS_FILE if configured.

    The configured path is suffixed with the worker pid so concurrent workers
    (TP/DP) write to separate files instead of clobbering one another. Runs at
    most once; a failure to open the file is logged and stdout output is kept.
    """
    global _metrics_file_attached
    if _metrics_file_attached or not envs.VLLM_RBLN_METRICS_FILE:
        return
    _metrics_file_attached = True
    root, ext = os.path.splitext(envs.VLLM_RBLN_METRICS_FILE)
    path = f"{root}.{os.getpid()}{ext}"
    try:
        logger.addHandler(make_file_handler(path))
    except OSError as e:
        _metrics_file_attached = False
        logger.warning("Failed to open metrics file %s: %s", path, e)


@dataclass
class StepMetrics:
    """Metrics for a single execution step."""

    latencies: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    host_times: list[int] = field(default_factory=list)
    device_times: list[int] = field(default_factory=list)
    ccl_times: list[int] = field(default_factory=list)
    prepare_times: list[int] = field(default_factory=list)

    def add_measurement(
        self,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
        prepare_time: int | None = None,
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
        if prepare_time is not None:
            self.prepare_times.append(prepare_time)

    def _without_outlier(self, values: list[T]) -> list[T]:
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
            self._without_outlier(self.latencies) if ignore_outlier else self.latencies
        )
        return sum(values) / len(values) * 1000 if values else 0.0

    def get_avg_throughput(self, ignore_outlier: bool = True) -> float:
        """Get average throughput in tokens/second,
        optionally ignoring one outlier."""
        if not self.latencies or not self.token_counts:
            return 0.0
        latencies = (
            self._without_outlier(self.latencies) if ignore_outlier else self.latencies
        )
        tokens = (
            self._without_outlier(self.token_counts)
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
            self._without_outlier(self.host_times)
            if ignore_outlier
            else self.host_times
        )
        return sum(values) / len(values) if values else 0.0

    def get_avg_device_time(self, ignore_outlier: bool = True) -> float:
        """Get average device time in microseconds,
        optionally ignoring one outlier."""
        values = (
            self._without_outlier(self.device_times)
            if ignore_outlier
            else self.device_times
        )
        return sum(values) / len(values) if values else 0.0

    def get_avg_ccl_time(self, ignore_outlier: bool = True) -> float:
        """Get average ccl time in microseconds,
        optionally ignoring one outlier."""
        values = (
            self._without_outlier(self.ccl_times) if ignore_outlier else self.ccl_times
        )
        return sum(values) / len(values) if values else 0.0

    def get_avg_prepare_time(self, ignore_outlier: bool = True) -> float:
        """Get average prepare time (PrepareInputs + PrepareOutputs around Run)
        in microseconds, optionally ignoring one outlier."""
        values = (
            self._without_outlier(self.prepare_times)
            if ignore_outlier
            else self.prepare_times
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
            if self.prepare_times:
                logger.info(
                    "  Average prepare time: %.2f us", self.get_avg_prepare_time()
                )
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
        prepare_time: int | None = None,
    ):
        """Add a latency and token count measurement."""
        self.metrics[request_id].add_measurement(
            latency,
            token_count,
            host_time,
            device_time,
            ccl_time,
            prepare_time,
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
        prepare_time: int | None = None,
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
        self.prefill_metrics.add_measurement(
            latency,
            token_count,
            host_time,
            device_time,
            ccl_time,
            prepare_time,
        )
        if request_id:
            self.prefill_metrics_by_request_id.add_measurement(
                request_id,
                latency,
                token_count,
                host_time,
                device_time,
                ccl_time,
                prepare_time,
            )

    def record_decode(
        self,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
        prepare_time: int | None = None,
        padded_decode: bool = False,
        request_ids: list[str] | None = None,
    ):
        """Record decode step metrics."""
        if self.check_dummy_request(request_ids):
            return
        metrics = self.padded_decode_metrics if padded_decode else self.decode_metrics
        metrics.add_measurement(
            latency,
            token_count,
            host_time,
            device_time,
            ccl_time,
            prepare_time,
        )

    def print_final_stats(self):
        _attach_metrics_file_handler()
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


@dataclass
class StepReport:
    """One execution step's timing before it is recorded into a tracker.

    Lets model and sampler timings be summed into a single combined
    measurement (merged_with) instead of being tracked separately.
    """

    latency: float
    token_count: int = 0
    host_time: int | None = None
    device_time: int | None = None
    ccl_time: int | None = None
    prepare_time: int | None = None
    is_prefill: bool = False
    padded_decode: bool = False
    request_ids: list[str] | None = None

    @classmethod
    def from_reports(
        cls,
        start_time: float,
        end_time: float,
        reports: list[dict] | None,
        **meta,
    ) -> "StepReport":
        host_time = device_time = ccl_time = prepare_time = None
        if reports:
            host_time = reports[0].get("total_host", None)
            device_time = reports[0].get("total_device", None)
            ccl_time = reports[0].get("total_ccl", None)
        if reports and len(reports) > 1:
            prepare_time = reports[1].get("prepare_input_us", 0) + reports[1].get(
                "prepare_output_us", 0
            )
        return cls(
            latency=end_time - start_time,
            host_time=host_time,
            device_time=device_time,
            ccl_time=ccl_time,
            prepare_time=prepare_time,
            **meta,
        )

    def merged_with(self, other: "StepReport | None") -> "StepReport":
        """Sum `other`'s timings into this step, keeping this step's metadata."""
        if other is None:
            return self

        def _add(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return replace(
            self,
            latency=self.latency + other.latency,
            host_time=_add(self.host_time, other.host_time),
            device_time=_add(self.device_time, other.device_time),
            ccl_time=_add(self.ccl_time, other.ccl_time),
            prepare_time=_add(self.prepare_time, other.prepare_time),
        )

    def record(self, performance_tracker: PerformanceTracker) -> None:
        if self.is_prefill:
            performance_tracker.record_prefill(
                self.latency,
                self.token_count,
                host_time=self.host_time,
                device_time=self.device_time,
                ccl_time=self.ccl_time,
                prepare_time=self.prepare_time,
                request_ids=self.request_ids,
            )
        else:
            performance_tracker.record_decode(
                self.latency,
                self.token_count,
                host_time=self.host_time,
                device_time=self.device_time,
                ccl_time=self.ccl_time,
                prepare_time=self.prepare_time,
                padded_decode=self.padded_decode,
                request_ids=self.request_ids,
            )


def collect_metrics(
    performance_tracker: PerformanceTracker,
    is_prefill: bool,
    start_time: float,
    end_time: float,
    reports: list[dict],
    token_count: int,
) -> None:
    StepReport.from_reports(
        start_time,
        end_time,
        reports,
        token_count=token_count,
        is_prefill=is_prefill,
    ).record(performance_tracker)
