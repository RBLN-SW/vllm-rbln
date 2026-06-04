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

import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from vllm_rbln import envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)
T = TypeVar("T", int, float)


@dataclass
class Metrics:
    latencies: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    host_times: list[int] = field(default_factory=list)
    device_times: list[int] = field(default_factory=list)
    ccl_times: list[int] = field(default_factory=list)
    prepare_times: list[int] = field(default_factory=list)

    def record(
        self,
        latency: float,
        token_count: int,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
        prepare_time: int | None = None,
    ) -> None:
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

    @property
    def call_count(self) -> int:
        return len(self.latencies)

    def _drop_outlier(self, values: list[T]) -> list[T]:
        if len(values) <= 1:
            return values
        mean = sum(values) / len(values)
        worst = max(range(len(values)), key=lambda i: abs(values[i] - mean))
        return [v for i, v in enumerate(values) if i != worst]

    def _avg(self, values: list[T], drop_outlier: bool = True) -> float:
        if drop_outlier:
            values = self._drop_outlier(values)
        return sum(values) / len(values) if values else 0.0

    def avg_latency_ms(self, drop_outlier: bool = True) -> float:
        return self._avg(self.latencies, drop_outlier) * 1000

    def avg_throughput(self, drop_outlier: bool = True) -> float:
        lats = self._drop_outlier(self.latencies) if drop_outlier else self.latencies
        toks = (
            self._drop_outlier(self.token_counts) if drop_outlier else self.token_counts
        )
        t = sum(lats)
        return sum(toks) / t if t > 0 else 0.0

    def avg_host_time_us(self, drop_outlier: bool = True) -> float:
        return self._avg(self.host_times, drop_outlier)

    def avg_device_time_us(self, drop_outlier: bool = True) -> float:
        return self._avg(self.device_times, drop_outlier)

    def avg_ccl_time_us(self, drop_outlier: bool = True) -> float:
        return self._avg(self.ccl_times, drop_outlier)

    def avg_prepare_time_us(self, drop_outlier: bool = True) -> float:
        return self._avg(self.prepare_times, drop_outlier)


try:
    import rebel  # type: ignore

    _REBEL_HAS_CAPTURE = hasattr(rebel, "capture_reports")
except ImportError:
    _REBEL_HAS_CAPTURE = False


class _TimingSpan:
    __slots__ = ("_metrics", "_token_count", "_reports", "_start", "_capture_ctx")

    def __init__(self, metrics: Metrics, token_count: int) -> None:
        self._metrics = metrics
        self._token_count = token_count
        self._reports: list[dict] | None = None
        self._start = 0.0

    def __enter__(self) -> "_TimingSpan":
        # Create capture_ctx on each __enter__ call: contextmanager-based objects
        # are exhausted after __exit__ and cannot be reused.
        self._capture_ctx = (
            rebel.capture_reports() if _REBEL_HAS_CAPTURE else nullcontext()
        )
        self._reports = self._capture_ctx.__enter__()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Record time before closing capture_ctx to exclude any internal
        # synchronization overhead inside rebel from the measured latency.
        latency = time.perf_counter() - self._start
        self._capture_ctx.__exit__(*args)
        host, device, ccl, prepare = _parse_reports(self._reports)
        self._metrics.record(latency, self._token_count, host, device, ccl, prepare)
        return False


class _NoopSpan:
    __slots__ = ()

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *_):
        return False


class ProfileSection(Enum):
    MODEL = ("model", True)  # tracked separately per phase (prefill / decode)
    SAMPLER = ("sampler", False)  # phase-agnostic; recorded into a single bucket

    def __init__(self, label: str, split_phase: bool) -> None:
        self.label = label
        self.split_phase = split_phase


class _PerformanceContext:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._metrics: dict[tuple[ProfileSection, bool | None], Metrics] = defaultdict(
            Metrics
        )

    def profile(
        self,
        is_prefill: bool = False,
        section: ProfileSection = ProfileSection.MODEL,
        token_count: int = 0,
    ) -> _TimingSpan:
        phase = is_prefill if section.split_phase else None
        return _TimingSpan(self._metrics[(section, phase)], token_count)

    def print_stats(self) -> None:
        def _label(section: ProfileSection, phase: bool | None) -> str:
            if phase is None:
                return section.label.upper()
            return f"{section.label.upper()} {'PREFILL' if phase else 'DECODE'}"

        sections = {
            _label(s, p): m
            for (s, p), m in sorted(
                self._metrics.items(), key=lambda x: (x[0][0].value, not x[0][1])
            )
        }
        _report_metrics(self.name, sections)


class _NoopPerformanceContext:
    def __init__(self, name: str | None = None) -> None:
        pass

    def profile(self, *args, **kwargs) -> _NoopSpan:
        return _NoopSpan()

    def print_stats(self) -> None:
        pass


def _metrics_file_path() -> str | None:
    if not envs.VLLM_RBLN_METRICS_FILE:
        return None

    root, ext = os.path.splitext(envs.VLLM_RBLN_METRICS_FILE)
    return f"{root}.{os.getpid()}{ext}"


def _write_metrics_file(lines: list[str]) -> None:
    if (path := _metrics_file_path()) is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
    except OSError as e:
        logger.warning("Failed to write metrics to file %s: %s", path, e)


def _render_metrics(label: str, m: Metrics) -> list[str]:
    if m.call_count == 0:
        return [f"{label} METRICS: No data recorded"]

    lines = [
        f"{label} METRICS:",
        f"  Total call counts  : {m.call_count}",
        f"  Average latency    : {m.avg_latency_ms():.2f} ms",
    ]
    if sum(m.token_counts) > 0:
        lines.extend(
            [
                f"  Total tokens       : {sum(m.token_counts)}",
                f"  Avg throughput     : {m.avg_throughput():.2f} tok/s",
            ]
        )
    if m.host_times:
        lines.append(f"  Avg host time     : {m.avg_host_time_us():.2f} us")
    if m.device_times:
        lines.append(f"  Avg device time   : {m.avg_device_time_us():.2f} us")
    if m.ccl_times:
        lines.append(f"  Avg ccl time      : {m.avg_ccl_time_us():.2f} us")
    if m.prepare_times:
        lines.append(f"  Avg prepare time  : {m.avg_prepare_time_us():.2f} us")

    return lines


def _render_metrics_report(name: str | None, sections: dict[str, Metrics]) -> list[str]:
    lines = [
        "=" * 40,
        f"PERFORMANCE STATISTICS{f' [{name}]' if name else ''}",
        "=" * 40,
    ]

    for label, metrics in sections.items():
        lines.extend(_render_metrics(label, metrics))
        lines.append("-" * 40)

    lines.append("=" * 40)
    return lines


def _report_metrics(name: str | None, sections: dict[str, Metrics]) -> None:
    lines = _render_metrics_report(name, sections)

    for line in lines:
        logger.info("%s", line)

    _write_metrics_file(lines)


def _parse_reports(
    reports: list[dict] | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Extract timing information from rebel.capture_reports() output."""
    if not reports:
        return None, None, None, None
    host_time = reports[0].get("total_host")
    device_time = reports[0].get("total_device")
    ccl_time = reports[0].get("total_ccl")
    prepare_time = (
        reports[1].get("prepare_input_us", 0) + reports[1].get("prepare_output_us", 0)
        if len(reports) > 1
        else None
    )
    return host_time, device_time, ccl_time, prepare_time


# Resolved once at import time via VLLM_RBLN_METRICS env var.
# When disabled, _NoopPerformanceContext is assigned so every profile()
# call returns a zero-overhead no-op span.
PerformanceContext: type[_PerformanceContext | _NoopPerformanceContext] = (
    _PerformanceContext if envs.VLLM_RBLN_METRICS else _NoopPerformanceContext
)
