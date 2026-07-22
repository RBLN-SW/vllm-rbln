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

import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

from vllm_rbln import envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)
T = TypeVar("T", int, float)


@dataclass
class Metrics:
    latencies: list[float] = field(default_factory=list)
    host_times: list[int] = field(default_factory=list)
    device_times: list[int] = field(default_factory=list)
    ccl_times: list[int] = field(default_factory=list)
    prepare_times: list[int] = field(default_factory=list)

    def record(
        self,
        latency: float,
        host_time: int | None = None,
        device_time: int | None = None,
        ccl_time: int | None = None,
        prepare_time: int | None = None,
    ) -> None:
        self.latencies.append(latency)
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

    def mean_latency_ms(self) -> float:
        return _mean(self.latencies) * 1000

    def latency_percentiles_ms(
        self, percentiles: tuple[float, ...] = (50.0, 90.0, 99.0)
    ) -> dict[str, float]:
        if not self.latencies:
            return {}
        arr = np.asarray(self.latencies, dtype=float) * 1000.0
        return {f"p{p:g}": float(np.percentile(arr, p)) for p in percentiles}

    def to_dict(self) -> dict:
        stats: dict = {
            "call_count": self.call_count,
            "mean_latency_ms": self.mean_latency_ms(),
            "latency_percentiles_ms": self.latency_percentiles_ms(),
        }
        timings = (
            ("mean_host_time_us", self.host_times),
            ("mean_device_time_us", self.device_times),
            ("mean_ccl_time_us", self.ccl_times),
            ("mean_prepare_time_us", self.prepare_times),
        )
        for key, values in timings:
            if values:
                stats[key] = _mean(values)
        return stats


@dataclass
class _Sample:
    """One span's timing, summable across model+sampler within a step."""

    latency: float
    host: int | None = None
    device: int | None = None
    ccl: int | None = None
    prepare: int | None = None
    phase: bool | None = None  # True=prefill, False=decode (set by the model span)

    def merged(self, other: "_Sample") -> "_Sample":
        def _add(a: int | None, b: int | None) -> int | None:
            return None if a is None and b is None else (a or 0) + (b or 0)

        return _Sample(
            latency=self.latency + other.latency,
            host=_add(self.host, other.host),
            device=_add(self.device, other.device),
            ccl=_add(self.ccl, other.ccl),
            prepare=_add(self.prepare, other.prepare),
            phase=self.phase,  # keep the model step's phase
        )


try:
    import rebel  # type: ignore

    _REBEL_HAS_CAPTURE = hasattr(rebel, "capture_reports")
except ImportError:
    _REBEL_HAS_CAPTURE = False


class _TimingSpan:
    __slots__ = ("_ctx", "_phase", "_is_sampler", "_reports", "_start", "_capture_ctx")

    def __init__(
        self, ctx: "_PerformanceContext", phase: bool | None, is_sampler: bool
    ) -> None:
        self._ctx = ctx
        self._phase = phase
        self._is_sampler = is_sampler
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
        end = time.perf_counter()
        latency = end - self._start
        self._capture_ctx.__exit__(*args)
        host, device, ccl, prepare = _parse_reports(self._reports)
        self._ctx._submit(
            _Sample(latency, host, device, ccl, prepare, self._phase),
            self._is_sampler,
            self._start,
            end,
        )
        return False


class _NoopSpan:
    __slots__ = ()

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *_):
        return False


class _PerformanceContext:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.rank_tag = _rank_tag()
        self._metrics: dict[bool, Metrics] = defaultdict(Metrics)
        self._pending: _Sample | None = None
        self._e2e_start: float | None = None
        self._e2e: dict[bool, Metrics] = defaultdict(Metrics)

    def profile_model(self, is_prefill: bool) -> _TimingSpan:
        # A prior model step with no sampler (intermediate chunked prefill,
        # non-last PP rank) leaves a pending report; record it model-only here.
        self._flush_pending()
        return _TimingSpan(self, phase=is_prefill, is_sampler=False)

    def profile_sampler(self) -> _TimingSpan:
        return _TimingSpan(self, phase=None, is_sampler=True)

    def _submit(
        self, sample: _Sample, is_sampler: bool, start: float, end: float
    ) -> None:
        if not is_sampler:
            self._pending = sample  # model: stash, wait for the sampler
            self._e2e_start = start
            return
        if self._pending is None:
            return  # sampler with no preceding model step; ignore
        self._record(self._pending.merged(sample))
        if self._e2e_start is not None:
            self._e2e[bool(self._pending.phase)].record(end - self._e2e_start)
        self._pending = None
        self._e2e_start = None

    def _flush_pending(self) -> None:
        if self._pending is not None:
            self._record(self._pending)
            self._pending = None
        self._e2e_start = None

    def _record(self, s: _Sample) -> None:
        self._metrics[bool(s.phase)].record(
            s.latency, s.host, s.device, s.ccl, s.prepare
        )

    def print_stats(self) -> None:
        self._flush_pending()  # record the final step if it had no sampler
        sections: dict[str, Metrics] = {}
        for phase, m in sorted(self._metrics.items(), key=lambda x: not x[0]):
            sections["PREFILL + SAMPLE" if phase else "DECODE + SAMPLE"] = m
        for phase, m in sorted(self._e2e.items(), key=lambda x: not x[0]):
            sections["PREFILL E2E" if phase else "DECODE E2E"] = m
        name = f"{self.name} | {self.rank_tag}" if self.rank_tag else self.name
        _report_metrics(name, sections)
        _write_metrics_json(self.name, self.rank_tag, sections)


class _NoopPerformanceContext:
    def __init__(self, name: str | None = None) -> None:
        pass

    def profile_model(self, *args, **kwargs) -> _NoopSpan:
        return _NoopSpan()

    def profile_sampler(self, *args, **kwargs) -> _NoopSpan:
        return _NoopSpan()

    def print_stats(self) -> None:
        pass


def _rank_tag() -> str:
    """Rank tag including only parallelism axes with degree > 1; '' if none."""
    parts = []
    try:
        from vllm.distributed import get_pp_group, get_tp_group

        tp = get_tp_group()
        if tp.world_size > 1:
            parts.append(f"TP{tp.rank_in_group}")
        pp = get_pp_group()
        if pp.world_size > 1:
            parts.append(f"PP{pp.rank_in_group}")
    except Exception:
        return ""
    try:
        from vllm.distributed import get_dp_group

        dp = get_dp_group()
        if dp.world_size > 1:
            parts.append(f"DP{dp.rank_in_group}")
    except Exception:
        pass  # DP group may not be initialized
    return " ".join(parts)


def _write_metrics_json(
    name: str | None, rank_tag: str, sections: dict[str, Metrics]
) -> None:
    if not envs.VLLM_RBLN_METRICS_DIR:
        return

    suffix = rank_tag.replace(" ", "_").lower()
    filename = f"metrics_{suffix}.json" if suffix else "metrics.json"
    path = os.path.join(envs.VLLM_RBLN_METRICS_DIR, filename)
    payload = {
        "name": name,
        "rank": rank_tag,
        "sections": {label: m.to_dict() for label, m in sections.items()},
    }

    try:
        os.makedirs(envs.VLLM_RBLN_METRICS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        logger.warning("Failed to write metrics JSON to %s: %s", path, e)


def _mean(values: list[T]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_metrics(label: str, m: Metrics) -> list[str]:
    if m.call_count == 0:
        return [f"{label} METRICS: No data recorded"]

    lines = [
        f"{label} METRICS:",
        f"  {'Total call counts':<25}: {m.call_count}",
        f"  {'Mean latency (ms)':<25}: {m.mean_latency_ms():.2f}",
    ]

    pct = m.latency_percentiles_ms()
    for key, value in pct.items():
        stat = "Median" if key == "p50" else key.upper()
        metric_label = f"{stat} latency (ms)"
        lines.append(f"  {metric_label:<25}: {value:.2f}")

    runtime_timings = [
        ("Mean host time (us)", m.host_times),
        ("Mean device time (us)", m.device_times),
        ("Mean ccl time (us)", m.ccl_times),
        ("Mean prepare time (us)", m.prepare_times),
    ]
    for runtime_label, values in runtime_timings:
        if values:
            lines.append(f"  {runtime_label:<25}: {_mean(values):.2f}")

    return lines


def _render_metrics_report(name: str | None, sections: dict[str, Metrics]) -> list[str]:
    lines = [
        "=" * 50,
        f"PERFORMANCE STATISTICS{f' [{name}]' if name else ''}",
        "=" * 50,
    ]

    for label, metrics in sections.items():
        lines.extend(_render_metrics(label, metrics))
        lines.append("-" * 50)

    lines.append("=" * 50)
    return lines


def _report_metrics(name: str | None, sections: dict[str, Metrics]) -> None:
    lines = _render_metrics_report(name, sections)
    logger.info("%s", "\n".join(lines))


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
