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
"""Shared wall-clock metrics for model execution passes."""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from vllm_rbln import envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

_CTX_ATTR = "_metrics_ctx"


def metrics_enabled() -> bool:
    return envs.VLLM_RBLN_METRICS


def get_performance_context(runner: object) -> "PerformanceContext":
    """The runner's context, built on first use.

    Deferred because the rank tag reads the parallelism groups, which are not
    initialised while the runner is still being constructed.
    """
    ctx = getattr(runner, _CTX_ATTR, None)
    if ctx is None:
        ctx = PerformanceContext("runner")
        setattr(runner, _CTX_ATTR, ctx)
    return ctx


class Phase(Enum):
    """Report phases, in report order; the value is the section label."""

    PREFILL = "PREFILL"
    DECODE = "DECODE"
    PADDED_DECODE = "PADDED DECODE"


@dataclass
class _Metrics:
    latencies: list[float] = field(default_factory=list)

    def record(self, latency: float) -> None:
        self.latencies.append(latency)

    @property
    def call_count(self) -> int:
        return len(self.latencies)

    def mean_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies) * 1000

    def latency_percentiles_ms(
        self, percentiles: tuple[float, ...] = (50.0, 90.0, 99.0)
    ) -> dict[str, float]:
        if not self.latencies:
            return {}
        arr = np.asarray(self.latencies, dtype=float) * 1000.0
        return {f"p{p:g}": float(np.percentile(arr, p)) for p in percentiles}

    def to_dict(self) -> dict:
        return {
            "call_count": self.call_count,
            "mean_latency_ms": self.mean_latency_ms(),
            "latency_percentiles_ms": self.latency_percentiles_ms(),
        }


class PerformanceContext:
    """One runner's measurements, accumulated a pass at a time."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.rank_tag = _rank_tag()
        self._graph: dict[Phase, _Metrics] = defaultdict(_Metrics)
        self._e2e: dict[Phase, _Metrics] = defaultdict(_Metrics)
        self._start: float | None = None
        self._phase: Phase | None = None
        self._graph_latency: float | None = None

    @property
    def in_pass(self) -> bool:
        return self._start is not None

    def start_pass(self) -> None:
        self._start = time.perf_counter()
        self._phase = None
        self._graph_latency = None

    def end_pass(self) -> None:
        end = time.perf_counter()
        start, self._start = self._start, None
        phase, self._phase = self._phase, None
        graph, self._graph_latency = self._graph_latency, None
        if start is None or phase is None or graph is None:
            # No pass was open, or it never reached the forward: an empty batch, KV
            # send/recv with no forward work, or a pass that raised.
            return
        self._e2e[phase].record(end - start)
        self._graph[phase].record(graph)

    def mark_phase(self, phase: Phase) -> None:
        if self._start is not None:
            self._phase = phase

    def add_graph_time(self, latency: float) -> None:
        if self._start is not None:
            self._graph_latency = (self._graph_latency or 0.0) + latency

    def print_stats(self) -> None:
        sections: dict[str, _Metrics] = {}
        for label, table in (("MODEL + SAMPLE", self._graph), ("E2E", self._e2e)):
            for phase in Phase:
                if phase in table:
                    sections[f"{label} ({phase.value})"] = table[phase]
        name = f"{self.name} | {self.rank_tag}" if self.rank_tag else self.name
        logger.info("%s", "\n".join(_render_metrics_report(name, sections)))
        _write_metrics_json(self.name, self.rank_tag, sections)


def _rank_tag() -> str:
    """Rank tag including only parallelism axes with degree > 1; '' if none."""

    def get_rank_info(group_name: str, get_group_func) -> str:
        try:
            group = get_group_func()
            if group.world_size > 1:
                return f"{group_name}{group.rank_in_group}"
        except (AssertionError, RuntimeError):
            return ""
        return ""

    parts = []

    from vllm.distributed import get_dp_group, get_ep_group, get_pp_group, get_tp_group

    for group_name, get_group_func in [
        ("TP", get_tp_group),
        ("PP", get_pp_group),
        ("DP", get_dp_group),
        ("EP", get_ep_group),
    ]:
        rank_info = get_rank_info(group_name, get_group_func)
        if rank_info:
            parts.append(rank_info)

    return " ".join(parts)


def _render_metrics(label: str, m: _Metrics) -> list[str]:
    if m.call_count == 0:
        return [f"{label} METRICS: No data recorded"]

    lines = [
        f"{label} METRICS:",
        f"  {'Total call counts':<25}: {m.call_count}",
        f"  {'Mean latency (ms)':<25}: {m.mean_latency_ms():.2f}",
    ]

    for key, value in m.latency_percentiles_ms().items():
        stat = "Median" if key == "p50" else key.upper()
        metric_label = f"{stat} latency (ms)"
        lines.append(f"  {metric_label:<25}: {value:.2f}")

    return lines


def _render_metrics_report(
    name: str | None, sections: dict[str, _Metrics]
) -> list[str]:
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


def _write_metrics_json(
    name: str | None, rank_tag: str, sections: dict[str, _Metrics]
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
        payload_str = json.dumps(payload, indent=2)
    except (TypeError, ValueError) as e:
        logger.warning("Failed to serialize metrics JSON: %s", e)
        return

    try:
        os.makedirs(envs.VLLM_RBLN_METRICS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload_str)
    except OSError as e:
        logger.warning("Failed to write metrics JSON to %s: %s", path, e)
