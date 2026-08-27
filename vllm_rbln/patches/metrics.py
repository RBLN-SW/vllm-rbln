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
"""Wall-clock metrics for the native runner, installed only under VLLM_RBLN_METRICS.

Three ranges are timed with perf_counter: the pass (execute_model through
sample_tokens), the model call, and the sampler call. The last two are reported as one
sum, so MODEL + SAMPLE carries the same call count as E2E and the difference of the two
means is the host overhead around the graphs. A pass is recorded only once its phase
and its graph time are both known, which is what keeps those counts equal.

The whole feature lives in this module so the runner carries no metrics code at all. A
range that is not a whole method -- the model call sits mid-way through execute_model --
is reached by wrapping the callable the runner holds, and the phase, which is a local of
execute_model, is read where the runner computes it.
"""

import functools
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

logger = init_logger(__name__)

_CTX_ATTR = "_metrics_ctx"


class _Phase(Enum):
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


class _PerformanceContext:
    """One runner's measurements, accumulated a pass at a time."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.rank_tag = _rank_tag()
        self._graph: dict[_Phase, _Metrics] = defaultdict(_Metrics)
        self._e2e: dict[_Phase, _Metrics] = defaultdict(_Metrics)
        self._start: float | None = None
        self._phase: _Phase | None = None
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

    def mark_phase(self, phase: _Phase) -> None:
        if self._start is not None:
            self._phase = phase

    def add_graph_time(self, latency: float) -> None:
        if self._start is not None:
            self._graph_latency = (self._graph_latency or 0.0) + latency

    def print_stats(self) -> None:
        sections: dict[str, _Metrics] = {}
        for label, table in (("MODEL + SAMPLE", self._graph), ("E2E", self._e2e)):
            for phase in _Phase:
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
        except Exception:
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


def _metrics_enabled() -> bool:
    return envs.VLLM_RBLN_METRICS


def _ctx(runner: RBLNModelRunner) -> _PerformanceContext:
    """The runner's context, built on first use.

    Deferred because the rank tag reads the parallelism groups, which are not
    initialised while the runner is still being constructed.
    """
    ctx = getattr(runner, _CTX_ATTR, None)
    if ctx is None:
        ctx = _PerformanceContext("runner")
        setattr(runner, _CTX_ATTR, ctx)
    return ctx


_execute_model = RBLNModelRunner.execute_model
_sample_tokens = RBLNModelRunner.sample_tokens
_sample = RBLNModelRunner._sample
_load_model = RBLNModelRunner.load_model
_determine_batch_execution_and_padding = (
    RBLNModelRunner._determine_batch_execution_and_padding
)
_shutdown = RBLNWorker.shutdown


@functools.wraps(_execute_model)
def execute_model(self, *args, **kwargs):
    ctx = _ctx(self)
    ctx.start_pass()
    output = _execute_model(self, *args, **kwargs)
    # The engine calls sample_tokens() only if execute_model() returned None, so a
    # non-None return means nobody else will end this pass.
    if output is not None:
        ctx.end_pass()
    return output


@functools.wraps(_sample_tokens)
def sample_tokens(self, *args, **kwargs):
    output = _sample_tokens(self, *args, **kwargs)
    _ctx(self).end_pass()
    return output


@functools.wraps(_sample)
def sample(self, *args, **kwargs):
    if self.is_intermediate_chunked_prefill:
        # Returns a placeholder without running the sampler, so there is no sampler
        # time to attribute -- only the cost of building that placeholder.
        return _sample(self, *args, **kwargs)
    start = time.perf_counter()
    output = _sample(self, *args, **kwargs)
    _ctx(self).add_graph_time(time.perf_counter() - start)
    return output


@functools.wraps(_load_model)
def load_model(self, *args, **kwargs):
    _load_model(self, *args, **kwargs)
    executable = self.model_executable

    def timed_executable(*call_args, **call_kwargs):
        ctx = _ctx(self)
        if not ctx.in_pass:
            # Warm-up and dummy runs: skip the clock reads as well as the record.
            return executable(*call_args, **call_kwargs)
        start = time.perf_counter()
        output = executable(*call_args, **call_kwargs)
        ctx.add_graph_time(time.perf_counter() - start)
        return output

    self.model_executable = timed_executable


@functools.wraps(_determine_batch_execution_and_padding)
def determine_batch_execution_and_padding(self, *args, **kwargs):
    result = _determine_batch_execution_and_padding(self, *args, **kwargs)
    batch_desc, _route, _num_tokens_across_dp = result
    if batch_desc is None:
        # A drained group runs nothing, so no pass opens and there is no phase to
        # attribute this step to.
        return result
    num_tokens_padded = batch_desc.num_tokens_padded
    if self.is_prefill:
        phase = _Phase.PREFILL
    elif num_tokens_padded == self.max_num_tokens:
        # num_tokens_padded is the MoE pad dimension, so matching the prefill dim means
        # this decode ran the prefill-sized graph. Only reachable under DP, where the
        # padding is computed at all. Changing how the runner pads means revisiting
        # this line.
        phase = _Phase.PADDED_DECODE
    else:
        phase = _Phase.DECODE
    # A no-op on the dummy run, which reaches this with no pass open.
    _ctx(self).mark_phase(phase)
    return result


@functools.wraps(_shutdown)
def shutdown(self):
    # The runner is built in init_device(), not __init__, so a worker that failed
    # before that has none and must still tear down.
    runner = getattr(self, "model_runner", None)
    ctx = getattr(runner, _CTX_ATTR, None)
    if ctx is not None:
        ctx.print_stats()
    _shutdown(self)


_RUNNER = "vllm_rbln.v1.worker.rbln_model_runner.RBLNModelRunner"
_WORKER = "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"


def _register_patches() -> None:
    for target, replacement, reason in (
        (
            f"{_RUNNER}.execute_model",
            execute_model,
            "Opens the measured pass. The runner cannot mark its own boundary without "
            "carrying metrics code on the serving path.",
        ),
        (
            f"{_RUNNER}.sample_tokens",
            sample_tokens,
            "Closes the measured pass, which spans two inbound calls and so has no "
            "single method to wrap.",
        ),
        (
            f"{_RUNNER}._sample",
            sample,
            "Times the sampler. The sampler call is not a method of its own, so the "
            "whole _sample is the narrowest wrappable range.",
        ),
        (
            f"{_RUNNER}.load_model",
            load_model,
            "Wraps the compiled model_executable once it exists; it is an instance "
            "attribute, so it cannot be replaced through the registry.",
        ),
        (
            f"{_RUNNER}._determine_batch_execution_and_padding",
            determine_batch_execution_and_padding,
            "Reads the pass phase where it is decided. The padded token dimension is a "
            "local of execute_model and is not observable from any wrapped callable.",
        ),
        (
            f"{_WORKER}.shutdown",
            shutdown,
            "Reports the collected metrics before teardown; the worker holds the only "
            "shutdown hook that still has the runner.",
        ),
    ):
        register_patch(
            target=target,
            reason=reason,
            key=f"vllm_rbln.patches.metrics.{target.rsplit('.', 1)[-1]}",
            owner_module="vllm_rbln.patches.metrics",
            condition=_metrics_enabled,
        )(replacement)


_register_patches()
