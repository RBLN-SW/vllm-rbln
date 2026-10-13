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
"""Wall-clock metrics for the vllm runner, installed only under VLLM_RBLN_METRICS.

Three ranges are timed with perf_counter: the pass (execute_model through
sample_tokens), the model call, and the sampler call with the bookkeeping over its
output. The last two are reported as one sum, so MODEL + SAMPLE carries the same call
count as E2E and the difference of the two means is the host overhead around the
graphs. The bookkeeping is in that sum because it holds the step's first blocking read
of the sampler output -- the cast that used to hold it inside _sample is gone, and
leaving the read untimed drops this section to dispatch cost while the E2E residual
absorbs the wait. Async defers that read out of the pass, so there it is dispatch.
A pass is recorded only once its phase and its graph time are both known, which is
what keeps those counts equal.

Spec decode moves that first blocking read again -- into the spec-only logits gather
in execute_model's postprocess block, which syncs on the target forward. The gather
is an inline block with no method to wrap, so timed_region folds the runner's
"postprocess" profiler region into the graph sum; it is empty without spec decode.
The drafter, which runs between _sample and _bookkeeping_sync, is deliberately not
folded: MODEL + SAMPLE stays the main-graph number, and the drafter's time -- under
EAGLE/MTP a draft-model forward, plus any sampler wait its first read absorbs --
shows up in the E2E residual.

Each recorded pass also gets a timeline: absolute CLOCK_MONOTONIC stamps at every
wrapped boundary (the DP padding exchange, the model call, the sampler, the
bookkeeping read, the hand-off and the runner's profiler regions), plus the pass's
thread CPU time and context switches. Durations alone cannot line ranks up or tell a
descheduled host thread from a device wait; the stamps can, because every rank on a
host reads the same clock. It goes to timeline_<rank>.json next to the metrics JSON,
whose shape is left untouched.

The whole feature lives in this module so the runner carries no metrics code at all. A
range that is not a whole method -- the model call sits mid-way through execute_model --
is reached by wrapping the callable the runner holds, and the phase, which is a local of
execute_model, is read where the runner computes it.
"""

import contextlib
import functools
import json
import os
import resource
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from vllm.sequence import IntermediateTensors
from vllm.v1.utils import record_function_or_nullcontext

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

logger = init_logger(__name__)

_CTX_ATTR = "_metrics_ctx"

# The timeline's clocks, bound here rather than read through ``time`` so the
# latency tests, which swap ``time`` for a fake perf_counter, leave them alone.
# perf_counter_ns is CLOCK_MONOTONIC on Linux: one clock for every process on
# the host, so the DP/EP ranks of one node line up without any offset.
_mono_ns = time.perf_counter_ns
_wall_ns = time.time_ns
_thread_cpu_ns = time.thread_time_ns


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
            # Per pass, in execution order: the aggregates above cannot tell one
            # slow pass from a drift or a bimodal split.
            "latencies_ms": [round(t * 1000.0, 3) for t in self.latencies],
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
        # One entry per recorded pass, in execution order; see _close_timeline.
        self._timeline: list[dict] = []
        self._marks: list[tuple[str, int]] | None = None
        self._pass_meta: dict | None = None

    @property
    def in_pass(self) -> bool:
        return self._start is not None

    def start_pass(self) -> None:
        self._start = time.perf_counter()
        self._phase = None
        self._graph_latency = None
        self._open_timeline()

    def end_pass(self) -> None:
        end = time.perf_counter()
        start, self._start = self._start, None
        phase, self._phase = self._phase, None
        graph, self._graph_latency = self._graph_latency, None
        if start is None or phase is None or graph is None:
            # No pass was open, or it never reached the forward: an empty batch, KV
            # send/recv with no forward work, or a pass that raised.
            self._close_timeline(None)
            return
        self._e2e[phase].record(end - start)
        self._graph[phase].record(graph)
        self._close_timeline(phase)

    def mark_phase(self, phase: _Phase) -> None:
        if self._start is not None:
            self._phase = phase

    def add_graph_time(self, latency: float) -> None:
        if self._start is not None:
            self._graph_latency = (self._graph_latency or 0.0) + latency

    def mark(self, name: str) -> None:
        """Stamp ``name`` on the open pass's timeline; a no-op outside a pass."""
        if self._marks is not None:
            self._marks.append((name, _mono_ns()))

    def _open_timeline(self) -> None:
        self._pass_meta = {
            "wall_start_ns": _wall_ns(),
            "thread_cpu_ns": _thread_cpu_ns(),
            "rusage": _thread_rusage(),
            "tid": threading.get_native_id(),
        }
        self._marks = [("pass_start", _mono_ns())]

    def _close_timeline(self, phase: _Phase | None) -> None:
        marks, self._marks = self._marks, None
        meta, self._pass_meta = self._pass_meta, None
        if phase is None or marks is None or meta is None:
            # Dropped exactly when the latencies drop the pass, so entry i of a
            # phase's timeline is entry i of that phase's latencies_ms.
            return
        marks.append(("pass_end", _mono_ns()))
        tid = threading.get_native_id()
        entry: dict = {
            "phase": phase.value,
            "index": self._e2e[phase].call_count - 1,
            "wall_start_ns": meta["wall_start_ns"],
            "marks": [[name, t] for name, t in marks],
            # Thread CPU time well below the pass's wall means the thread was
            # off-CPU: blocked on the device, or descheduled.
            "thread_cpu_ns": _thread_cpu_ns() - meta["thread_cpu_ns"],
            # A pass that opens and closes on different threads makes the
            # per-thread counters meaningless; the tids show when that happened.
            "tid": meta["tid"] if tid == meta["tid"] else [meta["tid"], tid],
        }
        before, after = meta["rusage"], _thread_rusage()
        if before is not None and after is not None:
            entry["nvcsw"] = after[0] - before[0]
            entry["nivcsw"] = after[1] - before[1]
        self._timeline.append(entry)

    def print_stats(self) -> None:
        sections: dict[str, _Metrics] = {}
        for label, table in (("MODEL + SAMPLE", self._graph), ("E2E", self._e2e)):
            for phase in _Phase:
                if phase in table:
                    sections[f"{label} ({phase.value})"] = table[phase]
        name = f"{self.name} | {self.rank_tag}" if self.rank_tag else self.name
        logger.info("%s", "\n".join(_render_metrics_report(name, sections)))
        _write_metrics_json(self.name, self.rank_tag, sections)
        _write_timeline_json(self.name, self.rank_tag, self._timeline)


def _thread_rusage() -> tuple[int, int] | None:
    """(voluntary, involuntary) context switches of the calling thread so far."""
    try:
        ru = resource.getrusage(resource.RUSAGE_THREAD)
    except (AttributeError, OSError, ValueError):
        # RUSAGE_THREAD is Linux-only.
        return None
    return ru.ru_nvcsw, ru.ru_nivcsw


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


# Describes the timeline file for whoever opens it without this module at hand.
_TIMELINE_SCHEMA = {
    "clock": "time.perf_counter_ns (CLOCK_MONOTONIC): absolute, shared by every "
    "process on the host, so ranks of one node compare directly",
    "passes": "recorded passes in execution order; (phase, index) is the position "
    "in that phase's latencies_ms of the metrics JSON",
    "marks": "[name, t_ns] in the order stamped; pass_start / pass_end bound the "
    "E2E latency, <region>.enter / .exit are the runner's profiler regions",
    "wall_start_ns": "time.time_ns at pass_start, to match node logs",
    "thread_cpu_ns": "CPU time of the pass's thread over the pass",
    "nvcsw / nivcsw": "voluntary / involuntary context switches of that thread "
    "over the pass (Linux only)",
}


def _write_timeline_json(name: str | None, rank_tag: str, timeline: list[dict]) -> None:
    """Write the per-pass timeline next to the metrics JSON, as its own file.

    Kept out of the metrics JSON so that file's shape stays what its readers
    expect; compact, since it is a few dozen marks per pass.
    """
    if not envs.VLLM_RBLN_METRICS_DIR or not timeline:
        return

    suffix = rank_tag.replace(" ", "_").lower()
    filename = f"timeline_{suffix}.json" if suffix else "timeline.json"
    path = os.path.join(envs.VLLM_RBLN_METRICS_DIR, filename)
    payload = {
        "name": name,
        "rank": rank_tag,
        "schema": _TIMELINE_SCHEMA,
        "passes": timeline,
    }

    try:
        payload_str = json.dumps(payload, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        logger.warning("Failed to serialize timeline JSON: %s", e)
        return

    try:
        os.makedirs(envs.VLLM_RBLN_METRICS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload_str)
    except OSError as e:
        logger.warning("Failed to write timeline JSON to %s: %s", path, e)


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
_bookkeeping_sync = RBLNModelRunner._bookkeeping_sync
_shutdown = RBLNWorker.shutdown


@functools.wraps(_execute_model)
def execute_model(self, *args, **kwargs):
    global _ACTIVE_CTX
    ctx = _ctx(self)
    _ACTIVE_CTX = ctx
    ctx.start_pass()
    output = _execute_model(self, *args, **kwargs)
    ctx.mark("execute_model.return")
    # A None pass is closed by sample_tokens, an IntermediateTensors pass by
    # send_handoff below; anything else nobody would end.
    if output is not None and not isinstance(output, IntermediateTensors):
        ctx.end_pass()
    return output


@functools.wraps(_sample_tokens)
def sample_tokens(self, *args, **kwargs):
    ctx = _ctx(self)
    # The gap from execute_model.return is the engine's, between the two calls.
    ctx.mark("sample_tokens.enter")
    output = _sample_tokens(self, *args, **kwargs)
    ctx.end_pass()
    return output


@functools.wraps(_bookkeeping_sync)
def bookkeeping_sync(self, *args, **kwargs):
    ctx = _ctx(self)
    ctx.mark("bookkeep.enter")
    start = time.perf_counter()
    try:
        return _bookkeeping_sync(self, *args, **kwargs)
    finally:
        ctx.add_graph_time(time.perf_counter() - start)
        ctx.mark("bookkeep.exit")


@functools.wraps(_sample)
def sample(self, *args, **kwargs):
    if self.is_intermediate_chunked_prefill:
        # Returns a placeholder without running the sampler, so there is no sampler
        # time to attribute -- only the cost of building that placeholder.
        return _sample(self, *args, **kwargs)
    ctx = _ctx(self)
    ctx.mark("sample.enter")
    start = time.perf_counter()
    output = _sample(self, *args, **kwargs)
    ctx.add_graph_time(time.perf_counter() - start)
    ctx.mark("sample.exit")
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
        ctx.mark("model_executable.enter")
        start = time.perf_counter()
        output = executable(*call_args, **call_kwargs)
        ctx.add_graph_time(time.perf_counter() - start)
        # Dispatch returns before the device finishes; the wait lands in the
        # first blocking read after it (bookkeep, or the spec postprocess).
        ctx.mark("model_executable.exit")
        return output

    self.model_executable = timed_executable


@functools.wraps(_determine_batch_execution_and_padding)
def determine_batch_execution_and_padding(self, *args, **kwargs):
    # Holds coordinate_batch_across_dp, the host-side DP exchange: a rank that
    # arrives late makes the others wait here, outside the graph sum.
    ctx = _ctx(self)
    ctx.mark("dp_padding.enter")
    result = _determine_batch_execution_and_padding(self, *args, **kwargs)
    ctx.mark("dp_padding.exit")
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
    ctx.mark_phase(phase)
    return result


_send_handoff = RBLNWorker._send_handoff


@functools.wraps(_send_handoff)
def send_handoff(self, *args, **kwargs):
    # The send waits on this stage's compute, so its wall carries the stage's time.
    # No try/finally: a raising hand-off leaves the pass open, like any raising pass.
    ctx = _ctx(self.model_runner)
    ctx.mark("handoff.enter")
    start = time.perf_counter()
    output = _send_handoff(self, *args, **kwargs)
    ctx.add_graph_time(time.perf_counter() - start)
    ctx.mark("handoff.exit")
    ctx.end_pass()
    return output


@functools.wraps(_shutdown)
def shutdown(self):
    # The runner is built in init_device(), not __init__, so a worker that failed
    # before that has none and must still tear down.
    runner = getattr(self, "model_runner", None)
    ctx = getattr(runner, _CTX_ATTR, None)
    if ctx is not None:
        ctx.print_stats()
    _shutdown(self)


_ACTIVE_CTX: _PerformanceContext | None = None

# The region that holds the spec-only logits gather (see the module docstring); it
# belongs to the graph sum, and is empty without spec decode.
_GRAPH_REGION = "rbln_model_runner: postprocess"


_REGION_PREFIX = "rbln_model_runner: "


@contextlib.contextmanager
def timed_region(name: str):
    ctx = _ACTIVE_CTX
    if ctx is None or not ctx.in_pass:
        with record_function_or_nullcontext(name):
            yield
        return
    # Every region is stamped on the timeline; only the graph region is folded
    # into the graph sum.
    label = name.removeprefix(_REGION_PREFIX)
    ctx.mark(f"{label}.enter")
    start = time.perf_counter()
    try:
        with record_function_or_nullcontext(name):
            yield
    finally:
        if name == _GRAPH_REGION:
            ctx.add_graph_time(time.perf_counter() - start)
        ctx.mark(f"{label}.exit")


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
            f"{_RUNNER}._bookkeeping_sync",
            bookkeeping_sync,
            "Holds the step's first blocking read of the sampler output, which the "
            "graph slice would otherwise miss.",
        ),
        (
            f"{_RUNNER}._determine_batch_execution_and_padding",
            determine_batch_execution_and_padding,
            "Reads the pass phase where it is decided. The padded token dimension is a "
            "local of execute_model and is not observable from any wrapped callable.",
        ),
        (
            f"{_WORKER}._send_handoff",
            send_handoff,
            "Times the hand-off and closes its pass; left outside the pass, "
            "the send hides a non-last rank's stage time.",
        ),
        (
            f"{_WORKER}.shutdown",
            shutdown,
            "Reports the collected metrics before teardown; the worker holds the only "
            "shutdown hook that still has the runner.",
        ),
        (
            "vllm_rbln.v1.worker.rbln_model_runner.record_function_or_nullcontext",
            timed_region,
            "Spec decode syncs on the forward in the spec-only logits gather, "
            "an inline block of execute_model with no method to wrap.",
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
