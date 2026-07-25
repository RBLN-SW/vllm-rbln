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

import json
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

    def record(self, report: "StepReport") -> None:
        if report.is_prefill:
            self.record_prefill(
                report.latency,
                report.token_count,
                host_time=report.host_time,
                device_time=report.device_time,
                ccl_time=report.ccl_time,
                prepare_time=report.prepare_time,
                request_ids=report.request_ids,
            )
        else:
            self.record_decode(
                report.latency,
                report.token_count,
                host_time=report.host_time,
                device_time=report.device_time,
                ccl_time=report.ccl_time,
                prepare_time=report.prepare_time,
                padded_decode=report.padded_decode,
                request_ids=report.request_ids,
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


# ITL breakdown phases, in chronological report order. "others" is not measured
# directly; it is the remainder (total step latency minus the sum of the
# measured phases) so the breakdown always sums back to the observed ITL.
ITL_MEASURED_PHASES = (
    "prepare_input",
    "model_forward",
    "postprocess",
    "sampler",
    "draft",
    "update_state",
)
ITL_PHASES = (*ITL_MEASURED_PHASES, "others")

# Step kinds. In DP+EP a rank steps in lockstep with its peers, so a local
# decode that coincides with any peer still in prefill (decode_mixed) is padded
# up to the prefill cadence and has a very different ITL from a step where every
# rank decodes (decode_all, the steady state). Prefill is keyed by this rank's
# own phase. For dp_size==1 every decode falls in decode_all.
ITL_KIND_PREFILL = "prefill"
ITL_KIND_DECODE_ALL = "decode_all"
ITL_KIND_DECODE_MIXED = "decode_mixed"
ITL_STEP_KINDS = (ITL_KIND_PREFILL, ITL_KIND_DECODE_ALL, ITL_KIND_DECODE_MIXED)


class ITLBreakdownTracker:
    """Breaks inter-token latency (ITL) down into execution phases.

    One scheduler step spans ``execute_model()`` + ``sample_tokens()``. Phase
    durations are accumulated into ``_current`` across both calls (via the
    runner's ``_itl_phase`` context manager) and committed to a per-step-kind
    aggregate by ``commit()`` at the end of the step. ``commit()`` derives the
    ``others`` phase as ``total - sum(measured)`` so no time is silently dropped,
    classifies the step (prefill / decode_all / decode_mixed), and accumulates
    the per-DP-rank token counts. A step that aborts before ``commit()`` (empty
    batch, chunked-prefill intermediate, PP mid-stage) is discarded by the next
    ``reset_step()``.
    """

    def __init__(self, name: str | None = None):
        self.name = name
        # step kind -> phase -> list of per-step latencies in milliseconds.
        self.latencies: dict[str, dict[str, list[float]]] = {
            kind: defaultdict(list) for kind in ITL_STEP_KINDS
        }
        # step kind -> per-DP-rank cumulative token totals (len == dp_size).
        self.token_totals: dict[str, list[int]] = {}
        # step kind -> number of steps contributing to token_totals.
        self.token_steps: dict[str, int] = defaultdict(int)
        # step kind -> per-step context length (mean / max seq_len over the
        # active batch). Recorded raw (one value per step, index-aligned with the
        # per-phase latency samples) so ITL can be plotted against context length
        # rather than blindly averaged over an uncontrolled length distribution.
        self.ctxlen_mean: dict[str, list[float]] = {k: [] for k in ITL_STEP_KINDS}
        self.ctxlen_max: dict[str, list[int]] = {k: [] for k in ITL_STEP_KINDS}
        # phase -> accumulated seconds for the in-flight step.
        self._current: dict[str, float] = defaultdict(float)

    def reset_step(self) -> None:
        """Drop any partially-measured step and start a fresh one."""
        self._current.clear()

    def add_phase(self, name: str, duration_s: float) -> None:
        """Accumulate ``duration_s`` seconds into phase ``name`` for this step.

        Additive so a phase entered more than once per step (or split across the
        two calls) sums into a single figure.
        """
        self._current[name] += duration_s

    @staticmethod
    def classify(is_prefill: bool, all_decode: bool) -> str:
        if is_prefill:
            return ITL_KIND_PREFILL
        return ITL_KIND_DECODE_ALL if all_decode else ITL_KIND_DECODE_MIXED

    def commit(
        self,
        *,
        total_s: float,
        is_prefill: bool,
        all_decode: bool,
        token_counts: list[int] | None,
        ctxlen_mean: float | None = None,
        ctxlen_max: int | None = None,
    ) -> None:
        """Commit the in-flight step's phase breakdown, then reset.

        ``total_s`` is the full step (ITL) latency; ``others`` is the remainder
        after the measured phases so the breakdown reconciles to it exactly.
        ``all_decode`` is the cross-DP "no rank is prefilling" flag and
        ``token_counts`` the per-DP-rank token workload for this step.
        ``ctxlen_mean`` / ``ctxlen_max`` are the mean / max context (KV) length
        over the step's active batch, recorded per step so ITL is comparable at
        matched context length.
        """
        kind = self.classify(is_prefill, all_decode)
        measured = sum(self._current.get(p, 0.0) for p in ITL_MEASURED_PHASES)
        target = self.latencies[kind]
        for phase in ITL_MEASURED_PHASES:
            target[phase].append(self._current.get(phase, 0.0) * 1000.0)
        target["others"].append(max(0.0, total_s - measured) * 1000.0)

        if ctxlen_mean is not None:
            self.ctxlen_mean[kind].append(round(ctxlen_mean, 1))
            self.ctxlen_max[kind].append(int(ctxlen_max if ctxlen_max is not None else ctxlen_mean))

        if token_counts:
            totals = self.token_totals.get(kind)
            if totals is None or len(totals) != len(token_counts):
                totals = [0] * len(token_counts)
                self.token_totals[kind] = totals
            for i, count in enumerate(token_counts):
                totals[i] += int(count)
            self.token_steps[kind] += 1

        self.reset_step()

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _show_stats(self, stat_type: str, kind: str) -> None:
        data = self.latencies[kind]
        counts = len(data.get(ITL_PHASES[0], []))
        if counts == 0:
            logger.info("%s ITL BREAKDOWN: No data recorded", stat_type)
            return
        total = sum(self._avg(data.get(p, [])) for p in ITL_PHASES)
        logger.info("%s ITL BREAKDOWN:", stat_type)
        logger.info("  Total steps: %d", counts)
        logger.info("  Average ITL: %.3f ms", total)
        for phase in ITL_PHASES:
            avg = self._avg(data.get(phase, []))
            pct = (avg / total * 100.0) if total > 0 else 0.0
            logger.info("  %-14s %8.3f ms (%5.1f%%)", phase, avg, pct)
        totals = self.token_totals.get(kind)
        steps = self.token_steps.get(kind, 0)
        if totals and steps > 0:
            per_rank = [t / steps for t in totals]
            logger.info(
                "  Avg tokens/step per DP rank: [%s] (total %.1f)",
                ", ".join(f"r{i}={v:.1f}" for i, v in enumerate(per_rank)),
                sum(per_rank),
            )
        cl = self.ctxlen_mean.get(kind, [])
        if cl:
            logger.info(
                "  Context length (mean seq_len): avg %.0f, range %.0f–%.0f over %d steps",
                self._avg(cl), min(cl), max(cl), len(cl),
            )

    def _kind_to_dict(self, kind: str, include_raw: bool) -> dict:
        data = self.latencies[kind]
        count = len(data.get(ITL_PHASES[0], []))
        phase_avgs = {p: self._avg(data.get(p, [])) for p in ITL_PHASES}
        total = sum(phase_avgs.values())
        phases: dict[str, dict] = {}
        for phase in ITL_PHASES:
            avg = phase_avgs[phase]
            entry = {
                "avg_ms": avg,
                "pct": (avg / total * 100.0) if total > 0 else 0.0,
            }
            if include_raw:
                entry["samples_ms"] = list(data.get(phase, []))
            phases[phase] = entry

        tokens: dict | None = None
        totals = self.token_totals.get(kind)
        steps = self.token_steps.get(kind, 0)
        if totals and steps > 0:
            per_rank = [t / steps for t in totals]
            tokens = {
                "steps": steps,
                "avg_per_rank": per_rank,
                "avg_total": sum(per_rank),
                "cumulative_per_rank": list(totals),
            }

        cl = self.ctxlen_mean.get(kind, [])
        clx = self.ctxlen_max.get(kind, [])
        context: dict | None = None
        if cl:
            context = {
                "avg_mean": self._avg(cl),
                "min": min(cl),
                "max": max(cl),
            }
            if include_raw:
                # Per-step raw values (index-aligned with phases[*].samples_ms),
                # so ITL can be plotted/regressed against context length.
                context["per_step_mean"] = list(cl)
                context["per_step_max"] = list(clx)

        return {
            "count": count,
            "avg_itl_ms": total,
            "phases": phases,
            "tokens": tokens,
            "context_length": context,
        }

    def to_dict(self, include_raw: bool = True) -> dict:
        """Serialize the collected ITL breakdown to a plain dict.

        ``include_raw`` keeps the per-step latency samples (``samples_ms``);
        set False for just the aggregates.
        """
        return {
            "name": self.name,
            "pid": os.getpid(),
            "phases_order": list(ITL_PHASES),
            "steps": {
                kind: self._kind_to_dict(kind, include_raw)
                for kind in ITL_STEP_KINDS
            },
        }

    def to_json(self, include_raw: bool = True, indent: int | None = 2) -> str:
        """Return the collected ITL breakdown as a JSON string."""
        return json.dumps(self.to_dict(include_raw=include_raw), indent=indent)

    def dump_json(self, path: str | None = None, include_raw: bool = True) -> str | None:
        """Write the ITL breakdown JSON to ``path`` (or VLLM_RBLN_METRICS_JSON_FILE).

        The worker pid is appended before the extension so concurrent TP/DP
        workers do not clobber one another. Returns the written path, or None
        when no path is configured or the write fails.
        """
        target = path or envs.VLLM_RBLN_METRICS_JSON_FILE
        if not target:
            return None
        root, ext = os.path.splitext(target)
        out_path = f"{root}.{os.getpid()}{ext or '.json'}"
        try:
            with open(out_path, "w") as f:
                f.write(self.to_json(include_raw=include_raw))
        except OSError as e:
            logger.warning("Failed to write ITL metrics JSON %s: %s", out_path, e)
            return None
        logger.info("Wrote ITL breakdown JSON to %s", out_path)
        return out_path

    def print_final_stats(self) -> None:
        _attach_metrics_file_handler()
        logger.info("=" * 80)
        if self.name:
            logger.info("FINAL ITL BREAKDOWN STATISTICS [%s]", self.name)
        else:
            logger.info("FINAL ITL BREAKDOWN STATISTICS")
        logger.info("=" * 80)
        self._show_stats("PREFILL", ITL_KIND_PREFILL)
        logger.info("-" * 40)
        self._show_stats("DECODE (all ranks decoding)", ITL_KIND_DECODE_ALL)
        logger.info("-" * 40)
        self._show_stats("DECODE (mixed: peer prefilling)", ITL_KIND_DECODE_MIXED)
        logger.info("=" * 80)
        # Also emit the machine-readable JSON when a target file is configured.
        self.dump_json()


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
        *,
        token_count: int = 0,
        is_prefill: bool = False,
        padded_decode: bool = False,
        request_ids: list[str] | None = None,
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
            token_count=token_count,
            host_time=host_time,
            device_time=device_time,
            ccl_time=ccl_time,
            prepare_time=prepare_time,
            is_prefill=is_prefill,
            padded_decode=padded_decode,
            request_ids=request_ids,
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


def collect_metrics(
    performance_tracker: PerformanceTracker,
    is_prefill: bool,
    start_time: float,
    end_time: float,
    reports: list[dict],
    token_count: int,
) -> None:
    performance_tracker.record(
        StepReport.from_reports(
            start_time,
            end_time,
            reports,
            token_count=token_count,
            is_prefill=is_prefill,
        )
    )
