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

"""Post-process a merged Perfetto trace and print a TTFT + decode breakdown
report.

Two views per metric:

  (a) Independent percentile distributions: for each metric (ttft, queue,
      service, decode_total, decode_step), the percentile values across the
      full request set. Useful for spotting which dimension dominates.

  (b) Per-TTFT-percentile breakdown: for the representative request at each
      TTFT percentile (p1, p5, ..., p99), report that same request's queue,
      service, decode totals, and average step. Useful for understanding
      *what kind of behaviour* a slow/fast request actually has.

Also reports:
  - Decode-step duration distribution across all individual decode events
    (median ≈ baseline step time; p99/max reveal prefill-induced stalls).
  - Cumulative time totals and one-line interpretation.

Used by `vllm-rbln bench serve --trace` to print a summary at end of run,
and re-exposed via `tools/analyze_trace.py` for offline analysis of any
merged trace JSON.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

PERCENTILES = (1, 5, 10, 50, 90, 95, 99)


def _parse_trace(events: list[dict]) -> tuple[dict[str, dict], list[int]]:
    """Returns (per_request_stats, all_decode_step_durations_us)."""
    per_req: dict[str, dict] = defaultdict(
        lambda: {
            "arrival_us": None,
            "queue_dur_us": 0,
            "prefill_dur_us": 0,
            "prefill_chunks": 0,
            "prefill_last_end_us": None,
            "decode_dur_us": 0,
            "decode_steps": 0,
        }
    )
    all_decode_step_us: list[int] = []
    for ev in events:
        ph = ev.get("ph")
        name = ev.get("name")
        if ph == "b" and name == "request":
            per_req[ev["id"]]["arrival_us"] = ev["ts"]
        elif ph == "X":
            rid = ev.get("tid")
            if rid is None:
                continue
            r = per_req[rid]
            dur = ev.get("dur", 0)
            ts = ev.get("ts", 0)
            if name == "queuing":
                r["queue_dur_us"] += dur
            elif name == "prefill":
                r["prefill_dur_us"] += dur
                r["prefill_chunks"] += 1
                end = ts + dur
                if r["prefill_last_end_us"] is None or end > r["prefill_last_end_us"]:
                    r["prefill_last_end_us"] = end
            elif name == "decode":
                r["decode_dur_us"] += dur
                r["decode_steps"] += 1
                all_decode_step_us.append(dur)
    return per_req, all_decode_step_us


def _build_rows(per_req: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for _rid, r in per_req.items():
        if r["arrival_us"] is None or r["prefill_last_end_us"] is None:
            continue
        ttft_ms = (r["prefill_last_end_us"] - r["arrival_us"]) / 1e3
        steps = r["decode_steps"]
        decode_ms = r["decode_dur_us"] / 1e3
        rows.append(
            {
                "ttft_ms": ttft_ms,
                "queue_ms": r["queue_dur_us"] / 1e3,
                "service_ms": r["prefill_dur_us"] / 1e3,
                "decode_total_ms": decode_ms,
                "decode_steps": steps,
                "decode_avg_step_ms": (decode_ms / steps) if steps > 0 else 0.0,
                "prefill_chunks": r["prefill_chunks"],
            }
        )
    return rows


def _pct(sorted_arr: list[float], p: float) -> float:
    n = len(sorted_arr)
    return sorted_arr[max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))]


def _row(
    label: str,
    vals: list[float],
    sorted_vals: list[float],
    width: int = 14,
    fmt: str = ".1f",
) -> None:
    mean = sum(vals) / len(vals) if vals else 0.0
    cells = [mean] + [_pct(sorted_vals, p) for p in PERCENTILES] + [sorted_vals[-1]]
    print(f"{label:<{width}}" + "".join(f"{c:>10{fmt}}" for c in cells))


def analyze_merged_trace(merged_path: str) -> dict[str, Any] | None:
    """Parse a merged Perfetto trace file and print a TTFT + decode report.

    Returns a dict with computed arrays (also printed), or None on failure.
    """
    if not os.path.exists(merged_path):
        return None
    try:
        with open(merged_path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None

    events = data.get("traceEvents", [])
    per_req, all_decode_us = _parse_trace(events)
    rows = _build_rows(per_req)
    if not rows:
        return None

    rows.sort(key=lambda r: r["ttft_ms"])
    n = len(rows)

    ttfts = [r["ttft_ms"] for r in rows]
    queues = [r["queue_ms"] for r in rows]
    services = [r["service_ms"] for r in rows]
    decode_totals = [r["decode_total_ms"] for r in rows]
    decode_avgs = [r["decode_avg_step_ms"] for r in rows]
    all_decode_ms = [d / 1e3 for d in all_decode_us]

    ttft_s = sorted(ttfts)
    queue_s = sorted(queues)
    service_s = sorted(services)
    decode_total_s = sorted(decode_totals)
    decode_avg_s = sorted(decode_avgs)
    decode_step_s = sorted(all_decode_ms)

    sep = "=" * 100
    print()
    print(sep)
    print(f" TTFT + decode analysis from {os.path.basename(merged_path)}")
    print(f" Requests: {n}    Decode steps total: {len(all_decode_ms):,}")
    print(sep)

    # ----- (a) Independent percentile distributions -----
    cols = ["mean"] + [f"p{p}" for p in PERCENTILES] + ["max"]
    header = f"{'metric':<14}" + "".join(f"{c:>10}" for c in cols)
    print("Distribution (independent percentiles):")
    print(header)
    print("-" * len(header))
    _row("ttft (ms)", ttfts, ttft_s)
    _row("queue (ms)", queues, queue_s)
    _row("service (ms)", services, service_s)
    _row("dec_total(ms)", decode_totals, decode_total_s)
    _row("dec_avg (ms)", decode_avgs, decode_avg_s)
    if all_decode_ms:
        _row("dec_step(ms)", all_decode_ms, decode_step_s)
    print()

    # ----- (b) Per-TTFT-percentile breakdown -----
    print(
        "Per-TTFT-percentile breakdown "
        "(representative request at each TTFT percentile):"
    )
    sub_header = (
        f"  {'pct':<6}{'ttft':>10}{'queue':>10}{'service':>10}"
        f"{'queue%':>9}{'service%':>10}{'dec_total':>12}"
        f"{'dec_steps':>11}{'avg_step':>10}"
    )
    print(sub_header)
    print("  " + "-" * (len(sub_header) - 2))
    for p in PERCENTILES:
        idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        r = rows[idx]
        t = r["ttft_ms"]
        q = r["queue_ms"]
        s = r["service_ms"]
        qp = q / t * 100 if t > 0 else 0.0
        sp = s / t * 100 if t > 0 else 0.0
        print(
            f"  p{p:<5}{t:>10.1f}{q:>10.1f}{s:>10.1f}"
            f"{qp:>8.1f}%{sp:>9.1f}%"
            f"{r['decode_total_ms']:>11.1f} "
            f"{r['decode_steps']:>10d}"
            f"{r['decode_avg_step_ms']:>10.2f}"
        )
    print()

    # ----- (c) Per-decode-step duration distribution -----
    if all_decode_ms:
        print("Decode-step duration distribution (across all decode events):")
        ds_header = f"  {'metric':<14}" + "".join(f"{c:>10}" for c in cols)
        print(ds_header)
        print("  " + "-" * (len(ds_header) - 2))
        _row("  step (ms)", all_decode_ms, decode_step_s, fmt=".2f")
        median_step = _pct(decode_step_s, 50)
        p99_step = _pct(decode_step_s, 99)
        max_step = decode_step_s[-1]
        print(
            f"  hint: median {median_step:.2f}ms ≈ baseline pure decode; "
            f"p99 {p99_step:.2f}ms / max {max_step:.2f}ms suggest "
            "prefill-induced stalls or other interruptions"
        )
        print()

    # ----- (d) Cumulative + interpretation -----
    total_ttft = sum(ttfts) / 1000.0
    total_queue = sum(queues) / 1000.0
    total_service = sum(services) / 1000.0
    total_decode = sum(decode_totals) / 1000.0
    print(f"Cumulative ({n} requests):")
    pct_q = total_queue / total_ttft * 100 if total_ttft > 0 else 0.0
    pct_s = total_service / total_ttft * 100 if total_ttft > 0 else 0.0
    print(f"  total TTFT    = {total_ttft:8.1f} s")
    print(f"  total queue   = {total_queue:8.1f} s ({pct_q:5.1f}% of TTFT)")
    print(f"  total service = {total_service:8.1f} s ({pct_s:5.1f}% of TTFT)")
    print(f"  total decode  = {total_decode:8.1f} s")
    print()

    mean_q = sum(queues) / len(queues)
    mean_t = sum(ttfts) / len(ttfts)
    queue_share = mean_q / mean_t * 100 if mean_t > 0 else 0.0
    if queue_share > 75:
        verdict = "QUEUE-DOMINATED (system is saturated; lower load or raise capacity)"
    elif queue_share > 40:
        verdict = "QUEUE-BOUNDED (approaching saturation)"
    else:
        verdict = "COMPUTE-BOUNDED (queue minimal; service-time dominates)"
    print(f"Interpretation: queue/TTFT mean = {queue_share:.1f}%  →  {verdict}")
    print(sep)
    print()

    return {
        "n": n,
        "ttft_ms": ttfts,
        "queue_ms": queues,
        "service_ms": services,
        "decode_total_ms": decode_totals,
        "decode_avg_step_ms": decode_avgs,
        "decode_step_ms": all_decode_ms,
        "total_ttft_s": total_ttft,
        "total_queue_s": total_queue,
        "total_service_s": total_service,
        "total_decode_s": total_decode,
    }
