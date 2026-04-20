#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
"""
Side-by-side parser for `vllm bench serve --save-result` JSONs produced by
`compare_disagg_vs_nondisagg.sh` (and any other A/B script that drops
result files named ``<tag>_rate<RATE>.json`` into a single directory).

Rows are grouped by rate, columns by tag (e.g. `disagg`, `nondisagg`).
Prints the standard throughput / TTFT / ITL / E2E metrics plus a per-rate
"wins N metrics" verdict when exactly two tags are present.

Usage:
    python compare_ec_results.py --result-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Metric -> (label, unit, lower_is_better)
METRICS = [
    ("request_throughput",  "req/s",         False),
    ("output_throughput",   "out tok/s",     False),
    ("mean_ttft_ms",        "TTFT mean",     True),
    ("median_ttft_ms",      "TTFT p50",      True),
    ("p99_ttft_ms",         "TTFT p99",      True),
    ("mean_itl_ms",         "ITL mean",      True),
    ("p99_itl_ms",          "ITL p99",       True),
    ("mean_e2el_ms",        "E2E mean",      True),
    ("p99_e2el_ms",         "E2E p99",       True),
]

FNAME_RE = re.compile(r"^(?P<tag>[^_]+)_rate(?P<rate>[0-9.]+)\.json$")


def load_results(result_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {rate: {tag: result_json}}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(result_dir.glob("*.json")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        with path.open() as f:
            data = json.load(f)
        out[m.group("rate")][m.group("tag")] = data
    return out


def fmt(x, unit: str) -> str:
    if x is None:
        return "  n/a"
    if unit in ("req/s", "out tok/s"):
        return f"{x:8.2f}"
    return f"{x:8.1f}"


def render_table(results: dict[str, dict[str, dict]]) -> None:
    if not results:
        print("  (no result JSONs found)")
        return

    # Stable column order: first-seen across rate buckets.
    tags: list[str] = []
    seen: set[str] = set()
    for _, by_tag in results.items():
        for t in by_tag:
            if t not in seen:
                tags.append(t)
                seen.add(t)

    for rate in sorted(results.keys(), key=float):
        by_tag = results[rate]
        print()
        print(f"── rate = {rate} req/s " + "─" * 50)
        header = f"  {'metric':<14}"
        for t in tags:
            header += f" | {t:>10}"
        if len(tags) == 2:
            header += f" | {'Δ (2-1)':>10} | {'Δ %':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key, label, _lower_better in METRICS:
            row = f"  {label:<14}"
            values: list[float | None] = []
            for t in tags:
                v = by_tag.get(t, {}).get(key)
                values.append(v if isinstance(v, (int, float)) else None)
                row += f" | {fmt(v, label):>10}"
            if len(tags) == 2 and all(v is not None for v in values):
                diff = values[1] - values[0]
                pct = diff / values[0] * 100 if values[0] else float("inf")
                row += f" | {diff:>10.2f} | {pct:>7.1f}%"
            print(row)

        if len(tags) == 2 and all(t in by_tag for t in tags):
            t1, t2 = tags
            wins = {t1: 0, t2: 0}
            for key, _, lower_better in METRICS:
                v1 = by_tag[t1].get(key)
                v2 = by_tag[t2].get(key)
                if not (isinstance(v1, (int, float)) and isinstance(v2, (int, float))):
                    continue
                if v1 == v2:
                    continue
                # Normalise so smaller-is-better regardless of metric polarity.
                a = v1 if lower_better else -v1
                b = v2 if lower_better else -v2
                if a < b:
                    wins[t1] += 1
                else:
                    wins[t2] += 1
            print()
            print(f"  → {t1} wins {wins[t1]} metrics, {t2} wins {wins[t2]} metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        required=True,
        type=Path,
        help="Directory of <tag>_rate<RATE>.json files "
             "(produced by vllm bench serve --save-result).",
    )
    args = parser.parse_args()

    results = load_results(args.result_dir)
    if not results:
        print(f"No matching JSON files in {args.result_dir}")
        return

    render_table(results)


if __name__ == "__main__":
    main()
