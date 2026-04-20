#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
"""
Parse `vllm bench serve --save-result` JSONs from the A/B EC connector runs
and print a side-by-side comparison.

Filenames produced by `compare_ec_connectors.sh`:
    <connector>_rate<RATE>.json       e.g. push_rate1.0.json, nixl_rate1.0.json

The rate is pulled from the filename so we can align rows across connectors.
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

FNAME_RE = re.compile(r"^(?P<connector>[^_]+)_rate(?P<rate>[0-9.]+)\.json$")


def load_results(result_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {rate: {connector: result_json}}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(result_dir.glob("*.json")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        with path.open() as f:
            data = json.load(f)
        out[m.group("rate")][m.group("connector")] = data
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

    connectors: list[str] = []
    seen: set[str] = set()
    for _, by_conn in results.items():
        for c in by_conn:
            if c not in seen:
                connectors.append(c)
                seen.add(c)

    for rate in sorted(results.keys(), key=float):
        by_conn = results[rate]
        print()
        print(f"── rate = {rate} req/s " + "─" * 50)
        header = f"  {'metric':<14}"
        for c in connectors:
            header += f" | {c:>10}"
        if len(connectors) == 2:
            header += f" | {'Δ (2-1)':>10} | {'Δ %':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key, label, lower_better in METRICS:
            row = f"  {label:<14}"
            values: list[float | None] = []
            for c in connectors:
                v = by_conn.get(c, {}).get(key)
                values.append(v if isinstance(v, (int, float)) else None)
                row += f" | {fmt(v, label):>10}"
            if len(connectors) == 2 and all(v is not None for v in values):
                diff = values[1] - values[0]
                pct = diff / values[0] * 100 if values[0] else float("inf")
                # Mark winner with * (push is col 1 here by default ordering).
                row += f" | {diff:>10.2f} | {pct:>7.1f}%"
            print(row)

        # Print a one-line verdict if exactly two connectors.
        if len(connectors) == 2 and all(c in by_conn for c in connectors):
            c1, c2 = connectors
            wins = {c1: 0, c2: 0}
            for key, _, lower_better in METRICS:
                v1 = by_conn[c1].get(key)
                v2 = by_conn[c2].get(key)
                if not (isinstance(v1, (int, float)) and isinstance(v2, (int, float))):
                    continue
                if v1 == v2:
                    continue
                better = v1 if lower_better else -v1
                other = v2 if lower_better else -v2
                if better < other:
                    wins[c1] += 1
                else:
                    wins[c2] += 1
            print()
            print(f"  → {c1} wins {wins[c1]} metrics, {c2} wins {wins[c2]} metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()

    results = load_results(args.result_dir)
    if not results:
        print(f"No matching JSON files in {args.result_dir}")
        return

    render_table(results)


if __name__ == "__main__":
    main()
