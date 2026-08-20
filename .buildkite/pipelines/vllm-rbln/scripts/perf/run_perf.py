#!/usr/bin/env python3
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

"""Turn a target file into a `vllm bench sweep serve` run.

The sweep itself -- starting the server, waiting for it, benchmarking, resetting
caches, tearing down, resuming -- is upstream's. This adds only what a target
needs and the sweep cannot express: the chip it runs on, the env it runs under,
and the device budget it fits in.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# The lanes share a directory, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lane import (  # noqa: E402
    build_cmd,
    device_count,
    devices_needed,
    host_chip,
    output_dir,
    publish_summary,
    run_logged,
    target_env,
    write_repro,
)

_HERE = Path(__file__).resolve().parent
_RR_BENCH = _HERE / "dp_round_robin_bench.py"


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f}s"


def _millis(ms: float) -> str:
    return f"{ms:.1f}ms"


# Out of the ~40 columns the benchmark writes. TTFT is in seconds because a long
# prompt puts it minutes away from the millisecond its field is named for.
_METRICS = (
    ("output_throughput", "out tok/s", lambda v: f"{v:.1f}"),
    ("total_token_throughput", "total tok/s", lambda v: f"{v:.1f}"),
    ("mean_ttft_ms", "TTFT mean", _seconds),
    ("median_ttft_ms", "p50", _seconds),
    ("p99_ttft_ms", "p99", _seconds),
    ("mean_tpot_ms", "TPOT mean", _millis),
    ("median_tpot_ms", "p50", _millis),
    ("p99_tpot_ms", "p99", _millis),
    ("duration", "dur", lambda v: f"{v:.0f}s"),
)


def _num(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _workload(row: dict[str, str]) -> dict[str, str]:
    """What has to match for two rows to be comparable. Derived from the token
    totals, not random_input_len, so a real-dataset target reads the same."""
    facts = {}
    if model := row.get("model_id"):
        facts["model"] = model
    prompts = _num(row, "num_prompts")
    if prompts:
        facts["prompts"] = f"{prompts:.0f}"
        for key, label in (
            ("total_input_tokens", "in"),
            ("total_output_tokens", "out"),
        ):
            if (total := _num(row, key)) is not None:
                facts[label] = f"{total / prompts:.0f}"
    if (conc := _num(row, "max_concurrency")) is not None:
        facts["concurrency"] = f"{conc:.0f}"
    return facts


def _read_rows(out: Path) -> list[dict[str, str]]:
    """Every run the sweep recorded for one launch. Its columns are the benchmark's
    own JSON keys plus run_number and the swept overrides."""
    results = out / "results"
    try:
        with (results / "summary.csv").open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        pass

    # summary.csv is written once, after the whole sweep finishes. A sweep that
    # died partway leaves only the per-combination files -- which is exactly when
    # the numbers it did get are worth reading.
    for pattern in ("*/summary.json", "*/run=*.json"):
        rows: list[dict[str, Any]] = []
        for path in sorted(results.glob(pattern)):
            with contextlib.suppress(OSError, ValueError):
                data = json.loads(path.read_text())
                rows.extend(data if isinstance(data, list) else [data])
        if rows:
            return rows
    return []


def summarize(
    target: str,
    chip: str | None,
    launches: dict[str, list[dict[str, str]]],
    skipped: dict[str, str],
) -> tuple[str, bool]:
    """One table for the whole target: its serve_params entries exist to be
    compared. A fact shared by every row moves to the header, leaving the table
    to what differs."""
    rows = [(launch, row) for launch, runs in launches.items() for row in runs]
    if not rows:
        note = "\n".join(f"- {name}: {why}" for name, why in skipped.items())
        return f"#### {target}\n\nno perf results.\n{note}", True

    shapes = [_workload(row) for _, row in rows]
    shared = {
        key: value
        for key, value in shapes[0].items()
        if all(shape.get(key) == value for shape in shapes[1:])
    }

    degraded = bool(skipped)
    lines = []
    for launch, row in rows:
        cells = [launch]
        for key, _, fmt in _METRICS:
            value = _num(row, key)
            cells.append(fmt(value) if value is not None else "--")
        completed, prompts = _num(row, "completed"), _num(row, "num_prompts")
        failed = _num(row, "failed") or 0
        done = f"{completed:.0f}/{prompts:.0f}" if completed and prompts else "--"
        cells.append(done)
        if failed or (completed and prompts and completed != prompts):
            degraded = True
        lines.append("| " + " | ".join(cells) + " |")

    titles = ["launch", *(title for _, title, _ in _METRICS), "done"]
    body = [
        f"#### {target}" + (f" -- {chip}" if chip else ""),
        "",
        " · ".join(f"{k} {v}" for k, v in shared.items()),
        "",
        "| " + " | ".join(titles) + " |",
        "|:--" + "|--:" * (len(titles) - 1) + "|",
        *lines,
    ]
    if skipped:
        body += ["", *(f"- {name}: {why}" for name, why in skipped.items())]
    return "\n".join(body), degraded


def affordable_serve_params(
    target: dict[str, Any], budget: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """The entries this host can run, and why each of the rest was left out --
    a launch missing from the summary should say so rather than not appear."""
    kept, skipped = {}, {}
    for name, overrides in target["serve_params"].items():
        needed = devices_needed({**target["serve"], **overrides}, target_env(target))
        if needed > budget:
            print(f"  skip {name}: needs {needed} NPUs, host has {budget}")
            skipped[name] = f"needs {needed} NPUs, host has {budget}"
            continue
        kept[name] = overrides
    return kept, skipped


def spread_across_ranks(bench_cmd: list[str], serve: dict[str, Any]) -> list[str]:
    """Under data parallelism, send the benchmark through the shim that cycles
    the rank per request: one client cannot otherwise reach more than one rank,
    and the server's balancer does not divide the load evenly enough."""
    dp = int(serve.get("data_parallel_size", 1))
    if dp <= 1:
        return bench_cmd
    return [sys.executable, str(_RR_BENCH), "--dp-size", str(dp), *bench_cmd]


def run_target(path: Path, out_dir: Path, run_id: str, passthrough: list[str]) -> int:
    target = yaml.safe_load(path.read_text())
    name = path.stem
    print(f"=== {name} ({path})")

    chips = target.get("chips")
    chip = host_chip()
    if chips and chip and chip not in chips:
        print(f"  skip: needs {'/'.join(chips)}, host is {chip}")
        return 0

    serve_params, skipped = affordable_serve_params(target, device_count())
    if not serve_params:
        print("  skip: no combination fits this host")
        return 0

    model = target["model"]
    serve_cmd = build_cmd(["vllm", "serve", "--model", model], target.get("serve", {}))
    bench_cmd = build_cmd(
        ["vllm", "bench", "serve", "--model", model], target.get("bench", {})
    )
    env = os.environ | target_env(target)

    # One sweep per serve_params entry, rather than one sweep over all of them:
    # an entry is one server launch, so this is what puts each launch's results,
    # summary and log in a directory of its own.
    status = 0
    launches: dict[str, list[dict[str, str]]] = {}
    for launch, overrides in serve_params.items():
        out = out_dir / run_id / name / launch
        print(f"--- {name} / {launch} -> {out}")
        rc = run_launch(
            out, {launch: overrides}, target, serve_cmd, bench_cmd, env, passthrough
        )
        status = max(status, rc)
        launches[launch] = _read_rows(out)
        if not launches[launch]:
            # The exit status separates a sweep that died from one that ran and
            # wrote nothing -- different logs to open.
            died = f" (sweep exited {rc}, see {out / 'perf.log'})" if rc else ""
            skipped[launch] = f"no results{died}"

    # A malformed row must not turn a lane that ran into a lane that failed.
    try:
        body, degraded = summarize(name, chip, launches, skipped)
    except Exception as exc:  # noqa: BLE001
        print(f"  summary: not built ({type(exc).__name__}: {exc})")
    else:
        publish_summary("perf", out_dir / run_id / name, body, degraded=degraded)
    return status


def run_launch(
    out: Path,
    serve_params: dict[str, Any],
    target: dict[str, Any],
    serve_cmd: list[str],
    bench_cmd: list[str],
    env: dict[str, str],
    passthrough: list[str],
) -> int:
    (overrides,) = serve_params.values()
    bench_cmd = spread_across_ranks(bench_cmd, {**target["serve"], **overrides})

    log = out / "perf.log"
    repro = write_repro(
        out,
        target_env(target),
        {"serve": build_cmd(serve_cmd, overrides)}
        | {
            name: build_cmd(bench_cmd, bench_overrides)
            for name, bench_overrides in target["bench_params"].items()
        },
    )
    print(f"  wrote {out / 'repro.sh'}")
    # Into the log too: reading an old run should not need the script beside it,
    # nor the console scrollback it was started from.
    with log.open("a", encoding="utf-8") as f:
        f.write(repro)

    with tempfile.TemporaryDirectory() as tmp:
        serve_file = Path(tmp) / "serve-params.json"
        bench_file = Path(tmp) / "bench-params.json"
        serve_file.write_text(json.dumps(serve_params))
        bench_file.write_text(json.dumps(target["bench_params"]))

        cmd = [
            "vllm",
            "bench",
            "sweep",
            "serve",
            "--serve-cmd",
            shlex.join(serve_cmd),
            "--bench-cmd",
            shlex.join(bench_cmd),
            "--serve-params",
            str(serve_file),
            "--bench-params",
            str(bench_file),
            # Without it the server and the benchmark write to /dev/null.
            "--show-stdout",
            "--output-dir",
            str(out),
            # The launch is the experiment, so its results stay beside the log and
            # the repro script rather than under a directory the sweep names.
            "--experiment-name",
            "results",
            "--server-ready-timeout",
            str(target.get("server_ready_timeout", 3600)),
            "--num-runs",
            str(target.get("num_runs", 1)),
            *passthrough,
        ]
        print(f"  {shlex.join(cmd)}")
        print(f"  logging to {log}")
        return run_logged(cmd, log, env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="target names to run (default: every file in --targets-dir)",
    )
    parser.add_argument("--targets-dir", type=Path, default=_HERE / "targets")
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="groups this run's launches; pass the same one with --resume",
    )
    args, passthrough = parser.parse_known_args()

    # Not a flag: an --output-dir here would reach the sweep through passthrough
    # and move its results out from under the launch directory.
    if bad := {"-o", "--output-dir"} & set(passthrough):
        parser.error(f"{', '.join(sorted(bad))} is set with PERF_OUTPUT_DIR instead")

    paths = sorted([*args.targets_dir.glob("*.yaml"), *args.targets_dir.glob("*.yml")])
    if args.targets:
        wanted = set(args.targets)
        paths = [p for p in paths if p.stem in wanted]
        if missing := wanted - {p.stem for p in paths}:
            parser.error(f"no such target: {', '.join(sorted(missing))}")
    if not paths:
        parser.error(f"no targets under {args.targets_dir}")

    out_dir = output_dir("PERF_OUTPUT_DIR", "perf-results")
    return max(run_target(p, out_dir, args.run_id, passthrough) for p in paths)


if __name__ == "__main__":
    sys.exit(main())
