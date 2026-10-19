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

"""Run isolated perf cases, generate CI jobs, and collect their results.

Each serve/bench pair gets its own upstream sweep and server process. CI jobs
write case artifacts; the summary job publishes one table per target and chip.
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
    case_status,
    device_count,
    devices_needed,
    download_results,
    host_chip,
    measured_pipeline,
    output_dir,
    publish_summary,
    run_logged,
    start_case,
    target_env,
    write_repro,
    write_result,
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


def _read_rows(out: Path) -> list[dict[str, Any]]:
    """Every run the sweep recorded for one case. Its columns are the benchmark's
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
    chip: str,
    cases: dict[tuple[str, str], dict[str, Any] | None],
) -> tuple[str, bool]:
    """Include every planned case, even when a job produced no artifact."""
    shapes = [
        _workload(row)
        for result in cases.values()
        if result is not None
        for row in result["rows"]
    ]
    shared = {
        key: value
        for key, value in (shapes[0] if shapes else {}).items()
        if all(shape.get(key) == value for shape in shapes[1:])
    }

    degraded = False
    lines = []
    for (launch, bench), result in cases.items():
        status = case_status(result)
        if result is None:
            rows, job = [], "--"
        else:
            rows = result["rows"]
            job = f"[log]({result['job_url']})" if result["job_url"] else "--"

        for row in rows or [{}]:
            cells = [launch, bench]
            for key, _, fmt in _METRICS:
                value = _num(row, key)
                cells.append(fmt(value) if value is not None else "--")
            completed, prompts = _num(row, "completed"), _num(row, "num_prompts")
            failed = _num(row, "failed") or 0
            done = (
                f"{completed:.0f}/{prompts:.0f}"
                if completed is not None and prompts is not None
                else "--"
            )
            row_status = status
            if failed or (
                completed is not None and prompts is not None and completed != prompts
            ):
                row_status = "incomplete requests" if status == "ok" else status
            degraded |= row_status != "ok"
            cells += [done, row_status, job]
            lines.append("| " + " | ".join(cells) + " |")

    titles = [
        "launch",
        "bench",
        *(title for _, title, _ in _METRICS),
        "done",
        "status",
        "job",
    ]
    body = [
        f"#### {target} -- {chip}",
        "",
        " · ".join(f"{k} {v}" for k, v in shared.items()),
        "",
        "| " + " | ".join(titles) + " |",
        "|:--|:--" + "|--:" * (len(_METRICS) + 1) + "|:--|:--|",
        *lines,
    ]
    return "\n".join(body), degraded


def spread_across_ranks(bench_cmd: list[str], serve: dict[str, Any]) -> list[str]:
    """Under data parallelism, send the benchmark through the shim that cycles
    the rank per request: one client cannot otherwise reach more than one rank,
    and the server's balancer does not divide the load evenly enough."""
    dp = int(serve.get("data_parallel_size", 1))
    if dp <= 1:
        return bench_cmd
    return [sys.executable, str(_RR_BENCH), "--dp-size", str(dp), *bench_cmd]


def run_case(
    out: Path,
    target: dict[str, Any],
    launch: str,
    bench: str,
    passthrough: list[str],
) -> int:
    result = start_case(out, resume="--resume" in passthrough)

    overrides = target["serve_params"][launch]
    serve = target["serve"] | overrides
    needed, available = devices_needed(serve, target_env(target)), device_count()
    if needed > available:
        raise ValueError(f"{launch} needs {needed} NPUs, host has {available}")

    model = target["model"]
    serve_cmd = build_cmd(["vllm", "serve", "--model", model], target["serve"])
    bench_cmd = build_cmd(
        ["vllm", "bench", "serve", "--model", model], target.get("bench", {})
    )
    bench_cmd = spread_across_ranks(bench_cmd, serve)
    bench_params = {bench: target["bench_params"][bench]}

    log = out / "perf.log"
    repro = write_repro(
        out,
        target_env(target),
        {
            "serve": build_cmd(serve_cmd, overrides),
            bench: build_cmd(bench_cmd, bench_params[bench]),
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
        serve_file.write_text(json.dumps({launch: overrides}))
        bench_file.write_text(json.dumps(bench_params))

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
            # Keep the sweep's files under the case's log and repro directory.
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
        rc = run_logged(cmd, log, os.environ | target_env(target))

    result["exit_code"] = rc
    result["rows"] = _read_rows(out)
    write_result(out, result)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "pipeline", "summary"):
        command = commands.add_parser(action)
        command.add_argument("targets", nargs="*", help="target names (default: all)")
        command.add_argument("--targets-dir", type=Path, default=_HERE / "targets")
        command.add_argument("--chip", required=action != "run")
        command.add_argument(
            "--run-id",
            default=os.environ.get("BUILDKITE_BUILD_ID")
            or datetime.now().strftime("%Y%m%d_%H%M%S"),
            help="shared by the cases and summary; pass the same ID to resume",
        )
        if action == "run":
            command.add_argument("--launch")
            command.add_argument("--bench")
        elif action == "summary":
            command.add_argument("--download", action="store_true")
    args, passthrough = parser.parse_known_args()
    if passthrough and args.action != "run":
        parser.error(f"unrecognized arguments: {shlex.join(passthrough)}")

    # Not a flag: an --output-dir here would reach the sweep through passthrough
    # and move its results out from under the launch directory.
    if bad := {"-o", "--output-dir"} & {arg.split("=", 1)[0] for arg in passthrough}:
        parser.error(f"{', '.join(sorted(bad))} is set with PERF_OUTPUT_DIR instead")

    paths = sorted([*args.targets_dir.glob("*.yaml"), *args.targets_dir.glob("*.yml")])
    if args.targets:
        wanted = set(args.targets)
        paths = [p for p in paths if p.stem in wanted]
        if missing := wanted - {p.stem for p in paths}:
            parser.error(f"no such target: {', '.join(sorted(missing))}")
    if not paths:
        parser.error(f"no targets under {args.targets_dir}")

    if args.action == "run":
        chip = host_chip()
        if chip is None:
            parser.error("cannot determine the host NPU chip")
        if args.chip is not None and args.chip != chip:
            parser.error(f"expected chip {args.chip}, host reports {chip}")
        args.chip = chip
    targets = {path.stem: yaml.safe_load(path.read_text()) for path in paths}
    targets = {
        name: target
        for name, target in targets.items()
        if not target.get("chips") or args.chip in target["chips"]
    }
    cases = [
        (name, launch, bench)
        for name, target in targets.items()
        for launch in target["serve_params"]
        for bench in target["bench_params"]
    ]
    if args.action == "run":
        for selected, column in ((args.launch, 1), (args.bench, 2)):
            if selected is not None:
                if selected not in {case[column] for case in cases}:
                    parser.error(
                        f"no such {'launch' if column == 1 else 'bench'}: {selected}"
                    )
                cases = [case for case in cases if case[column] == selected]
    if not cases:
        parser.error(f"no perf cases for {args.chip}")

    if args.action == "pipeline":
        pipeline = measured_pipeline(
            "perf",
            Path(__file__).resolve(),
            targets,
            cases,
            chip=args.chip,
            run_id=args.run_id,
            targets_dir=args.targets_dir,
        )
        print(json.dumps(pipeline))
        return 0

    base = output_dir("PERF_OUTPUT_DIR", "perf-results") / args.run_id / args.chip
    if args.action == "run":
        status = 0
        for name, launch, bench in cases:
            out = base / name / launch / bench
            print(f"--- {name} / {launch} / {bench} -> {out}", flush=True)
            rc = run_case(out, targets[name], launch, bench, passthrough)
            status = max(status, int(rc != 0))
        return status

    assert args.action == "summary"
    if args.download:
        download_results("perf", args.run_id, args.chip)
    for name in targets:
        results = {}
        for case_name, launch, bench in cases:
            if case_name == name:
                path = base / name / launch / bench / "result.json"
                results[launch, bench] = (
                    json.loads(path.read_text()) if path.exists() else None
                )
        body, degraded = summarize(name, args.chip, results)
        publish_summary("perf", base / name, body, chip=args.chip, degraded=degraded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
