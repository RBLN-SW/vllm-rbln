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

"""Score a target with lm-eval against a server this starts.

The sweep behind the perf lane cannot be reused: it reads back a result file in
its own format, which lm-eval does not write. Its ServerProcess can be, and is --
starting a server, polling it, and killing the process group it leaves behind are
the parts worth not reimplementing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
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
    target_env,
    write_repro,
)

_HERE = Path(__file__).resolve().parent


@dataclass
class _Summary:
    """One target's launches. They evaluate the same model and differ only in how
    it is parallelized, so their scores belong in one table, not one each."""

    chip: str | None = None
    facts: list[dict[str, str]] = field(default_factory=list)
    rows: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    degraded: bool = False


# The launch directory's parent is the target, per the layout run_target builds.
_SUMMARIES: dict[Path, _Summary] = {}


def _summary_for(out: Path, chip: str | None) -> _Summary:
    summary = _SUMMARIES.setdefault(out.parent, _Summary(chip=chip))
    return summary


def _note(out: Path, chip: str | None, note: str) -> None:
    """Why a launch has no numbers, so it is not simply absent from the table."""
    summary = _summary_for(out, chip)
    summary.notes.append(f"- {out.name}: {note}")
    summary.degraded = True


def _facts(payload: dict[str, Any], tasks: list[str]) -> dict[str, str]:
    """What has to match for two launches' scores to be comparable: a run capped
    at 200 of gsm8k's 1319 is not comparable with an uncapped one."""
    counts = payload.get("n-samples") or {}
    shots = payload.get("n-shot") or {}
    # Values carry their own label, since only the shared ones get printed.
    facts = {"model": payload.get("model_name") or "?"}
    for task in tasks:
        if (shot := shots.get(task)) is not None:
            facts[f"{task} shots"] = f"{task} {shot}-shot"
        if sampled := counts.get(task):
            effective = sampled.get("effective", "?")
            original = sampled.get("original", "?")
            facts[f"{task} samples"] = f"{effective} / {original} samples"
    return facts


def _record(
    out: Path, chip: str | None, payload: dict[str, Any], tasks: list[str]
) -> None:
    """One row per (task, metric, filter). lm-eval keys them as ``metric,filter``
    and puts the error bar under ``metric_stderr,filter``; without that bar a
    reader treats a move inside it as a regression."""
    summary = _summary_for(out, chip)
    summary.facts.append(_facts(payload, tasks))
    elapsed = "--"
    with contextlib.suppress(TypeError, ValueError, KeyError):
        elapsed = f"{float(payload['total_evaluation_time_seconds']):.0f}s"

    for task in tasks:
        scores = payload["results"].get(task) or {}
        for key, value in sorted(scores.items()):
            if "," not in key or "stderr" in key or not isinstance(value, float):
                continue
            metric, _, filter_name = key.partition(",")
            stderr = scores.get(f"{metric}_stderr,{filter_name}")
            error = f"±{stderr:.4f}" if isinstance(stderr, float) else "--"
            summary.rows.append(
                f"| {out.name} | {task} | {metric} | {filter_name} "
                f"| {value:.4f} | {error} | {elapsed} |"
            )


def flush_summaries() -> None:
    """Publish each target's launches as one table. A fact shared by every launch
    moves to the header, leaving the table to what differs."""
    for target_dir, summary in _SUMMARIES.items():
        try:
            _flush(target_dir, summary)
        except Exception as exc:  # noqa: BLE001 -- reporting never fails a lane
            print(f"  summary: not built ({type(exc).__name__}: {exc})")


def _flush(target_dir: Path, summary: _Summary) -> None:
    shared = {
        key: value
        for key, value in (summary.facts[0] if summary.facts else {}).items()
        if all(facts.get(key) == value for facts in summary.facts[1:])
    }
    chip = f" -- {summary.chip}" if summary.chip else ""
    body = [f"#### {target_dir.name}{chip}", ""]
    if shared:
        body += [" · ".join(shared.values()), ""]
    if summary.rows:
        body += [
            "| launch | task | metric | filter | value | stderr | time |",
            "|:--|:--|:--|:--|--:|--:|--:|",
            *summary.rows,
        ]
    else:
        body.append("no lm-eval results.")
    if summary.notes:
        body += ["", *summary.notes]
    publish_summary("lm-eval", target_dir, "\n".join(body), degraded=summary.degraded)


@contextlib.contextmanager
def env_applied(env: dict[str, str]) -> Iterator[None]:
    """Put a target's env in this process's own environment.

    ServerProcess spawns the server with `os.environ` and takes no env argument,
    so a variable that selects the model path -- VLLM_RBLN_USE_VLLM_MODEL above
    all -- reaches the server only from here."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


def eval_cmd(target: dict[str, Any], serve: dict[str, Any], out: Path) -> list[str]:
    """lm-eval against the served OpenAI endpoint. `local-completions` is the
    plain completions API, so the prompt is the task's own -- no chat template in
    the way of a score meant to be compared with published ones."""
    spec = target["eval"]
    base = f"http://{serve.get('host', 'localhost')}:{serve.get('port', 8000)}/v1"
    # Enough in flight to fill what the server was compiled for. Fewer and each
    # rank decodes a fraction of its batch, which only makes the run longer --
    # lm-eval's own default is 1, so this has to be set.
    concurrent = spec.get(
        "num_concurrent",
        int(serve.get("data_parallel_size", 1)) * int(serve.get("max_num_seqs", 8)),
    )
    model_args = ",".join(
        [
            f"model={target['model']}",
            f"base_url={base}/completions",
            f"num_concurrent={concurrent}",
            "tokenized_requests=False",
        ]
    )
    cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(spec["tasks"]),
        # A path ending in .json is the one form lm-eval does not append a
        # directory named after the model to; this one already names it.
        "--output_path",
        str(out / "results.json"),
    ]
    for flag in ("num_fewshot", "limit", "batch_size"):
        if flag in spec:
            cmd += [f"--{flag}", str(spec[flag])]
    # What every question was asked and answered, for when a score has to be
    # explained rather than compared. Off by default: it is a file per task.
    if spec.get("log_samples"):
        cmd.append("--log_samples")
    return cmd


def report(out: Path, target: dict[str, Any], chip: str | None) -> int:
    """Read what lm-eval wrote and say what it measured.

    Without an expected value this only records: a first run is what decides
    what to expect, and the same model scores differently on a chip that has to
    fall back to W8A16."""
    results = sorted(out.glob("results_*.json"))
    if not results:
        print(f"  no lm-eval results under {out}", file=sys.stderr)
        _note(out, chip, "no lm-eval results")
        return 1

    spec = target["eval"]
    # The whole payload: the counts the summary needs sit outside "results".
    payload = json.loads(results[-1].read_text())
    measured = payload["results"]
    expected = (spec.get("expected") or {}).get(chip)
    key = spec.get("filter", "exact_match,strict-match")
    status = 0
    for task in spec["tasks"]:
        scores = measured[task]
        # Every filter, not only the one compared against: a score that collapses
        # under strict-match but holds under flexible-extract is the answer being
        # formatted differently, not the model being wrong.
        print(
            f"  {task}: "
            + ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(scores.items())
                if isinstance(v, float) and "stderr" not in k
            )
        )
        if (value := scores.get(key)) is None:
            print(f"  {task}: no filter {key!r} in the results", file=sys.stderr)
            _record(out, chip, payload, spec["tasks"])
            _note(out, chip, f"`{key}` is not among the {task} filters")
            return 1
        line = f"  {task}: {key}={value:.4f}"
        if expected is None:
            print(f"{line}  (no expected value for {chip}; recorded only)")
            continue
        rtol = spec.get("rtol", 0.03)
        ok = abs(value - expected) <= rtol
        print(f"{line}  expected {expected:.4f} +-{rtol}  {'ok' if ok else 'FAILED'}")
        status = max(status, 0 if ok else 1)
    _record(out, chip, payload, spec["tasks"])
    return status


def run_launch(
    out: Path,
    target: dict[str, Any],
    serve: dict[str, Any],
    serve_cmd: list[str],
    chip: str | None,
) -> int:
    from vllm.benchmarks.sweep.server import ServerProcess

    out.mkdir(parents=True, exist_ok=True)
    cmd = eval_cmd(target, serve, out)
    # The repro needs the overlay spelled out -- lm-eval is not in the locked
    # environment -- and pinned to the version that produced this score. The api
    # extra is what the served-endpoint backend needs.
    pin = f"lm_eval[api]=={metadata.version('lm_eval')}"
    print(
        write_repro(
            out,
            target_env(target),
            {
                "serve": serve_cmd,
                "eval": ["uv", "run", "--no-sync", "--with", pin, *cmd],
            },
        )
    )

    # show_stdout, or the server's output -- the compile included -- goes to
    # /dev/null and a failure to start says only that it failed.
    server = ServerProcess(serve_cmd, [], show_stdout=True)
    server.server_cmd = serve_cmd
    with env_applied(target_env(target)), server:
        server.wait_until_ready(timeout=target.get("server_ready_timeout", 3600))
        status = subprocess.run(cmd).returncode

    return max(status, report(out, target, chip))


def run_target(path: Path, out_dir: Path, run_id: str) -> int:
    target = yaml.safe_load(path.read_text())
    name = path.stem
    print(f"=== {name} ({path})")

    chips = target.get("chips")
    chip = host_chip()
    if chips and chip and chip not in chips:
        print(f"  skip: needs {'/'.join(chips)}, host is {chip}")
        return 0

    budget = device_count()
    status = 0
    for launch, overrides in target["serve_params"].items():
        serve = {**target.get("serve", {}), **overrides}
        needed = devices_needed(serve, target_env(target))
        if needed > budget:
            print(f"  skip {launch}: needs {needed} NPUs, host has {budget}")
            continue
        out = out_dir / run_id / name / launch
        print(f"--- {name} / {launch} -> {out}")
        serve_cmd = build_cmd(
            ["vllm", "serve", "--model", target["model"]], target.get("serve", {})
        )
        status = max(
            status,
            run_launch(out, target, serve, build_cmd(serve_cmd, overrides), chip),
        )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="target names (default: all)")
    parser.add_argument("--targets-dir", type=Path, default=_HERE / "targets")
    # A fresh directory per run, so a launch that writes nothing cannot have an
    # earlier run's score read back as its own.
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    paths = sorted([*args.targets_dir.glob("*.yaml"), *args.targets_dir.glob("*.yml")])
    if args.targets:
        wanted = set(args.targets)
        paths = [p for p in paths if p.stem in wanted]
        if missing := wanted - {p.stem for p in paths}:
            parser.error(f"no such target: {', '.join(sorted(missing))}")
    if not paths:
        parser.error(f"no targets under {args.targets_dir}")

    out_dir = output_dir("LM_EVAL_OUTPUT_DIR", "lm-eval-results")
    # Materialized, or the summaries would be flushed while targets still run.
    statuses = [run_target(p, out_dir, args.run_id) for p in paths]
    flush_summaries()
    return max(statuses)


if __name__ == "__main__":
    sys.exit(main())
