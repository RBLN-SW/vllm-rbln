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

"""Run isolated lm-eval launches and collect their scores in a target summary."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from importlib import metadata
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
    start_case,
    target_env,
    write_repro,
    write_result,
)

_HERE = Path(__file__).resolve().parent


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


def summarize(
    target: str,
    chip: str,
    cases: dict[str, dict[str, Any] | None],
) -> tuple[str, bool]:
    """Show every planned launch, including missing and incomplete evaluations."""
    facts = [
        result["facts"]
        for result in cases.values()
        if result is not None and result["rows"]
    ]
    shared = {
        key: value
        for key, value in (facts[0] if facts else {}).items()
        if all(other.get(key) == value for other in facts[1:])
    }
    body = [f"#### {target} -- {chip}", ""]
    if shared:
        body += [" · ".join(shared.values()), ""]
    body += [
        "| launch | task | metric | filter | value | stderr | time | status | job |",
        "|:--|:--|:--|:--|--:|--:|--:|:--|:--|",
    ]
    degraded = False
    for launch, result in cases.items():
        status = case_status(result)
        degraded |= status != "ok"
        rows = result["rows"] if result is not None else []
        job_url = result["job_url"] if result is not None else None
        job = f"[log]({job_url})" if job_url else "--"
        for row in rows or [{}]:
            value, stderr, elapsed = (
                row.get("value"),
                row.get("stderr"),
                row.get("time"),
            )
            cells = [
                launch,
                row.get("task", "--"),
                row.get("metric", "--"),
                row.get("filter", "--"),
                f"{value:.4f}" if value is not None else "--",
                f"±{stderr:.4f}" if isinstance(stderr, float) else "--",
                f"{float(elapsed):.0f}s" if elapsed is not None else "--",
                status,
                job,
            ]
            body.append("| " + " | ".join(cells) + " |")
    return "\n".join(body), degraded


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


def report(out: Path, target: dict[str, Any], chip: str, result: dict[str, Any]) -> int:
    """Read what lm-eval wrote and say what it measured.

    Without an expected value this only records: a first run is what decides
    what to expect, and the same model scores differently on a chip that has to
    fall back to W8A16."""
    results = sorted(out.glob("results_*.json"))
    if not results:
        print(f"  no lm-eval results under {out}", file=sys.stderr)
        return 1

    spec = target["eval"]
    # The whole payload: the counts the summary needs sit outside "results".
    payload = json.loads(results[-1].read_text())
    measured = payload["results"]
    result["facts"] = _facts(payload, spec["tasks"])
    expected = (spec.get("expected") or {}).get(chip)
    key = spec.get("filter", "exact_match,strict-match")
    status = 0
    for task in spec["tasks"]:
        scores = measured[task]
        for name, value in sorted(scores.items()):
            if "," not in name or "stderr" in name or not isinstance(value, float):
                continue
            metric, _, filter_name = name.partition(",")
            result["rows"].append(
                {
                    "task": task,
                    "metric": metric,
                    "filter": filter_name,
                    "value": value,
                    "stderr": scores.get(f"{metric}_stderr,{filter_name}"),
                    "time": payload.get("total_evaluation_time_seconds"),
                }
            )
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
            status = 1
            continue
        line = f"  {task}: {key}={value:.4f}"
        if expected is None:
            print(f"{line}  (no expected value for {chip}; recorded only)")
            continue
        rtol = spec.get("rtol", 0.03)
        ok = abs(value - expected) <= rtol
        print(f"{line}  expected {expected:.4f} +-{rtol}  {'ok' if ok else 'FAILED'}")
        status = max(status, 0 if ok else 1)
    return status


def run_launch(
    out: Path,
    target: dict[str, Any],
    serve: dict[str, Any],
    serve_cmd: list[str],
) -> int:
    from vllm.benchmarks.sweep.server import ServerProcess

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
    with (
        env_applied(target_env(target)),
        ServerProcess(serve_cmd, [], show_stdout=True) as server,
    ):
        server.wait_until_ready(timeout=target.get("server_ready_timeout", 3600))
        status = subprocess.run(cmd).returncode

    return status


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
            help="shared by the cases and summary",
        )
        if action == "run":
            command.add_argument("--launch")
        elif action == "summary":
            command.add_argument("--download", action="store_true")
    args = parser.parse_args()

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
        (name, launch)
        for name, target in targets.items()
        for launch in target["serve_params"]
    ]
    if args.action == "run" and args.launch is not None:
        if args.launch not in {launch for _, launch in cases}:
            parser.error(f"no such launch: {args.launch}")
        cases = [case for case in cases if case[1] == args.launch]
    if not cases:
        parser.error(f"no lm-eval cases for {args.chip}")

    if args.action == "pipeline":
        pipeline = measured_pipeline(
            "lm-eval",
            Path(__file__).resolve(),
            targets,
            cases,
            chip=args.chip,
            run_id=args.run_id,
            targets_dir=args.targets_dir,
        )
        print(json.dumps(pipeline))
        return 0

    base = output_dir("LM_EVAL_OUTPUT_DIR", "lm-eval-results") / args.run_id / args.chip
    if args.action == "run":
        status = 0
        for name, launch in cases:
            out = base / name / launch
            print(f"--- {name} / {launch} -> {out}", flush=True)
            result = start_case(out)
            target = targets[name]
            serve = target["serve"] | target["serve_params"][launch]
            needed = devices_needed(serve, target_env(target))
            budget = device_count()
            if needed > budget:
                raise ValueError(f"{launch} needs {needed} NPUs, host has {budget}")
            serve_cmd = build_cmd(["vllm", "serve", "--model", target["model"]], serve)
            rc = run_launch(out, target, serve, serve_cmd)
            result["exit_code"] = max(
                int(rc != 0), report(out, target, args.chip, result)
            )
            write_result(out, result)
            status = max(status, result["exit_code"])
        return status

    assert args.action == "summary"
    if args.download:
        download_results("lm-eval", args.run_id, args.chip)
    for name in targets:
        results = {}
        for case_name, launch in cases:
            if case_name == name:
                path = base / name / launch / "result.json"
                results[launch] = (
                    json.loads(path.read_text()) if path.exists() else None
                )
        body, degraded = summarize(name, args.chip, results)
        publish_summary("lm-eval", base / name, body, chip=args.chip, degraded=degraded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
