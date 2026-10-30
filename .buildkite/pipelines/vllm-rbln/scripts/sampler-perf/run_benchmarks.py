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

"""Expand explicit YAML cases and sweeps; run each in a fresh process, sequentially."""

import argparse
import itertools
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path

import yaml
from benchmark_sampler import parse_args, write_result


def sampler_summary(manifest: dict) -> dict:
    """Build a batch matrix for both the job annotation and nightly report."""
    batches = sorted(
        {case["batch_size"] for case in manifest["cases"]},
        key=lambda value: (value is None, value or 0),
    )
    titles = ["Input", "Sampling", *[f"B={b or 'all'} (ms)" for b in batches]]
    grouped = {}
    passed = failed = incomplete = 0
    for case in manifest["cases"]:
        extra = ", ".join(
            f"{key}={value}"
            for key, value in case["sweep_values"].items()
            if key not in ("base", "batch_size")
        )
        sampling = case["base"] or "custom"
        if extra:
            sampling += f" ({extra})"
        key = (case["input_id"], sampling)
        cells = grouped.setdefault(key, {})
        metrics = case.get("summary")
        valid = (
            case["status"] == "passed"
            and case["returncode"] == 0
            and isinstance(metrics, dict)
            and metrics.get("sample_count", 0) > 0
            and all(
                isinstance(metrics.get(name), (int, float))
                and math.isfinite(metrics[name])
                and metrics[name] > 0
                for name in ("median_ns", "p90_ns")
            )
            and metrics["p90_ns"] >= metrics["median_ns"]
        )
        if valid:
            passed += 1
            value = f"{metrics['median_ns'] / 1_000_000:.6f}"
        elif case["status"] in ("not_started", "running"):
            incomplete += 1
            value = "INCOMPLETE" if case["status"] == "running" else "MISSING"
        else:
            failed += 1
            value = "FAILED"
        batch = case["batch_size"]
        if batch in cells:
            raise ValueError(f"duplicate sampler summary cell: {key}, B={batch}")
        cells[batch] = value
    rows = [
        [*key, *[cells.get(batch, "—") for batch in batches]]
        for key, cells in grouped.items()
    ]
    text = (
        f"{passed}/{len(manifest['cases'])} passed · {failed} failed · "
        f"{incomplete} incomplete · {manifest['chip'] or 'chip not measured'}\n"
        "Median latency (ms), including token transfer to CPU and completion wait."
    )
    markdown = ["| " + " | ".join(titles) + " |", "|" + "---|" * len(titles)]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    return {
        "text": text,
        "titles": titles,
        "rows": rows,
        "markdown": "\n".join(markdown),
        "degraded": bool(not manifest["cases"] or failed or incomplete),
    }


def summarize_results(manifest: dict, *, title: str = "Summary") -> None:
    """Print completed cases and retain their comparable metrics in the manifest."""
    headers = [
        "Case",
        "Status",
        "B",
        "V",
        "Dtype",
        "Samples",
        "Median(ms)",
        "P90(ms)",
        "Round CV(%)",
    ]
    rows = []
    passed = 0
    sample_count = 0
    for entry in manifest["cases"]:
        path = Path(entry["result"])
        entry.pop("summary", None)
        if entry["returncode"] is not None and path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            entry["status"] = result["status"]
            if result["status"] == "passed" and entry["returncode"] != 0:
                entry["status"] = "execution_failed"
            if entry["status"] == "passed":
                measurement = result["measurement"]
                summary = {
                    "shape": result["input"]["shape"],
                    "dtype": result["input"]["dtype"],
                    "sample_count": measurement["sample_count"],
                    "median_ns": measurement["median"],
                    "p90_ns": measurement["p90"],
                    "round_median_cv": measurement["round_median_cv"],
                }
                entry["summary"] = summary
                passed += 1
                sample_count += summary["sample_count"]
        elif entry["status"] == "passed":
            entry["status"] = "missing_result"
        summary = entry.get("summary")
        metrics = (
            [
                *map(str, summary["shape"]),
                summary["dtype"],
                str(summary["sample_count"]),
                f"{summary['median_ns'] / 1_000_000:.6f}",
                f"{summary['p90_ns'] / 1_000_000:.6f}",
                f"{summary['round_median_cv'] * 100:.2f}",
            ]
            if summary is not None
            else ["-"] * 7
        )
        rows.append([entry["case_id"], entry["status"], *metrics])
    completed = sum(entry["returncode"] is not None for entry in manifest["cases"])
    manifest["summary"] = {
        "planned": len(rows),
        "completed": completed,
        "passed": passed,
        "failed": completed - passed,
        "incomplete": len(rows) - completed,
        "sample_count": sample_count,
    }
    print(
        f"\n{title}: {completed}/{len(rows)} completed, {passed} passed, "
        f"{completed - passed} failed, {len(rows) - completed} incomplete"
    )
    print(
        "Latency includes sampler execution, token transfer to CPU and completion wait."
    )
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    for entry in manifest["cases"]:
        if entry["status"] != "passed":
            print(
                f"Failed: {entry['case_id']}; exit={entry['returncode']}; "
                f"log: {entry['log']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML with schema_version: 1, cases and optional named bases. "
        "Omitting base and sweep.base selects all defined bases in order. "
        "Use base or sweep.base for an explicit selection; "
        "paths use the YAML directory.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print existing results from --output-dir without launching processes; "
        "omit --config, --iterations and --dry-run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Results, child logs and manifest directory; "
        "relative paths use the working directory.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Measured calls per round; required here or in each expanded case. "
        "An explicit base or case value overrides this option.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI arguments and input paths; print expanded commands "
        "without launching benchmarks or creating results.",
    )
    args = parser.parse_args()
    on_ci = bool(os.environ.get("BUILDKITE"))
    output_dir = args.output_dir.resolve()
    if args.summary_only:
        if args.config is not None or args.iterations is not None or args.dry_run:
            parser.error("--summary-only accepts only --output-dir")
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        print(f"{'+++' if on_ci else '==='} Sampler benchmark summary", flush=True)
        summarize_results(manifest)
        raise SystemExit(
            1
            if manifest["summary"]["failed"] or manifest["summary"]["incomplete"]
            else 0
        )
    if args.config is None:
        parser.error("--config is required unless --summary-only is specified")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or not {"schema_version", "cases"} <= config.keys()
        or config.keys() - {"schema_version", "cases", "bases"}
        or config.get("schema_version") != 1
        or not isinstance(config.get("cases"), list)
        or not config.get("cases")
    ):
        parser.error("expected schema_version: 1 and a nonempty cases list")
    bases = config.get("bases", {})
    if not isinstance(bases, dict):
        parser.error("bases must map names to benchmark option mappings")
    for name, options in bases.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or not isinstance(options, dict)
            or {"id", "base", "sweep", "case_id", "output"} & options.keys()
        ):
            parser.error("each base must be a named option mapping without inheritance")
    script = Path(__file__).with_name("benchmark_sampler.py").resolve()
    commands = []
    identifiers = set()
    for case in config["cases"]:
        if not isinstance(case, dict) or "id" not in case:
            parser.error("each case must be a mapping with an id")
        name = case["id"]
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            parser.error(
                "case ids accept only letters, digits, underscores, dots and hyphens"
            )
        case_options = {
            key: value for key, value in case.items() if key not in ("id", "sweep")
        }
        if {"case_id", "output"} & case_options.keys():
            parser.error("case_id and output are assigned by the runner")
        sweep = case.get("sweep", {})
        if not isinstance(sweep, dict) or any(
            not isinstance(values, list) or not values for values in sweep.values()
        ):
            parser.error(f"{name}: sweep must map option names to nonempty lists")
        if sweep.keys() & case_options.keys():
            parser.error(
                f"{name}: an option cannot occur in both the case and its sweep"
            )
        if {"case_id", "output", "id", "sweep"} & sweep.keys():
            parser.error("sweep can contain only benchmark options")
        if bases and "base" not in case_options and "base" not in sweep:
            sweep = {"base": list(bases), **sweep}
        for index, combination in enumerate(itertools.product(*sweep.values())):
            overrides = {**case_options, **dict(zip(sweep, combination))}
            base_name = overrides.pop("base", None)
            if base_name is not None and (
                not isinstance(base_name, str) or base_name not in bases
            ):
                parser.error(f"{name}: unknown base {base_name!r}")
            options = {**bases.get(base_name, {}), **overrides}
            label = f"{name}-{base_name}" if base_name is not None else name
            case_id = f"{label}-{index + 1:03d}" if sweep else label
            if case_id == "manifest" or case_id in identifiers:
                parser.error(f"reserved or duplicate expanded case id: {case_id}")
            identifiers.add(case_id)
            if args.iterations is not None:
                options.setdefault("iterations", args.iterations)
            if "input_logits" in options:
                if not isinstance(options["input_logits"], str):
                    parser.error(f"{case_id}: input_logits must be a path string")
                options["input_logits"] = str(
                    (args.config.parent / options["input_logits"]).resolve()
                )
            arguments = [
                "--case-id",
                case_id,
                "--output",
                str(output_dir / f"{case_id}.json"),
            ]
            for key, value in options.items():
                if not isinstance(key, str) or not re.fullmatch(
                    r"[a-z][a-z0-9_]*", key
                ):
                    parser.error(f"invalid benchmark option name: {key!r}")
                flag = "--" + key.replace("_", "-")
                if key == "greedy":
                    if type(value) is not bool:
                        parser.error(f"{case_id}: greedy must be a boolean")
                    if value:
                        arguments.append(flag)
                elif isinstance(value, list):
                    if (
                        key not in ("weights", "means", "stds")
                        or not value
                        or any(type(item) not in (int, float) for item in value)
                    ):
                        parser.error(
                            f"{case_id}: use sweep for option lists; "
                            "only mixture parameters are list-valued"
                        )
                    arguments.extend([flag, *map(str, value)])
                elif value is None or type(value) not in (str, int, float):
                    parser.error(
                        f"{case_id}: unsupported value for {key}; omit unused options"
                    )
                else:
                    arguments.extend([flag, str(value)])
            parsed = parse_args(arguments)
            commands.append(
                {
                    "case_id": case_id,
                    "input_id": name,
                    "batch_size": parsed.batch_size,
                    "base": base_name,
                    "sweep_values": dict(zip(sweep, combination)),
                    "result": str(parsed.output),
                    "command": [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        str(script),
                        *arguments,
                    ],
                }
            )
    print(
        f"{len(commands)} cases; one fresh process per case, sequential execution",
        flush=True,
    )
    if args.dry_run:
        for entry in commands:
            print(shlex.join(entry["command"]), flush=True)
        return
    manifest = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "chip": None,
        "job_url": (
            f"{os.environ['BUILDKITE_BUILD_URL']}?job={os.environ['BUILDKITE_JOB_ID']}"
            if on_ci
            else None
        ),
        "cases": [
            dict(
                entry,
                status="not_started",
                returncode=None,
                log=str(output_dir / f"{entry['case_id']}.log"),
            )
            for entry in commands
        ],
    }
    write_result(output_dir / "manifest.json", manifest)
    for index, entry in enumerate(manifest["cases"], 1):
        entry["status"] = "running"
        write_result(output_dir / "manifest.json", manifest)
        print(
            f"{'---' if on_ci else '==>'} [{index}/{len(commands)}] {entry['case_id']}",
            flush=True,
        )
        print(shlex.join(entry["command"]), flush=True)
        result_path = Path(entry["result"])
        result_path.unlink(missing_ok=True)
        log_path = output_dir / f"{entry['case_id']}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log_path.chmod(0o600)
            with subprocess.Popen(
                entry["command"],
                cwd=script.parents[5],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
        entry["returncode"] = returncode
        entry["log"] = str(log_path)
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["status"] == "passed":
                manifest["chip"] = result["environment"]["chip"]
            entry["status"] = result["status"]
        else:
            entry["status"] = (
                "argument_validation_failed" if returncode == 2 else "missing_result"
            )
        summarize_results({"cases": [entry]}, title="Case result")
        if on_ci and entry["status"] != "passed":
            print("^^^ +++", flush=True)
        write_result(output_dir / "manifest.json", manifest)
        print(f"Result: {result_path}; log: {log_path}", flush=True)
    print(f"{'+++' if on_ci else '==='} Sampler benchmark summary", flush=True)
    summarize_results(manifest)
    write_result(output_dir / "manifest.json", manifest)
    matrix = sampler_summary(manifest)
    body = matrix["text"] + "\n\n" + matrix["markdown"]
    summary_path = output_dir / "summary.md"
    summary_path.touch(mode=0o600, exist_ok=True)
    summary_path.chmod(0o600)
    summary_path.write_text(body + "\n", encoding="utf-8")
    print(f"Results: {output_dir / 'manifest.json'}", flush=True)
    raise SystemExit(1 if matrix["degraded"] else 0)


if __name__ == "__main__":
    main()
