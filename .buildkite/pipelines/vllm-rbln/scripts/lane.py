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

"""Shared CI target snapshots, device budgets, commands, and results."""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_TARGET_DIRS = ("perf", "lm-eval", "sampler-perf")
_RSD_ENV = "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"
_RSD_SERVE = "rbln_num_devices_per_local_rank"
_PARALLEL_KEYS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "prefill_context_parallel_size",
    "data_parallel_size",
)


def host_chip() -> str | None:
    """The chip this host reports, or None when it cannot be resolved -- run
    every target then, rather than silently none."""
    forced = os.environ.get("RBLN_FORCE_NPU_NAME") or os.environ.get("RBLN_TARGET_SOC")
    if forced:
        return forced.strip().upper()
    try:
        import rebel

        return str(rebel.get_npu_name(0)).strip().upper()
    except Exception:
        return None


def device_count() -> int:
    """NPUs this run may use. The visible list wins when set, since a job may be
    given some of them. Both spellings are read because this runs before the
    platform plugin folds the deprecated one into the other."""
    visible_list = os.environ.get("RBLN_VISIBLE_DEVICES", "")
    raw = visible_list or os.environ.get("RBLN_DEVICES", "")
    visible = [d for d in raw.split(",") if d.strip()]
    return len(visible) or len(glob.glob("/dev/rbln*"))


def devices_needed(serve: dict[str, Any], env: dict[str, str]) -> int:
    """Mirrors RBLNWorker._init_device_env: DP ranks do not share, and every rank
    of vLLM's world size takes --rbln-num-devices-per-local-rank devices.

    Both spellings are read, because the variable still works until 0.14.0 and
    the worker reads both.
    """
    needed = int(serve.get(_RSD_SERVE, env.get(_RSD_ENV, 1)))
    for key in _PARALLEL_KEYS:
        needed *= int(serve.get(key, 1))
    return needed


def output_dir(env_name: str, default_name: str) -> Path:
    """Where a lane writes, taken only from the environment.

    Resolved, because the lane scripts cd to the repo root: a relative path
    printed here would look like it lands wherever the caller was."""
    return Path(os.environ.get(env_name) or default_name).resolve()


def build_cmd(base: list[str], params: dict[str, Any]) -> list[str]:
    """Spell flags exactly as the sweep's own overrides do, so a param that also
    appears here is replaced rather than passed twice."""
    from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem

    return ParameterSweepItem(params).apply_to_cmd(base)


def perf_cases(target: dict[str, Any]) -> list[tuple[str, str]]:
    launches = target["serve_params"]
    benches = target["bench_params"]
    for bench, params in benches.items():
        if "serve" not in params:
            continue
        selected = params["serve"]
        if (
            not isinstance(selected, list)
            or not selected
            or not all(isinstance(name, str) for name in selected)
        ):
            raise ValueError(f"{bench}: serve must be a non-empty list of launch names")
        if unknown := set(selected) - launches.keys():
            raise ValueError(
                f"{bench}: unknown serve names: {', '.join(sorted(unknown))}"
            )

    return [
        (launch, bench)
        for launch in launches
        for bench, params in benches.items()
        if "serve" not in params or launch in params["serve"]
    ]


def load_targets(directory: Path, chip: str, lane: str) -> dict[str, Any]:
    paths = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    if not paths:
        raise ValueError(f"No {lane} targets under {directory}")
    targets = {}
    for path in paths:
        target = yaml.safe_load(path.read_text())
        if not isinstance(target, dict) or not isinstance(target.get("model"), str):
            raise ValueError(f"{path}: model must be a string")
        fields = ("serve", "serve_params", "bench_params" if lane == "perf" else "eval")
        for field in fields:
            if not isinstance(target.get(field), dict):
                raise ValueError(f"{path}: {field} must be a mapping")
        if not target["serve_params"]:
            raise ValueError(f"{path}: no launches configured")
        if lane == "perf":
            if not perf_cases(target):
                raise ValueError(f"{path}: no perf cases configured")
        else:
            assert lane == "lm-eval"
            tasks = target["eval"].get("tasks")
            if (
                not isinstance(tasks, list)
                or not tasks
                or not all(isinstance(task, str) and task for task in tasks)
            ):
                raise ValueError(
                    f"{path}: eval.tasks must be a non-empty list of names"
                )
        for overrides in target["serve_params"].values():
            devices_needed(target["serve"] | overrides, target_env(target))
        if not target.get("chips") or chip in target["chips"]:
            targets[path.stem] = target
    if not targets:
        raise ValueError(f"No {lane} cases configured for {chip}")
    return targets


def restore_ci_targets(directory: Path) -> dict[str, Any]:
    with tarfile.open(directory / "snapshot.tar.gz") as archive:
        for name in _TARGET_DIRS:
            if (directory / name).exists():
                shutil.rmtree(directory / name)
        archive.extractall(directory, filter="data")
    source = json.loads((directory / "source.json").read_text())
    print(f"CI targets: {source['repo']}@{source['commit']}", flush=True)
    return source


_LANE_ENV = {
    # A launch compiles before it serves, which takes far longer than the
    # default this allows.
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
}


def target_env(target: dict[str, Any]) -> dict[str, str]:
    """What every launch runs under, plus what this target adds.

    Values are stringified because YAML reads a bare 1 as an int, and an env
    mapping with a non-string in it is a TypeError at spawn time. A target that
    names one of the lane variables overrides it."""
    return _LANE_ENV | {k: str(v) for k, v in target.get("env", {}).items()}


def write_repro(out: Path, env: dict[str, str], cases: dict[str, list[str]]) -> str:
    """Write a launch as a script, and return it for the log.

    Neither the sweep nor lm-eval prints anything that can be run by hand: the
    commands come out as python lists, and the env a target adds is never shown."""
    lines = [
        "#!/usr/bin/env bash",
        "# Reproduce this launch by hand: serve in one shell, then the rest in",
        "# another once the server reports ready.",
        "set -euo pipefail",
        "",
    ]
    if env:
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        lines += [f"export {exports}", ""]
    lines.append('case "${1:-}" in')
    for case, cmd in cases.items():
        lines += [f"{case})", f"  exec {shlex.join(cmd)}", "  ;;"]
    lines += [
        "*)",
        f'  echo "usage: $0 {{{"|".join(cases)}}}" >&2',
        "  exit 2",
        "  ;;",
        "esac",
    ]

    script = "\n".join(lines) + "\n"
    path = out / "repro.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    path.chmod(0o755)
    return script


def run_logged(cmd: list[str], log: Path, env: dict[str, str]) -> int:
    """Run cmd with its output going to the terminal and to `log`.

    With --show-stdout the server and the benchmark inherit this stream, so the
    file holds everything for this launch. PYTHONUNBUFFERED because python block-
    buffers its own prints when they do not go to a terminal, and a run that dies
    would take whatever is still buffered with it."""
    log.parent.mkdir(parents=True, exist_ok=True)
    # On CI the stream is also the build log, so tee shows it as well as saving
    # it. By hand the file is enough; tail it to watch.
    echo = (
        None
        if os.environ.get("BUILDKITE") or os.environ.get("LANE_ECHO")
        else subprocess.DEVNULL
    )
    tee = subprocess.Popen(["tee", "-a", str(log)], stdin=subprocess.PIPE, stdout=echo)
    try:
        return subprocess.run(
            cmd,
            stdout=tee.stdin,
            stderr=subprocess.STDOUT,
            env=env | {"PYTHONUNBUFFERED": "1"},
        ).returncode
    finally:
        assert tee.stdin is not None
        tee.stdin.close()
        tee.wait()


def agent_bin() -> str | None:
    """A containerized step gets the agent bind-mounted into the build root
    rather than installed, so `which` alone finds nothing."""
    if found := shutil.which("buildkite-agent"):
        return found
    roots = [
        os.environ.get("BUILDKITE_BIN_PATH"),
        os.environ.get("BUILDKITE_BUILD_PATH"),
        "/workspace",
    ]
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / "buildkite-agent"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def write_result(out: Path, result: dict[str, Any]) -> None:
    """Keep result.json readable when a job is interrupted during a write."""
    temporary = out / "result.tmp"
    temporary.write_text(json.dumps(result))
    temporary.replace(out / "result.json")


def start_case(out: Path, *, resume: bool = False) -> dict[str, Any]:
    if out.exists() and not resume:
        raise ValueError(f"Cannot overwrite {out}; use a new --run-id")
    out.mkdir(parents=True, exist_ok=True)
    build_url = os.environ.get("BUILDKITE_BUILD_URL")
    job_id = os.environ.get("BUILDKITE_JOB_ID")
    result: dict[str, Any] = {
        "exit_code": None,
        "job_url": f"{build_url}?job={job_id}" if build_url and job_id else None,
        "rows": [],
    }
    write_result(out, result)
    return result


def case_status(result: dict[str, Any] | None) -> str:
    if result is None:
        return "no result"
    if result["exit_code"] is None:
        return "incomplete"
    if result["exit_code"] != 0:
        return f"failed (exit {result['exit_code']})"
    if not result["rows"]:
        return "no results"
    return "ok"


def pytest_addoption(parser):
    parser.addoption("--ci-result-dir", help="Write structured CI test counts here")


def pytest_sessionstart(session):
    if directory := session.config.getoption("ci_result_dir"):
        start_case(Path(directory))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if directory := config.getoption("ci_result_dir"):
        out = Path(directory)
        result = json.loads((out / "result.json").read_text())
        result["exit_code"] = int(exitstatus)
        result["counts"] = {
            label: len(terminalreporter.stats.get(category, []))
            for label, category in (
                ("passed", "passed"),
                ("failed", "failed"),
                ("errors", "error"),
                ("skipped", "skipped"),
                ("xfailed", "xfailed"),
                ("xpassed", "xpassed"),
            )
        }
        write_result(out, result)


def download_results(lane: str, run_id: str, chip: str) -> None:
    agent = agent_bin()
    if agent is None:
        raise RuntimeError("buildkite-agent is required to download results")
    pattern = f"{lane}-results/{run_id}/{chip}/**/result.json"
    found = subprocess.run(
        [agent, "artifact", "search", pattern, "--allow-empty-results"],
        check=True,
        text=True,
        capture_output=True,
    )
    if found.stdout.strip():
        subprocess.run([agent, "artifact", "download", pattern, "."], check=True)


def measured_pipeline(
    lane: str,
    driver: Path,
    targets: dict[str, Any],
    cases: list[tuple[str, ...]],
    *,
    chip: str,
    run_id: str,
    targets_dir: Path,
    snapshot: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    template = yaml.safe_load((root / ".buildkite/vllm-rbln-nightly.yml").read_text())[
        f"_{lane.replace('-', '_')}"
    ]
    chip_name = chip.removeprefix("RBLN-")
    prefix = f"nightly-{lane}-{chip_name.lower()}"
    setup_commands = ()
    if snapshot:
        setup_commands = (
            "uv run --no-project --python 3.12 --with pyyaml python "
            + shlex.join(
                [
                    str(Path(__file__).resolve().relative_to(root)),
                    "restore",
                    "--run-id",
                    run_id,
                ]
            ),
        )
    driver = driver.relative_to(root)
    runner = driver.with_name(driver.stem.replace("_", "-") + ".sh")
    common = [
        "--chip",
        chip,
        "--run-id",
        run_id,
        "--targets-dir",
        os.path.relpath(targets_dir, root),
    ]
    steps = []
    for case in cases:
        name, launch = case[:2]
        target = targets[name]
        serve = target["serve"] | target["serve_params"][launch]
        selection = [
            arg
            for flag, value in zip(("--launch", "--bench"), case[1:])
            for arg in (flag, value)
        ]
        command = shlex.join(["bash", str(runner), name, *selection, *common])
        steps.append(
            template
            | {
                "label": " / ".join(case),
                "key": f"{prefix}-{'-'.join(case)}".replace(".", "_"),
                "resources": {
                    "npu": {
                        "count": devices_needed(serve, target_env(target)),
                        "product": chip,
                    }
                },
                "command": [*setup_commands, *template["command"], command],
            }
        )
    summary = template | {
        "label": "summary",
        "key": f"{prefix}-summary",
        "depends_on": [step["key"] for step in steps],
        "allow_dependency_failure": True,
        "timeout_in_minutes": 10,
        "artifact_paths": [f"{lane}-results/{run_id}/{chip}/**/summary.md"],
        "command": "uv run --no-project --python 3.12 --with pyyaml python "
        + shlex.join([str(driver), "summary", *targets, *common, "--download"]),
    }
    if setup_commands:
        summary["command"] = [*setup_commands, summary["command"]]
    for key in ("agents", "secrets", "plugins"):
        del summary[key]
    steps.append(summary)
    return {
        "steps": [
            {
                "group": f"[{chip_name}] {lane}",
                "key": prefix,
                "steps": steps,
            }
        ]
    }


def publish_summary(
    lane: str,
    target_dir: Path,
    body: str,
    *,
    chip: str | None = None,
    degraded: bool = False,
) -> None:
    """Report a target's results to the log, a file, and the build page.

    A summary reports on a run and is never part of its result, so nothing here
    is allowed to raise: a lane that measured for two hours must not fail because
    its numbers could not be written down."""
    try:
        _publish_summary(lane, target_dir, body, chip=chip, degraded=degraded)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"  summary: not published ({type(exc).__name__}: {exc})", flush=True)


def _publish_summary(
    lane: str, target_dir: Path, body: str, *, chip: str | None, degraded: bool
) -> None:
    """Cheapest channel first, so a failure in one still leaves the rest.

    One context per (lane, chip, target): a driver runs once per target, and the
    nightly runs a lane on two chips, so a context short of all three keeps only
    whichever job annotated last. Each carries its own style."""
    # `+++` expands the group, putting the table at the top of the step's log.
    on_ci = bool(os.environ.get("BUILDKITE"))
    heading = f"+++ {lane} summary" if on_ci else f"=== {lane} summary"
    print(f"{heading} -- {target_dir.name}")
    print(body, flush=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "summary.md").write_text(body, encoding="utf-8")

    if not on_ci:
        return
    agent = agent_bin()
    if agent is None:
        print("  summary: no buildkite-agent found; annotation skipped", flush=True)
        return
    # Say why on failure, or a missing annotation looks like one never built.
    done = subprocess.run(
        [
            agent,
            "annotate",
            "--context",
            "-".join(part for part in (lane, chip, target_dir.name) if part),
            "--style",
            "warning" if degraded else "info",
        ],
        input=body,
        text=True,
        capture_output=True,
        check=False,
    )
    if done.returncode:
        detail = done.stderr.strip() or done.stdout.strip()
        print(f"  summary: annotate failed ({done.returncode}): {detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or restore CI target snapshots"
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "restore"):
        command = commands.add_parser(action)
        if action == "prepare":
            command.add_argument("--chip", required=True)
        command.add_argument(
            "--run-id",
            default=os.environ.get("BUILDKITE_BUILD_ID")
            or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    args = parser.parse_args()
    snapshot = Path("ci-targets") / args.run_id
    if args.action == "restore":
        agent = agent_bin()
        if agent is None:
            raise RuntimeError("buildkite-agent is required to download CI targets")
        subprocess.run(
            [
                agent,
                "artifact",
                "download",
                str(snapshot / "snapshot.tar.gz"),
                ".",
                "--step",
                "nightly-targets-prepare",
            ],
            check=True,
        )
        restore_ci_targets(snapshot)
        return 0

    assert args.action == "prepare"
    agent = agent_bin() if os.environ.get("BUILDKITE") else None
    if os.environ.get("BUILDKITE") and agent is None:
        raise RuntimeError("buildkite-agent is required to pin CI targets")
    recorded = ""
    if agent:
        recorded = subprocess.run(
            [agent, "meta-data", "get", "ci-target-repo-source", "--default", ""],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if recorded:
        source = json.loads(recorded)
        repo, commit = source["repo"], source["commit"]
    else:
        repo = os.environ.get("CI_TARGET_REPO_URL")
        if not repo:
            parser.error("CI_TARGET_REPO_URL must name the target repository")
        commit = os.environ.get("CI_TARGET_REPO_COMMIT", "")
        source = {
            "repo": repo,
            "branch": os.environ.get("CI_TARGET_REPO_BRANCH") or "main",
            "requested_commit": commit,
        }
    if commit and not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit):
        parser.error("CI_TARGET_REPO_COMMIT must be a full commit SHA")
    ref = commit or f"refs/heads/{source['branch']}"
    git_env = os.environ | {"GIT_TERMINAL_PROMPT": "0"}
    if token := os.environ.get("GIT_PAT"):
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        git_env |= {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
        }
    snapshot.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary)
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "fetch",
                "--quiet",
                "--depth=1",
                repo,
                ref,
            ],
            env=git_env,
            check=True,
        )
        resolved = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "FETCH_HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit and resolved != commit.lower():
            raise ValueError(f"Expected target commit {commit}, got {resolved}")
        source["commit"] = resolved
        if agent and not recorded:
            subprocess.run(
                [
                    agent,
                    "meta-data",
                    "set",
                    "ci-target-repo-source",
                    json.dumps(source),
                ],
                check=True,
            )
        archive_path = checkout / "targets.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                resolved,
                *_TARGET_DIRS,
            ],
            check=True,
        )
        content = checkout / "content"
        with tarfile.open(archive_path) as archive:
            archive.extractall(content, filter="data")
        for name in ("perf", "lm-eval"):
            load_targets(content / name, args.chip, name)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("sampler-perf") / "run_benchmarks.py"),
                "--config",
                str(content / "sampler-perf" / "benchmark-configs.yaml"),
                "--output-dir",
                str(checkout / "sampler-validation"),
                "--dry-run",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        manifest = snapshot / "source.json"
        manifest.write_text(json.dumps(source, indent=2) + "\n")
        with tarfile.open(snapshot / "snapshot.tar.gz", "w:gz") as archive:
            for name in _TARGET_DIRS:
                archive.add(content / name, arcname=name)
            archive.add(manifest, arcname="source.json")
    restore_ci_targets(snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
