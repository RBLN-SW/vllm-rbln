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

"""What the perf and lm-eval drivers both need: which chip and how many devices
this host has, how a target's env and flags become a command, and how to run one
with its output kept."""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

_RSD_ENV = "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"
_PARALLEL_KEYS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
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
    """NPUs this run may use. RBLN_DEVICES wins when set, since a job may be
    given some of them."""
    visible = [d for d in os.environ.get("RBLN_DEVICES", "").split(",") if d.strip()]
    return len(visible) or len(glob.glob("/dev/rbln*"))


def devices_needed(serve: dict[str, Any], env: dict[str, str]) -> int:
    """Mirrors RBLNWorker._init_device_env: DP ranks do not share, and every
    (tp x pp) rank takes VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK devices."""
    needed = int(env.get(_RSD_ENV, 1))
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


_LANE_ENV = {
    # A launch compiles before it serves, which takes far longer than the
    # default this allows.
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    # Weights stay resident. Offloading them is what a lane measuring either
    # speed or a score is least willing to pay for.
    "VLLM_RBLN_DISABLE_OFFLOAD": "1",
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


def _agent_bin() -> str | None:
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


def publish_summary(
    lane: str, target_dir: Path, body: str, *, degraded: bool = False
) -> None:
    """Report a target's results to the log, a file, and the build page.

    A summary reports on a run and is never part of its result, so nothing here
    is allowed to raise: a lane that measured for two hours must not fail because
    its numbers could not be written down."""
    try:
        _publish_summary(lane, target_dir, body, degraded=degraded)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"  summary: not published ({type(exc).__name__}: {exc})", flush=True)


def _publish_summary(lane: str, target_dir: Path, body: str, *, degraded: bool) -> None:
    """Cheapest channel first, so a failure in one still leaves the rest.

    One context per (lane, target): a driver that runs once per target would
    otherwise leave only the last one, and each carries its own style."""
    # `+++` expands the group, putting the table at the top of the step's log.
    on_ci = bool(os.environ.get("BUILDKITE"))
    heading = f"+++ {lane} summary" if on_ci else f"=== {lane} summary"
    print(f"{heading} -- {target_dir.name}")
    print(body, flush=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "summary.md").write_text(body, encoding="utf-8")

    if not on_ci:
        return
    agent = _agent_bin()
    if agent is None:
        print("  summary: no buildkite-agent found; annotation skipped", flush=True)
        return
    # Say why on failure, or a missing annotation looks like one never built.
    done = subprocess.run(
        [
            agent,
            "annotate",
            "--context",
            f"{lane}-{target_dir.name}",
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
