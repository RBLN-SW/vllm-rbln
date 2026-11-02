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

"""Run examples unchanged, taking online server commands from their docstrings."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from lane import device_count, publish_summary, write_repro

from vllm_rbln import envs as rbln_envs

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples/vllm"
SPECDEC = "features/speculative_decoding/"
GROUPS = ("basic", "features", "specdec", "mtp", "dflash", "pooling")
DEVICES = {"features": 4, "mtp": 8, "dflash": 4}


@dataclass(frozen=True)
class Example:
    path: Path
    args: tuple[str, ...] = ()
    serve: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.path.relative_to(EXAMPLES).as_posix() + (
            f" ({self.args[-1]})" if self.args else ""
        )

    @property
    def command(self) -> list[str]:
        return ["uv", "run", "--no-sync", "python", str(self.path), *self.args]


def collect(group: str) -> list[Example]:
    directory = EXAMPLES / (
        "features/speculative_decoding"
        if group in ("specdec", "mtp", "dflash")
        else group
    )
    cases = []
    guard = ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body)
    for path in sorted(directory.rglob("*.py")):
        relative = path.relative_to(EXAMPLES).as_posix()
        if group == "features" and relative.startswith(SPECDEC):
            continue
        if group in ("specdec", "mtp", "dflash"):
            filename = (
                "spec_decode_offline.py"
                if group == "specdec"
                else f"{group}_offline.py"
            )
            if path.name != filename:
                continue
        tree = ast.parse(path.read_text(), filename=str(path))
        if not any(
            isinstance(node, ast.If) and ast.dump(node.test) == guard
            for node in tree.body
        ):
            continue
        doc = ast.get_docstring(tree, clean=False) or ""
        commands = re.findall(r"(?m)^([ \t]+vllm serve[^\n]*(?:\n[ \t]+[^\n]+)*)", doc)
        if len(commands) > 1 or ("vllm serve" in doc and not commands):
            raise ValueError(f"{relative}: expected one indented vllm serve command")
        serve = tuple(shlex.split(commands[0].replace("\\\n", ""))) if commands else ()
        if group == "specdec":
            cases.extend(
                Example(path, ("--method", method))
                for method in ("ngram", "eagle", "eagle3", "suffix")
            )
        else:
            cases.append(Example(path, serve=serve))
    if not cases:
        raise ValueError(f"No executable examples found for {group}: {directory}")
    return cases


@contextlib.contextmanager
def process(
    command: list[str], log: Path, env: dict[str, str]
) -> Generator[subprocess.Popen, None, None]:
    """Keep output and terminate descendants even when the parent exits first.

    The sweep's ServerProcess cannot redirect server output to a per-case log;
    offline examples also need process-group cleanup when they time out.
    """
    print(f"--- {shlex.join(command)}\nLog: {log}", flush=True)
    with subprocess.Popen(["tee", str(log)], stdin=subprocess.PIPE) as tee:
        assert tee.stdin is not None
        try:
            child = subprocess.Popen(
                command,
                stdout=tee.stdin,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=ROOT,
                start_new_session=True,
            )
            try:
                yield child
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        finally:
            tee.stdin.close()
        if tee.wait():
            raise RuntimeError(f"Failed to write log: {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument(
        "--list", action="store_true", help="List commands without running"
    )
    parser.add_argument(
        "--timeout", type=int, default=7200, help="Seconds per script/server startup"
    )
    args = parser.parse_args()
    cases = collect(args.group)
    if args.list:
        for case in cases:
            print(case.name)
            if case.serve:
                print(f"  server: {shlex.join(case.serve)}")
            print(f"  client: {shlex.join(case.command)}")
        print(f"Total: {len(cases)} cases")
        return 0

    needed = DEVICES.get(args.group, 1)
    if device_count() < needed:
        raise RuntimeError(f"{args.group} requires {needed} visible NPUs")
    # The deprecated TP_SIZE alias is still read by the device-count getter.
    runtime_variables = rbln_envs.environment_variables.keys() | {"VLLM_RBLN_TP_SIZE"}
    overrides = sorted(os.environ.keys() & runtime_variables)
    if overrides:
        raise RuntimeError(
            f"Examples require an environment without overrides: {overrides}"
        )
    out = ROOT / "example-results" / args.group
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONUNBUFFERED": "1"}
    launches: dict[tuple[str, ...], list[Example]] = {}
    for case in cases:
        launches.setdefault(case.serve, []).append(case)
    results = []
    for index, (serve, clients) in enumerate(launches.items()):
        launch = out / str(index)
        launch.mkdir(exist_ok=True)
        commands = {
            f"client-{client_index}": case.command
            for client_index, case in enumerate(clients)
        }
        if serve:
            commands = {"serve": list(serve)} | commands
        print(write_repro(launch, {}, commands), flush=True)
        with contextlib.ExitStack() as stack:
            if serve:
                try:
                    server = stack.enter_context(
                        process(list(serve), launch / "server.log", env)
                    )
                    host = (
                        serve[serve.index("--host") + 1]
                        if "--host" in serve
                        else "localhost"
                    )
                    port = "8000"
                    for flag in ("--port", "-p"):
                        if flag in serve:
                            port = serve[serve.index(flag) + 1]
                    deadline = time.monotonic() + args.timeout
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"Server exited with code {server.returncode}"
                            )
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Server was not ready within {args.timeout} seconds"
                            )
                        try:
                            with urlopen(
                                f"http://{host}:{port}/health", timeout=5
                            ) as response:
                                if response.status == 200:
                                    break
                        except (URLError, TimeoutError):
                            time.sleep(1)
                except (RuntimeError, TimeoutError) as exc:
                    for case in clients:
                        results.append(
                            {"name": case.name, "status": "FAIL", "error": str(exc)}
                        )
                    print(f"Server failed: {exc}", flush=True)
                    continue
            for client_index, case in enumerate(clients):
                log = launch / f"{client_index}-{case.path.stem}.log"
                started = time.monotonic()
                error = ""
                try:
                    with process(case.command, log, env) as child:
                        code = child.wait(timeout=args.timeout)
                    if code:
                        error = f"Exit code {code}"
                except subprocess.TimeoutExpired:
                    error = f"Timed out after {args.timeout} seconds"
                results.append(
                    {
                        "name": case.name,
                        "status": "FAIL" if error else "PASS",
                        "error": error,
                        "elapsed_seconds": time.monotonic() - started,
                        "log": str(log.relative_to(ROOT)),
                    }
                )
                print(f"{case.name}: {results[-1]['status']} {error}", flush=True)
    passed = sum(result["status"] == "PASS" for result in results)
    failed = passed != len(cases)
    summary = [
        f"{args.group}: {passed}/{len(cases)} passed"
        + ("" if failed else " — all passed"),
        "",
        "| Example | Result | Details |",
        "| --- | --- | --- |",
        *(f"| {r['name']} | {r['status']} | {r['error']} |" for r in results),
    ]
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    publish_summary("examples-vllm", out, "\n".join(summary), degraded=failed)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
