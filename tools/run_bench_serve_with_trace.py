#!/usr/bin/env python3
# Copyright 2025 Rebellions Inc. All rights reserved.
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

"""
Run `vllm bench serve` wrapped with Perfetto trace start/stop + sanity check.

Server-side patches are auto-applied by the `vllm_rbln` plugin, so the server
just needs to be started with the usual `vllm serve <args>` — the
`/v1/trace/start` and `/v1/trace/stop` endpoints are exposed automatically.

This wrapper handles the client-side orchestration: it verifies the endpoints
are present, brackets the bench run with trace start/stop calls, and merges
the per-PID trace files at the end. All unknown arguments are forwarded
transparently to `vllm bench serve`.

Usage:
    python run_bench_serve_with_trace.py <vllm bench serve args...>

Example:
    python run_bench_serve_with_trace.py \\
        --backend vllm --model MiniMaxAI/MiniMax-M2.5 \\
        --max-concurrency 32 --request-rate 4 \\
        --dataset-name random --random-input-len 512 --random-output-len 256 \\
        --num-prompts 640 --save-detailed --result-filename bench.json

Wrapper-only flags (not forwarded to vllm bench serve):
    --server-url URL    Server base URL (default: $SERVER_URL or http://localhost:8000)
    --server-cwd DIR    Where server writes per-pid trace files (default: auto-detect
                        from /v1/trace/start response; falls back to $PWD)
    --trace-output FILE Merged trace filename (default: trace_<timestamp>_merged.json)
    --no-trace          Skip trace start/stop/merge; run bench only
    --no-sanity         Skip /openapi.json sanity check
    --keep-pid-traces   Don't delete per-pid trace files after merge
    --wrapper-help      Show this help

Env vars (alternative to flags):
    SERVER_URL, SERVER_CWD, TRACE_OUTPUT,
    NO_TRACE=1, NO_SANITY=1, KEEP_PID_TRACES=1
"""

from __future__ import annotations

import contextlib
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _split_args(argv: list[str]) -> tuple[dict, list[str]]:
    """Pull wrapper-only flags from argv; everything else is forwarded."""
    opts: dict = {
        "server_url": os.environ.get("SERVER_URL", "http://localhost:8000"),
        "server_cwd": os.environ.get("SERVER_CWD", ""),
        "trace_output": os.environ.get("TRACE_OUTPUT", ""),
        "no_trace": os.environ.get("NO_TRACE", "0") == "1",
        "no_sanity": os.environ.get("NO_SANITY", "0") == "1",
        "keep_pid_traces": os.environ.get("KEEP_PID_TRACES", "0") == "1",
    }
    fwd: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--server-url":
            opts["server_url"] = argv[i + 1]
            i += 2
        elif a == "--server-cwd":
            opts["server_cwd"] = argv[i + 1]
            i += 2
        elif a == "--trace-output":
            opts["trace_output"] = argv[i + 1]
            i += 2
        elif a == "--no-trace":
            opts["no_trace"] = True
            i += 1
        elif a == "--no-sanity":
            opts["no_sanity"] = True
            i += 1
        elif a == "--keep-pid-traces":
            opts["keep_pid_traces"] = True
            i += 1
        elif a == "--wrapper-help":
            print(__doc__)
            sys.exit(0)
        else:
            fwd.append(a)
            i += 1
    return opts, fwd


def _http_post(url: str) -> dict:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _sanity_check(server_url: str) -> None:
    print(f"=== [1/5] Sanity check: trace endpoints on {server_url} ===")
    try:
        spec = _http_get_json(f"{server_url}/openapi.json")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  ERROR: failed to fetch {server_url}/openapi.json: {e}")
        sys.exit(1)
    paths = [p for p in spec.get("paths", {}) if "trace" in p]
    print(f"  trace paths exposed: {paths}")
    needed = {"/v1/trace/start", "/v1/trace/stop"}
    if not needed.issubset(paths):
        missing = needed - set(paths)
        print(f"  ERROR: missing endpoints {sorted(missing)}.")
        print(
            "         The server must have the vllm_rbln plugin loaded "
            "(it auto-applies tracing patches). Pass --no-trace / --no-sanity "
            "to bypass."
        )
        sys.exit(1)
    print("  OK\n")


def _start_trace(server_url: str) -> tuple[str, str]:
    """Returns (timestamp, server_cwd_from_response)."""
    print("=== [2/5] Starting Perfetto trace ===")
    resp = _http_post(f"{server_url}/v1/trace/start")
    print(f"  {json.dumps(resp)}")
    fpath = resp.get("file", "")
    ts = ""
    server_cwd = ""
    if fpath:
        m = re.search(r"\d{8}_\d{6}", os.path.basename(fpath))
        if m:
            ts = m.group(0)
        server_cwd = os.path.dirname(fpath)
    if not ts:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  timestamp: {ts}")
    if server_cwd:
        print(f"  server cwd: {server_cwd}")
    print()
    return ts, server_cwd


def _stop_trace(server_url: str) -> dict:
    print("=== [4/5] Stopping Perfetto trace ===")
    resp = _http_post(f"{server_url}/v1/trace/stop")
    print(f"  {json.dumps(resp)}\n")
    return resp


def _merge_pid_traces(server_cwd: str, ts: str, output: str, keep: bool) -> None:
    print("=== [5/5] Merging per-pid trace files ===")
    pattern = os.path.join(server_cwd, f"trace_{ts}_pid*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no per-pid trace files at {pattern}")
        print("  (was the trace active long enough? was any request processed?)\n")
        return

    # Import merge_traces from sibling module in tools/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from merge_traces import merge_traces  # type: ignore

    rc = merge_traces(files, output)
    if rc == 0:
        print(f"  merged → {output}")
        if not keep:
            for f in files:
                with contextlib.suppress(OSError):
                    os.unlink(f)
            print(
                "  cleaned up per-pid trace files (pass --keep-pid-traces to preserve)"
            )
    print()


def main() -> int:
    opts, fwd_args = _split_args(sys.argv[1:])

    # [1] sanity check
    if not opts["no_trace"] and not opts["no_sanity"]:
        _sanity_check(opts["server_url"])

    # [2] start trace
    ts = ""
    if not opts["no_trace"]:
        ts, auto_cwd = _start_trace(opts["server_url"])
        if not opts["server_cwd"]:
            opts["server_cwd"] = auto_cwd or os.getcwd()

    # [3] run vllm bench serve (forward all unknown args)
    print(f"=== [3/5] vllm bench serve {' '.join(fwd_args)} ===")
    bench_rc = subprocess.call(["vllm", "bench", "serve", *fwd_args])
    print(f"  bench exit code: {bench_rc}\n")

    # [4] stop trace
    if not opts["no_trace"]:
        try:
            _stop_trace(opts["server_url"])
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  WARNING: trace stop failed: {e}\n")

    # [5] merge
    if not opts["no_trace"] and ts:
        out = opts["trace_output"] or f"trace_{ts}_merged.json"
        _merge_pid_traces(opts["server_cwd"], ts, out, opts["keep_pid_traces"])

    return bench_rc


if __name__ == "__main__":
    sys.exit(main())
