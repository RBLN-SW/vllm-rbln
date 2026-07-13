#!/usr/bin/env python3
"""
Run vLLM P/D disaggregation once from a single terminal.

This script starts:
  1. CUDA prefill server
  2. RBLN decode server
  3. toy proxy server
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_CWD = Path(os.environ.get("PDD_RUN_CWD", SCRIPT_DIR)).expanduser().resolve()
WORK_REPO = Path(
    os.environ.get("PDD_VLLM_RBLN_REPO", SCRIPT_DIR)
).expanduser().resolve()
VENV_ROOT = Path(
    os.environ.get("PDD_VENV_ROOT", Path.home() / ".venvs")
).expanduser().resolve()

CUDA_PREFILL_VENV = Path(
    os.environ.get("PDD_PREFILL_VENV", VENV_ROOT / "pdd-prefill")
).expanduser().resolve()
RBLN_DECODE_VENV = Path(
    os.environ.get("PDD_DECODE_VENV", VENV_ROOT / "pdd-decode")
).expanduser().resolve()

CUDA_VLLM_BIN = CUDA_PREFILL_VENV / "bin" / "vllm"
RBLN_VLLM_BIN = RBLN_DECODE_VENV / "bin" / "vllm"
RBLN_PYTHON_BIN = RBLN_DECODE_VENV / "bin" / "python3"

PROXY_SCRIPT = (
    WORK_REPO
    / "tests"
    / "torch_compile"
    / "e2e"
    / "v1"
    / "kv_connector"
    / "nixl_integration"
    / "toy_proxy_server.py"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
HOST = "127.0.0.1"

PREFILL_PORT = 8100
DECODE_PORT = 8200
PROXY_PORT = 8192

PREFILL_NIXL_SIDE_CHANNEL_PORT = 5559
DECODE_NIXL_SIDE_CHANNEL_PORT = 5659

BLOCK_SIZE = 4096
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 1
TENSOR_PARALLEL_SIZE = 1
PREFILL_GPU_MEMORY_UTILIZATION = "0.20"

DEFAULT_PROMPT = "What is 2 + 10 ?"
DEFAULT_MAX_TOKENS = 8
DEFAULT_TEMPERATURE = 0.0

LOG_ROOT = Path(
    os.environ.get("PDD_LOG_ROOT", SCRIPT_DIR / "pdd_logs")
).expanduser().resolve()
READY_TIMEOUT_SEC = 20 * 60
HTTP_POLL_INTERVAL_SEC = 2.0
WAIT_STATUS_INTERVAL_SEC = 15.0
HTTP_GET_TIMEOUT_SEC = 5.0
COMPLETION_TIMEOUT_SEC = 10 * 60
TERMINATE_TIMEOUT_SEC = 30
LOG_TAIL_LINES = 80

MANAGED_PORTS = [
    PREFILL_PORT,
    DECODE_PORT,
    PROXY_PORT,
    PREFILL_NIXL_SIDE_CHANNEL_PORT,
    DECODE_NIXL_SIDE_CHANNEL_PORT,
]
MANAGED_PROCESS_MARKERS = (
    "vllm",
    "vllm::engine",
    "enginecore",
    "apiserver",
    "toy_proxy_server",
)

DECODE_KV_CONNECTOR_NAME = "RblnNixlConnector"



# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

VERBOSE = False
SHUTTING_DOWN = False


@dataclass
class ProcessInfo:
    name: str
    process: subprocess.Popen[str]
    log_path: Path
    log_file: Any
    thread: threading.Thread


PROCESS_INFO: dict[int, ProcessInfo] = {}


class ProcessDiedError(RuntimeError):
    def __init__(self, info: ProcessInfo, returncode: int):
        self.info = info
        self.returncode = returncode
        super().__init__(
            f"{info.name} exited unexpectedly with code {returncode}. "
            f"Log: {info.log_path}"
        )


def _stream_process_output(name: str, pipe: Any, log_file: Any) -> None:
    try:
        for line in iter(pipe.readline, ""):
            log_file.write(line)
            log_file.flush()
            if VERBOSE:
                sys.stdout.write(f"[{name}] {line}")
                sys.stdout.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass


def start_process(
    name: str,
    cmd: list[str],
    env: dict[str, str] | None,
    cwd: Path | str,
    log_path: Path | str,
) -> subprocess.Popen[str]:
    """Start one server process in the background and stream output to a log."""
    cwd = Path(cwd)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process_env = os.environ.copy()
    process_env.update({key: str(value) for key, value in (env or {}).items()})
    process_env.setdefault("PYTHONUNBUFFERED", "1")

    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    log_file.write(f"$ cd {cwd}\n")
    log_file.write(f"$ {shlex.join(cmd)}\n\n")
    log_file.flush()

    print(f"[start] {name}: log -> {log_path}")
    if VERBOSE:
        print(f"[start] {name}: {shlex.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise

    assert process.stdout is not None
    thread = threading.Thread(
        target=_stream_process_output,
        args=(name, process.stdout, log_file),
        daemon=True,
    )
    thread.start()

    PROCESS_INFO[process.pid] = ProcessInfo(
        name=name,
        process=process,
        log_path=log_path,
        log_file=log_file,
        thread=thread,
    )
    return process


def _tail_log(log_path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except FileNotFoundError:
        return f"(log file not found: {log_path})\n"
    except Exception as exc:
        return f"(could not read log file {log_path}: {exc})\n"


def _cmdline_for_pid(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _find_listening_port_owners(
    ports: list[int],
) -> list[tuple[int, int, str, str]]:
    try:
        result = subprocess.run(
            ["ss", "-ltnp"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return []

    wanted = set(ports)
    owners: list[tuple[int, int, str, str]] = []
    for line in result.stdout.splitlines():
        if "LISTEN" not in line:
            continue
        local_match = re.search(r"(?P<addr>\S+):(?P<port>\d+)\s+\S+:\*", line)
        if local_match is None:
            continue
        port = int(local_match.group("port"))
        if port not in wanted:
            continue

        for proc_match in re.finditer(r'"(?P<name>[^"]+)",pid=(?P<pid>\d+)', line):
            pid = int(proc_match.group("pid"))
            name = proc_match.group("name")
            owners.append((port, pid, name, _cmdline_for_pid(pid)))
    return owners


def _is_managed_process(name: str, cmdline: str) -> bool:
    haystack = f"{name} {cmdline}".lower()
    return any(marker in haystack for marker in MANAGED_PROCESS_MARKERS)


def cleanup_stale_managed_ports(ports: list[int]) -> None:
    owners = _find_listening_port_owners(ports)
    if not owners:
        return

    unmanaged = [
        (port, pid, name, cmdline)
        for port, pid, name, cmdline in owners
        if not _is_managed_process(name, cmdline)
    ]
    if unmanaged:
        details = "\n".join(
            f"- port {port}: pid={pid}, name={name}, cmd={cmdline or '(unknown)'}"
            for port, pid, name, cmdline in unmanaged
        )
        raise RuntimeError(
            "Required port is already in use by a non-vLLM process:\n"
            + details
            + "\nStop it manually or edit the port variables near the top "
            "of this script."
        )

    pids = sorted({pid for _, pid, _, _ in owners})
    print("[preflight] cleaning stale vLLM/proxy processes on managed ports:")
    for port, pid, name, cmdline in owners:
        print(f"[preflight] port {port}: pid={pid}, name={name}, cmd={cmdline}")

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        remaining = [
            owner
            for owner in _find_listening_port_owners(ports)
            if owner[1] in pids
        ]
        if not remaining:
            return
        time.sleep(0.5)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(1)
    remaining = [
        owner
        for owner in _find_listening_port_owners(ports)
        if owner[1] in pids
    ]
    if remaining:
        details = "\n".join(
            f"- port {port}: pid={pid}, name={name}, cmd={cmdline or '(unknown)'}"
            for port, pid, name, cmdline in remaining
        )
        raise RuntimeError("Could not clean stale managed port owners:\n" + details)


def _print_log_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> None:
    print(f"\n--- Last {lines} lines: {log_path} ---", file=sys.stderr)
    print(_tail_log(log_path, lines), file=sys.stderr, end="")
    print(f"--- End log tail: {log_path} ---\n", file=sys.stderr)


def _check_processes_alive() -> None:
    if SHUTTING_DOWN:
        return
    for info in list(PROCESS_INFO.values()):
        returncode = info.process.poll()
        if returncode is not None:
            raise ProcessDiedError(info, returncode)


def wait_for_http(name: str, url: str, timeout_sec: float) -> None:
    """Poll a GET endpoint until it returns HTTP 2xx."""
    print(f"[wait] {name}: polling {url}")
    deadline = time.monotonic() + timeout_sec
    next_status_at = 0.0
    last_error = "no response yet"

    while True:
        _check_processes_alive()

        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {name} at {url}. "
                f"Last error: {last_error}"
            )

        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(
                request,
                timeout=HTTP_GET_TIMEOUT_SEC,
            ) as response:
                status = response.getcode()
                if 200 <= status < 300:
                    print(f"[ready] {name}: HTTP {status}")
                    return
                last_error = f"HTTP {status}"
        except urllib.error.HTTPError as exc:
            body = exc.read(1000).decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body.strip()[:300]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        now = time.monotonic()
        if now >= next_status_at:
            elapsed = timeout_sec - max(0.0, deadline - now)
            remaining = max(0.0, deadline - now)
            print(
                f"[wait] {name}: still waiting "
                f"({elapsed:.0f}s elapsed, {remaining:.0f}s left). "
                f"Last error: {last_error}"
            )
            next_status_at = now + WAIT_STATUS_INTERVAL_SEC

        time.sleep(HTTP_POLL_INTERVAL_SEC)


def post_completion(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST one completion request and return the parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=COMPLETION_TIMEOUT_SEC,
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Completion request failed with HTTP {exc.code}:\n{body}"
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Completion response was not JSON:\n{raw}") from exc


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    """Terminate all started subprocesses, including their process groups."""
    global SHUTTING_DOWN
    SHUTTING_DOWN = True

    live_processes = [process for process in processes if process.poll() is None]
    if not live_processes:
        return

    print("[stop] terminating subprocesses...")

    for process in reversed(live_processes):
        info = PROCESS_INFO.get(process.pid)
        name = info.name if info else f"pid {process.pid}"
        print(f"[stop] SIGTERM -> {name}")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            process.terminate()

    deadline = time.monotonic() + TERMINATE_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in live_processes):
            break
        time.sleep(0.5)

    still_live = [process for process in live_processes if process.poll() is None]
    for process in reversed(still_live):
        info = PROCESS_INFO.get(process.pid)
        name = info.name if info else f"pid {process.pid}"
        print(f"[stop] SIGKILL -> {name}")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            process.kill()

    for process in live_processes:
        try:
            process.wait(timeout=5)
        except Exception:
            pass

    for info in list(PROCESS_INFO.values()):
        try:
            info.thread.join(timeout=2)
        except Exception:
            pass


def _build_prefill_cmd(
    model: str,
    served_model_name: str,
    block_size: int,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> list[str]:
    kv_config = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
        "kv_buffer_device": "cuda",
        "kv_load_failure_policy": "fail",
    }
    cmd = [
        str(CUDA_VLLM_BIN),
        "serve",
        model,
        "--host",
        HOST,
        "--port",
        str(PREFILL_PORT),
        "--enforce-eager",
        "--block-size",
        str(block_size),
        "--gpu-memory-utilization",
        PREFILL_GPU_MEMORY_UTILIZATION,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--served-model-name",
        served_model_name,
        "--kv-transfer-config",
        json.dumps(kv_config, separators=(",", ":")),
    ]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    return cmd


def _build_decode_cmd(
    model: str,
    served_model_name: str,
    block_size: int,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> list[str]:
    kv_config = {
        "kv_connector": DECODE_KV_CONNECTOR_NAME,
        "kv_role": "kv_consumer",
        "kv_buffer_device": "cpu",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "remote_nixl_memory_type": "VRAM",
            "rbln_external_kv_format": "host_visible_hnd_to_runtime_private",
            "rbln_external_kv_source_dtype": "bfloat16",
        },
    }
    cmd = [
        str(RBLN_VLLM_BIN),
        "serve",
        model,
        "--host",
        HOST,
        "--port",
        str(DECODE_PORT),
        "--block-size",
        str(block_size),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--served-model-name",
        served_model_name,
        "--kv-transfer-config",
        json.dumps(kv_config, separators=(",", ":")),
    ]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    return cmd


def _build_proxy_cmd() -> list[str]:
    return [
        str(RBLN_PYTHON_BIN),
        str(PROXY_SCRIPT),
        "--port",
        str(PROXY_PORT),
        "--prefiller-hosts",
        HOST,
        "--prefiller-ports",
        str(PREFILL_PORT),
        "--decoder-hosts",
        HOST,
        "--decoder-ports",
        str(DECODE_PORT),
    ]


def _validate_config() -> None:
    required_dirs = {
        "sota/vllm-rbln repo": WORK_REPO,
        "vllm_rbln package in local repo": WORK_REPO / "vllm_rbln",
    }
    required_files = {
        "CUDA vLLM executable": CUDA_VLLM_BIN,
        "RBLN vLLM executable": RBLN_VLLM_BIN,
        "RBLN python executable": RBLN_PYTHON_BIN,
        "toy proxy server script": PROXY_SCRIPT,
    }

    errors: list[str] = []
    for label, path in required_dirs.items():
        if not path.is_dir():
            errors.append(f"- Missing {label}: {path}")

    for label, path in required_files.items():
        if not path.exists():
            errors.append(f"- Missing {label}: {path}")
        elif label.endswith("executable") and not os.access(path, os.X_OK):
            errors.append(f"- Not executable ({label}): {path}")

    if errors:
        hint = (
            "\nEdit the configuration variables near the top of this file if "
            "any path has moved."
        )
        raise FileNotFoundError(
            "Required path check failed:\n" + "\n".join(errors) + hint
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start prefill/decode/proxy servers and run one completion.",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--prefill-model",
        default=None,
        help="Model path/name for the CUDA prefill server. Defaults to --model.",
    )
    parser.add_argument(
        "--decode-model",
        default=None,
        help="Model path/name for the RBLN decode server. Defaults to --model.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="OpenAI API model name accepted by both servers. Defaults to --model.",
    )
    parser.add_argument("--prefill-block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--decode-block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--prefill-max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--decode-max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--prefill-max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--decode-max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep servers running after the test request. Ctrl+C stops them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also stream subprocess logs to this terminal.",
    )
    parser.add_argument(
        "--ready-timeout-sec",
        type=float,
        default=READY_TIMEOUT_SEC,
        help="Per-endpoint readiness timeout.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for this run's log files.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate paths and print resolved commands without starting servers.",
    )
    parser.add_argument(
        "--no-cleanup-stale",
        action="store_true",
        help="Do not terminate stale vLLM/proxy processes occupying managed ports.",
    )
    return parser.parse_args()


def _extract_completion_text(response_json: dict[str, Any]) -> str | None:
    try:
        text = response_json["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None
    return text if isinstance(text, str) else str(text)


def _save_completion_outputs(
    log_dir: Path,
    payload: dict[str, Any],
    response_json: dict[str, Any],
) -> None:
    text = _extract_completion_text(response_json)
    (log_dir / "completion_request.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (log_dir / "completion_response.json").write_text(
        json.dumps(response_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if text is not None:
        (log_dir / "completion_text.txt").write_text(text, encoding="utf-8")


def _print_completion_response(
    payload: dict[str, Any],
    response_json: dict[str, Any],
    log_dir: Path,
) -> None:
    prompt = payload.get("prompt", "")
    text = _extract_completion_text(response_json)

    print("\n========== Prompt ==========")
    print(prompt)

    print("\n========== Full response JSON ==========")
    print(json.dumps(response_json, indent=2, ensure_ascii=False))

    print("\n========== Generated text: choices[0].text ==========")
    if text is None:
        print("(choices[0].text was not found in the response)")
    else:
        print(text)

    print("\n========== Saved files ==========")
    print(log_dir / "completion_request.json")
    print(log_dir / "completion_response.json")
    if text is not None:
        print(log_dir / "completion_text.txt")
    print("=================================\n")


def _print_resolved_config(
    prefill_model: str,
    decode_model: str,
    served_model_name: str,
    prefill_block_size: int,
    decode_block_size: int,
    prefill_max_model_len: int | None,
    decode_max_model_len: int | None,
    prefill_max_num_seqs: int | None,
    decode_max_num_seqs: int | None,
    log_dir: Path,
) -> None:
    print(f"[config] prefill model: {prefill_model}")
    print(f"[config] decode model:  {decode_model}")
    print(f"[config] served model:  {served_model_name}")
    print(f"[config] prefill block size: {prefill_block_size}")
    print(f"[config] decode block size:  {decode_block_size}")
    print(f"[config] prefill max model len: {prefill_max_model_len}")
    print(f"[config] decode max model len:  {decode_max_model_len}")
    print(f"[config] prefill max num seqs: {prefill_max_num_seqs}")
    print(f"[config] decode max num seqs:  {decode_max_num_seqs}")
    print(f"[config] repo:  {WORK_REPO}")
    print(f"[config] proxy: {PROXY_SCRIPT}")
    print(f"[config] logs:  {log_dir}")
    print(f"[config] decode connector: {DECODE_KV_CONNECTOR_NAME}")
    print("[config] prefill command:")
    print(
        "  "
        + shlex.join(
            _build_prefill_cmd(
                prefill_model,
                served_model_name,
                prefill_block_size,
                prefill_max_model_len,
                prefill_max_num_seqs,
            )
        )
    )
    print("[config] decode command:")
    print(
        "  "
        + shlex.join(
            _build_decode_cmd(
                decode_model,
                served_model_name,
                decode_block_size,
                decode_max_model_len,
                decode_max_num_seqs,
            )
        )
    )
    print("[config] proxy command:")
    print("  " + shlex.join(_build_proxy_cmd()))


def main() -> int:
    global VERBOSE

    args = _parse_args()
    VERBOSE = args.verbose

    processes: list[subprocess.Popen[str]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or (LOG_ROOT / f"run_{timestamp}")

    prefill_url = f"http://{HOST}:{PREFILL_PORT}/v1/models"
    decode_url = f"http://{HOST}:{DECODE_PORT}/v1/models"
    proxy_health_url = f"http://{HOST}:{PROXY_PORT}/healthcheck"
    completion_url = f"http://{HOST}:{PROXY_PORT}/v1/completions"

    try:
        _validate_config()

        prefill_model = args.prefill_model or args.model
        decode_model = args.decode_model or args.model
        served_model_name = args.served_model_name or args.model

        _print_resolved_config(
            prefill_model,
            decode_model,
            served_model_name,
            args.prefill_block_size,
            args.decode_block_size,
            args.prefill_max_model_len,
            args.decode_max_model_len,
            args.prefill_max_num_seqs,
            args.decode_max_num_seqs,
            log_dir,
        )
        if args.check_config:
            print("[check-config] OK. No servers were started.")
            return 0

        if not args.no_cleanup_stale:
            cleanup_stale_managed_ports(MANAGED_PORTS)

        prefill_env = {
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "UCX_NET_DEVICES": "all",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(PREFILL_NIXL_SIDE_CHANNEL_PORT),
        }
        decode_env = {
            "VLLM_RBLN_USE_VLLM_MODEL": "1",
            "VLLM_RBLN_COMPILE_MODEL": "1",
            "RBLN_DEVICES": "0",
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "UCX_NET_DEVICES": "all",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(DECODE_NIXL_SIDE_CHANNEL_PORT),
        }
        proxy_env = {}

        processes.append(
            start_process(
                "prefill",
                _build_prefill_cmd(
                    prefill_model,
                    served_model_name,
                    args.prefill_block_size,
                    args.prefill_max_model_len,
                    args.prefill_max_num_seqs,
                ),
                prefill_env,
                RUN_CWD,
                log_dir / "prefill.log",
            )
        )
        processes.append(
            start_process(
                "decode",
                _build_decode_cmd(
                    decode_model,
                    served_model_name,
                    args.decode_block_size,
                    args.decode_max_model_len,
                    args.decode_max_num_seqs,
                ),
                decode_env,
                RUN_CWD,
                log_dir / "decode.log",
            )
        )
        processes.append(
            start_process(
                "proxy",
                _build_proxy_cmd(),
                proxy_env,
                RUN_CWD,
                log_dir / "proxy.log",
            )
        )

        wait_for_http("prefill", prefill_url, args.ready_timeout_sec)
        wait_for_http("decode", decode_url, args.ready_timeout_sec)
        wait_for_http("proxy", proxy_health_url, args.ready_timeout_sec)

        payload = {
            "model": served_model_name,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }

        print(f"[test] POST {completion_url}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        response_json = post_completion(completion_url, payload)
        _save_completion_outputs(log_dir, payload, response_json)
        _print_completion_response(payload, response_json, log_dir)

        if args.keep_running:
            print("[keep-running] servers are still running.")
            print("[keep-running] press Ctrl+C to terminate all subprocesses.")
            while True:
                _check_processes_alive()
                time.sleep(5)

        return 0

    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C received.")
        return 130
    except ProcessDiedError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        _print_log_tail(exc.info.log_path)
        return 1
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        if VERBOSE:
            traceback.print_exc()
        if processes:
            for process in processes:
                info = PROCESS_INFO.get(process.pid)
                if info is not None:
                    _print_log_tail(info.log_path, lines=40)
        return 1
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
