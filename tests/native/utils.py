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

"""Shared helpers for the native (``VLLM_RBLN_USE_VLLM_MODEL=1``) suite."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec

# What "native" means: the switch that selects vLLM modelling over optimum.
# Deliberately nothing else -- pinning a knob to the value the source already
# defaults to would mean a flipped default goes unnoticed.
NATIVE_ENV = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_RBLN_USE_VLLM_MODEL": "1",
}

# Host description -- which NPUs exist, what SOC to compile for, OMP threads.
# CI and developer machines legitimately set these; scrubbing them would break
# device selection and make compile-only hosts fail in get_device_name().
HOST_ENV_PASSTHROUGH = frozenset(
    {
        "RBLN_DEVICES",
        "RBLN_FORCE_NPU_NAME",
        "RBLN_TARGET_SOC",
        "RBLN_NPUS_PER_DEVICE",
        "RBLN_NUM_THREADS",
    }
)

SCRUBBED_PREFIXES = ("VLLM_RBLN_", "RBLN_")

# Upstream knobs RblnPlatform itself branches on, so they are part of the
# native suite's input surface despite not being RBLN-named.
SCRUBBED_EXTRA = frozenset(
    {
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_DISABLE_COMPILE_CACHE",
        "VLLM_WORKER_MULTIPROC_METHOD",
    }
)


def is_scrubbed(key: str) -> bool:
    """Whether ``key`` is a behavior knob the suite must control itself."""
    if key in HOST_ENV_PASSTHROUGH:
        return False
    # The harness's own spawn-control vars (VLLM_RBLN_TEST_SPAWN_*) are not RBLN
    # behavior knobs; scrubbing them in a spawned child would drop the re-entry
    # guard and make the child spawn itself forever.
    if key.startswith("VLLM_RBLN_TEST_"):
        return False
    return key.startswith(SCRUBBED_PREFIXES) or key in SCRUBBED_EXTRA


def scrub_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Drop every behavior knob from ``environ`` in place and return what was
    dropped, so ``vllm_rbln.envs`` takes its defaults regardless of the shell."""
    target = os.environ if environ is None else environ
    removed = {key: target[key] for key in list(target) if is_scrubbed(key)}
    for key in removed:
        del target[key]
    return removed


# (token_ids, text) from a greedy run; the second form adds per-step top-k
# {token_id: logprob} for the tolerant comparison.
TokensText = tuple[list[int], str]
TokensTextLogprobs = tuple[list[int], str, list[dict[int, float]]]


def check_outputs_equal(
    *,
    outputs_0_lst: list[TokensText],
    outputs_1_lst: list[TokensText],
    name_0: str,
    name_1: str,
) -> None:
    """Assert two runs produced identical token ids for every prompt."""
    assert len(outputs_0_lst) == len(outputs_1_lst)
    for i, ((ids_0, text_0), (ids_1, text_1)) in enumerate(
        zip(outputs_0_lst, outputs_1_lst)
    ):
        if ids_0 != ids_1:
            raise AssertionError(
                f"prompt {i}:\n  {name_0}: {text_0!r} {ids_0}"
                f"\n  {name_1}: {text_1!r} {ids_1}"
            )


# Max logprob gap (nats) between the two picks for a divergence to count as a
# near-tie: ~0.5 nats means the preferred token was at most ~1.65x likelier.
NEAR_TIE_MAX_LOGPROB_GAP = 0.5


class NearTieWarning(UserWarning):
    """A greedy pick differed but both runs rated the two tokens near-equal."""


def _tie_gap(topk: dict[int, float], *, own: int, other: int) -> float | None:
    """``logprob(own) - logprob(other)`` within one run's top-k; None if absent."""
    if own not in topk or other not in topk:
        return None
    return topk[own] - topk[other]


def check_logprobs_close(
    *,
    outputs_0_lst: list[TokensTextLogprobs],
    outputs_1_lst: list[TokensTextLogprobs],
    name_0: str,
    name_1: str,
    max_logprob_gap: float = NEAR_TIE_MAX_LOGPROB_GAP,
) -> None:
    """Assert two runs agree, tolerating only genuine near-tie flips: at the
    first differing token each run must rank the other's pick within its top-k
    and within ``max_logprob_gap`` nats (else fail). Stops at first divergence."""
    assert len(outputs_0_lst) == len(outputs_1_lst)
    for i, ((ids_0, _, lps_0), (ids_1, _, lps_1)) in enumerate(
        zip(outputs_0_lst, outputs_1_lst)
    ):
        for pos in range(min(len(ids_0), len(ids_1))):
            t_0, t_1 = ids_0[pos], ids_1[pos]
            if t_0 == t_1:
                continue
            gap_0 = _tie_gap(lps_0[pos], own=t_0, other=t_1)
            gap_1 = _tie_gap(lps_1[pos], own=t_1, other=t_0)
            if gap_0 is None or gap_1 is None or max(gap_0, gap_1) > max_logprob_gap:
                raise AssertionError(
                    f"prompt {i} token {pos}: {name_0} picked {t_0}, {name_1} "
                    f"picked {t_1}; not a near-tie (gap {name_0}={gap_0}, "
                    f"{name_1}={gap_1} nats, threshold {max_logprob_gap})"
                )
            warnings.warn(
                f"prompt {i} token {pos}: near-tie flip -- {name_0} picked {t_0}, "
                f"{name_1} picked {t_1} (gap {max(gap_0, gap_1):.3f} nats)",
                NearTieWarning,
                stacklevel=2,
            )
            break


# -- per-test process isolation -------------------------------------------------
# Ported from vLLM tests/utils.py (issue #41415). The RBLN SDK accumulates
# device/runtime state across LLM instantiations in one process, so each
# engine-using test must run in a fresh interpreter (dev-legacy #629:
# "rebellions SDK somewhat requires LLM instance to be instantiated in a
# separated process"). spawn (not fork) is required: fork would inherit the
# parent's already-loaded RBLN runtime.

_P = ParamSpec("_P")
_SPAWN_CHILD_ENV = "VLLM_RBLN_TEST_SPAWN_CHILD"
# Path the module-batch child writes per-test outcomes to (one JSON obj/line),
# so the parent can attribute results back to each test of the single spawn.
_SPAWN_RESULTS_ENV = "VLLM_RBLN_TEST_SPAWN_RESULTS"


def _format_subprocess_exit(returncode: int) -> str:
    if returncode >= 0:
        return f"exit code {returncode}"
    try:
        return f"killed by {signal.Signals(-returncode).name} ({returncode})"
    except ValueError:
        return f"exit code {returncode}"


_CHILD_SCRIPT = (
    "import sys, importlib, traceback, cloudpickle\n"
    "try:\n"
    "    from _pytest.outcomes import Skipped\n"
    "except ImportError:\n"
    "    class Skipped(BaseException): pass\n"
    # conftest only runs in the parent; re-apply the RBLN plugins so patched
    # symbols are in place before the test module imports.
    "from vllm.plugins import load_general_plugins\n"
    "load_general_plugins()\n"
    "data = cloudpickle.loads(sys.stdin.buffer.read())\n"
    "mod = importlib.import_module(data['module'])\n"
    "parts = data['qualname'].split('.')\n"
    "target = mod\n"
    "for i, name in enumerate(parts):\n"
    "    target = getattr(target, name)\n"
    # instantiate a test class so its method binds (a bare function/method
    # otherwise misses `self`); module-level test functions skip this.
    "    if isinstance(target, type) and i < len(parts) - 1:\n"
    "        target = target()\n"
    "try:\n"
    "    target(*data['args'], **data['kwargs'])\n"
    "except Skipped:\n"
    "    sys.exit(0)\n"
    "except BaseException:\n"
    "    open(data['tb_file'], 'w').write(traceback.format_exc())\n"
    "    sys.exit(1)\n"
)


def run_test_in_spawned_process(
    module: str, qualname: str, args: tuple, kwargs: dict, label: str
) -> None:
    """Run ``module.qualname`` in a fresh spawned interpreter with cloudpickled
    args, re-raising the child's traceback on failure. The child re-applies the
    RBLN general plugins and inherits the parent's sys.path/env."""
    import cloudpickle

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tb", mode="wb") as tmp:
        tb_file = tmp.name
    try:
        payload = cloudpickle.dumps(
            {
                "module": module,
                "qualname": qualname,
                "args": args,
                "kwargs": kwargs,
                "tb_file": tb_file,
            }
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]
        )
        env[_SPAWN_CHILD_ENV] = "1"
        # start_new_session -> the child leads its own process group, so its
        # EngineCore grandchildren land in that group. After it exits, SIGKILL
        # the whole group so a leaked/crashed EngineCore cannot keep holding the
        # NPU for the next test (mirrors vLLM's fork setpgrp + killpg).
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT],
            stdin=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            proc.communicate(input=payload)
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        if proc.returncode != 0:
            import pytest

            try:
                tb = Path(tb_file).read_text()
            except OSError:
                tb = ""
            # The child's exception can't cross the process boundary as a live
            # object, so surface its traceback text instead of wrapping it in a
            # RuntimeError raised from this frame. pytrace=False hides this harness
            # frame; leading with the child's final line puts the real cause (its
            # type + message) in the one-line short summary.
            cause = (
                tb.strip().splitlines()[-1]
                if tb.strip()
                else f"{_format_subprocess_exit(proc.returncode)}; no traceback"
            )
            pytest.fail(
                f"[{label}] {cause}\n\n"
                + (tb or "<no Python traceback; see subprocess output above>"),
                pytrace=False,
            )
    finally:
        with contextlib.suppress(OSError):
            os.remove(tb_file)


# How often the parent re-reads the child's results file while waiting for the
# test it is currently reporting. Small enough to feel live, coarse enough that
# a 30-minute compile costs a negligible number of stat/read syscalls.
_SPAWN_POLL_INTERVAL_S = 0.2


class ModuleSpawn:
    """A running ``pytest <module>`` child, consumed one test at a time.

    The child appends a JSON line per event as it goes (see the native conftest's
    logreport hook), so the parent never has to wait for the whole file:
    :meth:`records_for` blocks only until *that* test's phase reports land.
    Waiting for the process instead would make the module's first test absorb the
    entire run, freezing the progress display until the file is done.

    Records are passed through as raw dicts. They are pytest's own serialized
    ``TestReport``s (``pytest_report_to_serializable``); the parent revives and
    re-emits them so the report the user sees is the child's real one, not a
    reconstruction -- which is what keeps the output identical to stock pytest.
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        results_file: str,
        log_path: str | None,
        nodeid_prefix: str = "",
    ) -> None:
        self._proc = proc
        self._results_file = results_file
        self.log_path = log_path
        # This module's rootdir-relative path. The child keys everything by the
        # intra-module nodeid (robust to how the file was addressed on its
        # command line), so the parent puts the prefix back to name a test.
        self.nodeid_prefix = nodeid_prefix
        # nodeid -> its phase reports in the order the child emitted them.
        self._reports: dict[str, list[dict]] = {}
        # nodeids whose teardown report arrived, i.e. that are fully reported.
        self._complete: set[str] = set()
        self._warnings: list[dict] = []
        self._offset = 0
        self._partial = ""
        self._reaped = False

    def records_for(self, nodeid: str) -> list[dict] | None:
        """Block until the child has fully reported ``nodeid`` and return its
        phase reports in order. None if the child exited before finishing it.

        The teardown report is the terminator, and a partial set counts as no
        result: a child that dies *inside* a test has already emitted that
        test's setup report, and emitting that alone would report neither a pass
        nor a failure -- the crashed test would vanish from the run entirely."""
        while True:
            self._drain()
            if nodeid in self._complete:
                return self._reports[nodeid]
            if self._reaped:
                return None
            if self._proc.poll() is not None:
                # Exited. Pick up anything written between the last read and the
                # exit before concluding this test was never fully reported.
                self._drain()
                self._reap()
                return self._reports[nodeid] if nodeid in self._complete else None
            time.sleep(_SPAWN_POLL_INTERVAL_S)

    def take_warnings(self) -> list[dict]:
        """Warnings the child has recorded since the last call, so the parent
        can re-emit them into its own warnings summary."""
        self._drain()
        pending, self._warnings = self._warnings, []
        return pending

    def finish(self) -> None:
        """Wait out the child and reap it. The child runs the WHOLE file even
        when the parent selected a subset, so it can still be driving the NPU
        after the parent's last selected test of that module -- the next module
        must not spawn on top of it."""
        if self._reaped:
            return
        with contextlib.suppress(OSError, ValueError):
            self._proc.wait()
        self._drain()
        self._reap()

    def close(self) -> None:
        """Finish, then drop the results file."""
        self.finish()
        with contextlib.suppress(OSError):
            os.remove(self._results_file)

    def _reap(self) -> None:
        # The child leads its own process group, so its EngineCore grandchildren
        # land in that group too: SIGKILL the group so a leaked or crashed one
        # cannot keep holding the NPU (mirrors vLLM's fork setpgrp + killpg).
        self._reaped = True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._proc.pid, signal.SIGKILL)

    def _drain(self) -> None:
        """Parse whatever whole lines have appeared since the last read. Read in
        binary and keep the trailing fragment: the parent can look at the file
        mid-write, and half a JSON line must not be parsed or skipped."""
        try:
            with open(self._results_file, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return
        lines = (self._partial + chunk.decode("utf-8", errors="replace")).split("\n")
        self._partial = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            with contextlib.suppress(ValueError, KeyError):
                rec = json.loads(line)
                if rec["kind"] == "warning":
                    self._warnings.append(rec)
                    continue
                nodeid = rec["nodeid"]
                self._reports.setdefault(nodeid, []).append(rec["report"])
                if rec["when"] == "teardown":
                    self._complete.add(nodeid)


def start_module_in_spawned_process(
    nodeids: list[str],
    *,
    nodeid_prefix: str = "",
    device_tensor: str | None,
    model_compile: bool,
    tb_style: str | None = None,
    maxfail: int = 0,
    stream_output: bool = False,
) -> ModuleSpawn:
    """Start one fresh spawned pytest over exactly ``nodeids`` (all from the
    same file) and return the handle the parent reads results from as they land.

    Lets a whole device-touching file share ONE spawn instead of spawning per
    test: the child is a real pytest run (so it resolves fixtures,
    parametrization and skips itself) with the spawn-child guard set, and the
    native conftest's logreport hook serializes each phase report to a file.

    The parent passes the tests it actually selected rather than the file, so
    -m/-k/--deselect mean what they say: handing over the file would run the
    deselected tests too (on the NPU, invisibly), and would re-run any unmarked
    test of a mixed file that the parent is already running in-process.
    --device-tensor / --model-compile are forwarded so the child's skip/run
    decisions match the parent's, and --tb / --maxfail so the reports it builds
    are rendered and cut off the way the parent was asked to.

    The child's raw stream is redirected to a log file rather than inherited.
    Inheriting it would put the WHOLE file's output -- every EngineCore
    traceback included -- into the parent's capture buffer for the *first* test
    of the module (the one whose pyfunc_call triggered this spawn), which pytest
    then discards when that test passes. Per-test output reaches the parent via
    each record's sections instead; the log file is the fallback for a child
    that dies without recording anything. ``stream_output`` (parent run with
    -s) inherits the stream instead, since nothing is capturing it then."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as tmp:
        results_file = tmp.name
    log_path: str | None = None
    if not stream_output:
        with tempfile.NamedTemporaryFile(
            delete=False, prefix="native-spawn-", suffix=".log", mode="w"
        ) as tmp:
            log_path = tmp.name

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]
    )
    env[_SPAWN_CHILD_ENV] = "1"
    env[_SPAWN_RESULTS_ENV] = results_file
    args = [
        sys.executable,
        "-m",
        "pytest",
        *nodeids,
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    if device_tensor is not None:
        args += ["--device-tensor", device_tensor]
    if model_compile:
        args.append("--model-compile")
    if tb_style:
        args.append(f"--tb={tb_style}")
    if maxfail:
        args.append(f"--maxfail={maxfail}")
    if stream_output:
        # Forward -s, or the child would still capture per test and the parent
        # would just stream the child's *report*. Turning the child's capture
        # off is the only way to see output from a test that dies hard (its
        # capture buffer dies with it). Costs the per-test sections -- fine,
        # since it is all on the terminal live.
        #
        # Silence the child's summary sections while we are at it. Its report is
        # now byte-identical to the one the parent re-emits, so leaving them on
        # prints every failure twice and reads like the test ran twice. Only the
        # rendering goes -- the reports themselves are still built and
        # serialized. (Not `-p no:terminal`: --tb, -q and friends are registered
        # by that very plugin, so disabling it both rejects those options and
        # drops the tb style the parent's traceback is built with.)
        args += ["-s", "--no-header", "--no-summary"]

    # The log handle can be closed right after the spawn: the child holds its
    # own dup of the fd for as long as it runs.
    with contextlib.ExitStack() as stack:
        log_fh = (
            stack.enter_context(open(log_path, "w", encoding="utf-8"))
            if log_path
            else None
        )
        proc = subprocess.Popen(
            args,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT if log_fh else None,
            start_new_session=True,
        )
    return ModuleSpawn(proc, results_file, log_path, nodeid_prefix)


def read_log_tail(log_path: str | None, max_lines: int = 50) -> str:
    """Last ``max_lines`` of the child's log, for a child that died without
    recording a result (a crash mid-test leaves nothing else to go on)."""
    if not log_path:
        return "<child output was streamed to the terminal (-s)>"
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return f"<child log unreadable: {log_path}>"
    tail = lines[-max_lines:]
    elided = len(lines) - len(tail)
    header = f"... ({elided} earlier lines elided)\n" if elided else ""
    return header + "\n".join(tail)


def spawn_new_process_for_each_test(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator form. Usually unnecessary -- the conftest auto-spawns every
    @model_compile test -- but available for explicit use."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        if os.environ.get(_SPAWN_CHILD_ENV) == "1":
            return f(*args, **kwargs)
        run_test_in_spawned_process(
            f.__module__, f.__qualname__, args, kwargs, f.__name__
        )

    return wrapper


def create_new_process_for_each_test(
    method: str = "spawn",
) -> Callable[[Callable[_P, None]], Callable[_P, None]]:
    """Decorator factory. RBLN only supports spawn -- fork would inherit the
    parent's already-loaded RBLN runtime and defeat the isolation."""
    assert method == "spawn", "the native suite only supports method='spawn'"
    return spawn_new_process_for_each_test
