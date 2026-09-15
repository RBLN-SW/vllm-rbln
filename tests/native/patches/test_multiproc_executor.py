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

"""Fail-close ordering: engine death is published before the workers end.

Pure logic: no worker process is started and no NPU is opened. A fake executor
records the order in which the patched ``shutdown`` touches the response queues
and the termination wait, which is the whole point of the patch -- upstream
waits first, so EngineCore stays parked on a response queue (and the API server
keeps answering 200) for the length of the grace / SIGTERM / SIGKILL ladder.
"""

import threading
from collections.abc import Callable

from vllm_rbln.patches import multiproc_executor as mpx


class _Queue:
    def __init__(self, log, name):
        self._log = log
        self._name = name
        self.shut = False

    def shutdown(self):
        self.shut = True
        self._log.append(f"mq:{self._name}")

    # death_writer.close() is the parent-side signal; record it the same way.
    close = shutdown


class _Proc:
    def __init__(self, name):
        self.name = name
        self.sentinel = object()
        self.exitcode = 70


class _Worker:
    def __init__(self, log, rank):
        self.proc = _Proc(f"VllmWorker-{rank}")
        self.death_writer = _Queue(log, f"death{rank}")
        self.worker_response_mq = _Queue(log, f"wresp{rank}")


class _Executor:
    """Just enough of MultiprocExecutor for the patched shutdown."""

    def __init__(self, log):
        self._log = log
        self.shutting_down = False
        self.is_failed = False
        self.failure_callback: Callable[[], None] | None = None
        self.workers = [_Worker(log, 0), _Worker(log, 1)]
        self.response_mqs = [_Queue(log, f"resp{i}") for i in range(2)]
        self.rpc_broadcast_mq = _Queue(log, "bcast")

    def shutdown(self):
        mpx.patched_shutdown(self)


def _install_fake_termination(monkeypatch, log, blocker=None):
    def fake_termination(procs):
        log.append("terminate:start")
        if blocker is not None:
            blocker.wait(timeout=5)
        log.append("terminate:done")

    monkeypatch.setattr(
        mpx.MultiprocExecutor,
        "_ensure_worker_termination",
        staticmethod(fake_termination),
    )


def test_response_queues_close_before_the_termination_wait(monkeypatch):
    """The whole point: EngineCore is woken before the ladder runs."""
    log: list[str] = []
    _install_fake_termination(monkeypatch, log)
    ex = _Executor(log)

    mpx.patched_shutdown(ex)

    assert "terminate:start" in log
    first_terminate = log.index("terminate:start")
    # Every queue EngineCore could be parked on is shut before the termination
    # wait begins. (The tail of shutdown() re-shuts self.response_mqs, as
    # upstream does, so compare the *first* close of each queue.)
    for name in ("mq:resp0", "mq:resp1", "mq:wresp0", "mq:wresp1"):
        assert name in log, (name, log)
        assert log.index(name) < first_terminate, (name, log)


def test_death_writers_close_first(monkeypatch):
    """Unchanged from upstream: children are signalled before anything else."""
    log: list[str] = []
    _install_fake_termination(monkeypatch, log)
    ex = _Executor(log)

    mpx.patched_shutdown(ex)

    death = [i for i, e in enumerate(log) if e.startswith("mq:death")]
    resp = [i for i, e in enumerate(log) if e.startswith("mq:resp")]
    assert max(death) < min(resp), log


def test_reentrant_shutdown_waits_for_the_termination(monkeypatch):
    """The dying engine's own teardown must not outrun the SIGKILL.

    A worker blocked in a collective only dies on the SIGKILL at the end of
    the ladder; if EngineCore's process exits first that worker is orphaned
    with its NPU context open.
    """
    log: list[str] = []
    blocker = threading.Event()
    _install_fake_termination(monkeypatch, log, blocker=blocker)
    ex = _Executor(log)

    monitor = threading.Thread(target=mpx.patched_shutdown, args=(ex,))
    monitor.start()
    # Wait until the monitor is inside the termination wait.
    for _ in range(500):
        if "terminate:start" in log:
            break
        threading.Event().wait(0.01)
    assert "terminate:start" in log

    returned = threading.Event()

    def engine_teardown():
        mpx.patched_shutdown(ex)  # re-entrant: shutting_down is already True
        returned.set()

    threading.Thread(target=engine_teardown).start()
    assert not returned.wait(0.3), (
        "re-entrant shutdown returned before the workers ended"
    )

    blocker.set()
    assert returned.wait(5), "re-entrant shutdown never returned"
    monitor.join(5)
    assert "terminate:done" in log


def test_reentrant_wait_is_bounded(monkeypatch):
    """A stuck termination wait must not wedge the engine's teardown forever."""
    log: list[str] = []
    blocker = threading.Event()  # never set
    _install_fake_termination(monkeypatch, log, blocker=blocker)
    monkeypatch.setattr(mpx, "_TERMINATION_JOIN_TIMEOUT_S", 0.2)
    ex = _Executor(log)

    threading.Thread(target=mpx.patched_shutdown, args=(ex,), daemon=True).start()
    for _ in range(500):
        if "terminate:start" in log:
            break
        threading.Event().wait(0.01)

    done = threading.Event()

    def engine_teardown():
        mpx.patched_shutdown(ex)
        done.set()

    threading.Thread(target=engine_teardown, daemon=True).start()
    assert done.wait(3), "bounded wait did not expire"
    blocker.set()


def test_failure_callback_runs_before_shutdown(monkeypatch):
    """An idle EngineCore is only woken by the callback, so it must come first."""
    log: list[str] = []
    _install_fake_termination(monkeypatch, log)
    ex = _Executor(log)
    ex.failure_callback = lambda: log.append("callback")

    # Drive the monitor body inline instead of spawning a thread.
    monkeypatch.setattr(
        mpx.multiprocessing.connection,
        "wait",
        lambda sentinels: [ex.workers[1].proc.sentinel],
    )
    mpx.patched_start_worker_monitor(ex, inline=True)

    assert ex.is_failed
    assert "callback" in log
    assert log.index("callback") < log.index("terminate:start"), log


def test_monitor_does_nothing_when_already_shutting_down(monkeypatch):
    log: list[str] = []
    _install_fake_termination(monkeypatch, log)
    ex = _Executor(log)
    ex.shutting_down = True
    ex.failure_callback = lambda: log.append("callback")

    monkeypatch.setattr(
        mpx.multiprocessing.connection,
        "wait",
        lambda sentinels: [ex.workers[0].proc.sentinel],
    )
    mpx.patched_start_worker_monitor(ex, inline=True)

    assert log == []
    assert not ex.is_failed
