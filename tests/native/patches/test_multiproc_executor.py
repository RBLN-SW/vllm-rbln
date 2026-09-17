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

Every test here fails against upstream's ``shutdown`` / ``start_worker_monitor``
run on the same fake; none of them pins behaviour the patch merely inherits.
"""

import threading
from collections.abc import Callable
from threading import Thread

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

    # The patched shutdown calls it through `self`, as upstream does, so the
    # fake lives on the fake executor rather than on MultiprocExecutor.
    monkeypatch.setattr(
        _Executor,
        "_ensure_worker_termination",
        staticmethod(fake_termination),
        raising=False,
    )


def _wait_for(log, entry, timeout_s=5.0):
    deadline = threading.Event()
    for _ in range(int(timeout_s / 0.01)):
        if entry in log:
            return
        deadline.wait(0.01)
    raise AssertionError(f"{entry!r} never appeared: {log}")


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


def test_worker_queues_close_between_the_death_writers_and_the_wait(monkeypatch):
    """Where the early close lands: after the exit signal, before the wait.

    Upstream closes the worker queues only after the wait, which fails the
    right-hand side. Closing them *before* the death writers would wake the
    engine even earlier but fails the left-hand side: the death writer is the
    child's exit signal, so the workers would sit through the grace period.
    """
    log: list[str] = []
    _install_fake_termination(monkeypatch, log)
    ex = _Executor(log)

    mpx.patched_shutdown(ex)

    death = [i for i, e in enumerate(log) if e.startswith("mq:death")]
    wresp = [i for i, e in enumerate(log) if e.startswith("mq:wresp")]
    assert max(death) < min(wresp) < log.index("terminate:start"), log


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

    monitor = Thread(target=mpx.patched_shutdown, args=(ex,))
    monitor.start()
    _wait_for(log, "terminate:start")

    returned = threading.Event()

    def engine_teardown():
        mpx.patched_shutdown(ex)  # re-entrant: shutting_down is already True
        returned.set()

    Thread(target=engine_teardown).start()
    assert not returned.wait(0.3), (
        "re-entrant shutdown returned before the workers ended"
    )

    blocker.set()
    assert returned.wait(5), "re-entrant shutdown never returned"
    monitor.join(5)
    assert "terminate:done" in log


def test_reentrant_wait_is_bounded(monkeypatch):
    """A stuck termination wait must not wedge the engine's teardown forever.

    The re-entrant call must hold for the bound and then give up: returning at
    once is upstream's behaviour (it never waits), never returning is a wedge.
    """
    log: list[str] = []
    blocker = threading.Event()  # never set
    _install_fake_termination(monkeypatch, log, blocker=blocker)
    monkeypatch.setattr(mpx, "_termination_join_timeout_s", lambda: 0.5)
    ex = _Executor(log)

    Thread(target=mpx.patched_shutdown, args=(ex,), daemon=True).start()
    _wait_for(log, "terminate:start")

    done = threading.Event()

    def engine_teardown():
        mpx.patched_shutdown(ex)
        done.set()

    Thread(target=engine_teardown, daemon=True).start()
    assert not done.wait(0.2), "re-entrant shutdown did not wait at all"
    assert done.wait(3), "bounded wait did not expire"
    blocker.set()


def test_reentrant_wait_bound_tracks_the_grace_period(monkeypatch):
    """The bound follows VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS at call time.

    A fixed bound would expire before the SIGKILL once the grace period is
    raised past it, and the engine would exit first -- the case this patch
    exists to prevent.
    """
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
    base = mpx._termination_join_timeout_s()
    assert base == 5 + mpx._SIGTERM_WAIT_S + mpx._JOIN_SLACK_S

    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "26")
    raised = mpx._termination_join_timeout_s()
    assert raised == base + 21
    # Strictly after the SIGKILL, whatever the grace period is.
    assert raised > 26 + mpx._SIGTERM_WAIT_S


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


def test_engine_woken_by_callback_waits_for_the_termination(monkeypatch):
    """The path the engine actually takes, end to end through the monitor.

    The monitor fires, the callback wakes the engine, and the engine's own
    ``shutdown()`` -- re-entrant, from another thread -- must block until the
    ladder finishes. Upstream's monitor has not run the callback when the wait
    starts, so the first assertion fails there; a patch without the re-entrant
    wait fails the second.
    """
    log: list[str] = []
    blocker = threading.Event()
    _install_fake_termination(monkeypatch, log, blocker=blocker)
    ex = _Executor(log)

    engine_returned = threading.Event()

    def engine_teardown():
        mpx.patched_shutdown(ex)  # EngineCore.shutdown() in its `finally`
        engine_returned.set()

    def callback():
        log.append("callback")
        Thread(target=engine_teardown).start()

    ex.failure_callback = callback
    monkeypatch.setattr(
        mpx.multiprocessing.connection,
        "wait",
        lambda sentinels: [ex.workers[1].proc.sentinel],
    )
    Thread(
        target=mpx.patched_start_worker_monitor,
        args=(ex,),
        kwargs={"inline": True},
        daemon=True,
    ).start()
    _wait_for(log, "terminate:start")

    assert "callback" in log, log
    assert log.index("callback") < log.index("terminate:start"), log
    assert not engine_returned.wait(0.3), "engine outran the termination wait"

    blocker.set()
    assert engine_returned.wait(5), "engine teardown never returned"
    assert "terminate:done" in log
