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
"""Publish engine death when a worker dies, not after the workers are gone.

``MultiprocExecutor.shutdown`` closes the worker response queues only after
``_ensure_worker_termination`` has walked the grace / SIGTERM / SIGKILL ladder.
EngineCore's main thread is parked in ``MessageQueue.acquire_read`` on one of
those queues, so for that whole window it cannot reach its fatal handler, does
not publish ``ENGINE_CORE_DEAD``, and the API server answers 200. Under DP the
ranks that did not fault keep serving through it, so one fault yields a mix of
successes and failures instead of failing everything.

Closing the queues first collapses the window. ``_ensure_worker_termination``
still runs to completion, but the engine now races ahead of it, so ``shutdown``
is made re-entrant-blocking: a worker stuck in a collective dies only on the
SIGKILL at the end of that wait, and an EngineCore process that exits before
then orphans it with its NPU context open.
"""

# Standard
import multiprocessing.connection
import threading
import weakref
from threading import Thread

# Third Party
import vllm.envs as envs
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

# First Party
from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

# Upstream's `_ensure_worker_termination` waits the configurable grace period,
# then 4 s after SIGTERM, then SIGKILLs. The engine's wait on that ladder is
# derived from the same numbers when the wait begins, so raising the grace
# period cannot leave the engine free to exit before the SIGKILL.
_SIGTERM_WAIT_S = 4.0
_JOIN_SLACK_S = 5.0


def _termination_join_timeout_s() -> float:
    return envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS + _SIGTERM_WAIT_S + _JOIN_SLACK_S

_terminated_lock = threading.Lock()


@register_patch(
    target="vllm.v1.executor.multiproc_executor.MultiprocExecutor.shutdown",
    reason=(
        "Upstream closes the worker response queues only after it has waited "
        "for the worker processes to end, so EngineCore stays parked on one "
        "of them and cannot publish ENGINE_CORE_DEAD until that wait "
        "finishes. There is no hook between the two, and under DP the ranks "
        "that did not fault keep answering 200 for that window."
    ),
    key="vllm_rbln.patches.multiproc_executor.shutdown",
    owner_module="vllm_rbln.patches.multiproc_executor",
)
def patched_shutdown(self: MultiprocExecutor) -> None:
    with _terminated_lock:
        terminated: threading.Event = (
            getattr(self, "_rbln_terminated", None) or threading.Event()
        )
        self._rbln_terminated = terminated
        # Decided under the lock: the engine thread, woken by the failure
        # callback, can arrive here while the monitor thread is still between
        # creating the Event and setting the flag.
        first_caller = not getattr(self, "shutting_down", False)
        if first_caller:
            self.shutting_down = True

    if first_caller:
        workers = getattr(self, "workers", None)
        logger.debug("[shutdown] Executor: start worker_count=%d", len(workers or []))

        if workers:
            for w in workers:
                if w.death_writer is not None:
                    w.death_writer.close()
                    w.death_writer = None

            # Ahead of the termination wait, unlike upstream: acquire_read
            # raises once a queue is marked shutting down, which is what lets
            # EngineCore reach its fatal handler now rather than after it.
            for mq in list(getattr(self, "response_mqs", None) or []):
                mq.shutdown()
            for w in workers:
                if w.worker_response_mq is not None:
                    w.worker_response_mq.shutdown()
                    w.worker_response_mq = None

            try:
                self._ensure_worker_termination([w.proc for w in workers])
            finally:
                terminated.set()
        else:
            terminated.set()

    elif not terminated.is_set():
        # Re-entrant call from EngineCore.shutdown() in the dying engine's
        # own `finally`, which now runs while the monitor thread is still
        # ending the workers. Wait, or the process exits before the SIGKILL.
        timeout_s = _termination_join_timeout_s()
        if not terminated.wait(timeout=timeout_s):
            logger.warning(
                "[shutdown] Executor: workers still being terminated after "
                "%.0fs; continuing engine teardown",
                timeout_s,
            )

    if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
        rpc_broadcast_mq.shutdown()
        self.rpc_broadcast_mq = None
    if response_mqs := getattr(self, "response_mqs", None):
        for mq in response_mqs:
            mq.shutdown()
        self.response_mqs = []

    logger.debug_once("[shutdown] Executor: complete")


@register_patch(
    target="vllm.v1.executor.multiproc_executor.MultiprocExecutor.start_worker_monitor",
    reason=(
        "Upstream calls the engine failure callback after shutdown(), which "
        "blocks until the workers have ended. The callback is the only thing "
        "that wakes an idle EngineCore, so until it runs the engine stays "
        "alive and serving even though the executor already knows a worker "
        "is gone."
    ),
    key="vllm_rbln.patches.multiproc_executor.start_worker_monitor",
    owner_module="vllm_rbln.patches.multiproc_executor",
)
def patched_start_worker_monitor(self: MultiprocExecutor, inline: bool = False) -> None:
    workers = self.workers
    self_ref = weakref.ref(self)

    def monitor_workers() -> None:
        sentinels = [h.proc.sentinel for h in workers]
        died = multiprocessing.connection.wait(sentinels)
        _self = self_ref()
        if not _self or getattr(_self, "shutting_down", False):
            logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
            return
        _self.is_failed = True
        proc_name = next(h.proc.name for h in workers if h.proc.sentinel == died[0])
        logger.error(
            "Worker proc %s died unexpectedly, shutting down executor.",
            proc_name,
        )
        callback = _self.failure_callback
        if callback is not None:
            _self.failure_callback = None
            callback()
        _self.shutdown()

    if not inline:
        Thread(
            target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
        ).start()
        return

    monitor_workers()
