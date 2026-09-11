# SPDX-License-Identifier: Apache-2.0
"""End the worker process when model execution raises in a multi-process layout.

Why this module exists
----------------------
With a single worker (``uni`` executor) vLLM already treats any exception from
``execute_model`` / ``sample_tokens`` as fatal: it propagates into EngineCore,
the engine logs a fatal error and shuts down, in-flight requests get 500 and
the replica exits so the orchestrator can restart it.

With separate worker processes (``mp`` executor) the same exception does not
get there:

* ``WorkerProc.worker_busy_loop`` catches every ``Exception``, logs it and keeps
  looping, so the worker process stays alive.
* The FAILURE reply to ``execute_model`` is parked in a future and only
  re-raised when the paired ``sample_tokens`` reply is ``None``; a worker whose
  ``execute_model`` raised answers ``sample_tokens`` with an empty
  ``ModelRunnerOutput`` instead, so the exception is dropped.
* The healthy ranks meanwhile block inside the forward collective waiting for
  the failed rank, and the engine waits for their reply until
  ``VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`` (300 s) expires, while ``/health``
  still answers 200.

An aborted ("guilty") NPU context is the case that matters: after the firmware
aborts the context every later submission is rejected, so the worker can never
make progress again. The executor learns about a broken worker in exactly one
reliable way, the worker *process* dying (``MultiprocExecutor``'s worker
monitor watches the process sentinel). This module makes that happen.

Policy
------
Any exception escaping a worker RPC entry point ends the process. No attempt is
made to classify the error: that is the single-worker contract, and a worker
that cannot execute a step is not serving anyway. Recoverable conditions must be
handled below this layer (LMCache turns RDS read/write failures into cache
misses; vLLM's ``kv_load_failure_policy`` decides whether a miss is recomputed).
The hard exit is limited to the ``mp`` executor; under ``uni`` the exception
already reaches the engine directly and ending the process early would only lose
the engine's diagnostics.
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

# Exit code for a worker that aborts itself. Distinct from vLLM's own codes so
# the reason is visible in the parent's log and in a container exit status.
WORKER_DEVICE_FAILURE_EXIT_CODE = 70


def _enabled() -> bool:
    """On by default; set the env var to 0/false/no to keep the old behaviour.

    Read per call rather than cached so a running deployment can be flipped by
    restarting a replica with a different value, and so tests can toggle it.
    """
    return os.environ.get("VLLM_RBLN_FAIL_FAST_ON_DEVICE_ERROR", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _worker_runs_in_its_own_process(worker: Any) -> bool:
    """True when ``worker`` is driven over RPC by a separate engine process.

    Read from the parallel config: vLLM resolves the executor backend to ``mp``
    for multi-worker layouts (including every DP rank) and ``uni`` when the
    single worker is hosted by EngineCore itself. When the layout cannot be
    determined the answer is True, because ending the process is the only
    failure signal that is guaranteed to be acted on.
    """
    try:
        backend = worker.vllm_config.parallel_config.distributed_executor_backend
    except AttributeError:
        return True
    return backend != "uni"


def abort_worker(exc: BaseException, *, where: str) -> None:
    """Log why this worker is stopping, then end the process immediately.

    ``os._exit`` rather than ``raise SystemExit``: the normal teardown path runs
    the KV-connector and offload cleanup, and that cleanup can itself block on
    the very device that just stopped accepting work. Skipping it is safe here
    because the driver reclaims the context when the process goes away, which
    is what the parent is waiting to observe.
    """
    logger.error(
        "RBLN worker %d: %s raised %s: %s. Ending this worker process with exit "
        "code %d so the executor sees the failure and the engine shuts down "
        "instead of waiting for a reply that will never come.",
        os.getpid(),
        where,
        type(exc).__name__,
        exc,
        WORKER_DEVICE_FAILURE_EXIT_CODE,
        exc_info=exc,
    )
    logging.shutdown()  # flush handlers; os._exit skips atexit and buffers
    os._exit(WORKER_DEVICE_FAILURE_EXIT_CODE)


def fail_fast_on_device_error(function: _F) -> _F:
    """Wrap a worker RPC entry point so an exception ends the process."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            worker = args[0] if args else None
            if _enabled() and _worker_runs_in_its_own_process(worker):
                abort_worker(exc, where=function.__qualname__)
            raise

    return guarded  # type: ignore[return-value]
