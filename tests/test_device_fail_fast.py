# SPDX-License-Identifier: Apache-2.0
"""Worker fail-fast: an exception in a multi-process worker ends the process.

Pure logic: no NPU is opened and no process is ended -- ``os._exit`` is patched
so the test observes the exit code the worker would have used.
"""

import pytest

from vllm_rbln.v1.worker import device_fail_fast
from vllm_rbln.v1.worker.device_fail_fast import (
    WORKER_DEVICE_FAILURE_EXIT_CODE,
    fail_fast_on_device_error,
)


@pytest.fixture
def exits(monkeypatch):
    """Collect exit codes instead of ending the test process."""
    codes: list[int] = []

    def fake_exit(code):
        codes.append(code)
        raise SystemExit(code)  # mimic os._exit: control never returns

    monkeypatch.setattr(device_fail_fast.os, "_exit", fake_exit)
    return codes


class _Worker:
    """Just enough of RBLNWorker for the decorator: the parallel config."""

    def __init__(self, backend, error):
        parallel = type("Parallel", (), {"distributed_executor_backend": backend})()
        self.vllm_config = type("Config", (), {"parallel_config": parallel})()
        self._error = error

    @fail_fast_on_device_error
    def execute_model(self):
        if self._error is not None:
            raise self._error
        return "output"


# The runtime reports an aborted context as a plain RuntimeError; a software
# bug arrives the same way. Both end the process: that is the single-worker
# contract, and neither lets the step complete.
ERRORS = [
    RuntimeError("SysError(125): [cmd_dispatcher] SubmitJob() failed, rc=125."),
    RuntimeError("SysError(5): [cmd_dispatcher] Failed to WaitForCompletion, seq=283."),
    RuntimeError("Logical device rbln: 1 is not assigned (1 logical device(s))."),
    ValueError("unrelated software error"),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: str(e)[:24])
def test_multiproc_worker_ends_its_process_on_any_exception(exits, error):
    with pytest.raises(SystemExit):
        _Worker("mp", error).execute_model()
    assert exits == [WORKER_DEVICE_FAILURE_EXIT_CODE]


def test_in_engine_worker_lets_the_exception_reach_the_engine(exits):
    """With the uni executor the exception already kills the replica via EngineCore."""
    with pytest.raises(RuntimeError):
        _Worker("uni", RuntimeError("SubmitJob() failed")).execute_model()
    assert exits == []


def test_unknown_layout_prefers_ending_the_process(exits):
    @fail_fast_on_device_error
    def execute_model():
        raise RuntimeError("no self, no config")

    with pytest.raises(SystemExit):
        execute_model()
    assert exits == [WORKER_DEVICE_FAILURE_EXIT_CODE]


def test_success_path_is_untouched(exits):
    assert _Worker("mp", None).execute_model() == "output"
    assert exits == []


def test_can_be_disabled_for_debugging(exits, monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_FAIL_FAST_ON_DEVICE_ERROR", "0")
    with pytest.raises(RuntimeError):
        _Worker("mp", RuntimeError("SubmitJob() failed")).execute_model()
    assert exits == []
