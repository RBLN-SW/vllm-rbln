# SPDX-License-Identifier: Apache-2.0
"""RBLNModelRunner KV-connector preemption call-convention test.

vLLM 0.18 -> 0.22 changed ``KVConnector.handle_preemptions`` from taking a
``preempted_req_ids: set[str]`` to taking the connector metadata
(``MultiConnector`` asserts ``MultiKVConnectorMetadata``). RBLNModelRunner.
execute_model still uses the 0.18 convention and passes
``scheduler_output.preempted_req_ids`` (a set), so on any real KV-cache
preemption the 0.22 MultiConnector raises AssertionError.

These tests reproduce that with dummy sub-connectors (no NIXL / serve needed):
  - BEFORE: passing the set -> AssertionError.
  - AFTER:  passing MultiKVConnectorMetadata -> dispatched to each sub-connector.
"""

# Third Party
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)


class _DummyConn:
    def __init__(self) -> None:
        self.received = "UNSET"

    def handle_preemptions(self, arg) -> None:  # type: ignore[no-untyped-def]
        self.received = arg


class _DummyMeta(KVConnectorMetadata):
    pass


def _multi(children):  # type: ignore[no-untyped-def]
    # Bypass __init__ (needs a full VllmConfig); handle_preemptions only reads
    # self._connectors, so this exercises the real 0.22 assert logic.
    mc = MultiConnector.__new__(MultiConnector)
    mc._connectors = list(children)
    return mc


def test_multiconnector_rejects_preempted_reqids_set():
    """BEFORE fix: the runner passes scheduler_output.preempted_req_ids (a set);
    0.22 MultiConnector.handle_preemptions asserts MultiKVConnectorMetadata."""
    mc = _multi([_DummyConn(), _DummyConn()])
    with pytest.raises(AssertionError):
        mc.handle_preemptions({"req-0", "req-1"})


def test_multiconnector_accepts_metadata():
    """AFTER fix: passing kv_connector_metadata dispatches per sub-connector."""
    c0, c1 = _DummyConn(), _DummyConn()
    mc = _multi([c0, c1])
    m0, m1 = _DummyMeta(), _DummyMeta()
    mc.handle_preemptions(MultiKVConnectorMetadata(metadata=(m0, m1)))
    assert c0.received is m0
    assert c1.received is m1


# --- RBLNModelRunner._handle_kv_connector_preemptions (the fix) ---------------
# Third Party
import types  # noqa: E402

# First Party
import vllm_rbln.v1.worker.rbln_model_runner as rmr  # noqa: E402

_RUNNER_FIX = rmr.RBLNModelRunner._handle_kv_connector_preemptions


def _sched_output(preempted, metadata):  # type: ignore[no-untyped-def]
    return types.SimpleNamespace(
        preempted_req_ids=preempted, kv_connector_metadata=metadata
    )


def test_runner_passes_metadata_not_reqid_set(monkeypatch):
    """The fix: runner forwards scheduler_output.kv_connector_metadata (NOT the
    preempted_req_ids set) to handle_preemptions."""
    spy = _DummyConn()
    monkeypatch.setattr(rmr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(rmr, "get_kv_transfer_group", lambda: spy)
    meta = MultiKVConnectorMetadata(metadata=())
    _RUNNER_FIX(_sched_output({"r0"}, meta))
    assert spy.received is meta  # metadata, not {"r0"}


def test_runner_skips_when_metadata_none(monkeypatch):
    """Warm-up phase: kv_connector_metadata is None -> connector not called."""
    spy = _DummyConn()
    monkeypatch.setattr(rmr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(rmr, "get_kv_transfer_group", lambda: spy)
    _RUNNER_FIX(_sched_output({"r0"}, None))
    assert spy.received == "UNSET"


def test_runner_skips_when_no_preemptions(monkeypatch):
    """No preempted requests -> connector not called."""
    spy = _DummyConn()
    monkeypatch.setattr(rmr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(rmr, "get_kv_transfer_group", lambda: spy)
    _RUNNER_FIX(_sched_output(set(), MultiKVConnectorMetadata(metadata=())))
    assert spy.received == "UNSET"
