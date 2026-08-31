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

# RblnNixlConnector's construction guards, role -> scheduler/worker wiring and
# finalize delegation, with the base __init__ and sub-connectors patched out.

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.connector as cm
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.connector import (
    RblnNixlConnector,
)


def _vllm_config(*, kv_buffer_device="cpu", engine_id="engine-0", has_transfer=True):
    transfer = (
        SimpleNamespace(engine_id=engine_id, kv_buffer_device=kv_buffer_device)
        if has_transfer
        else None
    )
    return SimpleNamespace(kv_transfer_config=transfer)


@pytest.fixture
def isolated_connector(monkeypatch):
    """Construct RblnNixlConnector with the upstream base __init__ neutralized
    and the sub-connectors faked, so only the guards + wiring execute. Returns a
    builder(vllm_config, role)."""
    monkeypatch.setattr(cm.KVConnectorBase_V1, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(cm, "RblnNixlConnectorScheduler", lambda *a, **k: "SCHEDULER")
    monkeypatch.setattr(cm, "RblnNixlConnectorWorker", lambda *a, **k: "WORKER")

    def build(vllm_config, role=KVConnectorRole.SCHEDULER):
        connector = object.__new__(RblnNixlConnector)
        RblnNixlConnector.__init__(connector, vllm_config, role, {"kv_cache": 1})
        return connector

    return build


class TestConstructionGuards:
    def test_requires_kv_transfer_config(self, isolated_connector):
        with pytest.raises(AssertionError):
            isolated_connector(_vllm_config(has_transfer=False))

    def test_requires_engine_id(self, isolated_connector):
        with pytest.raises(AssertionError):
            isolated_connector(_vllm_config(engine_id=None))

    def test_rejects_unknown_kv_buffer_device(self, isolated_connector):
        with pytest.raises(AssertionError, match="kv_buffer_device"):
            isolated_connector(_vllm_config(kv_buffer_device="gpu"))

    @pytest.mark.parametrize("kv_buffer_device", ["cpu", "rbln"])
    def test_accepts_supported_kv_buffer_devices(
        self, isolated_connector, kv_buffer_device
    ):
        # Both host-bounce ("cpu") and D2D ("rbln") pass the guard and construct
        # all the way through to the role wiring.
        connector = isolated_connector(_vllm_config(kv_buffer_device=kv_buffer_device))
        assert connector.connector_scheduler == "SCHEDULER"


class TestRoleWiring:
    def test_scheduler_role_builds_scheduler_only(self, isolated_connector):
        connector = isolated_connector(_vllm_config(), role=KVConnectorRole.SCHEDULER)
        assert connector.connector_scheduler == "SCHEDULER"
        assert connector.connector_worker is None

    def test_worker_role_builds_worker_only(self, isolated_connector):
        connector = isolated_connector(_vllm_config(), role=KVConnectorRole.WORKER)
        assert connector.connector_worker == "WORKER"
        assert connector.connector_scheduler is None


class TestFinalizeDelegation:
    def test_delegates_to_worker_when_present(self):
        calls = []
        connector = object.__new__(RblnNixlConnector)
        connector.connector_worker = SimpleNamespace(
            finalize_kv_cache_registration=lambda: calls.append(1)
        )
        connector.finalize_kv_cache_registration()
        assert calls == [1]

    def test_noop_on_scheduler_role(self):
        # Scheduler role has no worker; finalize must be a safe no-op.
        connector = object.__new__(RblnNixlConnector)
        connector.connector_worker = None
        connector.finalize_kv_cache_registration()  # must not raise
