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

# RblnNixlPullConnector's construction guards, role -> scheduler/worker wiring,
# finalize delegation, the side-channel keying it hands upstream, and the read it
# holds until a submission is in flight, with the upstream __init__ and
# sub-connectors patched out.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.connector as cm
import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.connector import (
    RblnNixlPullConnector,
    RblnNixlPushConnector,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    flush_deferred_loads,
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
    """Construct RblnNixlPullConnector with the upstream base __init__ neutralized
    and the sub-connectors faked, so only the guards + wiring execute. Returns a
    builder(vllm_config, role, use_device_tensor=True)."""
    monkeypatch.setattr(cm.KVConnectorBase_V1, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(
        cm, "RblnNixlPullConnectorScheduler", lambda *a, **k: "SCHEDULER"
    )
    monkeypatch.setattr(cm, "RblnNixlPullConnectorWorker", lambda *a, **k: "WORKER")

    def build(vllm_config, role=KVConnectorRole.SCHEDULER, use_device_tensor=True):
        monkeypatch.setattr(envs, "VLLM_RBLN_USE_DEVICE_TENSOR", use_device_tensor)
        connector = object.__new__(RblnNixlPullConnector)
        RblnNixlPullConnector.__init__(connector, vllm_config, role, {"kv_cache": 1})
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

    def test_requires_device_tensor(self, isolated_connector):
        with pytest.raises(AssertionError, match="VLLM_RBLN_USE_DEVICE_TENSOR"):
            isolated_connector(_vllm_config(), use_device_tensor=False)


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
        connector = object.__new__(RblnNixlPullConnector)
        connector.connector_worker = SimpleNamespace(
            finalize_kv_cache_registration=lambda: calls.append(1)
        )
        connector.finalize_kv_cache_registration()
        assert calls == [1]

    def test_noop_on_scheduler_role(self):
        # Scheduler role has no worker; finalize must be a safe no-op.
        connector = object.__new__(RblnNixlPullConnector)
        connector.connector_worker = None
        connector.finalize_kv_cache_registration()  # must not raise


class TestSetXferHandshakeMetadataPpAware:
    # Producer side: every (pp_rank, tp_rank) shard must reach the side channel.
    #
    # EngineCore hands the merged worker dicts to
    # ``set_xfer_handshake_metadata_pp_aware``. Upstream's implementation rejects
    # pp_rank > 0 and keys by tp_rank alone; this connector flattens the pair into
    # the rank a consumer asks for in ``_nixl_handshake``.
    #

    @staticmethod
    def _connector(tp_size):
        c = object.__new__(RblnNixlPullConnector)
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = tp_size
        c._vllm_config = vllm_config
        return c

    def _flatten(self, metadata, *, tp_size):
        c = self._connector(tp_size)
        with patch.object(
            RblnNixlPullConnector, "set_xfer_handshake_metadata"
        ) as forward:
            c.set_xfer_handshake_metadata_pp_aware(metadata)
        forward.assert_called_once()
        return forward.call_args[0][0]

    def test_single_stage_reduces_to_tp_rank(self):
        # pp_size == 1: flat rank == tp_rank, i.e. upstream's behavior.
        assert self._flatten({(0, 0): "m0", (0, 1): "m1"}, tp_size=2) == {
            0: "m0",
            1: "m1",
        }

    def test_pp_stages_get_distinct_flat_ranks(self):
        assert self._flatten({(0, 0): "s0", (1, 0): "s1"}, tp_size=1) == {
            0: "s0",
            1: "s1",
        }
        assert self._flatten(
            {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}, tp_size=2
        ) == {0: "a", 1: "b", 2: "c", 3: "d"}

    def test_pp_rank_gt_zero_is_accepted(self):
        # Upstream would raise here; a PP-aware connector must not.
        assert self._flatten({(3, 0): "s3"}, tp_size=1) == {3: "s3"}

    def test_collision_is_rejected(self):
        # A tp_size disagreeing with the reported ranks would silently drop a
        # shard; fail loudly instead.
        with pytest.raises(ValueError, match="Duplicate handshake metadata"):
            self._flatten({(0, 1): "a", (1, 0): "b"}, tp_size=1)


class TestConnectorWiring:
    @pytest.fixture
    def push_connector(self, monkeypatch):
        monkeypatch.setattr(
            cm.KVConnectorBase_V1, "__init__", lambda self, *a, **k: None
        )
        monkeypatch.setattr(
            cm, "RblnNixlPushConnectorScheduler", lambda *a, **k: "SCHEDULER"
        )
        monkeypatch.setattr(cm, "RblnNixlPushConnectorWorker", lambda *a, **k: "WORKER")
        monkeypatch.setattr(envs, "VLLM_RBLN_USE_DEVICE_TENSOR", True)

        def build(role, kv_buffer_device="rbln"):
            vllm_config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    engine_id="engine-0", kv_buffer_device=kv_buffer_device
                )
            )
            connector = object.__new__(RblnNixlPushConnector)
            RblnNixlPushConnector.__init__(
                connector, vllm_config, role, {"kv_cache": 1}
            )
            return connector

        return build

    def test_worker_role_builds_the_push_worker(self, push_connector):
        connector = push_connector(KVConnectorRole.WORKER)
        assert connector.connector_worker == "WORKER"
        assert connector.connector_scheduler is None

    def test_scheduler_role_builds_the_push_scheduler(self, push_connector):
        connector = push_connector(KVConnectorRole.SCHEDULER)
        assert connector.connector_scheduler == "SCHEDULER"
        assert connector.connector_worker is None

    def test_shares_the_construction_guards(self, push_connector):
        # The guards live on the shared connector base; one of them is enough to
        # pin that this direction goes through it.
        with pytest.raises(AssertionError, match="kv_buffer_device"):
            push_connector(KVConnectorRole.WORKER, kv_buffer_device="gpu")


class TestDeferredLoad:
    """Holding a read until a model submission is in flight to run it behind."""

    @pytest.fixture
    def worker_connector(self, isolated_connector, monkeypatch):
        def build():
            connector = isolated_connector(_vllm_config(), role=KVConnectorRole.WORKER)
            connector.connector_worker = SimpleNamespace(
                started=[],
                start_load_kv=lambda meta: connector.connector_worker.started.append(
                    meta
                ),
            )
            return connector

        return build

    @staticmethod
    def _ctx(attn_metadata):
        return SimpleNamespace(attn_metadata=attn_metadata)

    def test_the_read_is_held(self, worker_connector):
        connector = worker_connector()
        connector._connector_metadata = "META"

        connector.start_load_kv(self._ctx({"layer.0": "ATTN"}))

        assert connector.connector_worker.started == []

    def test_upstream_is_never_reached_at_this_point(
        self, worker_connector, monkeypatch
    ):
        # Upstream issues here; this override replaces that rather than adding to
        # it. Patching the upstream method so a rename or a re-route on that side
        # fails here (monkeypatch raises by default).
        connector = worker_connector()
        connector._connector_metadata = "META"
        delegated = []
        monkeypatch.setattr(
            cm.NixlPullConnector,
            "start_load_kv",
            lambda self, forward_context, **kw: delegated.append(forward_context),
        )

        connector.start_load_kv(self._ctx({"layer.0": "ATTN"}))

        assert delegated == []

    def test_flush_issues_the_held_read_once(self, worker_connector):
        connector = worker_connector()
        connector._connector_metadata = "META"
        connector.start_load_kv(self._ctx(None))

        connector.flush_deferred_load()

        assert connector.connector_worker.started == ["META"]

    def test_flush_again_issues_nothing(self, worker_connector):
        # A second flush in the same round -- the dummy step and the next
        # execute_model both call it -- must not read every block twice.
        connector = worker_connector()
        connector._connector_metadata = "META"
        connector.start_load_kv(self._ctx(None))
        connector.flush_deferred_load()

        connector.flush_deferred_load()

        assert connector.connector_worker.started == ["META"]

    def test_flush_with_nothing_held_issues_nothing(self, worker_connector):
        connector = worker_connector()

        connector.flush_deferred_load()

        assert connector.connector_worker.started == []

    def test_the_helper_reaches_this_connector(self, worker_connector):
        # The runner flushes through the helper, which skips anything without the
        # protocol -- so going through it is what proves this connector is reached.
        connector = worker_connector()
        connector._connector_metadata = "META"
        connector.start_load_kv(self._ctx({"layer.0": "ATTN"}))

        flush_deferred_loads(connector)

        assert connector.connector_worker.started == ["META"]
