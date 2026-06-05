# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for RblnNixlDirectConnector (direct D2D KV transfer).

Tests cover:
- Worker: deferred NIXL registration lifecycle (defer → finalize, idempotency)
- Worker: host-bounce removal (no host xfer buffer / copy ops)
- Connector: role-based delegation of finalize / runtime holder
- Platform & factory: direct path is discoverable
"""

from unittest.mock import MagicMock

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_direct_connector import (  # noqa: E501
    RblnNixlDirectConnector,
    RblnNixlDirectConnectorWorker,
)


def _create_direct_worker():
    """Create an RblnNixlDirectConnectorWorker skeleton without running
    __init__ (which requires the nixl_rbln package and a live NIXL agent),
    mirroring the RblnNixlConnectorScheduler tests in
    test_pd_disaggregation.py."""
    worker = object.__new__(RblnNixlDirectConnectorWorker)
    worker._pending_kv_caches = None
    worker.use_host_buffer = False
    return worker


def _create_direct_connector(worker):
    """Create an RblnNixlDirectConnector skeleton with the given worker
    (None = scheduler role)."""
    connector = object.__new__(RblnNixlDirectConnector)
    connector.connector_worker = worker
    connector.connector_scheduler = None
    return connector


class TestDeferredRegistration:
    """register_kv_caches must not touch NIXL before warm-up; the actual
    registration runs exactly once in finalize_kv_cache_registration."""

    def test_register_defers_until_finalize(self):
        worker = _create_direct_worker()
        impl_calls = []
        worker._register_kv_caches_impl = impl_calls.append

        kv_caches = {"layer.0": MagicMock()}
        worker.register_kv_caches(kv_caches)

        # Nothing registered yet; caches are stashed.
        assert impl_calls == []
        assert worker._pending_kv_caches is kv_caches

        worker.finalize_kv_cache_registration()

        assert impl_calls == [kv_caches]
        assert worker._pending_kv_caches is None

    def test_finalize_is_idempotent(self):
        worker = _create_direct_worker()
        impl_calls = []
        worker._register_kv_caches_impl = impl_calls.append
        worker.register_kv_caches({"layer.0": MagicMock()})

        worker.finalize_kv_cache_registration()
        worker.finalize_kv_cache_registration()

        assert len(impl_calls) == 1

    def test_finalize_without_pending_is_noop(self):
        worker = _create_direct_worker()
        impl_calls = []
        worker._register_kv_caches_impl = impl_calls.append

        worker.finalize_kv_cache_registration()

        assert impl_calls == []


class TestHostBounceRemoval:
    """The direct path registers device memory with NIXL; it must never
    allocate a host staging buffer or wire host copy ops."""

    def test_initialize_host_xfer_buffer_allocates_nothing(self):
        worker = _create_direct_worker()
        # Even if the config asked for a host buffer, the direct path
        # ignores it (logs and moves on).
        worker.use_host_buffer = True

        worker.initialize_host_xfer_buffer({"layer.0": MagicMock()})

        assert not getattr(worker, "host_xfer_buffers", None)

    def test_set_host_xfer_buffer_ops_is_noop(self):
        worker = _create_direct_worker()

        worker.set_host_xfer_buffer_ops(MagicMock())

        assert not hasattr(worker, "copy_blocks")


class TestConnectorRoleDelegation:
    """finalize / set_runtime_holder act on the worker and are no-ops on
    the scheduler side."""

    def test_scheduler_role_is_noop(self):
        connector = _create_direct_connector(worker=None)

        # Must not raise despite connector_worker being None.
        connector.finalize_kv_cache_registration()
        connector.set_runtime_holder([MagicMock()])

    def test_worker_role_delegates(self):
        worker = MagicMock()
        connector = _create_direct_connector(worker)

        connector.finalize_kv_cache_registration()
        worker.finalize_kv_cache_registration.assert_called_once_with()

        runtime_holder = [MagicMock()]
        connector.set_runtime_holder(runtime_holder)
        assert worker._runtime_holder is runtime_holder


def test_platform_supports_rbln_kv_buffer_for_nixl():
    """Both the host-bounce ('cpu') and direct ('rbln') kv_buffer_device
    values must be accepted, in both directions."""
    from vllm_rbln.platform import RblnPlatform

    assert RblnPlatform.get_nixl_supported_devices() == {
        "cpu": ("cpu", "rbln"),
        "rbln": ("rbln", "cpu"),
    }


def test_direct_connector_registered_in_factory():
    import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    cls = KVConnectorFactory.get_connector_class_by_name("RblnNixlDirectConnector")
    assert cls is RblnNixlDirectConnector
