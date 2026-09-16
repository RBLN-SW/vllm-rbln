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

from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnector,
    NixlPullConnector,
    NixlPushConnector,
)

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_scheduler import (
    RblnNixlPullConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_scheduler import (
    RblnNixlPushConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker import (
    RblnNixlPushConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    SupportsDeferredLoad,
    SupportsKVCacheRegistrationFinalize,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlConnectorBase(NixlBaseConnector, SupportsKVCacheRegistrationFinalize):
    """RBLN's NIXL KV connector. A single worker runs both paths and branches
    internally on `kv_transfer_config.kv_buffer_device`:

    * `"cpu"`  → host-bounce: page-aligned host staging, RDMA over DRAM
      via the RBLN NIXL backend's `ibv_reg_mr` path.
    * `"rbln"` → D2D: RBLN NIXL backend's `ibv_reg_dmabuf_mr` path on
      the device memory exported by the `nixl_rbln` adapter; no host
      staging.

    Both paths use the same RBLN backend / RDMA NICs; the only
    difference is which memory segment (DRAM_SEG vs VRAM_SEG) is
    registered. Both require `VLLM_RBLN_USE_DEVICE_TENSOR=1`.

    A direction subclass builds the scheduler or worker for its role; this
    class leaves both unset."""

    connector_scheduler: RblnNixlSchedulerBase | None
    connector_worker: RblnNixlWorkerBase | None

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # NOTE(RBLN): skip past NixlBaseConnector.__init__ to the connector
        # base: everything it sets is set below, and its kv_role deprecation
        # warning does not apply -- both roles live on one connector here.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        kv_buffer_device = vllm_config.kv_transfer_config.kv_buffer_device
        assert kv_buffer_device in ("cpu", "rbln"), (
            f"{type(self).__name__} requires kv_buffer_device in "
            f"{{'cpu', 'rbln'}}; got {kv_buffer_device!r}."
        )
        assert envs.VLLM_RBLN_USE_DEVICE_TENSOR, (
            f"{type(self).__name__} requires VLLM_RBLN_USE_DEVICE_TENSOR=1."
        )
        self.kv_cache_config = kv_cache_config
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.connector_scheduler = None
        self.connector_worker = None

    def finalize_kv_cache_registration(self) -> None:
        """Run the worker's deferred NIXL registration after warm-up
        materializes the KV cache backing memory. No-op on host-bounce."""
        if self.connector_worker is not None:
            self.connector_worker.finalize_kv_cache_registration()


class RblnNixlPullConnector(
    RblnNixlConnectorBase, NixlPullConnector, SupportsDeferredLoad
):
    """Pull-based (READ) RBLN NIXL KV transfer connector.

    Registered under `RblnNixlConnector` as well: that is the name the read path
    shipped under and what deployments carry in `kv_transfer_config`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._deferred_load_meta: NixlConnectorMetadata | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = RblnNixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = RblnNixlPullConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Keep this step's read; `flush_deferred_load` issues it.

        Safe because the scheduler withholds a request until its load is
        reported finished, so the read and the forward touch disjoint blocks.
        `clear_connector_metadata` rebinds that field to None, so the object
        kept here survives the step.

        The assert holds because every `execute_model` flushes on entry: losing
        a held read strands its request as surely as never issuing one.
        """
        assert self._deferred_load_meta is None
        self._deferred_load_meta = self._connector_metadata

    def flush_deferred_load(self) -> None:
        """Issue a held read, or nothing if none is held.

        A request is listed for receive once, so a read nobody issues strands it
        for good -- hence every site that can be a step's last chance flushes.

        That another step comes at all rests on the scheduler counting
        `skipped_waiting` as unfinished work, which a paused one does not: both
        pause states leave it out, holding a read until the unpause.

        The replay is the whole of `start_load_kv`, lease arming and heartbeats
        included; their deadlines are absolute, so only their arrival shifts.
        """
        meta = self._deferred_load_meta
        if meta is None:
            return
        self._deferred_load_meta = None
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(meta)


class RblnNixlPushConnector(
    RblnNixlConnectorBase, NixlPushConnector, SupportsDeferredLoad
):
    """Push-based (WRITE) RBLN NIXL KV transfer connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._deferred_load_meta: NixlConnectorMetadata | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = RblnNixlPushConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = RblnNixlPushConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    def wait_for_save(self) -> None:
        """Take the closed prefill for the writer, after the host copy.

        Upstream does the host-staging copy here; taking it first would let a
        write read a buffer still being filled, so the order this call site
        fixes is the invariant.
        """
        super().wait_for_save()
        assert isinstance(self.connector_worker, RblnNixlPushConnectorWorker)
        assert isinstance(self._connector_metadata, RblnNixlConnectorMetadata)
        self.connector_worker.start_early_push(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Keep this step's work; `flush_deferred_load` runs it.

        An offer covers blocks a prefill closed on the step before this one, and
        that forward has not retired at this point: the model call this step is
        about to make is what it retires behind. Held so the release sits past
        that call.
        """
        self._deferred_load_meta = self._connector_metadata

    def flush_deferred_load(self) -> None:
        """Release the held offers and drive the worker, or do nothing.

        The release stays ahead of the handover the worker adopts, so a request
        whose handover lands on this step is written by its offer as well -- the
        two are successive batches of it, not duplicates.

        A step with no forward reaches no flush site of its own; the next step's
        entry site runs what it holds, so nothing is stranded.
        """
        meta = self._deferred_load_meta
        if meta is None:
            return
        self._deferred_load_meta = None
        assert isinstance(self.connector_worker, RblnNixlPushConnectorWorker)
        self.connector_worker.release_early_offers()
        self.connector_worker.start_load_kv(meta)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Drain an early write whose source blocks are about to be reused.

        Runs ahead of the forward that would overwrite them.
        """
        super().handle_preemptions(kv_connector_metadata)
        assert isinstance(kv_connector_metadata, RblnNixlConnectorMetadata)
        assert isinstance(self.connector_worker, RblnNixlPushConnectorWorker)
        self.connector_worker.flush_early_sends(kv_connector_metadata.push_early_flush)
