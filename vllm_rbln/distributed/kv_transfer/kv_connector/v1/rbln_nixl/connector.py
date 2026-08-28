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
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnector,
    NixlConnectorMetadata,
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
    SupportsKVCacheRegistrationFinalize,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
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

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Serve every producer shard, including pipeline-parallel stages.

        Upstream rejects `pp_rank > 0` and keys the side channel by `tp_rank`
        alone, so PP stages would overwrite each other. Flatten to the rank a
        peer actually asks for in `_nixl_handshake`, using this engine's
        `tp_size`; at `pp_size == 1` that is upstream's key again.
        """
        tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        flattened: dict[int, KVConnectorHandshakeMetadata] = {}
        for (pp_rank, tp_rank), rank_metadata in metadata.items():
            flat_rank = pp_rank * tp_size + tp_rank
            if flat_rank in flattened:
                raise ValueError(
                    "Duplicate handshake metadata for flat rank "
                    f"{flat_rank} (pp_rank={pp_rank}, tp_rank={tp_rank}); "
                    f"tensor_parallel_size={tp_size} disagrees with the ranks "
                    "reported by the workers."
                )
            flattened[flat_rank] = rank_metadata
        self.set_xfer_handshake_metadata(flattened)

    def finalize_kv_cache_registration(self) -> None:
        """Run the worker's deferred NIXL registration after warm-up
        materializes the KV cache backing memory. No-op on host-bounce."""
        if self.connector_worker is not None:
            self.connector_worker.finalize_kv_cache_registration()


class RblnNixlPullConnector(RblnNixlConnectorBase, NixlPullConnector):
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
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = RblnNixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = RblnNixlPullConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )


class RblnNixlPushConnector(RblnNixlConnectorBase, NixlPushConnector):
    """Push-based (WRITE) RBLN NIXL KV transfer connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
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

        The base does the host-staging copy here; taking it first would let a
        write read a buffer still being filled, so the order this call site
        fixes is the invariant.
        """
        super().wait_for_save()
        assert isinstance(self.connector_worker, RblnNixlPushConnectorWorker)
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_early_push(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Let the previous step's offers go, then start this step's work.

        Ahead of `super()`, which adopts the handover: a request whose handover
        lands on this step has to be written by its offer rather than dropped
        along with it.
        """
        assert isinstance(self.connector_worker, RblnNixlPushConnectorWorker)
        self.connector_worker.release_early_offers()
        super().start_load_kv(forward_context, **kwargs)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Drain an early write whose source blocks are about to be reused.

        Runs ahead of the forward that would overwrite them.
        """
        super().handle_preemptions(kv_connector_metadata)
        assert isinstance(kv_connector_metadata, RblnNixlConnectorMetadata)
        assert isinstance(self.connector_worker, RblnNixlPushConnectorWorker)
        self.connector_worker.flush_early_sends(kv_connector_metadata.push_early_flush)
