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

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnector,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.scheduler import (
    RblnNixlConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.worker import (
    RblnNixlConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    SupportsKVCacheRegistrationFinalize,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlConnector(NixlConnector, SupportsKVCacheRegistrationFinalize):
    """RBLN's NIXL KV connector. A single `RblnNixlConnectorWorker` runs
    both paths and branches internally on
    `kv_transfer_config.kv_buffer_device`:

    * `"cpu"`  → host-bounce: page-aligned host staging, RDMA over DRAM
      via the RBLN NIXL backend's `ibv_reg_mr` path.
    * `"rbln"` → D2D: RBLN NIXL backend's `ibv_reg_dmabuf_mr` path on
      the device memory exported by the `nixl_rbln` adapter; no host
      staging.

    Both paths use the same RBLN backend / RDMA NICs; the only
    difference is which memory segment (DRAM_SEG vs VRAM_SEG) is
    registered. Both require `VLLM_RBLN_USE_DEVICE_TENSOR=1`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        kv_buffer_device = vllm_config.kv_transfer_config.kv_buffer_device
        assert kv_buffer_device in ("cpu", "rbln"), (
            "RblnNixlConnector requires kv_buffer_device in "
            f"{{'cpu', 'rbln'}}; got {kv_buffer_device!r}."
        )
        assert envs.VLLM_RBLN_USE_DEVICE_TENSOR, (
            "RblnNixlConnector requires VLLM_RBLN_USE_DEVICE_TENSOR=1."
        )
        self.kv_cache_config = kv_cache_config
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: RblnNixlConnectorScheduler | None = (
                RblnNixlConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: RblnNixlConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = RblnNixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    def finalize_kv_cache_registration(self) -> None:
        """Run the worker's deferred NIXL registration after warm-up
        materializes the KV cache backing memory. No-op on host-bounce."""
        if self.connector_worker is not None:
            self.connector_worker.finalize_kv_cache_registration()
