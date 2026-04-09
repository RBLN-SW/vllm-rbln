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

"""RBLN KV connector for vLLM.

Provides RBLNLMCacheConnectorV1 as a drop-in replacement for the upstream
LMCacheConnectorV1Dynamic.  This connector uses RBLNServiceFactory (which
injects RBLNConnector) instead of the upstream VllmServiceFactory (which
requires CUDA).
"""

from collections import defaultdict
from typing import Any, Optional

import torch
from lmcache.integration.vllm.lmcache_connector_v1 import (
    LMCacheConnectorV1Dynamic,
)
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.manager import LMCacheManager
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.model_executor.models.utils import extract_layer_index

from vllm_rbln.logger import init_logger

from .lmcache.rbln_connector import RBLNConnector
from .lmcache.rbln_lmcache_service_factory import RBLNServiceFactory

logger = init_logger(__name__)


class RBLNLMCacheConnectorV1Impl(LMCacheConnectorV1Impl):
    """RBLN-specific LMCache connector implementation.

    Overrides the upstream LMCacheConnectorV1Impl to use
    RBLNServiceFactory instead of the upstream VllmServiceFactory.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        from lmcache.integration.vllm.utils import (
            lmcache_get_or_create_config,
        )
        from lmcache.v1.config import LMCacheEngineConfig

        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = RBLNServiceFactory(
            config,
            vllm_config,
            role.name.lower(),
        )
        self._manager = LMCacheManager(
            config,
            service_factory,
            connector=self,
        )

        self._manager.start_services()
        self._init_connector_state(role, vllm_config, config)
        self._setup_metrics()

        from lmcache import utils
        from vllm.version import __version__ as VLLM_VERSION

        logger.info(
            "RBLN LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def set_runtime_holder(self, runtime_holder: list) -> None:
        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector
        connector.set_runtime_holder(runtime_holder)
        logger.info("RBLN runtime_holder set on RBLNConnector (lazy)")

    def set_runtime(self, runtime) -> None:
        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector
        connector.set_runtime(runtime)
        logger.info("RBLN runtime set on RBLNConnector")

    @staticmethod
    def _normalize_kv_shape(
        kv_caches: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Normalize RBLN 6D KV cache tensors to the 5D format LMCache expects.

        RBLN attention backend produces shape [2, B, H, 1, S, D].
        Upstream LMCache expects [2, B, S, H, D].
        """
        normalized = {}
        for name, tensor in kv_caches.items():
            if tensor.dim() == 6 and tensor.shape[3] == 1:
                tensor = tensor.squeeze(3).permute(0, 1, 3, 2, 4).contiguous()
            normalized[name] = tensor
        return normalized

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        normalized_kv_caches = self._normalize_kv_shape(kv_caches)
        super().register_kv_caches(normalized_kv_caches)

        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector
        index2name: dict[int, list[str]] = defaultdict(list)
        for layer_name in kv_caches:
            index2name[extract_layer_index(layer_name)].append(layer_name)
        kv_cache_names = [names[0] for _, names in sorted(index2name.items())]
        connector.set_kv_cache_names(kv_cache_names)
        logger.info(
            "Set %d kv_cache_names on RBLNConnector",
            len(kv_cache_names),
        )

        sorted_kv_caches = {name: kv_caches[name] for name in kv_cache_names}
        connector.initialize_xfer_buffers(sorted_kv_caches)


class RBLNLMCacheConnectorV1(LMCacheConnectorV1Dynamic):
    """RBLN KV connector for vLLM v1.

    Inherits from LMCacheConnectorV1Dynamic and only overrides:
    - __init__: to use RBLNLMCacheConnectorV1Impl instead of upstream impl
    - set_runtime_holder / set_runtime: RBLN-specific runtime injection
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ):
        if kv_cache_config is not None:
            KVConnectorBase_V1.__init__(
                self,
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        else:
            KVConnectorBase_V1.__init__(
                self,
                vllm_config=vllm_config,
                role=role,
            )
        self._lmcache_engine = RBLNLMCacheConnectorV1Impl(vllm_config, role, self)

    def set_runtime_holder(self, runtime_holder: list) -> None:
        self._lmcache_engine.set_runtime_holder(runtime_holder)

    def set_runtime(self, runtime) -> None:
        self._lmcache_engine.set_runtime(runtime)

    def get_connector(self) -> RBLNConnector:
        return self._lmcache_engine.lmcache_engine.gpu_connector
