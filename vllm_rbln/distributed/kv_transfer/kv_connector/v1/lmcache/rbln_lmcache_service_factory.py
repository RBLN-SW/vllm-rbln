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

"""RBLN-specific vLLM v1 adapter for LMCache.

Provides RBLNServiceFactory that injects RBLNConnector instead of the
CUDA-based GPU connector.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.metadata import LMCacheMetadata

from vllm_rbln.logger import init_logger

from .rbln_connector import RBLNConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _rbln_calculate_local_rank_and_world_size(
    vllm_config: "VllmConfig",
) -> tuple[int, int]:
    """Calculate local rank and world size for RBLN NPU.

    Unlike the upstream version which calls torch.cuda.device_count(),
    this uses rebel.get_npu_count() for RBLN devices.
    """
    parallel_config = vllm_config.parallel_config
    global_world_size = parallel_config.world_size

    try:
        import rebel

        num_devices = rebel.get_npu_count()
    except Exception:
        num_devices = 1

    if global_world_size <= num_devices:
        return parallel_config.rank, parallel_config.world_size
    else:
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        local_world_size = global_world_size // pp_size
        assert local_world_size == tp_size, (
            "LMCache assumes TP is intra-node and PP is inter-node. "
            f"Got local_world_size={local_world_size}, tp_size={tp_size}"
        )
        local_rank = parallel_config.rank % local_world_size
        return local_rank, local_world_size


class RBLNServiceFactory(VllmServiceFactory):
    """RBLN-specific service factory for LMCache.

    Overrides the upstream VllmServiceFactory to:
    - Use RBLN NPU rank calculation instead of CUDA device counting
    - Create RBLNConnector instead of the CUDA-based GPU connector
    """

    def get_or_create_metadata(self) -> Optional[LMCacheMetadata]:
        if self.metadata is not None:
            return self.metadata

        from lmcache.integration.vllm.utils import (
            calculate_draft_layers,
            mla_enabled,
            validate_mla_config,
        )

        try:
            from vllm.utils.torch_utils import get_kv_cache_torch_dtype
        except ImportError:
            from vllm.utils import get_kv_cache_torch_dtype

        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        cache_config = self.vllm_config.cache_config

        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        use_mla = mla_enabled(model_config)
        validate_mla_config(self.lmcache_config, use_mla)

        num_layer = model_config.get_num_layers(parallel_config)
        num_draft_layers = calculate_draft_layers(self.vllm_config)
        num_layer += num_draft_layers
        chunk_size = self.lmcache_config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_shape = (
            num_layer,
            1 if use_mla else 2,
            chunk_size,
            num_kv_head,
            head_size,
        )

        logger.info(
            "RBLN LMCache metadata: num_layer=%d, chunk_size=%d, "
            "num_kv_head=%d, head_size=%d, hidden_dim=%d, "
            "use_mla=%s, kv_shape=%s, num_draft_layers=%d",
            num_layer,
            chunk_size,
            num_kv_head,
            head_size,
            num_kv_head * head_size,
            use_mla,
            kv_shape,
            num_draft_layers,
        )

        engine_id = None
        kv_connector_extra_config = None
        if hasattr(self.vllm_config, "kv_transfer_config"):
            kv_transfer_config = self.vllm_config.kv_transfer_config
            if kv_transfer_config is not None:
                engine_id = getattr(kv_transfer_config, "engine_id", None)
                kv_connector_extra_config = getattr(
                    kv_transfer_config,
                    "kv_connector_extra_config",
                    None,
                )

        if self.role == "scheduler":
            local_worker_id = parallel_config.rank
            local_world_size = parallel_config.world_size
        else:
            local_worker_id, local_world_size = (
                _rbln_calculate_local_rank_and_world_size(self.vllm_config)
            )

        self.metadata = LMCacheMetadata(
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            local_world_size=local_world_size,
            worker_id=parallel_config.rank,
            local_worker_id=local_worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=self.role,
            served_model_name=model_config.served_model_name,
            chunk_size=self.lmcache_config.chunk_size,
            engine_id=engine_id,
            kv_connector_extra_config=kv_connector_extra_config,
        )
        return self.metadata

    def get_or_create_lmcache_engine(self) -> Optional[LMCacheEngine]:
        self._ensure_metadata()
        assert self.metadata is not None

        if (
            self.role == "scheduler"
            and not self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            from lmcache.observability import PrometheusLogger

            PrometheusLogger.GetOrCreate(
                self.metadata,
                config=self.lmcache_config,
            )
            return None

        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            self.lmcache_engine = curr_engine
            return curr_engine

        if self.role == "scheduler":
            tpg = SimpleNamespace()
            tpg.broadcast = lambda tensor, src: tensor
            tpg.broadcast_object = lambda obj, src: obj
            rbln_connector = None
        else:
            from vllm.distributed.parallel_state import get_tp_group

            tpg = get_tp_group()

            num_layer = self.metadata.kv_shape[0]
            num_kv_head = self.metadata.kv_shape[3]
            head_size = self.metadata.kv_shape[4]
            rbln_connector = RBLNConnector(
                num_layers=num_layer,
                num_kv_heads=num_kv_head,
                head_dim=head_size,
                block_size=self.vllm_config.cache_config.block_size,
                dtype=self.metadata.kv_dtype,
            )

            logger.info(
                "Created RBLNConnector: layers=%d, kv_heads=%d, "
                "head_dim=%d, block_size=%d",
                num_layer,
                num_kv_head,
                head_size,
                self.vllm_config.cache_config.block_size,
            )

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_config,
            self.metadata,
            rbln_connector,
            tpg.broadcast,
            tpg.broadcast_object,
        )
        self.lmcache_engine = engine

        if (
            self.role == "scheduler"
            and self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            assert engine.save_only_first_rank or (
                self.lmcache_config.get_extra_config_value(
                    "remote_enable_mla_worker_id_as0", self.metadata.use_mla
                )
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine
