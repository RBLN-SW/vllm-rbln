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

"""RBLN LMCache KV connector for vLLM.

Provides RBLNLMCacheConnectorV1 as a drop-in replacement for the upstream
LMCacheConnectorV1Dynamic, with RBLNConnector handling NPU <-> CPU KV cache
transfers via the rebel runtime API.

Usage in vLLM config:
    kv_connector = "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector.RBLNLMCacheConnectorV1"
"""

from collections import defaultdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch
from rebel.kv_cache import aligned_tensor
from typing_extensions import override
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.model_executor.models.utils import extract_layer_index

from lmcache.integration.vllm.lmcache_connector_v1 import (
    LMCacheConnectorV1Dynamic,
)
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# RBLNConnector: NPU <-> CPU KV cache transfer via rebel runtime API
# ---------------------------------------------------------------------------


class RBLNConnector(GPUConnectorInterface):
    """RBLN NPU connector for KV cache transfer.

    Uses the rebel runtime's block-level fetch_kv_cache / update_kv_cache
    APIs with slot_mapping to transfer data between the paged NPU KV cache
    and CPU memory objects.

    The runtime is obtained lazily from ``runtime_holder``, a mutable list
    that the RBLN compile backend populates after model compilation.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        kv_cache_names: Optional[List[str]] = None,
        runtime_holder: Optional[list] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim_size = num_kv_heads * head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.kv_cache_names = kv_cache_names or []
        self._runtime_holder: list = runtime_holder if runtime_holder is not None else []

        # Per-layer aligned transfer buffers, keyed by kv_cache_name.
        self._xfer_buffers: dict[str, torch.Tensor] = {}

    @property
    def runtime(self):
        """Lazily resolve the low-level rebel runtime from the holder."""
        if not self._runtime_holder:
            return None
        rt = self._runtime_holder[0]
        # Unwrap DynamoRuntime -> PyRblnSyncRuntime.
        try:
            from rebel.sync_runtime import DynamoRuntime

            if isinstance(rt, DynamoRuntime):
                return rt._runtime_handle
        except ImportError:
            pass
        return rt

    def set_runtime_holder(self, runtime_holder: list) -> None:
        """Replace the runtime holder reference."""
        self._runtime_holder = runtime_holder

    def set_runtime(self, runtime) -> None:
        """Eagerly set the runtime."""
        self._runtime_holder = [runtime]

    def set_kv_cache_names(self, kv_cache_names: List[str]) -> None:
        """Set the KV cache layer name mapping."""
        self.kv_cache_names = kv_cache_names

    def initialize_xfer_buffers(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Create per-layer aligned transfer buffers from actual KV cache shapes."""
        self._xfer_buffers = {}
        for layer_name, kv_cache in kv_caches.items():
            self._xfer_buffers[layer_name] = aligned_tensor(
                kv_cache.numel()
            ).reshape(kv_cache.shape)
        logger.info(
            "Initialized %d xfer_buffers from kv_caches",
            len(self._xfer_buffers),
        )

    def _group_slots_by_block(
        self, slot_mapping: torch.Tensor
    ) -> dict[int, list[tuple[int, int]]]:
        """Group slot indices by their block index.

        Returns:
            Dict mapping block_idx -> list of (offset_in_block, token_idx).
        """
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for token_idx, slot in enumerate(slot_mapping.tolist()):
            block_idx = slot // self.block_size
            offset = slot % self.block_size
            groups[block_idx].append((offset, token_idx))
        return groups

    @override
    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize connector state from vLLM."""
        if "runtime_holder" in kwargs:
            self._runtime_holder = kwargs["runtime_holder"]
        elif "runtime" in kwargs:
            self._runtime_holder = [kwargs["runtime"]]
        if "kv_cache_names" in kwargs:
            self.kv_cache_names = kwargs["kv_cache_names"]

    # ------------------------------------------------------------------
    # D2H: NPU -> CPU
    # ------------------------------------------------------------------

    @override
    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs
    ):
        """Copy KV cache from RBLN NPU to CPU memory (D2H)."""
        assert memory_obj.tensor is not None
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, (
            "slot_mapping is required for RBLN D2H transfer"
        )
        assert self.runtime is not None, (
            "rebel runtime is required for RBLN D2H transfer"
        )
        assert len(self.kv_cache_names) > 0, (
            "kv_cache_names must be set before D2H transfer"
        )

        num_tokens = end - start
        hidden = self.hidden_dim_size
        dst = memory_obj.tensor.reshape(
            2, self.num_layers, num_tokens, hidden
        )

        chunk_slots = slot_mapping[start:end]
        block_groups = self._group_slots_by_block(chunk_slots)
        assert self._xfer_buffers, (
            "xfer_buffers not initialized -- call initialize_xfer_buffers first"
        )
        xfer_buffers = self._xfer_buffers
        rt = self.runtime

        logger.info(
            "[from_gpu/D2H] start=%d, end=%d, num_tokens=%d, "
            "num_blocks=%d, block_indices=%s",
            start, end, num_tokens,
            len(block_groups), list(block_groups.keys()),
        )

        for block_idx, slots in block_groups.items():
            for layer_idx, kv_name in enumerate(self.kv_cache_names):
                buf = xfer_buffers[kv_name]
                rt.fetch_kv_cache(
                    buf.data_ptr(), block_idx, 0, self.block_size, kv_name,
                )
                for offset, token_idx in slots:
                    if buf.dim() == 6:
                        # [2, num_blocks, num_kv_heads, 1, block_size, head_dim]
                        dst[0, layer_idx, token_idx, :] = buf[
                            0, block_idx, :, 0, offset, :
                        ].reshape(-1)
                        dst[1, layer_idx, token_idx, :] = buf[
                            1, block_idx, :, 0, offset, :
                        ].reshape(-1)
                    else:
                        # [2, num_blocks, block_size, num_kv_heads, head_dim]
                        dst[0, layer_idx, token_idx, :] = buf[
                            0, block_idx, offset
                        ].reshape(-1)
                        dst[1, layer_idx, token_idx, :] = buf[
                            1, block_idx, offset
                        ].reshape(-1)

    # ------------------------------------------------------------------
    # H2D: CPU -> NPU
    # ------------------------------------------------------------------

    @override
    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs
    ):
        """Copy KV cache from CPU memory to RBLN NPU (H2D).

        Uses read-modify-write per block to avoid overwriting tokens
        outside this chunk.
        """
        assert memory_obj.tensor is not None
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, (
            "slot_mapping is required for RBLN H2D transfer"
        )
        assert self.runtime is not None, (
            "rebel runtime is required for RBLN H2D transfer"
        )
        assert len(self.kv_cache_names) > 0, (
            "kv_cache_names must be set before H2D transfer"
        )

        num_tokens = end - start
        hidden = self.hidden_dim_size
        src = memory_obj.tensor.reshape(
            2, self.num_layers, num_tokens, hidden
        )

        chunk_slots = slot_mapping[start:end]
        block_groups = self._group_slots_by_block(chunk_slots)
        assert self._xfer_buffers, (
            "xfer_buffers not initialized -- call initialize_xfer_buffers first"
        )
        xfer_buffers = self._xfer_buffers
        rt = self.runtime

        logger.info(
            "[to_gpu/H2D] start=%d, end=%d, num_tokens=%d, "
            "num_blocks=%d, block_indices=%s",
            start, end, num_tokens,
            len(block_groups), list(block_groups.keys()),
        )

        for block_idx, slots in block_groups.items():
            for layer_idx, kv_name in enumerate(self.kv_cache_names):
                buf = xfer_buffers[kv_name]
                # Read-modify-write: fetch current layer's block first
                rt.fetch_kv_cache(
                    buf.data_ptr(), block_idx, 0, self.block_size, kv_name,
                )
                # Overlay source data onto the fetched block
                for offset, token_idx in slots:
                    if buf.dim() == 6:
                        # [2, num_blocks, num_kv_heads, 1, block_size, head_dim]
                        buf[0, block_idx, :, 0, offset, :] = src[
                            0, layer_idx, token_idx, :
                        ].reshape(self.num_kv_heads, self.head_dim)
                        buf[1, block_idx, :, 0, offset, :] = src[
                            1, layer_idx, token_idx, :
                        ].reshape(self.num_kv_heads, self.head_dim)
                    else:
                        # [2, num_blocks, block_size, num_kv_heads, head_dim]
                        buf[0, block_idx, offset] = src[
                            0, layer_idx, token_idx, :
                        ].reshape(self.num_kv_heads, self.head_dim)
                        buf[1, block_idx, offset] = src[
                            1, layer_idx, token_idx, :
                        ].reshape(self.num_kv_heads, self.head_dim)
                # Write back the modified block
                rt.update_kv_cache(
                    buf.data_ptr(), block_idx, 0, self.block_size, kv_name,
                )

    # ------------------------------------------------------------------
    # Batched operations
    # ------------------------------------------------------------------

    @override
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """Batched D2H transfer."""
        for memory_obj, s, e in zip(memory_objs, starts, ends):
            self.from_gpu(memory_obj, s, e, **kwargs)

    @override
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        """Batched H2D transfer."""
        if memory_objs is None or starts is None or ends is None:
            return
        for memory_obj, s, e in zip(memory_objs, starts, ends):
            self.to_gpu(memory_obj, s, e, **kwargs)

    @override
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Returns: [kv_size=2, num_layers, num_tokens, hidden_dim]"""
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size]
        )


def CreateRBLNConnector(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_names: Optional[List[str]] = None,
    runtime_holder: Optional[list] = None,
) -> RBLNConnector:
    """Factory function to create an RBLN connector."""
    return RBLNConnector(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        kv_cache_names=kv_cache_names,
        runtime_holder=runtime_holder,
    )


# ---------------------------------------------------------------------------
# RBLNLMCacheManager: injects RBLNConnector instead of CUDA GPU connector
# ---------------------------------------------------------------------------


def _rbln_calculate_local_rank_and_world_size(
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Calculate local rank and world size for RBLN NPU.

    Uses rebel.get_npu_count() instead of torch.cuda.device_count().
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


class RBLNLMCacheManager(LMCacheManager):
    """LMCacheManager subclass that uses RBLNConnector for RBLN NPU."""

    def _create_lmcache_engine(self, role: str) -> LMCacheEngine:
        """Create LMCacheEngine with RBLNConnector."""
        from lmcache.integration.vllm.utils import ENGINE_NAME, mla_enabled

        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            return curr_engine

        assert self._vllm_config is not None, (
            "vllm_config required for vLLM mode"
        )

        try:
            from vllm.utils.torch_utils import get_kv_cache_torch_dtype
        except ImportError:
            from vllm.utils import get_kv_cache_torch_dtype

        from vllm.distributed.parallel_state import get_tp_group

        model_config = self._vllm_config.model_config
        parallel_config = self._vllm_config.parallel_config
        cache_config = self._vllm_config.cache_config

        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        use_mla = mla_enabled(model_config)
        self._validate_mla_config(use_mla)

        num_layer = model_config.get_num_layers(parallel_config)
        num_draft_layers = self._calculate_draft_layers()
        num_layer += num_draft_layers
        chunk_size = self._config.chunk_size
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
            "RBLN LMCache init: num_layer=%d, chunk_size=%d, "
            "num_kv_head=%d, head_size=%d, hidden_dim=%d, "
            "use_mla=%s, kv_shape=%s, num_draft_layers=%d",
            num_layer, chunk_size, num_kv_head, head_size,
            num_kv_head * head_size, use_mla, kv_shape, num_draft_layers,
        )

        engine_id = None
        kv_connector_extra_config = None
        if hasattr(self._vllm_config, "kv_transfer_config"):
            kv_transfer_config = self._vllm_config.kv_transfer_config
            if kv_transfer_config is not None:
                engine_id = getattr(kv_transfer_config, "engine_id", None)
                kv_connector_extra_config = getattr(
                    kv_transfer_config,
                    "kv_connector_extra_config",
                    None,
                )

        local_worker_id, local_world_size = (
            _rbln_calculate_local_rank_and_world_size(self._vllm_config)
        )
        metadata = LMCacheMetadata(
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            local_world_size=local_world_size,
            worker_id=parallel_config.rank,
            local_worker_id=local_worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=role,
            served_model_name=model_config.served_model_name,
            chunk_size=self._config.chunk_size,
            engine_id=engine_id,
            kv_connector_extra_config=kv_connector_extra_config,
        )

        if role == "scheduler":
            tpg = SimpleNamespace()
            tpg.broadcast = lambda tensor, src: tensor
            tpg.broadcast_object = lambda obj, src: obj
            rbln_connector = None
        else:
            tpg = get_tp_group()
            rbln_connector = CreateRBLNConnector(
                num_layers=num_layer,
                num_kv_heads=num_kv_head,
                head_dim=head_size,
                block_size=cache_config.block_size,
                dtype=kv_dtype,
            )
            logger.info(
                "Created RBLNConnector: layers=%d, kv_heads=%d, "
                "head_dim=%d, block_size=%d",
                num_layer, num_kv_head, head_size, cache_config.block_size,
            )

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self._config,
            metadata,
            rbln_connector,
            tpg.broadcast,
            tpg.broadcast_object,
        )

        if (
            role == "scheduler"
            and self._config.enable_scheduler_bypass_lookup
        ):
            assert (
                engine.save_only_first_rank
                or self._config.get_extra_config_value(
                    "remote_enable_mla_worker_id_as0", metadata.use_mla
                )
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine


# ---------------------------------------------------------------------------
# RBLNLMCacheConnectorV1Impl: RBLN-specific LMCache connector implementation
# ---------------------------------------------------------------------------


class RBLNLMCacheConnectorV1Impl(LMCacheConnectorV1Impl):
    """RBLN-specific LMCache connector implementation.

    Uses RBLNLMCacheManager instead of the upstream LMCacheManager.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        # Skip LMCacheConnectorV1Impl.__init__ to avoid upstream
        # LMCacheManager instantiation (which calls get_vllm_torch_dev
        # and fails on RBLN).
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = (
            vllm_config.parallel_config.tensor_parallel_size
        )

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

        self._manager = RBLNLMCacheManager(
            config=config,
            vllm_config=vllm_config,
            role=role.name.lower(),
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
        """Pass the runtime_holder list reference to RBLNConnector."""
        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector
        connector.set_runtime_holder(runtime_holder)
        logger.info("RBLN runtime_holder set on RBLNConnector (lazy)")

    def set_runtime(self, runtime) -> None:
        """Eagerly set the rebel runtime."""
        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector
        connector.set_runtime(runtime)
        logger.info("RBLN runtime set on RBLNConnector")

    @staticmethod
    def _normalize_kv_shape(
        kv_caches: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Normalize RBLN 6D KV cache tensors to the 5D format LMCache expects.

        RBLN: [2, B, H, 1, S, D] -> LMCache: [2, B, S, H, D]
        """
        normalized = {}
        for name, tensor in kv_caches.items():
            if tensor.dim() == 6 and tensor.shape[3] == 1:
                tensor = tensor.squeeze(3).permute(0, 1, 3, 2, 4).contiguous()
            normalized[name] = tensor
        return normalized

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches and extract layer names for runtime API."""
        normalized_kv_caches = self._normalize_kv_shape(kv_caches)
        super().register_kv_caches(normalized_kv_caches)

        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector
        index2name: dict[int, list[str]] = defaultdict(list)
        for layer_name in kv_caches:
            index2name[extract_layer_index(layer_name)].append(layer_name)
        kv_cache_names = [
            names[0]
            for _, names in sorted(index2name.items())
        ]
        connector.set_kv_cache_names(kv_cache_names)
        logger.info(
            "Set %d kv_cache_names on RBLNConnector",
            len(kv_cache_names),
        )

        # Initialize per-layer xfer buffers from original (pre-normalize)
        # kv_caches so buffer shapes match what the runtime expects.
        sorted_kv_caches = {
            name: kv_caches[name] for name in kv_cache_names
        }
        connector.initialize_xfer_buffers(sorted_kv_caches)


# ---------------------------------------------------------------------------
# RBLNLMCacheConnectorV1: vLLM entry point connector
# ---------------------------------------------------------------------------


class RBLNLMCacheConnectorV1(LMCacheConnectorV1Dynamic):
    """RBLN KV connector for vLLM v1.

    Usage in vllm-rbln:
        Set kv_connector to this class path in vLLM config.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ):
        if kv_cache_config is not None:
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        else:
            super().__init__(vllm_config=vllm_config, role=role)
        # Replace upstream impl with RBLN version
        self._lmcache_engine = RBLNLMCacheConnectorV1Impl(
            vllm_config, role, self
        )

    def set_runtime(self, runtime) -> None:
        """Set the rebel runtime. Called by vllm-rbln model runner."""
        self._lmcache_engine.set_runtime(runtime)
