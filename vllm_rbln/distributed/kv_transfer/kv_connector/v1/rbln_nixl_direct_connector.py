# Copyright 2025 Rebellions Inc. All rights reserved.
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
"""Direct (D2D) RBLN NIXL connector — sibling of RblnNixlConnector with no CPU
host-staging/bounce: registers KV cache tensors straight with NIXL's RBLN
backend (NPU dmabuf MR) via the nixl_rbln adapter. Opt in with
kv_transfer_config.kv_connector="RblnNixlDirectConnector".
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    nixl_connector as _up,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorRole,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
    RblnNixlConnector,
    RblnNixlConnectorScheduler,
    RblnNixlConnectorWorker,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlDirectConnector(RblnNixlConnector):
    """Same scheduler-side behavior as RblnNixlConnector; the
    difference is the worker, which registers KV caches directly
    instead of bouncing through a CPU staging buffer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # Do NOT call RblnNixlConnector.__init__: it asserts
        # kv_buffer_device != "rbln". The direct path *requires* "rbln",
        # so we replicate the small base body here, swapping in the
        # direct worker and the opposite assert.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        assert vllm_config.kv_transfer_config.kv_buffer_device == "rbln", (
            "RblnNixlDirectConnector is device-to-device only "
            "(kv_buffer_device='rbln'). For the host-bounce path "
            "(kv_buffer_device='cpu') use RblnNixlConnector."
        )
        self.kv_cache_config = kv_cache_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = RblnNixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = RblnNixlDirectConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    def finalize_kv_cache_registration(self) -> None:
        """Run the worker's deferred NIXL registration. Invoked by
        RBLNWorker.compile_or_warm_up_model after warm-up has allocated
        the KV cache physical views (see
        RblnNixlDirectConnectorWorker.register_kv_caches for why this is
        deferred). No-op on the scheduler side."""
        if self.connector_worker is not None:
            self.connector_worker.finalize_kv_cache_registration()

    def set_runtime_holder(self, runtime_holder) -> None:
        """Receive the model runner's runtime_holder (propagated by
        RBLNModelRunner._propagate_runtime_holder). The worker reads the
        RblnContext pointer straight from the live runtime, avoiding a
        device-keyed global Context lookup. No-op on the scheduler side."""
        if self.connector_worker is not None:
            self.connector_worker._runtime_holder = runtime_holder


class RblnNixlDirectConnectorWorker(RblnNixlConnectorWorker):
    """Worker that registers KV caches via the `nixl_rbln` adapter
    rather than allocating per-layer CPU staging buffers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        assert vllm_config.kv_transfer_config.kv_buffer_device == "rbln", (
            "RblnNixlDirectConnectorWorker requires kv_buffer_device='rbln'."
        )
        super().__init__(vllm_config, engine_id, kv_cache_config)
        try:
            import nixl_rbln  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "RblnNixlDirectConnector requires the 'nixl-rbln' "
                "adapter package to be installed in the same venv. "
                f"(import failed: {e})"
            ) from e

        # Direct path registers NPU memory, not host DRAM. The platform
        # reports "DRAM" (host-bounce default) so we override here.
        self.nixl_memory_type = "VRAM"
        self.nixl_backends = ["RBLN"]
        self._runtime_holder = None

    # ---- host-bounce removal ------------------------------------------

    def initialize_host_xfer_buffer(
        self, kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Direct path: there is no host xfer buffer. NIXL talks to
        NPU memory through the dmabuf MR the RBLN plugin built; no
        staging needed."""
        if self.use_host_buffer:
            logger.info(
                "RblnNixlDirectConnector: ignoring kv_buffer_device='cpu' "
                "— direct-vmem path registers NPU memory with NIXL "
                "directly, no host staging buffer needed.",
            )

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp) -> None:
        """No copy ops to wire — see `initialize_host_xfer_buffer`."""
        return

    # ---- deferred registration (post-warmup) --------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Stash the caches; register in finalize_kv_cache_registration().

        Called before warm-up, when the KV vmem entries are still EMPTY (no
        physical view) so vaddr->dva would fail. RBLNWorker calls finalize
        after warm-up has materialized the (lifetime-stable) physical views.
        """
        self._pending_kv_caches = kv_caches
        logger.info(
            "RblnNixlDirectConnector: deferring registration of %d KV "
            "cache layer(s) until after warm-up (physical views are not "
            "allocated yet).",
            len(kv_caches),
        )

    def finalize_kv_cache_registration(self) -> None:
        """Run the deferred NIXL registration. Idempotent; a no-op if
        registration already ran or there is nothing pending. Called by
        RBLNWorker after warm_up_model() has allocated the KV cache
        physical views."""
        pending = getattr(self, "_pending_kv_caches", None)
        if pending is None:
            return
        self._pending_kv_caches = None
        self._register_kv_caches_impl(pending)

    # ---- the actual direct register path ------------------------------

    def _register_kv_caches_impl(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Direct-vmem variant of NixlConnectorWorker.register_kv_caches:
        build the upstream topology, hand the logical K/V regions to
        nixl_rbln.register_kv_regions (vaddr->dva, chiplet sharding, MR reg),
        and feed the returned physical tables into the base transfer state.
        """
        import nixl_rbln

        self.kv_topo = _up.TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,
            remote_block_size=self._block_size,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = _up.compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.kv_topo.cross_layers_blocks
        )

        # Device id for the RBLN backend's RblnContext.
        device_id = next(iter(kv_caches.values())).get_device()
        assert device_id >= 0, (
            "RblnNixlDirectConnector: KV cache is not an 'rbln' device tensor "
            "(is VLLM_RBLN_USE_DEVICE_TENSOR=1 set?)."
        )

        # Direct path never stages through a host buffer.
        assert not self.use_host_buffer
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, (
            "host_xfer_buffer should not be initialized when "
            f"kv_buffer_device is {self.kv_buffer_device}"
        )

        logger.info(
            "Registering KV_Caches (direct vmem). use_mla: %s, "
            "kv_buffer_device: %s, device_id: %d",
            self.use_mla,
            self.kv_buffer_device,
            device_id,
        )

        tensor_size_bytes = None
        # Logical K/V regions (entry_tensor, byte_offset, full_block_len) for
        # nixl-rbln, which does vaddr->dva, chiplet sharding and MR registration
        # and returns the physical tables — so this connector never sees dvas or
        # the chiplet count, only torch tensors and logical byte offsets.
        regions: list[tuple[Any, int, int]] = []
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs[layer_name]
            if isinstance(layer_spec, _up.UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.kv_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, _up.MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate regions.
            physical_page_size = physical_page_size // len(cache_list)
            if self.kv_topo._cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, _up.MambaSpec)
                else self.num_blocks
            )
            curr_tensor_size_bytes = num_blocks * physical_page_size
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes

            # Materialize the vmem physical view: device-tensor KV caches are
            # torch.empty(device="rbln") (EMPTY, no physical view) until first
            # written, but vaddr->dva needs it. A one-time zero_() allocates it
            # (harmless); the vmem entry is stable for the worker's lifetime.
            cache_or_caches.zero_()

            # Collect this entry's logical K/V regions. K and V are views into
            # one vmem entry (V at entry_base + K_size); pass the entry tensor
            # plus the view's byte offset so nixl-rbln converts the entry base
            # (and its chiplet shards) once.
            entry_base_vaddr = cache_or_caches.data_ptr()
            for cache in cache_list:
                region_offset = cache.data_ptr() - entry_base_vaddr
                if isinstance(layer_spec, _up.MambaSpec):
                    full_block_len = (
                        physical_page_size
                        // self._physical_blocks_per_logical_kv_block
                    )
                else:
                    full_block_len = physical_page_size

                assert cache.shape[0] == num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                if not self.use_mla:
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                regions.append((cache_or_caches, region_offset, full_block_len))

        # RblnContext pointer straight from the live model runtime (propagated
        # via set_runtime_holder). No device-keyed global-Context fallback:
        # the runtime holder must be set by the time we register.
        assert self._runtime_holder, (
            "RblnNixlDirectConnector: runtime_holder not set — RBLNModelRunner "
            "must propagate it via set_runtime_holder before registration."
        )
        rbln_ctx_ptr = self._runtime_holder[0]._runtime_handle.get_context().rbln_ctx_ptr

        # Delegate vmem-vaddr -> NPU-dva, chiplet sharding and MR registration
        # to nixl-rbln. It registers one whole-entry MR per chiplet shard and
        # returns the physical transfer tables (base addrs + block lens),
        # already chiplet-expanded so the base connector's descriptor math is
        # correct without this connector knowing the chiplet count.
        xfer = nixl_rbln.register_kv_regions(
            self.nixl_wrapper, regions, device_id, mem=self.nixl_memory_type,
            rbln_ctx_ptr=rbln_ctx_ptr,
        )
        self.device_id = device_id
        self.block_len_per_layer = list(xfer.block_lens)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = xfer.base_addrs
        self._registered_descs.append(xfer.reg_handle)
        assert len(self.block_len_per_layer) == len(xfer.base_addrs)

        self.num_regions = len(xfer.base_addrs)
        if self.kv_topo.is_kv_layout_blocks_first:
            self.num_regions *= 2
        self.num_descs = self.num_regions * self.num_blocks

        logger.info(
            "RblnNixlDirectConnector: registered %d transfer region(s) "
            "across %d chiplet shard(s) (K/V split).",
            self.num_regions, xfer.n_shards,
        )

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = _up.NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
        )
        assert self.compat_hash is not None
        encoder = _up.msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = _up.NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )
