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

import msgspec
import numpy as np
import rebel
import torch
from rebel.kv_cache import aligned_tensor
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlConnectorWorker(NixlConnectorWorker):
    """RBLN's KV connector worker.

    The runner filters `kv_caches` to one Full-attention canonical layer
    per HMA pool before `register_kv_caches`, so upstream's
    `cache.shape[0] == num_blocks` invariant holds without a bigger
    override (see `RBLNModelRunner._select_canonical_kv_layers_per_pool`).

    Not supported: pure-SWA single-group with `sliding_window < block_size`
    under a KV connector — the canonical-layer fallback picks the SWA
    layer (kernel granularity), whose `cache.shape[0]` mismatches
    `num_blocks`. Non-disagg serving is unaffected.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Pick the NIXL transport backend.
        #   * nixl-rbln installed  → use RBLN backend on both paths
        #     (host-bounce DRAM_SEG / D2D VRAM_SEG via ibv_reg{,_dmabuf}_mr).
        #   * nixl-rbln absent     → upstream defaults stand (UCX backend,
        #     DRAM only). D2D (`kv_buffer_device="rbln"`) requires the
        #     RBLN backend and is rejected here.
        try:
            import nixl_rbln  # noqa: F401

            self._use_rbln_nixl_backend = True
        except ImportError:
            self._use_rbln_nixl_backend = False

        if self._use_rbln_nixl_backend:
            self.nixl_backends = ["RBLN"]
            # D2D registers VRAM (device dmabuf); host-bounce keeps DRAM.
            if self.kv_buffer_device == "rbln":
                self.nixl_memory_type = "VRAM"
        elif self.kv_buffer_device == "rbln":
            raise RuntimeError(
                "kv_buffer_device='rbln' (D2D) requires the 'nixl-rbln' "
                "adapter package; install it or set kv_buffer_device='cpu' "
                "to fall back to the upstream NIXL (UCX) host-bounce path."
            )
        else:
            logger.info(
                "RblnNixlConnectorWorker: nixl-rbln not available — "
                "using upstream NIXL (UCX) on the host-bounce path."
            )

        # `RblnPlatform.device_type = "cpu"` makes upstream skip the host
        # buffer; restore it — NIXL cannot register RBLN device memory.
        self.use_host_buffer = self.kv_buffer_device == "cpu"

        self._pending_kv_caches: dict[str, torch.Tensor] | None = None

        # Pin to logical values. Upstream would otherwise multiply by the
        # attention backend's kernel ratio, which doesn't reflect per-spec
        # ratios in hybrid models.
        self.num_blocks = self.kv_cache_config.num_blocks
        self.block_size = self.vllm_config.cache_config.block_size
        self._physical_blocks_per_logical_kv_block = 1
        self._logical_num_blocks = self.num_blocks

        # SWA-side desc layout: publish a second `sliding_window`-length
        # descriptor range alongside the Full-length range at the same
        # NIXL base addresses, so SWA groups transport only the actually-
        # populated prefix over RDMA (the runtime always pins SWA's
        # kernel slot 0 at the block's base offset). Storage stays Full-
        # tiled, host h2d/d2h still moves the full block — only
        # `register_local_xfer_handler` / `add_remote_agent` /
        # `_compute_desc_ids` consult `_sw_ratio` / `_group_specs`.
        # `_sw_ratio is None` collapses every override to upstream's
        # Full-only desc layout.
        self._group_specs: list[Any] = []
        self._sw_ratio: int | None = None
        if envs.VLLM_RBLN_NIXL_SWA_VIEW_OPT:
            self._group_specs = [
                g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
            ]
            for spec in self._group_specs:
                if not isinstance(spec, SlidingWindowSpec):
                    continue
                assert spec.block_size % spec.sliding_window == 0
                ratio = spec.block_size // spec.sliding_window
                if ratio == 1:
                    continue
                if self._sw_ratio is None:
                    self._sw_ratio = ratio
                else:
                    assert self._sw_ratio == ratio, (
                        "RBLN NIXL connector assumes a single SWA ratio "
                        f"across groups, got {self._sw_ratio} vs {ratio}"
                    )
            if self._sw_ratio is not None:
                logger.info(
                    "VLLM_RBLN_NIXL_SWA_VIEW_OPT=1: trimming SWA-group "
                    "RDMA payload by 1/%d (sliding_window-sized descs "
                    "alongside Full descs at shared base addrs).",
                    self._sw_ratio,
                )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Wire KV caches into NIXL.

        D2D (`kv_buffer_device="rbln"`): stash and defer; backing memory
        isn't materialized until warm-up. Backend creation happens in
        `_register_kv_caches_impl` via `nixl_rbln.register_kv_regions`.

        Host-bounce (`kv_buffer_device="cpu"`): when nixl-rbln is
        available, create the RBLN backend on the agent so upstream's
        `register_memory(..., backends=["RBLN"])` resolves. Otherwise
        fall straight through to upstream's UCX-backed registration —
        the host xfer buffers are plain DRAM in either case.
        """
        if self.kv_buffer_device == "rbln":
            self._pending_kv_caches = kv_caches
            logger.info(
                "RblnNixlConnectorWorker (D2D): deferring registration of "
                "%d KV cache layer(s) until after warm-up.",
                len(kv_caches),
            )
            return
        if self._use_rbln_nixl_backend:
            import nixl_rbln

            nixl_rbln.ensure_rbln_backend(self.nixl_wrapper, device_id=0)
        super().register_kv_caches(kv_caches)

    def finalize_kv_cache_registration(self) -> None:
        """Run the deferred D2D registration. No-op on host-bounce and
        on re-entry (idempotent via `_pending_kv_caches`)."""
        if self._pending_kv_caches is None:
            return
        pending = self._pending_kv_caches
        self._pending_kv_caches = None
        self._register_kv_caches_impl(pending)

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Allocate one rebel-aligned host buffer per layer."""
        assert self.kv_cache_layout == "HND", (
            "RBLN NIXL Connector only supports HND layout"
        )
        xfer_buffers: dict[str, torch.Tensor] = {}

        def _aligned_like(kv_cache: torch.Tensor) -> torch.Tensor:
            """Page-aligned host buffer with `kv_cache`'s shape and dtype.
            `aligned_tensor` only knows fp16 (numpy has no bfloat16), so
            we size by byte count and view-cast to the target dtype."""
            bytes_needed = kv_cache.numel() * kv_cache.element_size()
            assert bytes_needed % 2 == 0, (
                "kv_cache byte footprint must be a multiple of 2 "
                f"(aligned_tensor backing dtype), got {bytes_needed}"
            )
            raw_fp16 = aligned_tensor(bytes_needed // 2)
            return raw_fp16.view(kv_cache.dtype).view(kv_cache.shape)

        try:
            for layer_name, kv_cache in kv_caches.items():
                xfer_buffers[layer_name] = _aligned_like(kv_cache)
        except MemoryError as e:
            logger.error("RblnNixlConnectorWorker gets %s", e)
            raise

        keys_preview = list(xfer_buffers.keys())
        if len(keys_preview) > 8:
            keys_preview = keys_preview[:4] + ["..."] + keys_preview[-4:]
        logger.info(
            "Host xfer buffers allocated: %d pool(s) (keys e.g. %s)",
            len(xfer_buffers),
            keys_preview,
        )

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used.

        Overrides upstream only to drop its `device_type == "cpu"` early
        return: RblnPlatform reports `device_type == "cpu"` yet still needs
        the host-buffer copies wired up on the host-bounce path.
        """
        if self.kv_buffer_device != "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _register_kv_caches_impl(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Direct variant of NixlConnectorWorker.register_kv_caches:
        build the upstream topology, hand the logical K/V regions to
        `nixl_rbln.register_kv_regions` (address translation, sharding,
        MR reg), and feed the returned transfer tables into the base
        transfer state.
        """
        import nixl_rbln

        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )

        # Device id for the RBLN backend's RblnContext.
        sample_kv_cache = next(iter(kv_caches.values()))
        device_id = sample_kv_cache.get_device()
        assert device_id >= 0, (
            "RblnNixlConnectorWorker (D2D): KV cache is not an 'rbln' "
            "device tensor (is VLLM_RBLN_USE_DEVICE_TENSOR=1 set?)."
        )

        # Direct path never stages through a host buffer.
        assert not self.use_host_buffer
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, (
            "host_xfer_buffer should not be initialized when "
            f"kv_buffer_device is {self.kv_buffer_device}"
        )

        logger.info(
            "Registering KV_Caches (direct). use_mla: %s, "
            "kv_buffer_device: %s, device_id: %d",
            self.use_mla,
            self.kv_buffer_device,
            device_id,
        )

        tensor_size_bytes = None
        # Logical K/V regions (entry_tensor, byte_offset, full_block_len)
        # for nixl-rbln.
        regions: list[tuple[Any, int, int]] = []
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs[layer_name]
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate
            # regions.
            physical_page_size = physical_page_size // len(cache_list)
            if self.transfer_topo._cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            curr_tensor_size_bytes = num_blocks * physical_page_size
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes

            # Materialize the backing memory of kv_cache.
            cache_or_caches.zero_()

            # Collect this entry's logical K/V regions.
            entry_base_addr = cache_or_caches.data_ptr()
            for cache in cache_list:
                region_offset = cache.data_ptr() - entry_base_addr
                if isinstance(layer_spec, MambaSpec):
                    full_block_len = (
                        physical_page_size // self._physical_blocks_per_logical_kv_block
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

        rbln_ctx_ptr = rebel.context_of(sample_kv_cache).rbln_ctx_ptr

        # Delegate sharding and MR registration to nixl-rbln. It registers
        # one whole-entry MR per shard and returns the transfer tables
        # (base addrs + block lens), already shard-expanded so the base
        # connector's descriptor math is correct without this connector
        # knowing the shard count.
        xfer = nixl_rbln.register_kv_regions(
            self.nixl_wrapper,
            regions,
            device_id,
            mem=self.nixl_memory_type,
            rbln_ctx_ptr=rbln_ctx_ptr,
        )
        self.device_id = device_id
        self.block_len_per_layer = list(xfer.block_lens)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = xfer.base_addrs
        self._registered_descs.append(xfer.reg_handle)
        assert len(self.block_len_per_layer) == len(xfer.base_addrs)

        self.num_regions = len(xfer.base_addrs)
        if self.transfer_topo.is_kv_layout_blocks_first:
            self.num_regions *= 2
        self.num_descs = self.num_regions * self.num_blocks

        logger.info(
            "RblnNixlConnectorWorker (D2D): registered %d transfer "
            "region(s) across %d shard(s) (K/V split).",
            self.num_regions,
            xfer.n_shards,
        )

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
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
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    # ------------------------------------------------------------------
    # Hybrid Full + SWA desc layout (RDMA payload only)
    # ------------------------------------------------------------------
    #
    # The runner registers one canonical Full-attention layer per HMA
    # pool, so the underlying NIXL memory regions are Full-sized
    # (block_size bytes per region per block). When
    # `VLLM_RBLN_NIXL_SWA_VIEW_OPT=1` is set and at least one group is
    # SWA, two desc ranges are published at the same base addresses:
    #
    #   [0, num_full_descs):
    #       Full-size descriptors (length `block_size`).  Read by
    #       Full-attention groups.
    #
    #   [num_full_descs, 2 * num_full_descs):
    #       SWA-size descriptors at the same base addresses (length
    #       `sliding_window`).  Read by SWA groups, which only need the
    #       first `sliding_window` bytes — the runtime always pins
    #       SWA's kernel slot 0 at the block's base offset, so the SWA
    #       payload is a contiguous prefix of the Full block.
    #
    # `_compute_desc_ids` routes per-group block lists into the
    # right range. When `_sw_ratio is None` (env off, or no SWA group,
    # or degenerate ratio == 1) every method below collapses to
    # upstream's single Full-range layout.
    #
    # Host-side h2d/d2h still copies the full block — only the over-
    # the-wire RDMA payload is trimmed.  The garbage tail SWA receives
    # back into the SWA-layer block is never read by the kernel
    # (attention reads only `[0, sliding_window)`), and the canonical
    # filter guarantees the storage's Full alias is never co-allocated
    # to the same block id (scheduler keeps Full/SWA block-id pools
    # disjoint).

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        if self._sw_ratio is None:
            # No SWA view opt: upstream's Full-only desc layout applies.
            return super().register_local_xfer_handler(block_size)
        assert self.transfer_topo is not None
        assert not self.transfer_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout (K and V in "
            "separate regions), not FlashInfer."
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba layers."

        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        num_blocks = self.num_blocks * block_size_ratio
        blocks_data: list[tuple[int, int, int]] = []

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses but `sliding_window`-sized.
        length_divisors = [1] if self._sw_ratio is None else [1, self._sw_ratio]
        for divisor in length_divisors:
            for i, base_addr in enumerate(local_base_addresses):
                kv_block_len = (
                    self.get_backend_aware_kv_block_len(
                        layer_idx=i, first_split=True, mamba_view=False
                    )
                    // block_size_ratio
                    // divisor
                )
                stride = self.block_len_per_layer[i] // block_size_ratio
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * stride
                    blocks_data.append((addr, kv_block_len, self.device_id))

        logger.debug(
            "Created %s local blocks (%s) for engine %s rank %s",
            len(blocks_data),
            "Full + SWA" if self._sw_ratio is not None else "Full",
            self.engine_id,
            self.tp_rank,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        if self._sw_ratio is None:
            # No SWA view opt: upstream handles Full-only remote descs.
            return super().add_remote_agent(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )
        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank %s already "
                "exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][remote_tp_rank]

        # vLLM 0.22 base.add_remote_agent registers the remote engine in the
        # TransferTopology and builds its TPMapping BEFORE any
        # block_size_ratio / tp_ratio / get_engine_info lookup (used below and
        # in _validate_remote_agent_handshake). The _sw_ratio-is-None path
        # delegates to super() which does this; the SWA-view-opt path must
        # replicate the prelude or get_engine_info() raises KeyError.
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=nixl_agent_meta.block_size,
                remote_block_len=nixl_agent_meta.block_lens[0],
                remote_physical_blocks_per_logical=(
                    nixl_agent_meta.physical_blocks_per_logical_kv_block
                ),
            ),
        )
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        kv_topo = self.transfer_topo
        assert not kv_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout."
        )
        assert not self.use_mla, "RBLN NIXL connector does not support MLA."

        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        indexes_into_remote = (
            not self.transfer_topo.is_kv_replicated(engine_id) and tp_ratio > 0
        )

        # Heterogeneous TP path (P TP > D TP): logically split own regions
        # into |tp_ratio| chunks. Mirrors upstream; preserved verbatim
        # because RBLN may run heterogeneous TP in the future.
        if (
            tp_ratio < 0
            and not self.use_mla
            and tp_ratio not in self.src_xfer_handles_by_tp_ratio
        ):
            self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
            for i in range(-tp_ratio):
                split_blocks_data = []
                for memory_region in self.src_blocks_data:
                    addr, local_block_len, own_tp_rank = memory_region
                    remote_block_len = local_block_len // (-tp_ratio)
                    addr = addr + i * remote_block_len
                    split_blocks_data.append((addr, remote_block_len, own_tp_rank))
                descs = self.nixl_wrapper.get_xfer_descs(
                    split_blocks_data, self.nixl_memory_type
                )
                handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
                self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)

        blocks_data: list[tuple[int, int, int]] = []
        num_blocks = nixl_agent_meta.num_blocks

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses (same `page_size` stride — the
        # remote tensor's physical block stride is still Full-sized),
        # shorter desc length.
        length_divisors = [1] if self._sw_ratio is None else [1, self._sw_ratio]
        for divisor in length_divisors:
            for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
                local_block_len = self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=True, mamba_view=False
                )
                remote_kv_block_len = local_block_len // block_size_ratio
                if block_size_ratio > 1:
                    local_block_len = remote_kv_block_len
                if tp_ratio < 0 and not self.use_mla:
                    local_block_len = local_block_len // (-tp_ratio)
                desc_len = local_block_len // divisor
                rank_offset = (
                    self.tp_rank % tp_ratio * remote_kv_block_len
                    if indexes_into_remote
                    else 0
                )
                page_size = nixl_agent_meta.block_lens[i]
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * page_size + rank_offset
                    blocks_data.append((addr, desc_len, nixl_agent_meta.device_id))

        logger.debug(
            "Created %s remote blocks (%s) for dst engine %s "
            "remote rank %s local rank %s",
            len(blocks_data),
            "Full + SWA" if self._sw_ratio is not None else "Full",
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

        if block_size_ratio > 1:
            self.src_xfer_handles_by_block_size[nixl_agent_meta.block_size] = (
                self.register_local_xfer_handler(nixl_agent_meta.block_size)[0]
            )

        return remote_agent_name

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        if self._sw_ratio is None:
            # No SWA view opt: upstream's Full/SSM desc layout applies.
            return super()._compute_desc_ids(
                block_ids,
                dst_num_blocks,
                block_size_ratio,
                physical_blocks_per_logical,
            )

        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)

        region_ids = np.arange(self.num_regions)[:, None]
        num_full_descs = self.num_regions * num_blocks
        all_descs: list[np.ndarray] = []
        for g, group in enumerate(block_ids):
            if not group:
                continue
            is_sw = isinstance(self._group_specs[g], SlidingWindowSpec)
            offset = num_full_descs if is_sw else 0
            group_arr = np.asarray(group)[None, :]
            all_descs.append((region_ids * num_blocks + group_arr + offset).flatten())
        return np.concatenate(all_descs) if all_descs else np.empty(0, dtype=int)
