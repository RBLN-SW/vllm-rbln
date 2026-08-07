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

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import rebel
import torch
import zmq
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
    GET_META_MSG,
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlAgentMetadata,
    rbln_pp_compat_hash,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
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

    Supported prefill -> decode topologies
    --------------------------------------
    The asserts scattered through this file each guard one axis; this is the
    combined picture. `D2D` is `kv_buffer_device="rbln"`, `host` is `"cpu"`.

      axis                        D2D                      host-bounce
      TP equal (P_TP == D_TP)     yes, positional pairing  yes (upstream)
      TP  P_TP <  D_TP            yes, head-band matched   yes (upstream)
      TP  P_TP >  D_TP            yes, head-band matched,  yes (upstream)
                                  fanned in over the peer
                                  ranks that hold our band,
                                  each region cut into as
                                  many pieces as it spans
                                  peer slices — only while
                                  one chiplet area does not
                                  straddle two peer RANKS,
                                  i.e. P_TP/D_TP <=
                                  min(chiplets, heads per
                                  decode rank)
      PP on prefill               yes, layer-band matched — composes with the
                                  head-band matching above, so a PP prefill may
                                  also have a smaller TP than the decode side
                                  (a LARGER one is rejected under PP: fanning
                                  in and staging both at once is untested)
      PP on decode                yes, if P_PP is a multiple of D_PP, but only
                                  with equal TP: both axes split on both sides
                                  at once is untested and rejected
      DP either side              yes — a DP replica is a separate engine_id
                                  and side-channel port; invisible here
      EP either side              yes — EP does not shard the KV cache

    Why D2D is the restricted one: it publishes one region PER CHIPLET AREA,
    so heads are area-major and region i means different heads on two peers
    with different TP. Host-bounce registers one logical full-shape buffer per
    layer, which is exactly upstream's model, so it inherits upstream's wider
    support and is the fallback for anything above marked NO.

    Also unsupported anywhere: SWA view-opt combined with either heterogeneous
    TP or PP; Mamba/SSM with PP; cross-layer-blocks with PP; more than one
    KV-cache group with PP; unequal P/D block sizes.
    """

    compat_hash: str | None
    xfer_handshake_metadata: NixlHandshakePayload | None

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

        # --- Chiplet geometry of one KV entry (D2D only) ---
        # Set from nixl_rbln.register_kv_regions. Host-bounce registers logical
        # full-shape buffers and never expands per area, so the defaults below
        # are its permanent (and correct) values.
        self._kv_areas: int = 1
        self._kv_slices: int = 1
        self._kv_slice_ids: list[int] = []

        # --- Pipeline-parallel (PP) P/D state (empty / inert for pp_size == 1) ---
        # Per remote producer shard, the ordered KV-cache layer names it owns,
        # keyed by engine_id -> global_rank (= pp_rank * tp_size + tp_rank,
        # == pp_rank when tensor parallelism is off).
        self._remote_shard_layer_names: defaultdict[str, dict[int, tuple[str, ...]]] = (
            defaultdict(dict)
        )
        # engine_id -> producer pp_size (discovered at handshake).
        self._remote_pp_size: dict[str, int] = {}
        # engine_id -> the producer stages (flat global ranks) whose layers this
        # rank owns; drives _read_blocks_for_req.
        self._overlapping_ranks: defaultdict[str, list[int]] = defaultdict(list)
        # Per producer shard, a local xfer dlist scoped to that shard's local
        # region subset, keyed by (engine_id, global_rank, block_size); and the
        # shard's per-region KV-group ids, keyed by (engine_id, global_rank).
        self.src_xfer_handles_by_remote: dict[tuple[str, int, int], int] = {}
        self._shard_region_group_ids: dict[tuple[str, int], tuple[int, ...]] = {}
        # How many descriptors each of that shard's regions is cut into (1
        # unless our chiplet area is coarser than the peer's).
        self._shard_desc_split: dict[tuple[str, int], int] = {}
        # Ordered local KV-cache layer names (one per layer), captured at
        # register_kv_caches.
        self.local_seen_layer_names: list[str] = []

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
        # Capture the ordered local layer names before any deferral so the PP
        # metadata publish (and the consumer-side name->region matching) can
        # use them; the D2D path re-uses these at finalize time.
        self.local_seen_layer_names = list(kv_caches.keys())
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
        # Re-wrap the base-published handshake metadata with this stage's PP
        # identity + owned layer names (no-op degrade for pp_size == 1).
        if self.xfer_handshake_metadata is not None:
            base_agent_metadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
                self.xfer_handshake_metadata.agent_metadata_bytes
            )
            self._publish_pp_handshake_metadata(base_agent_metadata, kv_caches.keys())

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

        # n_shards = physical chiplet areas; slices = distinct logical slices.
        # They differ when the compiler replicates a KV head across chiplets
        # (fewer heads than chiplets), which is what the byte arithmetic in
        # register_kv_regions divides by. Kept for diagnosis and for the
        # region-pairing guard.
        self._kv_areas = xfer.n_shards
        self._kv_slices = xfer.slices
        self._kv_slice_ids = list(xfer.slice_ids)
        logger.info(
            "RblnNixlConnectorWorker (D2D): registered %d transfer "
            "region(s) across %d chiplet area(s), %d logical slice(s)%s.",
            self.num_regions,
            xfer.n_shards,
            xfer.slices,
            " -- KV heads are replicated across chiplets"
            if xfer.n_shards != xfer.slices
            else "",
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
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
        )
        # Publish the handshake metadata wrapped with this stage's PP identity
        # and owned layer names (degrades to no-PP for pp_size == 1).
        self._publish_pp_handshake_metadata(
            agent_metadata, self.device_kv_caches.keys()
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
        *,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
        split: int = 1,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        if self._sw_ratio is None:
            if registered_layer_names is None and peer_areas is None and split == 1:
                # No SWA view opt, whole-engine peer: upstream's Full-only
                # layout, one handle covering every region.
                return super().register_local_xfer_handler(block_size)
            # Per-peer shard: register only the regions this peer serves --
            # a pipeline stage's layers, a finer-grained TP peer's chiplet
            # areas, or both -- and cut each into `split` pieces when our area
            # is coarser than the peer's.
            return self._register_shard_local_xfer_handler(
                block_size, registered_layer_names or self.local_seen_layer_names,
                peer_areas=peer_areas, split=split,
            )
        assert registered_layer_names is None and peer_areas is None and split == 1, (
            "RBLN NIXL: SWA view-opt is not supported with pipeline "
            "parallelism or heterogeneous tensor parallelism"
        )
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
        # _sw_ratio is not None here (the None case returned early above).
        length_divisors = [1, self._sw_ratio]
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

    @staticmethod
    def _slice_head_bounds(
        tp_rank: int, tp_size: int, total_kv_heads: int, areas: int, slices: int
    ) -> tuple[int, int]:
        """(first head this shard owns, heads per logical slice).

        A shard owns ``total_kv_heads // tp_size`` heads and the compiler cuts
        them into ``slices`` equal pieces, one per chiplet area -- except that
        when the shard owns fewer heads than the device has chiplets, several
        areas carry a replica of the same piece (``areas // slices`` of them,
        with the replication axis innermost, hence
        ``slice_id = area // (areas // slices)``).

        So logical slice ``s`` of this shard covers heads
        ``[base + s * per_slice, + per_slice)``.
        """
        assert total_kv_heads % tp_size == 0, (
            f"KV heads {total_kv_heads} not divisible by TP size {tp_size}"
        )
        heads_per_rank = total_kv_heads // tp_size
        assert slices > 0 and heads_per_rank % slices == 0, (
            f"{heads_per_rank} heads not divisible into {slices} slices"
        )
        assert areas % slices == 0, (
            f"{areas} chiplet areas not a whole number of {slices} slices"
        )
        return tp_rank * heads_per_rank, heads_per_rank // slices

    def _build_head_matched_remote(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
    ) -> list[tuple[int, int, int]]:
        """Remote descriptors for a peer whose TP degree differs from ours.

        Emitted in LOCAL region order, so the positional pairing that
        ``_compute_desc_ids`` relies on keeps holding and nothing downstream has
        to change. Upstream instead walks the remote's own region list and
        applies one global ``rank_offset``, which assumes the remote keeps this
        rank's heads contiguous inside a single region -- true for its
        one-region-per-layer model, false once a region is one chiplet area and
        the heads are area-major.

        For each local region we find the remote slice whose head range covers
        ours and address into it. Two things differ from positional pairing and
        both occur in practice:
          * the remote area index is not the local one (P TP2 -> D TP4: local
            area 2 carries head 1, which is remote area 1);
          * the wanted heads may start partway into the remote area (P TP1 ->
            D TP4: the remote area holds 2 heads, we want the second).

        `registered_layer_names` makes this a PP shard: the peer is one pipeline
        stage owning a slice of the layers, so we emit only our regions for
        those layers, in the peer's own layer order, and index its region list
        by position WITHIN the stage. Left None the peer owns every layer and
        the two layer orders coincide. The two axes are independent -- layers
        select which regions participate, heads select where inside the peer
        each one reads -- which is why they compose by nesting.
        """
        assert self.transfer_topo is not None
        total_heads = self.transfer_topo.total_num_kv_heads
        areas_l, slices_l = self._kv_areas, self._kv_slices
        areas_r = nixl_agent_meta.kv_areas
        slices_r = nixl_agent_meta.kv_slices

        base_l, per_slice_l = self._slice_head_bounds(
            self.tp_rank, self.transfer_topo.tp_size, total_heads, areas_l, slices_l
        )
        base_r, per_slice_r = self._slice_head_bounds(
            remote_tp_rank, remote_tp_size, total_heads, areas_r, slices_r
        )
        # Pieces per region when our area is coarser than the peer's; 1 otherwise.
        split = self._head_split(per_slice_l, per_slice_r)

        replicas_l = areas_l // slices_l
        replicas_r = areas_r // slices_r
        num_blocks = nixl_agent_meta.num_blocks
        remote_bases = nixl_agent_meta.kv_caches_base_addr
        remote_lens = nixl_agent_meta.block_lens

        # (local logical region, its position in the peer's region list). A
        # logical region is one K or V of one layer, before chiplet expansion.
        # Without a PP shard the peer owns every layer and the two coincide;
        # with one, only our regions for the peer's layers take part, ordered
        # as the peer ordered them.
        n_logical_l = len(self.block_len_per_layer) // areas_l
        if registered_layer_names is None:
            logical_pairs = [(i, i) for i in range(n_logical_l)]
        else:
            per_layer = self._regions_per_layer() // areas_l
            logical_pairs = [
                (layer_l * per_layer + c, pos * per_layer + c)
                for pos, layer_l in enumerate(
                    self._local_region_indices_for_layer_names(registered_layer_names)
                )
                for c in range(per_layer)
            ]

        # Which of our areas this peer holds. Every one of them unless the peer
        # has MORE TP ranks, in which case it holds only its share and the
        # others are read from its siblings (see _fan_in_peer_areas).
        areas_iter = range(areas_l) if peer_areas is None else peer_areas

        out: list[tuple[int, int, int]] = []
        # Order must match the local descriptor list exactly: logical region,
        # then chiplet area (see _shard_local_region_ids), and within a region
        # block-major, matching _compute_desc_ids' region_id * num_blocks + b.
        for logical_l, logical_r in logical_pairs:
            for area_l in areas_iter:
                out.extend(
                    self._head_matched_desc(
                        region_id=logical_l * areas_l + area_l,
                        logical_r=logical_r,
                        area_l=area_l,
                        geom=(base_l, per_slice_l, replicas_l),
                        peer=(base_r, per_slice_r, replicas_r, slices_r),
                        areas_r=areas_r,
                        remote_bases=remote_bases,
                        remote_lens=remote_lens,
                        device_id=nixl_agent_meta.device_id,
                        num_blocks=num_blocks,
                    )
                )
        assert len(out) == len(logical_pairs) * len(areas_iter) * num_blocks * split
        return out

    def _fan_in_peer_areas(
        self, remote_tp_rank: int, remote_tp_size: int
    ) -> list[int] | None:
        """Which local chiplet areas live on this peer, when it has MORE TP.

        ``None`` when the peer holds every one of our areas (``tp_ratio > 0``),
        so callers can pass the result through without branching.

        With a finer-grained producer our head band is spread over
        ``|tp_ratio|`` of its ranks, so a transfer to any one of them must
        carry only the areas whose heads that rank actually owns -- otherwise
        we would read the same bytes from every peer and get whichever answer
        landed last.

        The partition is only well defined while an area does not straddle two
        peers, i.e. while its ``per_slice_l`` heads fit inside one peer's
        ``total // remote_tp_size``. That is the ``k <= min(chiplets, heads
        per local rank)`` bound; past it a single region would have to be split
        across agents, which the one-handle-per-peer model cannot express.
        """
        assert self.transfer_topo is not None
        if self.transfer_topo.tp_ratio(remote_tp_size) > 0:
            return None
        total_heads = self.transfer_topo.total_num_kv_heads
        base_l, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.transfer_topo.tp_size,
            total_heads,
            self._kv_areas,
            self._kv_slices,
        )
        heads_per_remote = total_heads // remote_tp_size
        if per_slice_l > heads_per_remote:
            raise RuntimeError(
                "RBLN NIXL D2D: this rank's chiplet area spans "
                f"{per_slice_l} KV heads but each peer rank owns only "
                f"{heads_per_remote}, so one area would straddle several "
                "peers. Reduce the prefill tensor-parallel size (the ratio "
                "must not exceed the smaller of the chiplet count and this "
                "rank's head count) or use kv_buffer_device='cpu'."
            )
        replicas_l = self._kv_areas // self._kv_slices
        return [
            area_l
            for area_l in range(self._kv_areas)
            if (base_l + (area_l // replicas_l) * per_slice_l) // heads_per_remote
            == remote_tp_rank
        ]

    def _head_matched_desc(
        self,
        *,
        region_id: int,
        logical_r: int,
        area_l: int,
        geom: tuple[int, int, int],
        peer: tuple[int, int, int, int],
        areas_r: int,
        remote_bases: list[int],
        remote_lens: list[int],
        device_id: int,
        num_blocks: int,
    ) -> list[tuple[int, int, int]]:
        """Descriptors for one local region: ``split`` per block.

        ``split`` is 1 unless our area is COARSER than the peer's -- then its
        heads are spread over several of the peer's slices and the region has
        to be read in that many pieces. Order is block-major, piece-minor, to
        match the local list `_register_shard_local_xfer_handler` builds.
        """
        base_l, per_slice_l, replicas_l = geom
        base_r, per_slice_r, replicas_r, slices_r = peer

        head = base_l + (area_l // replicas_l) * per_slice_l
        desc_len = self.get_backend_aware_kv_block_len(
            layer_idx=region_id, first_split=True, mamba_view=False
        )
        # Heads per piece: whichever side cuts finer decides. Equal granularity
        # and a coarser PEER both give one piece; a coarser LOCAL area gives
        # per_slice_l / per_slice_r of them.
        split = self._head_split(per_slice_l, per_slice_r)
        per_piece = per_slice_l // split
        sub_len = desc_len // split

        out: list[tuple[int, int, int]] = []
        pieces: list[tuple[int, int]] = []
        for j in range(split):
            # The remote slice covering this piece's first head, and how far
            # into it we start. Then take that slice's FIRST replica --
            # replicas hold identical bytes, so reading one is enough.
            slice_r, head_within = divmod(head + j * per_piece - base_r, per_slice_r)
            if not 0 <= slice_r < slices_r:
                raise RuntimeError(
                    f"RBLN NIXL D2D: local head {head + j * per_piece} is "
                    f"outside the peer's range (it owns heads {base_r}.."
                    f"{base_r + per_slice_r * slices_r - 1})."
                )
            remote_region = logical_r * areas_r + slice_r * replicas_r
            page = remote_lens[remote_region]
            if page % per_slice_r != 0:
                raise RuntimeError(
                    f"RBLN NIXL D2D: peer region {remote_region} block length "
                    f"{page}B does not split into {per_slice_r} heads."
                )
            # Heads are contiguous inside a block, so skipping `head_within` of
            # them is a plain byte offset. Zero whenever the two sides cut heads
            # at the same granularity or we are the coarser side; non-zero when
            # the peer's area is coarser.
            head_offset = head_within * (page // per_slice_r)
            if sub_len + head_offset > page:
                raise RuntimeError(
                    f"RBLN NIXL D2D: local region {region_id} piece {j} wants "
                    f"{sub_len}B at +{head_offset}B, past the end of the peer's "
                    f"{page}B region {remote_region}."
                )
            pieces.append((remote_bases[remote_region] + head_offset, page))

        for block_id in range(num_blocks):
            for base, page in pieces:
                out.append((base + block_id * page, sub_len, device_id))
        return out

    def _peer_head_split(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> int:
        """`_head_split` for a peer, from its advertised chiplet geometry."""
        if not self._is_head_matched_peer(remote_tp_size):
            return 1
        assert self.transfer_topo is not None
        total_heads = self.transfer_topo.total_num_kv_heads
        _, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.transfer_topo.tp_size,
            total_heads,
            self._kv_areas,
            self._kv_slices,
        )
        _, per_slice_r = self._slice_head_bounds(
            0,
            remote_tp_size,
            total_heads,
            nixl_agent_meta.kv_areas,
            nixl_agent_meta.kv_slices,
        )
        return self._head_split(per_slice_l, per_slice_r)

    @staticmethod
    def _head_split(per_slice_l: int, per_slice_r: int) -> int:
        """How many pieces one of our regions is read in.

        Our area carries ``per_slice_l`` heads, the peer's ``per_slice_r``. When
        ours is coarser its heads live in several of the peer's slices, and
        since a descriptor names one contiguous range on each side, the region
        has to be transferred in that many pieces.
        """
        if per_slice_l <= per_slice_r:
            return 1
        if per_slice_l % per_slice_r:
            raise RuntimeError(
                f"RBLN NIXL D2D: this rank's slice spans {per_slice_l} KV heads "
                f"and the peer's {per_slice_r}, which does not divide it; the "
                "two sides must cut heads at commensurate granularities."
            )
        return per_slice_l // per_slice_r

    def _register_remote_engine_prelude(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Replicate upstream ``add_remote_agent``'s prelude.

        vLLM 0.22 base.add_remote_agent registers the remote engine in the
        TransferTopology and builds its TPMapping BEFORE any block_size_ratio /
        tp_ratio / get_engine_info lookup (used by the callers below and by
        ``_validate_remote_agent_handshake``). Paths that do not delegate to
        super() must do this themselves or get_engine_info() raises KeyError.
        """
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            nixl_agent_meta.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=nixl_agent_meta.block_size,
                remote_block_len=nixl_agent_meta.block_lens[0],
                remote_physical_blocks_per_logical=(
                    nixl_agent_meta.physical_blocks_per_logical_kv_block
                ),
            ),
        )
        self.tp_mappings[nixl_agent_meta.engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

    def _add_remote_agent_head_matched(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
    ) -> str:
        """Register a peer with a different TP degree, matching on head bands.

        `registered_layer_names` narrows this to one pipeline stage's layers;
        the caller is then `_nixl_handshake`, once per overlapping stage.

        A peer with MORE TP ranks holds only part of our head band, so the
        descriptors cover just the areas it owns (`_fan_in_peer_areas`) and the
        siblings supply the rest.
        """
        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        self._register_remote_engine_prelude(nixl_agent_meta, remote_tp_size)
        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        # Under PP the caller keys shards by the flat global rank
        # (pp_rank * tp_size + tp_rank); the head band depends only on the
        # tp_rank part. Modulo is a no-op on the non-PP path.
        peer_tp_rank = remote_tp_rank % remote_tp_size
        blocks_data = self._build_head_matched_remote(
            nixl_agent_meta,
            peer_tp_rank,
            remote_tp_size,
            registered_layer_names=registered_layer_names,
            peer_areas=self._fan_in_peer_areas(peer_tp_rank, remote_tp_size),
        )
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )
        logger.info(
            "RblnNixlConnectorWorker: head-matched %d remote desc(s) from "
            "%s rank %d (peer TP %d, %d area(s)/%d slice(s); local TP %d, "
            "%d area(s)/%d slice(s)).",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            remote_tp_size,
            nixl_agent_meta.kv_areas,
            nixl_agent_meta.kv_slices,
            self.transfer_topo.tp_size,
            self._kv_areas,
            self._kv_slices,
        )
        return remote_agent_name

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        if self._sw_ratio is None:
            assert self.transfer_topo is not None
            # tp_ratio is pure arithmetic on the two TP sizes, so it is safe to
            # ask before the engine is registered.
            tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
            if tp_ratio == 1:
                # Homogeneous TP: local region i IS remote region i, which is
                # what upstream's positional descriptor math assumes.
                return super().add_remote_agent(
                    nixl_agent_meta, remote_tp_rank, remote_tp_size
                )
            if tp_ratio != 1 and not self.use_host_buffer:
                # Different TP degrees, either direction. tp_ratio > 1: the
                # peer's regions carry wider head bands than ours, so pair by
                # head range instead of by position. tp_ratio < 0: the peer
                # carries only part of our band, so this transfer covers just
                # the areas it owns and its siblings supply the rest.
                assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
                return self._add_remote_agent_head_matched(
                    nixl_agent_meta, remote_tp_rank, remote_tp_size
                )
            # host-bounce: one logical region per layer, upstream's model holds.
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

        self._register_remote_engine_prelude(nixl_agent_meta, remote_tp_size)

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

        # This is the SWA view-opt path only; fan-in (tp_ratio < 0) reaches
        # _add_remote_agent_head_matched instead. The two have never been
        # combined -- _check_d2d_region_pairing already rejects SWA with any
        # unequal TP, so this only documents the assumption locally.
        assert tp_ratio >= 0, (
            "RBLN NIXL SWA view-opt does not support remote TP > local TP "
            f"(tp_ratio={tp_ratio})."
        )

        blocks_data: list[tuple[int, int, int]] = []
        num_blocks = nixl_agent_meta.num_blocks

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses (same `page_size` stride — the
        # remote tensor's physical block stride is still Full-sized),
        # shorter desc length.
        # _sw_ratio is not None here (the None case returned early above).
        length_divisors = [1, self._sw_ratio]
        for divisor in length_divisors:
            for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
                local_block_len = self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=True, mamba_view=False
                )
                remote_kv_block_len = local_block_len // block_size_ratio
                if block_size_ratio > 1:
                    local_block_len = remote_kv_block_len
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

        # The SWA desc formula below indexes physical blocks directly; the
        # connector pins one physical block per logical block, so the
        # physical_blocks_per_logical argument does not apply here.
        assert physical_blocks_per_logical == 1, (
            "RBLN NIXL connector assumes physical_blocks_per_logical == 1"
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

    # ------------------------------------------------------------------
    # Pipeline-parallel (PP) P/D over NIXL
    # ------------------------------------------------------------------
    #
    # Under sync-scheduling PP the producer runs pipeline_parallel_size
    # stages, each owning a contiguous band of layers. Every stage
    # advertises its (pp_rank, tp_rank) shard and the KV-cache layer names
    # it registered; the consumer matches each shard to its own local KV
    # regions BY NAME (the owned layer range is derived locally on each
    # side, never sent over the wire) and reads each overlapping shard with
    # a shard-scoped local/remote descriptor pair. All of this collapses to
    # the upstream single-stage path for pp_size == 1.

    def _check_pp_constraints(self) -> None:
        if self.vllm_config.parallel_config.pipeline_parallel_size <= 1:
            return
        assert self.transfer_topo is not None
        if self.transfer_topo.cross_layers_blocks:
            raise RuntimeError(
                "RBLN NIXL: cross-layer-blocks mode is not supported with "
                "pipeline_parallel_size > 1."
            )
        if self._has_mamba:
            raise RuntimeError(
                "RBLN NIXL: hybrid (Mamba/SSM) models are not supported with "
                "pipeline_parallel_size > 1 over NIXL P/D."
            )
        if self._sw_ratio is not None:
            raise RuntimeError(
                "RBLN NIXL: sliding-window view-opt (VLLM_RBLN_NIXL_SWA_VIEW_OPT) "
                "is not supported with pipeline_parallel_size > 1."
            )

    def _publish_pp_handshake_metadata(
        self, base_meta: NixlAgentMetadata, registered_layer_names
    ) -> None:
        self._check_pp_constraints()
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        pp_rank = get_pp_group().rank_in_group if pp_size > 1 else 0
        pp_meta = RblnNixlAgentMetadata(
            engine_id=base_meta.engine_id,
            agent_metadata=base_meta.agent_metadata,
            device_id=base_meta.device_id,
            kv_caches_base_addr=base_meta.kv_caches_base_addr,
            num_blocks=base_meta.num_blocks,
            block_lens=base_meta.block_lens,
            kv_cache_layout=base_meta.kv_cache_layout,
            block_size=base_meta.block_size,
            ssm_sizes=base_meta.ssm_sizes,
            attn_backend_name=base_meta.attn_backend_name,
            physical_blocks_per_logical_kv_block=(
                base_meta.physical_blocks_per_logical_kv_block
            ),
            pp_rank=pp_rank,
            pp_size=pp_size,
            registered_layer_names=list(registered_layer_names),
            kv_areas=self._kv_areas,
            kv_slices=self._kv_slices,
        )
        base_hash = self.compat_hash
        assert base_hash is not None
        self.compat_hash = rbln_pp_compat_hash(base_hash)
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.Encoder().encode(pp_meta),
        )

    def _query_agent_meta(
        self, sock: "zmq.Socket", remote_rank: int, expected_engine_id: str
    ) -> RblnNixlAgentMetadata:
        sock.send(msgspec.msgpack.encode((GET_META_MSG, remote_rank)))
        handshake_payload = msgspec.msgpack.Decoder(NixlHandshakePayload).decode(
            sock.recv()
        )
        assert self.compat_hash is not None
        if (
            self.enforce_compat_hash
            and handshake_payload.compatibility_hash != self.compat_hash
        ):
            raise RuntimeError(
                "NIXL compatibility hash mismatch "
                f"(local={self.compat_hash}, "
                f"remote={handshake_payload.compatibility_hash}). Prefill and "
                "decode instances have incompatible configurations."
            )
        metadata = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            handshake_payload.agent_metadata_bytes
        )
        if metadata.engine_id != expected_engine_id:
            raise RuntimeError(
                "Remote NIXL agent engine ID mismatch. "
                f"Expected {expected_engine_id}, received {metadata.engine_id}."
            )
        return metadata

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        # Background thread needs a device context (see base _nixl_handshake).
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        assert self.transfer_topo is not None
        p_remote_tp_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        path = make_zmq_path("tcp", host, port)
        remote_rank_to_agent_name: dict[int, str] = {}

        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # ms; avoid hang on dead server

            # Bootstrap: the first shard (pp_rank 0) advertises pp_size.
            first_rank = p_remote_tp_ranks[0]
            metas = {
                first_rank: self._query_agent_meta(sock, first_rank, expected_engine_id)
            }
            pp_size = metas[first_rank].pp_size

            if pp_size > 1:
                if self._sw_ratio is not None:
                    raise RuntimeError(
                        "RBLN NIXL: sliding-window attention combined with "
                        "pipeline-parallel P/D is not supported."
                    )
                tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
                if tp_ratio < 0:
                    raise RuntimeError(
                        "RBLN NIXL: pipeline-parallel P/D does not support a "
                        f"peer with a larger tensor-parallel size (peer "
                        f"{remote_tp_size} > local {self.transfer_topo.tp_size})."
                    )
                local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
                if tp_ratio != 1 and local_pp > 1:
                    # Each axis is handled: layers by name matching, heads by
                    # _build_head_matched_remote. Splitting BOTH on BOTH sides
                    # at once is untested, so keep it out until something needs
                    # it -- decode is TP/DP-only in the topologies we run.
                    raise RuntimeError(
                        "RBLN NIXL: heterogeneous tensor parallelism "
                        f"(tp_ratio={tp_ratio}) combined with pipeline "
                        f"parallelism on BOTH sides (local pp={local_pp}) is "
                        "not supported."
                    )

            for pp_rank in range(pp_size):
                for remote_tp_rank in p_remote_tp_ranks:
                    global_rank = pp_rank * remote_tp_size + remote_tp_rank
                    if global_rank in metas:
                        metadata = metas[global_rank]
                    else:
                        metadata = self._query_agent_meta(
                            sock, global_rank, expected_engine_id
                        )
                    names = tuple(metadata.registered_layer_names)
                    self._remote_shard_layer_names[expected_engine_id][global_rank] = (
                        names
                    )
                    if pp_size == 1:
                        # No pipeline parallelism: single-stage registration,
                        # exactly as the non-PP path (read path also delegates to
                        # base for pp_size == 1).
                        remote_rank_to_agent_name[global_rank] = self.add_remote_agent(
                            metadata, global_rank, remote_tp_size
                        )
                        split = self._peer_head_split(metadata, remote_tp_size)
                        if not self._is_fan_in_peer(remote_tp_size) and split == 1:
                            continue
                        # Either the peer holds only part of our head band
                        # (fan-in) or our chiplet area is coarser than its, so
                        # each region is read in pieces. Base's single
                        # whole-engine handle describes neither. Take the same
                        # per-shard read state the PP path uses -- narrowed on
                        # the head axis instead of the layer axis -- and record
                        # the peer so the read path visits it.
                        self._register_shard_read_state(
                            expected_engine_id,
                            global_rank,
                            metadata.block_size,
                            names,
                            peer_areas=self._fan_in_peer_areas(
                                global_rank % remote_tp_size, remote_tp_size
                            ),
                            split=split,
                        )
                        self._overlapping_ranks[expected_engine_id].append(global_rank)
                        continue

                    # PP: only build/register producer stages whose layers this
                    # rank actually owns. base.add_remote_agent -> _build_fa_remote
                    # indexes the *local* block_len_per_layer by the *remote*
                    # region position, so it is only safe when n_remote <= n_local.
                    # Overlapping stages satisfy that (symmetric same-stage: equal;
                    # asymmetric fan-in: remote is a sub-band of this larger band);
                    # a non-overlapping peer stage may be larger under an uneven
                    # split (e.g. 62 layers / 4 = 16/16/15/15) and would index out
                    # of range -- and this rank never reads it -- so skip it.
                    indices = self._local_region_indices_for_layer_names(names)
                    if 0 < len(indices) < len(names):
                        raise RuntimeError(
                            f"RBLN NIXL PP: producer stage {global_rank} "
                            f"({len(names)} layers) partially overlaps this "
                            f"decode rank's band ({len(indices)} owned). "
                            "The prefill pipeline-parallel size must be >= "
                            "the decode pipeline-parallel size and an integer "
                            "multiple of it."
                        )
                    if not indices:
                        continue
                    if self._is_head_matched_peer(remote_tp_size):
                        # Producer has fewer TP ranks: its regions carry wider
                        # head bands, so pair them by head range. Restricted to
                        # this stage's layers -- the two axes compose.
                        remote_rank_to_agent_name[global_rank] = (
                            self._add_remote_agent_head_matched(
                                metadata, global_rank, remote_tp_size,
                                registered_layer_names=names,
                            )
                        )
                    else:
                        remote_rank_to_agent_name[global_rank] = self.add_remote_agent(
                            metadata, global_rank, remote_tp_size
                        )
                    self._register_shard_read_state(
                        expected_engine_id,
                        global_rank,
                        metadata.block_size,
                        names,
                        # 1 under PP today (fan-in with PP is rejected above),
                        # but derived rather than assumed.
                        split=self._peer_head_split(metadata, remote_tp_size),
                    )
                    self._overlapping_ranks[expected_engine_id].append(global_rank)
        self._remote_pp_size[expected_engine_id] = pp_size
        return remote_rank_to_agent_name

    def _register_shard_read_state(
        self,
        engine_id: str,
        global_rank: int,
        block_size: int,
        registered_layer_names: tuple[str, ...],
        peer_areas: list[int] | None = None,
        split: int = 1,
    ) -> None:
        handle, _ = self.register_local_xfer_handler(
            block_size,
            registered_layer_names=registered_layer_names,
            peer_areas=peer_areas,
            split=split,
        )
        self.src_xfer_handles_by_remote[(engine_id, global_rank, block_size)] = handle
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "RBLN NIXL per-shard reads support a single KV-cache group"
        )
        region_ids = self._shard_local_region_ids(
            registered_layer_names, peer_areas=peer_areas
        )
        self._shard_region_group_ids[(engine_id, global_rank)] = tuple(
            0 for _ in region_ids
        )
        self._shard_desc_split[(engine_id, global_rank)] = split

    def _local_region_indices_for_layer_names(
        self, registered_layer_names: tuple[str, ...] | list[str]
    ) -> list[int]:
        positions_by_name: dict[str, list[int]] = defaultdict(list)
        for local_idx, layer_name in enumerate(self.local_seen_layer_names):
            positions_by_name[layer_name].append(local_idx)

        occurrences_by_name: dict[str, int] = defaultdict(int)
        local_indices: list[int] = []
        for layer_name in registered_layer_names:
            occurrence = occurrences_by_name[layer_name]
            occurrences_by_name[layer_name] += 1
            matches = positions_by_name.get(layer_name, [])
            if occurrence >= len(matches):
                continue
            local_indices.append(matches[occurrence])
        return local_indices

    def _regions_per_layer(self) -> int:
        num_layers = len(self.local_seen_layer_names)
        assert num_layers > 0 and self.num_regions % num_layers == 0, (
            f"num_regions={self.num_regions} not divisible by num_layers={num_layers}"
        )
        return self.num_regions // num_layers

    def _shard_local_region_ids(
        self,
        registered_layer_names: tuple[str, ...] | list[str],
        peer_areas: list[int] | None = None,
    ) -> list[int]:
        """Our region ids that take part in a transfer with one peer.

        Two independent filters, matching the two axes a peer can be narrower
        on. `registered_layer_names` keeps the layers a pipeline stage owns;
        `peer_areas` keeps the chiplet areas whose heads a finer-grained
        tensor-parallel peer owns. Region ids run logical-region-major,
        area-minor (`(layer * K/V) * areas + area`), which is why the area
        filter is a test on `k % areas`.

        The order here IS the descriptor order: `_register_shard_local_xfer_handler`
        walks this list to build the local dlist and `_build_head_matched_remote`
        walks the same axes in the same nesting to build the remote one.
        """
        rpl = self._regions_per_layer()
        layer_indices = self._local_region_indices_for_layer_names(
            registered_layer_names
        )
        if peer_areas is None:
            keep = range(rpl)
        else:
            areas = self._kv_areas
            wanted = set(peer_areas)
            keep = [k for k in range(rpl) if k % areas in wanted]
        return [layer_idx * rpl + k for layer_idx in layer_indices for k in keep]

    def _register_shard_local_xfer_handler(
        self,
        block_size: int,
        registered_layer_names: tuple[str, ...] | list[str],
        peer_areas: list[int] | None = None,
        split: int = 1,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        assert self.transfer_topo is not None
        assert not self.transfer_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout (K and V in separate "
            "regions), not FlashInfer."
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba."

        block_size_ratio = self.block_size // block_size
        num_blocks = self.num_blocks * block_size_ratio
        all_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        region_ids = self._shard_local_region_ids(
            registered_layer_names, peer_areas=peer_areas
        )

        blocks_data: list[tuple[int, int, int]] = []
        for region_id in region_ids:
            base_addr = all_base_addrs[region_id]
            kv_block_len = (
                self.get_backend_aware_kv_block_len(
                    layer_idx=region_id, first_split=True, mamba_view=False
                )
                // block_size_ratio
            )
            stride = self.block_len_per_layer[region_id] // block_size_ratio
            # `split` > 1 when our chiplet area is coarser than the peer's: its
            # heads sit in several of the peer's slices, so the region is read
            # in that many contiguous pieces. Heads are contiguous inside a
            # block, so the pieces are just consecutive byte ranges here -- it
            # is the REMOTE side that scatters (_head_matched_desc). Nesting
            # (block, piece) must match there exactly.
            sub_len = kv_block_len // split
            for block_id in range(num_blocks):
                for j in range(split):
                    blocks_data.append(
                        (
                            base_addr + block_id * stride + j * sub_len,
                            sub_len,
                            self.device_id,
                        )
                    )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )

    def _get_block_descs_ids_for_shard(
        self,
        engine_id: str,
        global_rank: int,
        num_blocks: int,
        block_ids: BlockIds,
    ) -> np.ndarray:
        region_group_ids = self._shard_region_group_ids[(engine_id, global_rank)]
        split = self._shard_desc_split.get((engine_id, global_rank), 1)
        desc_ids: list[np.ndarray] = []
        for region_id, group_id in enumerate(region_group_ids):
            group_arr = np.asarray(block_ids[group_id], dtype=np.int64)
            if group_arr.size == 0:
                continue
            block_ix = region_id * num_blocks + group_arr
            if split == 1:
                desc_ids.append(block_ix)
                continue
            # Both dlists are laid out region-major, then block, then piece, so
            # one block becomes `split` consecutive descriptors.
            desc_ids.append(
                (block_ix[:, None] * split + np.arange(split, dtype=np.int64)).ravel()
            )
        if not desc_ids:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(desc_ids)

    def _check_d2d_region_pairing(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject D2D topologies whose chiplet region lists cannot be paired.

        On the D2D path a KV entry is published as one region PER CHIPLET AREA
        (see ``nixl_rbln.register_kv_regions``). Equal TP degrees make local
        region i and remote region i the same head band, which is what
        upstream's positional descriptor math assumes. A peer with FEWER TP
        ranks is handled by ``_add_remote_agent_head_matched``, which pairs by
        head range instead. What is left unsupported:

        1. A peer with SO MANY more TP ranks that one of our chiplet areas
           straddles two of them. A peer with more TP ranks is otherwise fine --
           ``_fan_in_peer_areas`` splits our areas across its ranks -- but once
           an area no longer fits inside one peer's head band, a single region
           would have to be divided across two agents, which the
           one-handle-per-peer model cannot express. The bound is checked there,
           not here, because it needs the measured chiplet geometry.
        2. Different region counts, e.g. a peer with a different chiplet count.
           Region count is independent of TP -- it is layers x K/V x chiplets --
           so a mismatch means the geometry itself differs, and neither the
           positional nor the head-matched pairing is meaningful. Upstream would
           only notice at transfer time, via
           ``assert len(local_block_descs_ids) == len(remote_block_descs_ids)``.
        3. SWA view-opt with heterogeneous TP. The two-descriptor-range layout
           and head matching have not been combined.

        Host-bounce is unaffected and deliberately not checked: it registers one
        logical full-shape host buffer per layer, never the per-area list, so
        upstream's contiguous-region model holds -- verified to produce correct
        output over ``kv_buffer_device=cpu`` for every case above.
        """
        if self.use_host_buffer:
            return
        assert self.transfer_topo is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        if tp_ratio != 1 and self._sw_ratio is not None:
            raise RuntimeError(
                "RBLN NIXL D2D: sliding-window view-opt "
                "(VLLM_RBLN_NIXL_SWA_VIEW_OPT) is not supported with "
                f"heterogeneous tensor parallelism (tp_ratio={tp_ratio})."
            )
        n_remote = len(nixl_agent_meta.kv_caches_base_addr)
        n_local = len(self.block_len_per_layer)
        if n_remote != n_local:
            raise RuntimeError(
                f"RBLN NIXL D2D: peer advertises {n_remote} KV regions but this "
                f"worker has {n_local}. Region count does not depend on TP, so "
                "both sides must expand identically (same chiplet count and "
                "same number of KV layers)."
            )

    def _is_head_matched_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer is served by ``_build_head_matched_remote``.

        Any unequal TP degree, in either direction. Equal degrees keep
        upstream's positional pairing (``tp_ratio == 1``).
        """
        if self.use_host_buffer or self._sw_ratio is not None:
            return False
        assert self.transfer_topo is not None
        return self.transfer_topo.tp_ratio(remote_tp_size) != 1

    def _is_fan_in_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer has MORE TP ranks than us, so we gather from it
        together with its siblings."""
        if self.use_host_buffer:
            return False
        assert self.transfer_topo is not None
        return self.transfer_topo.tp_ratio(remote_tp_size) < 0

    def _validate_head_matched_handshake(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Handshake checks for a head-matched peer.

        Upstream's equivalent cannot be used here. It asserts
        ``remote_block_len == local_block_len * tp_ratio``, which holds only
        for its one-region-per-layer model, where the peer's single region
        carries tp_ratio times our heads. After chiplet expansion BOTH sides
        publish the same number of regions -- one per area -- so the ratio of
        per-area block lengths is the ratio of heads per area, which is
        unrelated to tp_ratio:

            P TP1 -> D TP2   heads/area 2 -> 1, tp_ratio 2   (agree by luck)
            P TP1 -> D TP4   heads/area 2 -> 1, tp_ratio 4   (disagree)
            P TP2 -> D TP4   heads/area 1 -> 1, tp_ratio 2   (disagree)

        The invariant that does hold, and that the descriptor arithmetic
        actually depends on, is that a single KV head occupies the same number
        of bytes per block on both sides -- same block_size, head_dim, dtype.
        """
        assert self.transfer_topo is not None
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        if block_size_ratio != 1:
            raise RuntimeError(
                "RBLN NIXL D2D with heterogeneous TP requires equal P/D block "
                f"sizes (got block_size_ratio={block_size_ratio})."
            )
        if nixl_agent_meta.kv_cache_layout != self.kv_cache_layout:
            raise RuntimeError(
                "RBLN NIXL D2D: peer KV layout "
                f"{nixl_agent_meta.kv_cache_layout!r} != local "
                f"{self.kv_cache_layout!r}."
            )
        total_heads = self.transfer_topo.total_num_kv_heads
        _, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.transfer_topo.tp_size,
            total_heads,
            self._kv_areas,
            self._kv_slices,
        )
        _, per_slice_r = self._slice_head_bounds(
            0, remote_tp_size, total_heads,
            nixl_agent_meta.kv_areas, nixl_agent_meta.kv_slices,
        )
        local_len = self.block_len_per_layer[0]
        remote_len = nixl_agent_meta.block_lens[0]
        if local_len * per_slice_r != remote_len * per_slice_l:
            raise RuntimeError(
                "RBLN NIXL D2D: a KV head occupies "
                f"{local_len / per_slice_l:.0f}B per block here but "
                f"{remote_len / per_slice_r:.0f}B on the peer "
                f"(local {local_len}B over {per_slice_l} head(s), remote "
                f"{remote_len}B over {per_slice_r}). Block size, head_dim and "
                "dtype must match across P and D."
            )

    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        if getattr(nixl_agent_meta, "pp_size", 1) <= 1:
            self._check_d2d_region_pairing(nixl_agent_meta, remote_tp_size)
            if self._is_head_matched_peer(remote_tp_size):
                assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
                self._validate_head_matched_handshake(
                    nixl_agent_meta, remote_tp_size
                )
                return
            super()._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)
            return

        assert self.transfer_topo is not None
        remote_engine_id = nixl_agent_meta.engine_id
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size
        # TP used to be barred outright here. A producer with FEWER TP ranks is
        # now matched per head band (_build_head_matched_remote); only the other
        # direction remains impossible, and _nixl_handshake already rejects it
        # before we get here.
        pp_tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        assert pp_tp_ratio > 0, (
            "PP over NIXL P/D does not support a peer with a larger TP size."
        )
        if pp_tp_ratio != 1:
            # Same head-geometry invariant as the non-PP head-matched path; the
            # layer axis does not change what a head costs per block.
            assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
            self._validate_head_matched_handshake(nixl_agent_meta, remote_tp_size)
        assert self.transfer_topo.block_size_ratio(nixl_agent_meta.block_size) == 1, (
            "PP over NIXL P/D requires equal P/D block sizes."
        )
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
        rpl = self._regions_per_layer()
        n_remote = len(nixl_agent_meta.kv_caches_base_addr)
        assert (
            n_remote > 0
            and n_remote % rpl == 0
            and n_remote <= len(self.block_len_per_layer)
        ), (
            f"PP shard advertised {n_remote} KV regions, not a valid "
            f"sub-multiple of this consumer's {len(self.block_len_per_layer)} "
            f"regions (regions/layer={rpl})."
        )

    def _read_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        pp_size = self._remote_pp_size.get(engine_id, 1)
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        # Two reasons to walk peers one at a time instead of issuing base's one
        # whole-engine transfer: a pipeline-parallel producer (each stage owns
        # some layers) and a finer-grained tensor-parallel one (each rank owns
        # some heads). Either way the per-shard descriptor lists were built to
        # pair up at handshake.
        if pp_size == 1 and not self._is_fan_in_peer(remote_info.remote_tp_size):
            return super()._read_blocks_for_req(req_id, meta)

        assert (
            self.transfer_topo.block_size_ratio(remote_info.remote_block_size) == 1
        ), "RBLN NIXL per-shard read path requires equal P/D block sizes"
        remote_block_size = remote_info.remote_block_size

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids, remote_info.remote_physical_blocks_per_logical
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
        prefix_hit = len(local_block_ids) == 0
        n_prompt_blocks = sum(len(g) for g in remote_block_ids)

        if not prefix_hit:
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                local_block_ids,
                remote_block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )

        n_read_blocks = sum(len(g) for g in local_block_ids)
        logger.info(
            "per-shard read req %s: pp_size=%d prompt_blocks=%d read_blocks=%d "
            "prefix_skipped=%d%s",
            req_id,
            pp_size,
            n_prompt_blocks,
            n_read_blocks,
            n_prompt_blocks - n_read_blocks,
            " (full prefix hit, notif only)" if prefix_hit else "",
        )

        for global_rank in self._overlapping_ranks[engine_id]:
            if prefix_hit:
                agent_name = self._remote_agents[engine_id][global_rank]
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
                continue

            remote_descs = self._get_block_descs_ids_for_shard(
                engine_id,
                global_rank,
                self.dst_num_blocks[engine_id],
                remote_block_ids,
            )
            local_descs = self._get_block_descs_ids_for_shard(
                engine_id, global_rank, self.num_blocks, local_block_ids
            )
            assert len(local_descs) == len(remote_descs)
            local_handle = self.src_xfer_handles_by_remote[
                (engine_id, global_rank, remote_block_size)
            ]
            remote_handle = self.dst_xfer_side_handles[engine_id][global_rank]

            handle = None
            try:
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "READ",
                    local_handle,
                    local_descs,
                    remote_handle,
                    remote_descs,
                    notif_msg=notif_id,
                )
                self.nixl_wrapper.transfer(handle)
                self._recving_transfers[req_id].append(handle)
            except Exception as e:
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=req_id,
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=engine_id,
                    remote_pp_rank=global_rank,
                )
                self._handle_failed_transfer(req_id, handle)
