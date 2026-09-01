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
from dataclasses import replace
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
    NixlBaseConnectorWorker,
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
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlAgentMetadata,
    rbln_compat_hash,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlWorkerBase(NixlBaseConnectorWorker):
    """Everything the transfer direction does not decide: memory registration,
    the handshake, region pairing, descriptor construction, topology guards.
    Mixed with whichever direction class moves the bytes.

    Supported prefill -> decode topologies
    --------------------------------------
    Peers pair by what they actually hold, on two axes that compose: KV heads
    (`_build_head_matched_remote`) and layers (`_layer_overlap`). Any TP or PP
    degree on either side works within these bounds. DP and EP are invisible --
    a replica is its own engine, and EP does not shard the KV cache.

      * the two pipeline sizes must tile each other
      * both sides pipelined requires equal TP, and a pipelined peer may not
        have MORE TP ranks
      * D2D only: one of OUR chiplet areas must fit inside a single peer rank's
        band (heads per area <= total KV heads / peer TP)
      * each side's TP degree must divide the model's KV heads; below that
        upstream replicates one head across ranks, which no head band names
      * D2D only: the coarser side's heads per area must be a whole multiple of
        the finer side's, or a descriptor would carry part of a head

    Not supported
    -------------
      * D2D only: unequal TP with MLA (`_check_mla_constraints`)
      * unequal P/D block sizes with unequal TP or with PP; the equal-TP,
        non-pipelined case is upstream's and unaffected
      * sliding-window attention with any model parallelism across P/D --
        unequal TP or PP on either side
      * with PP: Mamba/SSM and cross-layer-blocks
      * more than one KV-cache group on any per-shard transfer -- that is,
        whenever a peer serves less than a whole engine
    """

    compat_hash: str | None
    xfer_handshake_metadata: NixlHandshakePayload | None

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # nixl-rbln present -> RBLN backend (host-bounce DRAM_SEG / D2D VRAM_SEG);
        # absent -> upstream UCX/DRAM defaults, and D2D (kv_buffer_device="rbln")
        # is rejected below since it needs the RBLN backend.
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
                "RBLN NIXL: nixl-rbln not available — "
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
        # rank owns; the per-shard transfer path walks exactly these.
        self._overlapping_ranks: defaultdict[str, list[int]] = defaultdict(list)
        # Per producer shard, a local xfer dlist scoped to that shard's local
        # region subset, keyed by (engine_id, global_rank, block_size); and the
        # shard's per-region KV-group ids, keyed by (engine_id, global_rank).
        self.src_xfer_handles_by_remote: dict[tuple[str, int, int], int] = {}
        # Which of those entries point at a handle upstream owns, so cleanup
        # drops the entry without releasing what other peers still use.
        self._borrowed_src_handles: set[tuple[str, int, int]] = set()
        self._shard_region_group_ids: dict[tuple[str, int], tuple[int, ...]] = {}
        # How many descriptors each of that shard's regions is cut into
        # (_head_split).
        self._shard_descs_per_block: dict[tuple[str, int], int] = {}
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

        # SWA view-opt: publish a second sliding_window-length desc range at the
        # same NIXL base addrs as the Full range, so SWA groups transport only the
        # populated prefix (kernel slot 0 is pinned at the block base). Storage and
        # host copies stay Full; _sw_ratio is None collapses to upstream Full-only.
        # See the "Hybrid Full + SWA desc layout" section below.
        self._group_specs: list[Any] = [
            g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
        ]
        # Whether the model has a sliding window at all, which decides the model
        # parallelism guards; `_sw_ratio` is the view-opt's desc layout and only
        # ever set when that flag is on.
        self._has_swa = any(
            isinstance(spec, SlidingWindowSpec) for spec in self._group_specs
        )
        self._sw_ratio: int | None = None
        if self._has_swa and envs.VLLM_RBLN_NIXL_SWA_VIEW_OPT:
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
                # Fail at startup rather than at the first handshake: the
                # two-desc-range layout below and a key-only latent have not
                # been combined.
                if self.use_mla:
                    raise RuntimeError(
                        "RBLN NIXL: VLLM_RBLN_NIXL_SWA_VIEW_OPT is not "
                        "supported with a sliding-window MLA cache."
                    )
                logger.info(
                    "VLLM_RBLN_NIXL_SWA_VIEW_OPT=1: trimming SWA-group "
                    "RDMA payload by 1/%d (sliding_window-sized descs "
                    "alongside Full descs at shared base addrs).",
                    self._sw_ratio,
                )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Wire KV caches into NIXL.

        D2D defers: its backing memory is not materialized until warm-up, so
        the real registration runs from `finalize_kv_cache_registration`.
        Host-bounce buffers are plain DRAM and register now; where nixl-rbln
        is installed the RBLN backend only has to exist first, so that
        upstream's `register_memory(..., backends=["RBLN"])` resolves.
        """
        # Capture the ordered local layer names before any deferral so the PP
        # metadata publish (and the consumer-side name->region matching) can
        # use them; the D2D path re-uses these at finalize time.
        self.local_seen_layer_names = list(kv_caches.keys())
        if self.kv_buffer_device == "rbln":
            self._pending_kv_caches = kv_caches
            logger.info(
                "RBLN NIXL (D2D): deferring registration of "
                "%d KV cache layer(s) until after warm-up.",
                len(kv_caches),
            )
            return
        if self._use_rbln_nixl_backend:
            import nixl_rbln

            nixl_rbln.ensure_rbln_backend(self.nixl_wrapper, device_id=0)
        super().register_kv_caches(kv_caches)
        # Re-wrap upstream's published handshake metadata with this stage's PP
        # identity + owned layer names (no-op degrade for pp_size == 1).
        if self.xfer_handshake_metadata is not None:
            base_agent_metadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
                self.xfer_handshake_metadata.agent_metadata_bytes
            )
            self._publish_handshake_metadata(base_agent_metadata, kv_caches.keys())

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
        # MLA has no head axis to order, which is why upstream advertises no
        # required layout for it and the resolved value is meaningless here.
        # The allocator below is shape-agnostic either way.
        assert self.use_mla or self.kv_cache_layout == "HND", (
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
            logger.error("RBLN NIXL: %s", e)
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
        MR reg), and feed the returned transfer tables into upstream's
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
            "RBLN NIXL (D2D): KV cache is not an 'rbln' "
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
        # REPLICATE flag per logical region, expanded to the chiplet-expanded
        # transfer table below (see _region_is_mla).
        logical_mla: list[bool] = []
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs[layer_name]
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            is_mla_region = isinstance(
                layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            )
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate
            # regions. MLA is key-only, so `cache_list` holds one.
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
            if tensor_size_bytes is None and not is_mla_region:
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

                # A pure-SWA single group with `sliding_window < block_size`
                # trips this: the canonical-layer fallback picks the SWA layer,
                # whose block count is not the group's. Non-disagg is unaffected.
                assert cache.shape[0] == num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                if not is_mla_region:
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All non-MLA kv cache tensors must have the same size"
                    )
                regions.append((cache_or_caches, region_offset, full_block_len))
                logical_mla.append(is_mla_region)

        rbln_ctx_ptr = rebel.context_of(sample_kv_cache).rbln_ctx_ptr

        # Delegate sharding and MR registration to nixl-rbln. It registers
        # one whole-entry MR per shard and returns the transfer tables
        # (base addrs + block lens), already shard-expanded so upstream's
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

        # Upstream keys REPLICATE vs SPLIT off this list and indexes it 1:1 with
        # block_len_per_layer, which is chiplet-expanded here -- so the flags are
        # too, every area of a logical region carrying the same latent.
        areas = max(xfer.n_shards, 1)
        self._region_is_mla = [is_mla for is_mla in logical_mla for _ in range(areas)]
        assert len(self._region_is_mla) == len(self.block_len_per_layer), (
            f"{len(logical_mla)} logical region(s) over {areas} area(s) do not "
            f"account for {len(self.block_len_per_layer)} transfer region(s); "
            "mislabelling one would silently pick the wrong descriptor layout."
        )

        self.num_regions = len(xfer.base_addrs)
        if self.transfer_topo.is_kv_layout_blocks_first:
            # Blocks-first layout doubles the region count (K/V split), like the
            # upstream's virtually_split_kv_in_blocks -- except for key-only MLA
            # regions, which have no V half. Inert while the connector rejects
            # blocks-first outright (FA layout only).
            self.num_regions = sum(
                1 if self._is_region_replicated(i) else 2
                for i in range(len(self._region_is_mla))
            )
        self.num_descs = self.num_regions * self.num_blocks

        # Areas vs slices: see RblnNixlAgentMetadata. Held for the descriptor
        # arithmetic and the region-pairing guard.
        self._kv_areas = xfer.n_shards
        self._kv_slices = xfer.slices
        logger.info(
            "RBLN NIXL (D2D): registered %d transfer "
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
        # Republish with what a peer needs to pair by content: the layer names
        # this shard registered and the chiplet geometry they expanded into.
        self._publish_handshake_metadata(agent_metadata, self.device_kv_caches.keys())

    # ------------------------------------------------------------------
    # Hybrid Full + SWA desc layout (RDMA payload only)
    # ------------------------------------------------------------------
    #
    # Regions are Full-sized. With VLLM_RBLN_NIXL_SWA_VIEW_OPT and an SWA group,
    # two desc ranges share the base addrs: [0, N) Full-length, [N, 2N)
    # sliding_window-length. SWA groups read only the prefix, so RDMA moves less
    # while the host copy still moves whole blocks. _compute_desc_ids routes each
    # group to its range; _sw_ratio None collapses to Full-only. Safe because the
    # tail SWA writes back is never read and the Full/SWA block-id pools are
    # disjoint.

    def register_local_xfer_handler(
        self,
        block_size: int,
        *,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
        split: int = 1,
        region_ids: list[int] | None = None,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        if self._sw_ratio is None:
            if registered_layer_names is None and peer_areas is None and split == 1:
                # No SWA view opt, whole-engine peer: upstream's Full-only
                # layout, one handle covering every region.
                return super().register_local_xfer_handler(block_size)
            # Per-peer shard: only the regions this peer serves, each cut into
            # `split` pieces (_shard_local_region_ids, _head_split).
            return self._register_shard_local_xfer_handler(
                block_size,
                registered_layer_names or self.local_seen_layer_names,
                peer_areas=peer_areas,
                split=split,
                region_ids=region_ids,
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
            "Full + SWA",
            self.engine_id,
            self.tp_rank,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )

    # ------------------------------------------------------------------
    # The head axis: which KV heads a chiplet area holds
    # ------------------------------------------------------------------
    #
    # A region is one chiplet area, so region i names different heads on two
    # peers as soon as their TP degrees differ. Everything below answers one
    # question -- for each of our areas, where in the peer do its heads live --
    # and an area may be a replica rather than a distinct slice, so a head width
    # comes from slices; areas only count regions.

    @staticmethod
    def _slice_head_bounds(
        tp_rank: int,
        tp_size: int,
        total_kv_heads: int,
        areas: int,
        slices: int,
        *,
        side: str,
    ) -> tuple[int, int]:
        """(first head this shard owns, heads per logical slice).

        The compiler cuts a shard's heads into ``slices`` pieces, one per
        chiplet area -- but a shard owning fewer heads than the device has
        chiplets gets ``areas // slices`` replicas of each, the replication
        axis innermost (``slice_id = area // (areas // slices)``).

        Callers pass a peer's advertised geometry as well as this rank's, so
        the three below refuse a pairing rather than assert an invariant;
        ``side`` says whose numbers failed.
        """
        if total_kv_heads % tp_size:
            raise RuntimeError(
                f"RBLN NIXL: the {side} tensor-parallel size {tp_size} does not "
                f"divide the model's {total_kv_heads} KV heads; upstream then "
                "replicates one head across ranks and a head band would be a "
                "fraction of a head, which no descriptor names."
            )
        heads_per_rank = total_kv_heads // tp_size
        if slices <= 0 or heads_per_rank % slices:
            raise RuntimeError(
                f"RBLN NIXL: the {side} shard owns {heads_per_rank} KV heads cut "
                f"into {slices} logical slice(s), which does not divide them; the "
                "compiler gives every slice the same head count."
            )
        if areas % slices:
            raise RuntimeError(
                f"RBLN NIXL: the {side} shard reports {areas} chiplet area(s) over "
                f"{slices} logical slice(s), which does not divide them; areas "
                "carry whole slices, replicated when a shard owns fewer heads "
                "than the device has chiplets."
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

        Emitted in LOCAL region order, so ``_compute_desc_ids``'s positional
        pairing keeps holding. Upstream instead walks the peer's list with one
        global ``rank_offset``, which assumes it holds our heads contiguously
        in a single region -- false once a region is one chiplet area and heads
        are area-major: P TP2 -> D TP4 reads local area 2 from remote area 1.

        ``registered_layer_names`` narrows this to one pipeline stage, its
        region list indexed by position within the stage; None means the peer
        owns every layer.
        """
        assert self.transfer_topo is not None
        total_heads = self.transfer_topo.total_num_kv_heads
        areas_l, slices_l = self._kv_areas, self._kv_slices
        areas_r = nixl_agent_meta.kv_areas
        slices_r = nixl_agent_meta.kv_slices

        base_l, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.transfer_topo.tp_size,
            total_heads,
            areas_l,
            slices_l,
            side="local",
        )
        base_r, per_slice_r = self._slice_head_bounds(
            remote_tp_rank,
            remote_tp_size,
            total_heads,
            areas_r,
            slices_r,
            side="peer",
        )
        split = self._head_split(per_slice_l, per_slice_r)

        replicas_l = areas_l // slices_l
        replicas_r = areas_r // slices_r
        num_blocks = nixl_agent_meta.num_blocks
        remote_bases = nixl_agent_meta.kv_caches_base_addr
        remote_lens = nixl_agent_meta.block_lens

        # (local logical region, its position in the peer's region list); a
        # logical region is one K or V of one layer, before chiplet expansion.
        n_logical_l = len(self.block_len_per_layer) // areas_l
        if registered_layer_names is None:
            logical_pairs = [(i, i) for i in range(n_logical_l)]
        else:
            per_layer = self._regions_per_layer() // areas_l
            # peer_pos indexes the peer's own list (see _layer_overlap).
            logical_pairs = [
                (layer_l * per_layer + c, peer_pos * per_layer + c)
                for peer_pos, layer_l in self._layer_overlap(registered_layer_names)
                for c in range(per_layer)
            ]

        # Which of our areas this peer holds (see _fan_in_peer_areas).
        areas_iter = range(areas_l) if peer_areas is None else peer_areas

        out: list[tuple[int, int, int]] = []
        # Axis order is `_shard_local_region_ids`'; within a region, block-major
        # to match _compute_desc_ids' region_id * num_blocks + b.
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
        # The same count `_register_shard_local_xfer_handler` builds locally
        # and `_shard_descs_per_block` records.
        assert len(out) == len(logical_pairs) * len(areas_iter) * num_blocks * split
        return out

    def _fan_in_peer_areas(
        self, remote_tp_rank: int, remote_tp_size: int
    ) -> list[int] | None:
        """Which local chiplet areas live on this peer, when it has MORE TP.

        ``None`` when the peer holds all of them, so callers need no branch.

        Our band spreads over ``|tp_ratio|`` of its ranks, so a transfer to one
        must carry only the areas whose heads that rank owns -- otherwise every
        peer yields the same bytes and the last to land wins.
        """
        assert self.transfer_topo is not None
        if self.use_host_buffer:
            # Host staging registers one logical full-shape buffer per layer,
            # so there are no chiplet areas to divide and nothing below applies.
            return None
        if self.transfer_topo.tp_ratio(remote_tp_size) > 0:
            return None
        total_heads = self.transfer_topo.total_num_kv_heads
        base_l, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.transfer_topo.tp_size,
            total_heads,
            self._kv_areas,
            self._kv_slices,
            side="local",
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
        """Descriptors for one local region: ``split`` (`_head_split`) per block.

        Order is block-major, piece-minor, to match the local list
        `_register_shard_local_xfer_handler` builds.
        """
        base_l, per_slice_l, replicas_l = geom
        base_r, per_slice_r, replicas_r, slices_r = peer

        head = base_l + (area_l // replicas_l) * per_slice_l
        desc_len = self.get_backend_aware_kv_block_len(
            layer_idx=region_id, first_split=True, mamba_view=False
        )
        split = self._head_split(per_slice_l, per_slice_r)
        per_piece = per_slice_l // split
        sub_len = desc_len // split

        # Replicas of a slice hold identical bytes, so reading any one of them
        # answers; take the first.

        out: list[tuple[int, int, int]] = []
        pieces: list[tuple[int, int]] = []
        for j in range(split):
            # The remote slice covering this piece's first head, and how far
            # into it we start.
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
            # them is a plain byte offset.
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
            side="local",
        )
        _, per_slice_r = self._slice_head_bounds(
            0,
            remote_tp_size,
            total_heads,
            nixl_agent_meta.kv_areas,
            nixl_agent_meta.kv_slices,
            side="peer",
        )
        return self._head_split(per_slice_l, per_slice_r)

    @staticmethod
    def _head_split(per_slice_l: int, per_slice_r: int) -> int:
        """How many pieces one of our regions is read in.

        A descriptor names one contiguous range on each side, so an area coarser
        than the peer's slice has to be transferred in as many pieces as the
        peer spreads its heads over.
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

        Upstream registers the remote engine in the TransferTopology and builds
        its TPMapping before any block_size_ratio / tp_ratio / get_engine_info
        lookup, which the callers below and ``_validate_remote_agent_handshake``
        also make. A path that does not delegate to super() has to do this
        itself or get_engine_info() raises KeyError.
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

        `registered_layer_names` narrows this to one pipeline stage's layers,
        `_fan_in_peer_areas` to the chiplet areas a finer peer owns.
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
            "RBLN NIXL: head-matched %d remote desc(s) from "
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
            if self._is_head_matched_peer(remote_tp_size):
                # Different TP degrees, either direction: pair by head range
                # instead of by position (_build_head_matched_remote).
                assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
                return self._add_remote_agent_head_matched(
                    nixl_agent_meta, remote_tp_rank, remote_tp_size
                )
            # Equal TP, or host staging's one region per layer: local region i
            # IS remote region i, which upstream's descriptor math assumes.
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

        # SWA view-opt never meets fan-in: unequal TP is head-matched, and
        # _check_d2d_region_pairing rejects SWA with any of it.
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
            "Full + SWA",
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
    # The handshake, and the layer axis it establishes
    # ------------------------------------------------------------------
    #
    # A peer answers per shard: one per (pp_rank, tp_rank), advertising the layer
    # names it registered and the chiplet geometry of its regions. We match those
    # names against our own -- each side derives its owned range, neither sends
    # it -- and a shard serving less than a whole engine gets its own descriptor
    # lists, addressed by the head arithmetic above. When nothing is narrowed the
    # transfer delegates to upstream's whole-engine handle.

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
        if self._has_swa:
            raise RuntimeError(
                "RBLN NIXL: sliding-window attention is not supported with "
                "pipeline_parallel_size > 1."
            )

    def _publish_handshake_metadata(
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
        self.compat_hash = rbln_compat_hash(base_hash)
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.Encoder().encode(pp_meta),
        )

    def _query_agent_meta(
        self, sock: "zmq.Socket", remote_rank: int, expected_engine_id: str
    ) -> RblnNixlAgentMetadata:
        sock.send(msgspec.msgpack.encode((GET_META_MSG, remote_rank)))
        try:
            handshake_payload = msgspec.msgpack.Decoder(NixlHandshakePayload).decode(
                sock.recv()
            )
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            raise RuntimeError(
                "Failed to decode NixlHandshakePayload; this likely indicates an "
                f"incompatibility between connector versions. Error: {e}"
            ) from e
        assert self.compat_hash is not None
        if (
            self.enforce_compat_hash
            and handshake_payload.compatibility_hash != self.compat_hash
        ):
            raise RuntimeError(
                "NIXL compatibility hash mismatch "
                f"(local={self.compat_hash}, "
                f"remote={handshake_payload.compatibility_hash}). Prefill and "
                "decode instances have incompatible configurations (vLLM "
                "version, model, dtype, KV cache layout, attention backend, "
                "etc.). Disable this check with --kv-transfer-config "
                '\'{"kv_connector_extra_config": '
                '{"enforce_handshake_compat": false}}\''
            )
        try:
            metadata = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
                handshake_payload.agent_metadata_bytes
            )
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            raise RuntimeError(
                f"Failed to decode RblnNixlAgentMetadata. Error: {e}"
            ) from e
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
        """Handshake with every shard of one peer engine.

        Runs on upstream's single-worker handshake executor, outside its lock;
        the one thing published under that lock is this method's return value,
        which upstream's done callback assigns to ``_remote_agents[engine_id]``.
        The read path reaches a peer only once that key exists, which is what
        makes the per-shard state written here visible to it. So every write has
        to land BEFORE the return -- state published after it, or from another
        thread, would be read half-built.
        """
        # Background thread needs a device context (see upstream _nixl_handshake).
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        assert self.transfer_topo is not None
        p_remote_tp_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        path = make_zmq_path("tcp", host, port)
        remote_rank_to_agent_name: dict[int, str] = {}
        overlapping: list[int] = []

        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # ms; avoid hang on dead server

            # Bootstrap: the first shard (pp_rank 0) advertises pp_size.
            first_rank = p_remote_tp_ranks[0]
            metas = {
                first_rank: self._query_agent_meta(sock, first_rank, expected_engine_id)
            }
            pp_size = metas[first_rank].pp_size

            # Guard on either side's pipeline, not just the peer's: the peer
            # runs none in the reverse shape, where ours is the finer one.
            local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
            if pp_size > 1 or local_pp > 1:
                if self._has_swa:
                    raise RuntimeError(
                        "RBLN NIXL: sliding-window attention combined with "
                        "pipeline-parallel P/D is not supported."
                    )
                wide, narrow = max(pp_size, local_pp), min(pp_size, local_pp)
                if wide % narrow:
                    raise RuntimeError(
                        "RBLN NIXL: pipeline-parallel P/D requires one side's "
                        f"pipeline size to be a multiple of the other's (peer "
                        f"{pp_size}, local {local_pp}); otherwise a stage's "
                        "layers straddle two of ours with no whole band to pair."
                    )
                tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
                if tp_ratio != 1 and pp_size > 1 and local_pp > 1:
                    # Either axis alone is handled -- layers by name matching,
                    # heads by _build_head_matched_remote -- but splitting both
                    # on both sides at once has no descriptor path.
                    raise RuntimeError(
                        "RBLN NIXL: heterogeneous tensor parallelism "
                        f"(tp_ratio={tp_ratio}) combined with pipeline "
                        f"parallelism on BOTH sides (peer pp={pp_size}, local "
                        f"pp={local_pp}) is not supported."
                    )
                if tp_ratio < 0 and pp_size > 1:
                    # The peer splits layers AND holds our heads across
                    # several of its ranks; host staging then borrows upstream's
                    # split (_base_fan_in_handle), which asserts a full region
                    # list a stage does not have.
                    raise RuntimeError(
                        "RBLN NIXL: a pipeline-parallel peer with a larger "
                        f"tensor-parallel size (peer {remote_tp_size} > local "
                        f"{self.transfer_topo.tp_size}) is not supported."
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
                    # Two overlaps, and the peer's pipeline size decides
                    # neither: which of our layers it holds, and which of our
                    # chiplet areas. Both narrow a whole-engine handle.
                    overlap = self._layer_overlap(names)
                    if not overlap:
                        continue
                    # The peer's stage reaches past our band, so several of our
                    # ranks pair with it -- a count the transfer path has to
                    # carry, since the peer frees a request's blocks by it.
                    partial = len(overlap) < len(names)
                    split = self._peer_head_split(metadata, remote_tp_size)
                    fan_in = self._is_fan_in_peer(remote_tp_size)

                    if self._is_head_matched_peer(remote_tp_size):
                        # Different TP degrees: pair by head range, over the
                        # layers we share.
                        remote_rank_to_agent_name[global_rank] = (
                            self._add_remote_agent_head_matched(
                                metadata,
                                global_rank,
                                remote_tp_size,
                                registered_layer_names=names,
                            )
                        )
                    else:
                        # Equal TP delegates to upstream, which needs a wider
                        # stage trimmed to our band (_trim_agent_meta_to_layers).
                        remote_rank_to_agent_name[global_rank] = self.add_remote_agent(
                            self._trim_agent_meta_to_layers(metadata, overlap)
                            if partial
                            else metadata,
                            global_rank,
                            remote_tp_size,
                        )

                    if not (pp_size > 1 or partial or fan_in or split > 1):
                        # Nothing is narrowed: upstream's whole-engine handle
                        # describes this peer, so the transfer path delegates.
                        continue
                    self._register_shard_xfer_state(
                        expected_engine_id,
                        global_rank,
                        metadata.block_size,
                        names,
                        peer_areas=self._fan_in_peer_areas(
                            global_rank % remote_tp_size, remote_tp_size
                        ),
                        split=split,
                        remote_tp_size=remote_tp_size,
                    )
                    overlapping.append(global_rank)
        # Published once, not accumulated: a handshake that raises partway
        # leaves no entry, so the retry that follows starts from empty instead
        # of appending its shards a second time and reading every block twice.
        self._overlapping_ranks[expected_engine_id] = overlapping
        self._remote_pp_size[expected_engine_id] = pp_size
        return remote_rank_to_agent_name

    def _base_fan_in_handle(
        self,
        engine_id: str,
        global_rank: int,
        block_size: int,
        region_ids: list[int],
        remote_tp_size: int,
    ) -> int | None:
        """Upstream's own head-band split of our regions, for host staging.

        Host staging registers one logical full-shape buffer per layer and has
        no chiplet areas to select, so a shard's descriptors would cover every
        peer's band at once and NIXL rejects the pairing on length. Upstream
        already built that split while registering the peer -- one handle per
        producer rank, in `all_source_ranks` order -- so borrow it. `None` when
        nothing needs narrowing, leaving the caller on the shard handler.
        """
        assert self.transfer_topo is not None
        if not self.use_host_buffer or block_size != self.block_size:
            return None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        if tp_ratio >= 0:
            return None
        handles = self.src_xfer_handles_by_tp_ratio[tp_ratio]
        plan = self.tp_mappings[engine_id]
        assert region_ids == list(range(self.num_regions)), (
            "RBLN NIXL: borrowing upstream's split needs it to describe the "
            "same regions in the same order, but this peer narrows ours to "
            f"{region_ids} of {self.num_regions}"
        )
        assert len(handles) == len(plan.all_source_ranks)
        # A pipeline-parallel peer cannot reach here (rejected during the
        # handshake), so the rank we hold is the peer's TP rank as planned.
        return handles[plan.all_source_ranks.index(global_rank)]

    def _register_shard_xfer_state(
        self,
        engine_id: str,
        global_rank: int,
        block_size: int,
        registered_layer_names: tuple[str, ...],
        peer_areas: list[int] | None = None,
        split: int = 1,
        remote_tp_size: int = 1,
    ) -> None:
        # Compute the local region ids once and reuse them for the handler
        # (PP context is always the shard path: SWA + PP is rejected earlier).
        region_ids = self._shard_local_region_ids(
            registered_layer_names, peer_areas=peer_areas
        )
        key = (engine_id, global_rank, block_size)
        handle = self._base_fan_in_handle(
            engine_id, global_rank, block_size, region_ids, remote_tp_size
        )
        if handle is not None:
            self._borrowed_src_handles.add(key)
        else:
            handle, _ = self.register_local_xfer_handler(
                block_size,
                registered_layer_names=registered_layer_names,
                peer_areas=peer_areas,
                split=split,
                region_ids=region_ids,
            )
        self.src_xfer_handles_by_remote[key] = handle
        n_groups = len(self.kv_cache_config.kv_cache_groups)
        assert n_groups == 1, (
            "RBLN NIXL per-shard transfers support a single KV-cache group, "
            f"got {n_groups}"
        )
        self._shard_region_group_ids[(engine_id, global_rank)] = (0,) * len(region_ids)
        self._shard_descs_per_block[(engine_id, global_rank)] = split

    def _cleanup_remote_engine(
        self, engine_id: str, *, log_eviction: bool = True
    ) -> None:
        """Drop this engine's per-stage PP state along with upstream's.

        Leaving per-stage entries behind would let a re-handshake read from a
        stage this engine no longer serves. The per-stage local dlist handles are
        ours to release too, one per stage with nothing else referring to them --
        except a borrowed one, which is upstream's (`_base_fan_in_handle`) and
        shared across peers, so only the entry goes.
        """
        for key in [k for k in self.src_xfer_handles_by_remote if k[0] == engine_id]:
            handle = self.src_xfer_handles_by_remote.pop(key)
            if key in self._borrowed_src_handles:
                self._borrowed_src_handles.discard(key)
            else:
                self.nixl_wrapper.release_dlist_handle(handle)
        for skey in [k for k in self._shard_region_group_ids if k[0] == engine_id]:
            del self._shard_region_group_ids[skey]
        for skey in [k for k in self._shard_descs_per_block if k[0] == engine_id]:
            del self._shard_descs_per_block[skey]
        self._remote_shard_layer_names.pop(engine_id, None)
        self._overlapping_ranks.pop(engine_id, None)
        self._remote_pp_size.pop(engine_id, None)
        super()._cleanup_remote_engine(engine_id, log_eviction=log_eviction)

    def _trim_agent_meta_to_layers(
        self, nixl_agent_meta: RblnNixlAgentMetadata, overlap: list[tuple[int, int]]
    ) -> NixlAgentMetadata:
        """Trim a peer stage to the layers this rank owns.

        Upstream pairs remote region i with local region i, so a stage holding
        more layers than our band has to be presented as just that band.

        Every field describing the layers moves together: trimmed regions with
        the full layer list would report the wrong regions per layer to anything
        dividing one by the other.
        """
        rpl = self._regions_per_layer()
        peer_positions = [peer_pos for peer_pos, _ in overlap]
        start, end = peer_positions[0], peer_positions[-1] + 1
        if peer_positions != list(range(start, end)):
            raise RuntimeError(
                "RBLN NIXL PP: this rank owns a non-contiguous part of producer "
                f"stage layers {peer_positions}; the pipelines must divide the "
                "same layer sequence."
            )
        lo, hi = start * rpl, end * rpl
        trimmed: dict[str, Any] = {
            "kv_caches_base_addr": nixl_agent_meta.kv_caches_base_addr[lo:hi],
            "block_lens": nixl_agent_meta.block_lens[lo:hi],
        }
        trimmed["registered_layer_names"] = list(
            nixl_agent_meta.registered_layer_names[start:end]
        )
        return replace(nixl_agent_meta, **trimmed)

    def _layer_overlap(
        self, registered_layer_names: tuple[str, ...] | list[str]
    ) -> list[tuple[int, int]]:
        """Pair a peer stage's layers with ours: ``(peer position, local index)``.

        The peer position indexes ITS OWN list, which is what addresses its
        regions -- a stage wider than our band is read at that offset instead of
        from its start.
        """
        positions_by_name: dict[str, list[int]] = defaultdict(list)
        for local_idx, layer_name in enumerate(self.local_seen_layer_names):
            positions_by_name[layer_name].append(local_idx)

        occurrences_by_name: dict[str, int] = defaultdict(int)
        pairs: list[tuple[int, int]] = []
        for peer_pos, layer_name in enumerate(registered_layer_names):
            occurrence = occurrences_by_name[layer_name]
            occurrences_by_name[layer_name] += 1
            matches = positions_by_name.get(layer_name, [])
            if occurrence >= len(matches):
                continue
            pairs.append((peer_pos, matches[occurrence]))
        return pairs

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

        One filter per axis a peer can be narrower on: its layers, and the
        chiplet areas whose heads it owns. Region ids run logical-region-major,
        area-minor (`(layer * K/V) * areas + area`), which is why the area
        filter is a test on `k % areas`.

        The order here IS the descriptor order -- the local dlist and
        `_build_head_matched_remote` walk these axes in the same nesting.
        """
        rpl = self._regions_per_layer()
        layer_indices = [
            local for _, local in self._layer_overlap(registered_layer_names)
        ]
        keep: list[int]
        if peer_areas is None:
            keep = list(range(rpl))
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
        region_ids: list[int] | None = None,
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
        if region_ids is None:
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
            # The pieces are consecutive byte ranges on this side -- it is the
            # REMOTE side that scatters.
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
        per_block = self._shard_descs_per_block.get((engine_id, global_rank), 1)
        # Converted once, not once per region: this runs per request, and every
        # region of a layer names the same group.
        group_arrays = [np.asarray(g, dtype=np.int64) for g in block_ids]
        desc_ids: list[np.ndarray] = []
        for region_id, group_id in enumerate(region_group_ids):
            group_arr = group_arrays[group_id]
            if group_arr.size == 0:
                continue
            block_ix = region_id * num_blocks + group_arr
            if per_block == 1:
                desc_ids.append(block_ix)
                continue
            # Both dlists are laid out region-major, then block, then piece, so
            # one block becomes `per_block` consecutive descriptors on each side.
            desc_ids.append(
                (
                    block_ix[:, None] * per_block + np.arange(per_block, dtype=np.int64)
                ).ravel()
            )
        if not desc_ids:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(desc_ids)

    def _check_d2d_region_pairing(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject D2D peers whose region lists cannot be paired.

        D2D publishes one region per chiplet area, so region i is the same head
        band only at equal TP; unequal TP pairs by head range instead
        (``_add_remote_agent_head_matched``), which a sliding window is not
        supported with. The other refusal compares regions per layer, not
        totals -- a peer holding more layers is the reverse pipeline shape, and
        upstream would catch a real mismatch at transfer time only.

        Host-bounce has no per-area list, so none of it applies.
        """
        if self.use_host_buffer:
            return
        assert self.transfer_topo is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        if tp_ratio != 1 and self._has_swa:
            raise RuntimeError(
                "RBLN NIXL D2D: sliding-window attention is not supported with "
                f"heterogeneous tensor parallelism (tp_ratio={tp_ratio})."
            )
        n_remote = len(nixl_agent_meta.kv_caches_base_addr)
        peer_layers = len(nixl_agent_meta.registered_layer_names)
        local_rpl = self._regions_per_layer()
        if not peer_layers:
            peer_layers, local_rpl = 1, len(self.block_len_per_layer)
        if n_remote % peer_layers or n_remote // peer_layers != local_rpl:
            raise RuntimeError(
                f"RBLN NIXL D2D: peer publishes {n_remote} KV regions over "
                f"{peer_layers} layer(s) but this worker publishes {local_rpl} "
                "per layer. Regions per layer are K/V times the chiplet count "
                "and depend on neither parallel size, so a mismatch means the "
                "geometry itself differs and no pairing is meaningful."
            )

    def _is_head_matched_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer is served by ``_build_head_matched_remote``.

        Any unequal TP degree, in either direction, on D2D without SWA
        view-opt. ``tp_ratio`` is pure arithmetic on the two TP sizes, so this
        is safe to ask before the engine is registered.
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
        """The byte invariant a peer with a different TP degree has to meet.

        Upstream's check scales a region by the two sides' heads per RANK, which
        holds only for its one-region-per-layer model. After chiplet expansion a
        region is one area, so the per-area ratio is what governs -- the two
        agree at P TP1 -> D TP2 and diverge past it. What does hold, and what the
        descriptor arithmetic needs, is that one KV head costs the same bytes per
        block on both sides.

        Host staging reaches this too, through a pipelined peer: that path never
        calls upstream's own check, so this is its only per-head check. The
        arithmetic holds there because its single area is the whole shard.
        """
        assert self.transfer_topo is not None
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        if block_size_ratio != 1:
            raise RuntimeError(
                "RBLN NIXL: heterogeneous TP requires equal P/D block "
                f"sizes (got block_size_ratio={block_size_ratio})."
            )
        if nixl_agent_meta.kv_cache_layout != self.kv_cache_layout:
            raise RuntimeError(
                "RBLN NIXL: peer KV layout "
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
            side="local",
        )
        _, per_slice_r = self._slice_head_bounds(
            0,
            remote_tp_size,
            total_heads,
            nixl_agent_meta.kv_areas,
            nixl_agent_meta.kv_slices,
            side="peer",
        )
        local_len = self.block_len_per_layer[0]
        remote_len = nixl_agent_meta.block_lens[0]
        if local_len * per_slice_r != remote_len * per_slice_l:
            raise RuntimeError(
                "RBLN NIXL: a KV head occupies "
                f"{local_len / per_slice_l:.0f}B per block here but "
                f"{remote_len / per_slice_r:.0f}B on the peer "
                f"(local {local_len}B over {per_slice_l} head(s), remote "
                f"{remote_len}B over {per_slice_r}). Block size, head_dim and "
                "dtype must match across P and D."
            )

    def _check_mla_constraints(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject the MLA topologies whose descriptor math is not established.

        MLA is REPLICATE and key-only, which upstream's positional pairing
        already expresses. What the refusals below add is head-band matching,
        where bands computed from the configured KV-head count give plausible
        wrong descriptors, and a peer whose chiplet geometry differs from ours.
        """
        if not self.use_mla:
            return
        assert self.transfer_topo is not None
        if self._is_head_matched_peer(remote_tp_size):
            raise RuntimeError(
                "RBLN NIXL D2D: MLA is not supported with heterogeneous tensor "
                f"parallelism (peer TP {remote_tp_size}, local "
                f"{self.transfer_topo.tp_size})."
            )
        # Positional pairing needs both sides to expand a logical region the same
        # way; each derives it from its own device buffers, so a mismatch shifts
        # the block stride and moves wrong bytes without failing.
        peer_geometry = (nixl_agent_meta.kv_areas, nixl_agent_meta.kv_slices)
        if peer_geometry != (self._kv_areas, self._kv_slices):
            raise RuntimeError(
                "RBLN NIXL: MLA chiplet geometry differs between P and D "
                f"(peer {peer_geometry[0]} area(s)/{peer_geometry[1]} slice(s), "
                f"local {self._kv_areas}/{self._kv_slices})."
            )

    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
        self._check_mla_constraints(nixl_agent_meta, remote_tp_size)
        self._check_d2d_region_pairing(nixl_agent_meta, remote_tp_size)
        if nixl_agent_meta.pp_size <= 1:
            if self._is_head_matched_peer(remote_tp_size):
                self._validate_head_matched_handshake(nixl_agent_meta, remote_tp_size)
                return
            super()._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)
            return

        assert self.transfer_topo is not None
        remote_engine_id = nixl_agent_meta.engine_id
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size
        # A producer with FEWER TP ranks is matched per head band; the other
        # direction never reaches here, rejected during the handshake.
        pp_tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        assert pp_tp_ratio > 0, (
            "PP over NIXL P/D does not support a peer with a larger TP size."
        )
        if pp_tp_ratio != 1:
            # Same head-geometry invariant as the non-PP head-matched path; the
            # layer axis does not change what a head costs per block.
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

    def _xfer_notif_id(
        self, engine_id: str, remote_request_id: str, remote_tp_size: int
    ) -> bytes:
        """Notification carrying how many of our ranks pair with one peer rank.

        NOTE(RBLN): upstream sends its own TP size, which the peer divides by its
        own to learn how many of us to hear from before freeing the request's
        blocks. A finer pipeline here multiplies that, each stage pairing with
        the same peer rank, so send it in the unit the peer already divides by:
        ours times the peer's TP. The two agree whenever the pipelines match.
        """
        remote_pp = self._remote_pp_size.get(engine_id, 1)
        local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
        peers = max(1, self.world_size // remote_tp_size) * max(
            1, local_pp // remote_pp
        )
        return f"{remote_request_id}:{peers * remote_tp_size}".encode()
