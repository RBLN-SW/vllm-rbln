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

from collections.abc import Iterable
from typing import Any

import msgspec
import rebel
import torch
from rebel.kv_cache import aligned_tensor
from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    RblnNixlAgentMetadata,
    rbln_compat_hash,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    RblnNixlWorkerState,
)
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class RblnNixlRegistrationMixin(RblnNixlWorkerState):
    """The registration lifetime: turning our KV tensors into a region table and
    registering that memory with NIXL.

    A mixin rather than a module of functions, for the same reason as the
    handshake lifetime -- upstream calls register_kv_caches and the host-buffer
    hooks on the worker.
    """

    #: Whether nixl-rbln is installed, so the RBLN backend can be asked for.
    _use_rbln_nixl_backend: bool
    _pending_kv_caches: dict[str, torch.Tensor] | None

    def _check_pp_constraints(self) -> None:
        if self.vllm_config.parallel_config.pipeline_parallel_size <= 1:
            return
        if self.topo.cross_layers_blocks:
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

    def _is_region_replicated(self, region_idx: int) -> bool:
        return super()._is_region_replicated(self._viewed_region(region_idx))

    def _layer_kv_heads(self, layer_name: str) -> int | None:
        """Model-wide KV heads of one layer, or None where no head band names it.

        Upstream floors the per-rank share at 1, so below one head per rank it
        fits more than one declared count -- a draft's as well as the target's.
        """
        layer_spec = self._unwrapped_layer_spec(layer_name)
        if isinstance(layer_spec, MambaSpec):
            return None
        known = {self.model_config.get_total_num_kv_heads()}
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is not None:
            draft_model_config = speculative_config.draft_model_config
            if draft_model_config is not None:
                known.add(draft_model_config.get_total_num_kv_heads())
        candidates = {
            total
            for total in known
            if max(1, total // self.world_size) == layer_spec.num_kv_heads
        }
        if len(candidates) != 1:
            return None
        total = candidates.pop()
        return None if total < self.world_size else total

    def _logical_head_bands(
        self, layer_regions: list[tuple[str, int]]
    ) -> list[int | None]:
        """One entry per logical region: the head band of the layer it belongs to."""
        return [
            self._layer_kv_heads(name)
            for name, regions in layer_regions
            for _ in range(regions)
        ]

    def _layer_page_sizes(self, layer_names: Iterable[str]) -> set[int]:
        """The distinct per-layer page sizes among these layers."""
        return {
            self._unwrapped_layer_spec(name).page_size_bytes for name in layer_names
        }

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
            kv_split_axis=self._kv_split_axis,
        )
        base_hash = self.compat_hash
        assert base_hash is not None
        self.compat_hash = rbln_compat_hash(
            base_hash,
            writes_into_peer=self._writes_into_peer,
            speculative_config=self.vllm_config.speculative_config,
        )
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.Encoder().encode(pp_meta),
        )

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
            self.vllm_config, self.backend_name, self.topo.cross_layers_blocks
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

        page_sizes = self._layer_page_sizes(xfer_buffers)
        mla_layers = {
            name
            for name in xfer_buffers
            if isinstance(
                self._unwrapped_layer_spec(name),
                (MLAAttentionSpec, SlidingWindowMLASpec),
            )
        }
        if mla_layers and len(mla_layers) != len(xfer_buffers):
            # Refused here rather than as a block-count mismatch far from its cause.
            raise RuntimeError(
                "RBLN NIXL: KV caches mix MLA and non-MLA layers "
                f"({sorted(mla_layers)} against the rest), which the region "
                "layout cannot express -- the K/V split is chosen once for the "
                "whole engine."
            )
        if self.topo.cross_layers_blocks and len(page_sizes) > 1:
            # The page is scaled by the KV-cache tensor count, so with sizes that
            # differ the product describes no layer.
            raise RuntimeError(
                "RBLN NIXL: cross-layer blocks require one page size for every "
                f"layer, got {sorted(page_sizes)}."
            )

        # Logical K/V regions (entry_tensor, byte_offset, full_block_len)
        # for nixl-rbln.
        regions: list[tuple[Any, int, int]] = []
        # REPLICATE flag per logical region, expanded to the chiplet-expanded
        # transfer table below (see _region_is_mla).
        logical_mla: list[bool] = []
        # None where a region has no head axis to compare against, which the
        # derivation below can never read as a non-head cut.
        logical_kv_heads: list[int | None] = []
        # How many logical regions each layer contributes, for the head bands.
        layer_regions: list[tuple[str, int]] = []
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._unwrapped_layer_spec(layer_name)
            cache_list = self.topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            layer_regions.append((layer_name, len(cache_list)))
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
            if self.topo.cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
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
                # Replaces upstream's equal-size check: a spec disagreeing with
                # what was allocated fails here instead of shifting every
                # descriptor past block 0.
                if not isinstance(layer_spec, MambaSpec) and not (
                    self.topo.cross_layers_blocks
                ):
                    region_bytes = cache.numel() * cache.element_size()
                    assert region_bytes == num_blocks * full_block_len, (
                        f"layer {layer_name} region of {region_bytes}B is not "
                        f"{num_blocks} blocks of {full_block_len}B"
                    )
                regions.append((cache_or_caches, region_offset, full_block_len))
                logical_mla.append(is_mla_region)
                logical_kv_heads.append(
                    layer_spec.num_kv_heads
                    if isinstance(layer_spec, AttentionSpec)
                    else None
                )

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
        self._logical_region_kv_heads = self._logical_head_bands(layer_regions)
        # `slice_ids` is per area, and replicas of one slice share an id, so
        # the DISTINCT ids over a region's areas are its own slice count.
        self._logical_region_slices = [
            len(set(xfer.slice_ids[r * areas : (r + 1) * areas]))
            for r in range(len(logical_mla))
        ]

        self.num_regions = len(xfer.base_addrs)
        if self.topo.is_kv_layout_blocks_first:
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
        # Downstream divides by this.
        assert self._kv_slices > 0, (
            f"the plugin reported {self._kv_slices} logical slice(s) per shard"
        )

        # A head axis of extent one cannot be cut, so the compiler replicates
        # instead and more than one slice there has no other explanation. Above
        # one head both axes fit the same count, so an axis that is not derived
        # stays HEAD rather than guessed.
        region_cut = [
            (heads, self._logical_region_slices[r])
            for r, heads in enumerate(logical_kv_heads)
        ]
        region_non_head = {heads == 1 and slices > 1 for heads, slices in region_cut}
        if len(region_non_head) > 1:
            raise RuntimeError(
                "RBLN NIXL (D2D): this engine's KV regions were not all cut on "
                "the same axis, which one advertised geometry cannot describe. "
                f"(head count, distinct slice ids) per logical region: {region_cut}."
            )
        self._kv_split_axis = (
            KVSplitAxis.NON_HEAD if region_non_head == {True} else KVSplitAxis.HEAD
        )
        logger.info(
            "RBLN NIXL (D2D): registered %d transfer region(s) across %d chiplet "
            "area(s), %d logical slice(s), cut on the %s axis%s.",
            self.num_regions,
            xfer.n_shards,
            xfer.slices,
            self._kv_split_axis.name,
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

    def _unwrapped_layer_spec(self, layer_name: str) -> Any:
        """This layer's own spec.

        Layers sharing an attention type but not a size arrive as one
        `UniformTypeKVCacheSpecs`, whose page size is the SUM over the group.
        """
        layer_spec = self._layer_specs[layer_name]
        if isinstance(layer_spec, UniformTypeKVCacheSpecs):
            return layer_spec.kv_cache_specs[layer_name]
        return layer_spec

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
        page_sizes = self._layer_page_sizes(kv_caches)
        if len(page_sizes) > 1:
            # TODO(RBLN): delete once the pinned vLLM drops that assert --
            # upstream already did, while standardising the KV-cache layout.
            raise RuntimeError(
                "RBLN NIXL: host staging cannot register KV caches whose "
                f"per-layer size differs (got {sorted(page_sizes)}), which is "
                "what a speculative draft model with its own "
                "num_key_value_heads produces. Upstream's "
                "`register_kv_caches` asserts one size for every non-MLA "
                "tensor and this path delegates to it. Use "
                "kv_buffer_device='rbln'."
            )
        super().register_kv_caches(kv_caches)
        # Every layer contributes the same number of regions (its K/V halves), so
        # the count follows from the transfer table upstream just filled. The D2D
        # path collects the list while it builds the regions instead.
        names = list(kv_caches.keys())
        per_layer, remainder = divmod(len(self.block_len_per_layer), len(names))
        assert remainder == 0, (
            f"{len(self.block_len_per_layer)} transfer region(s) do not divide "
            f"among {len(names)} layer(s)"
        )
        self._logical_region_kv_heads = self._logical_head_bands(
            [(name, per_layer) for name in names]
        )
        # Re-wrap upstream's published handshake metadata with this stage's PP
        # identity + owned layer names (no-op degrade for pp_size == 1).
        if self.xfer_handshake_metadata is not None:
            base_agent_metadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
                self.xfer_handshake_metadata.agent_metadata_bytes
            )
            self._publish_handshake_metadata(base_agent_metadata, kv_caches.keys())

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
