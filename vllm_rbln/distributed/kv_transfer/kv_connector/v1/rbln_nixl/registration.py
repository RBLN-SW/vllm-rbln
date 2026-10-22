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

from collections import Counter
from collections.abc import Iterable
from typing import Any

import msgspec
import rebel
import torch
from rebel.kv_cache import aligned_tensor
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
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    ChipletSplitAxis,
    RblnNixlAgentMetadata,
    rbln_compat_hash,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    AreaPlace,
    RblnNixlWorkerState,
    RegionGeometry,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_transfer_topology import (
    RblnTransferTopology,
)
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.utils import canonical_kv_layers

logger = init_logger(__name__)

# Chiplets the compiler picks an attention op's shard mode for, and the lane
# group a context-cut shard's columns have to tile. `_region_geometries`
# mirrors the rule the pinned rebel-compiler applies to its packed-KV attention
# ops; that rule is not an interface the compiler exports, so re-check it when
# the pin moves.
RSD_SHARDS = 4
_LANES = 64


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
    #: None where nobody named one, which is not the same as a width of 0.
    _stripe_width: int | None

    @property
    def _backend_extra(self) -> dict[str, int]:
        """What the adapter takes beyond the arguments every call passes.

        0 is a width the adapter accepts and a missing one means "keep the
        plugin default", so the test is on absence, not on truthiness. Both
        calls that reach the adapter read it from here.
        """
        if self._stripe_width is None:
            return {}
        return {"stripe_width": self._stripe_width}

    def _check_pp_constraints(self) -> None:
        if self.vllm_config.parallel_config.pipeline_parallel_size <= 1:
            return
        if self.topo.cross_layers_blocks:
            raise RuntimeError(
                "RBLN NIXL: cross-layer-blocks mode is not supported with "
                "pipeline_parallel_size > 1."
            )
        if self._shape.has_swa:
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

    def _publish_handshake_metadata(self, base_meta: NixlAgentMetadata) -> None:
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
            block_strides=base_meta.block_strides,
            kv_cache_layout=base_meta.kv_cache_layout,
            block_size=base_meta.block_size,
            ssm_sizes=base_meta.ssm_sizes,
            attn_backend_name=base_meta.attn_backend_name,
            physical_blocks_per_logical_kv_block=(
                base_meta.physical_blocks_per_logical_kv_block
            ),
            # vllm 0.30 put the region ledger and the context-parallel sizes on
            # the wire. Re-wrapping without them advertises the dataclass
            # defaults, so a peer would describe our regions by its own
            # geometry instead of reading ours.
            region_num_blocks=base_meta.region_num_blocks,
            region_group_ids=base_meta.region_group_ids,
            region_names=base_meta.region_names,
            region_mem_types=base_meta.region_mem_types,
            dcp_size=base_meta.dcp_size,
            pcp_size=base_meta.pcp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            kv_area_slices=[
                [place.slice_id for place in geometry.areas]
                for geometry in self._logical_region_geometry
            ],
            kv_split_axes=[geometry.axis for geometry in self._logical_region_geometry],
            kv_per_block=self._kv_per_block,
            # Zero where this shard holds no sliding-window group, and where its
            # groups disagree -- neither cuts a window range of its own. A peer
            # that does cuts OUR addresses by ITS grid, so the second case goes
            # unrefused; no model here produces it.
            swa_kernel_block=(
                next(iter(self._swa_kernel_blocks))
                if len(self._swa_kernel_blocks) == 1
                else 0
            ),
        )
        base_hash = self.compat_hash
        assert base_hash is not None
        self.compat_hash = rbln_compat_hash(
            base_hash,
            writes_into_peer=self._writes_into_peer,
            cross_layers_blocks=self.topo.cross_layers_blocks,
            speculative_config=self.vllm_config.speculative_config,
        )
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.Encoder().encode(pp_meta),
        )

    def _register_kv_caches_impl(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Direct variant of NixlConnectorWorker.register_kv_caches:
        build the upstream topology, lay each logical K/V region over the
        chiplet areas its entry's storage is held in, register one MR per area,
        and feed the resulting transfer tables into upstream's transfer state.
        """
        import nixl_rbln

        self.transfer_topo = RblnTransferTopology(
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
            self.vllm_config,
            self.backend_name,
            transfer_mode=self._TRANSFER_MODE,
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

        # Logical K/V regions (entry_tensor, layer_spec, byte_offset,
        # full_block_len), which `_region_geometries` lays over the chiplet
        # areas its entry's storage sits in.
        regions: list[tuple[Any, Any, int, int]] = []
        # REPLICATE flag per logical region, expanded to the chiplet-expanded
        # transfer table (see `_region_is_mla`).
        logical_mla: list[bool] = []
        # This rank's share, not the model-wide band; None where the layer has
        # no attention heads.
        logical_kv_heads: list[int | None] = []
        # How many logical regions each layer contributes, for the head bands.
        layer_regions: list[tuple[str, int]] = []
        kv_per_block_seen: int | None = None
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
            if not is_mla_region and not isinstance(layer_spec, MambaSpec):
                # One region back means the block holds both; two means each
                # has its own. Read off what registered, not the config that
                # chose it. Anything else makes `2 //` yield 0, which surfaces
                # as a division far from the layout that caused it.
                assert len(cache_list) in (1, 2), (
                    f"{layer_name}: an attention layer registers K and V as "
                    f"one region or as two, got {len(cache_list)}"
                )
                kv_per_block = 2 // len(cache_list)
                assert kv_per_block_seen in (None, kv_per_block), (
                    "attention layers disagree on whether K and V share a "
                    "block; the descriptor arithmetic is model-wide"
                )
                kv_per_block_seen = kv_per_block
                self._kv_per_block = kv_per_block
            if self.topo.cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )

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
                regions.append(
                    (cache_or_caches, layer_spec, region_offset, full_block_len)
                )
                logical_mla.append(is_mla_region)
                logical_kv_heads.append(
                    layer_spec.num_kv_heads
                    if isinstance(layer_spec, AttentionSpec)
                    else None
                )

        rbln_ctx_ptr = rebel.context_of(sample_kv_cache).rbln_ctx_ptr

        nixl_rbln.ensure_rbln_backend(
            self.nixl_wrapper,
            device_id,
            rbln_ctx_ptr=rbln_ctx_ptr,
            **self._backend_extra,
        )
        geometries, area_descs = self._region_geometries(regions)
        self._registered_descs.append(
            self.nixl_wrapper.register_memory(
                [(addr, size, device_id, "") for addr, size in area_descs],
                mem_type=self.nixl_memory_type,
                backends=["RBLN"],
            )
        )
        self.device_id = device_id
        self._logical_region_geometry = geometries
        places = [place for geometry in geometries for place in geometry.areas]
        self.block_len_per_layer = [place.block_len for place in places]
        # An area holds one contiguous run of blocks, so a block's stride is
        # its own length. Upstream keeps the two apart because a region can be
        # a strided window into a larger tensor; on RBLN it never is, and the
        # D2D descriptor builder already reads addresses as
        # `base + block * block_len`.
        self.block_stride_per_layer = list(self.block_len_per_layer)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = [
            place.device_addr for place in places
        ]
        # Transfer region `logical * areas + area` is one area of one logical
        # region, which needs every entry over the same number of areas.
        area_counts = {len(geometry.areas) for geometry in geometries}
        if len(area_counts) != 1:
            raise RuntimeError(
                "RBLN NIXL (D2D): KV entries are laid over different chiplet "
                f"area counts {sorted(area_counts)}; a transfer region is "
                "numbered by one."
            )
        (areas,) = area_counts
        self._kv_areas = areas

        # Upstream keys REPLICATE vs SPLIT off this list and indexes it 1:1 with
        # block_len_per_layer, which is chiplet-expanded here -- so the flags are
        # too, every area of a logical region carrying the same latent.
        self._region_is_mla = [is_mla for is_mla in logical_mla for _ in range(areas)]
        # vllm 0.30.0 added four per-region lists upstream builds while walking
        # the caches. This path registers the areas itself, so they are filled
        # here, in the same logical-major, area-minor order as
        # `_region_is_mla`. Every RBLN region is one area of one layer's cache,
        # so they hold the whole block count and one memory type.
        self.region_names = [
            name for name, regions in layer_regions for _ in range(regions * areas)
        ]
        self.region_num_blocks = [self.num_blocks] * len(self.block_len_per_layer)
        self.region_mem_types = [self.nixl_memory_type] * len(self.block_len_per_layer)
        # Transfer-group index, not cache-group index: upstream's descriptor
        # machinery indexes `transfer_groups`, and the two lists diverge as soon
        # as a group opts out of KV transfer.
        group_of = self.kv_cache_config.transfer_group_index_by_layer
        self.region_group_ids = [group_of[name] for name in self.region_names]

        self._logical_region_kv_heads = self._logical_head_bands(layer_regions)
        self._logical_region_local_heads = list(logical_kv_heads)

        self.num_regions = len(places)
        if self.topo.virtually_split_kv_in_blocks:
            # A Mamba state indexes its two regions separately, which doubles
            # the count -- except for key-only MLA regions, which have no second
            # half. Unreachable: upstream returns this for a Mamba group alone,
            # and `register_kv_caches` refuses one.
            self.num_regions = sum(
                1 if self._is_region_replicated(i) else 2
                for i in range(len(self._region_is_mla))
            )
        self.num_descs = self.num_regions * self.num_blocks

        # The spec carries `block_size` and `sliding_window`; which of the two
        # the kernel addresses the cache in is a property of the compiled
        # graph, and the view the runner handed over is the frame it reads.
        self._swa_kernel_blocks = {
            view[self.topo.kv_token_dim] for _, view in self._swa_group_views
        }
        if self._shape.has_swa:
            logger.info(
                "RBLN NIXL: the sliding-window kernel addresses this cache in "
                "blocks of %s token(s).",
                sorted(self._swa_kernel_blocks),
            )
        full_groups = sum(
            not isinstance(spec, SlidingWindowSpec) for spec in self._group_specs
        )
        # Streaming names part of a request for its own reason and needs the
        # same shape to do it: one group whose blocks it counts in, and any
        # other group described by a range of its own.
        cutter = "chunk_mode" if self._shape.chunk_mode else "a streamed write"
        # One group is all it asks: `_own_engine_layout` is already true for
        # every multi-group engine that writes part of a request, and both
        # counts come off `transfer_groups`, so a second term could only repeat
        # this.
        if self._shape.writes_part_of_a_request and full_groups != 1:
            raise RuntimeError(
                f"RBLN NIXL (D2D): {cutter} needs one full-attention KV-cache "
                "group, and any other group to be a sliding window whose view "
                f"it can extend. Got groups={len(self._group_specs)}, "
                f"full={full_groups}, swa={self._shape.has_swa}, "
                f"sw_ratio={self._shape.window_ratio}."
            )
        # The whole-engine lists name a block's chunks without naming which
        # span holds the request's last token, so a block cut into several
        # spans would send chunks past the request's own blocks. A streamed
        # write cuts no block but still counts its coverage in those spans,
        # which these lists were never shown to place. A head cut has one span.
        region_axes = [geometry.axis for geometry in geometries]
        if (
            self._shape.writes_part_of_a_request
            and self._own_engine_layout
            and ChipletSplitAxis.TOKEN in region_axes
        ):
            raise RuntimeError(
                f"RBLN NIXL (D2D): {cutter} on a sliding-window engine needs "
                "a head cut, which leaves one token range a block. This cache "
                f"is cut on the axes {[a.name for a in region_axes]} into "
                f"{areas} area(s)."
            )
        cut_on = Counter(axis.name for axis in region_axes)
        slice_counts = sorted({geometry.slices for geometry in geometries})
        logger.info(
            "RBLN NIXL (D2D): registered %d transfer region(s) across %d chiplet "
            "area(s), %s logical slice(s), cut on the %s%s.",
            self.num_regions,
            areas,
            "/".join(map(str, slice_counts)),
            f"{next(iter(cut_on))} axis"
            if len(cut_on) == 1
            else " and ".join(
                f"{name} axis for {n} logical region(s)" for name, n in cut_on.items()
            ),
            " -- some chiplet areas hold replicas of a slice"
            if slice_counts != [areas]
            else "",
        )

        # One grid per range for this engine. Each descriptor list derives its
        # chunk grid from the block size it was built with; these are the ones
        # a transfer reads back, and they have to answer for the list it
        # selects in. The window range has no such parameter, so the builders
        # read the value parked here rather than ask twice.
        self._chunk_grid = self._shard_chunk_grid(
            range(self.num_regions), block_size=self.block_size, split=1
        )
        self._window_grid_cut = self._window_grid()
        # Said once here rather than per peer: every list counts a block in
        # the same chunks, and a peer whose block holds a different count is
        # refused at the handshake.
        if self._shape.chunk_mode:
            chunks_per_block = self._chunks_per_block(self._chunk_grid)
            logger.info(
                "RBLN NIXL (D2D): chunk_mode on: %d-token chunks, %d a block.",
                self.block_size // chunks_per_block,
                chunks_per_block,
            )
        else:
            logger.info("RBLN NIXL (D2D): chunk_mode off.")

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        # The 0.30.0 read path looks its own engine up in these the way it
        # looks a peer up, so registering has to seed them here as well.
        self.dst_region_num_blocks[self.engine_id] = self.region_num_blocks
        self.dst_region_group_ids[self.engine_id] = self.region_group_ids
        self.dst_region_mem_types[self.engine_id] = self.region_mem_types
        self._uses_region_group_mapping = len(set(self.region_group_ids)) > 1
        self.dst_uses_region_group_mapping[self.engine_id] = (
            self._uses_region_group_mapping
        )
        self._mixed_mem_types = len(set(self.region_mem_types)) > 1

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
            block_strides=self.block_stride_per_layer,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            # The region ledger vllm 0.30 put on the wire. Upstream fills these
            # inside the `register_kv_caches` this path replaces, so leaving
            # them out advertises the dataclass defaults and a peer describes
            # our regions by its own geometry.
            region_num_blocks=self.region_num_blocks,
            region_group_ids=self.region_group_ids,
            region_names=self.region_names,
            region_mem_types=self.region_mem_types,
            dcp_size=self.dcp_size,
            pcp_size=self.pcp_size,
        )
        # Republish with the chiplet geometry the region ledger cannot express.
        self._publish_handshake_metadata(agent_metadata)

    def _region_geometries(
        self, regions: list[tuple[torch.Tensor, Any, int, int]]
    ) -> tuple[list[RegionGeometry], list[tuple[int, int]]]:
        """Lay every logical region over the chiplet areas its storage sits in.

        The device answers one buffer per chiplet and says nothing about how
        they were cut, so the areas, their addresses and the slice count are
        read while only the dim is derived -- from the predicate the attention
        op picks its shard mode by, over every frame an op reads the storage
        in. A storage is walked once: a layer's K and V, and the layers that
        alias it, share its cut. Returns the geometries in `regions` order and
        one (address, size) per distinct area, which is the memory-region list
        only this walk knows.
        """
        import nixl_rbln

        head_dim, token_dim = self.topo.kv_head_dim, self.topo.kv_token_dim
        cuts: dict[int, tuple[ChipletSplitAxis, int, tuple[tuple[int, int], ...]]] = {}
        areas: list[tuple[int, int]] = []
        for entry, spec, _, _ in regions:
            base_addr = entry.data_ptr()
            if base_addr in cuts:
                continue
            shape = tuple(entry.shape)
            if entry.storage_offset() != 0:
                raise RuntimeError(
                    f"RBLN NIXL (D2D): a KV entry of shape {shape} starts "
                    f"{entry.storage_offset()} element(s) into its storage, "
                    "whose areas are read from its first byte."
                )
            # The adapter's own call: it settles the entry on the device
            # before asking, and names the rebel version that answers.
            buffers = nixl_rbln.entry_areas(base_addr)
            sizes = {buffer.size for buffer in buffers}
            if len(sizes) != 1:
                raise RuntimeError(
                    f"RBLN NIXL (D2D): a KV entry of shape {shape} sits on "
                    f"chiplet areas of {sorted(sizes)}B; a cut of it gives "
                    "every area the same."
                )
            (area_size,) = sizes
            shards = len(buffers)
            if shards not in (1, RSD_SHARDS):
                raise RuntimeError(
                    f"RBLN NIXL (D2D): a KV entry of shape {shape} sits on "
                    f"{shards} chiplet area(s), and the compiler picks its "
                    f"shard mode for {RSD_SHARDS}, so which piece an area "
                    "holds does not follow."
                )
            entry_bytes = entry.numel() * entry.element_size()
            slices, remainder = divmod(entry_bytes, area_size)
            if remainder or not slices or shards % slices:
                raise RuntimeError(
                    f"RBLN NIXL (D2D): a KV entry of {entry_bytes}B sits on "
                    f"{shards} chiplet area(s) of {area_size}B, which do not "
                    "cut it evenly."
                )
            axis, slice_ids = ChipletSplitAxis.HEAD, (0,) * shards
            if slices > 1:
                if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                    raise RuntimeError(
                        f"RBLN NIXL (D2D): a KV entry of shape {shape} is cut "
                        f"into {slices} piece(s), and {type(spec).__name__} is "
                        "not an op whose shard contract this path holds."
                    )
                # Every frame an op reads this storage in: the entry's own,
                # then the window views the runner handed over. They share one
                # cut, so they have to agree on the dim and on its length.
                frames = [(shape, isinstance(spec, SlidingWindowSpec))] + [
                    (view, True)
                    for addr, view in self._swa_group_views
                    if addr == base_addr and view != shape
                ]
                axes = set()
                for frame, reads_a_window in frames:
                    if head_dim is None:
                        # A latent has no head to cut: CP is the op's only mode.
                        axes.add(ChipletSplitAxis.TOKEN)
                        continue
                    heads, tokens = frame[head_dim], frame[token_dim]
                    if reads_a_window:
                        cols = tokens // shards
                        cp_cuts = (
                            bool(cols)
                            and tokens % shards == 0
                            and (cols % _LANES == 0 or _LANES % cols == 0)
                        )
                        can_tp = heads >= shards or not cp_cuts
                    else:
                        # The QK matmul carries its output channels in 64-wide
                        # blocks, so a shard of the partition has to hold one.
                        can_tp = heads >= shards or not tokens >= shards * _LANES
                    axes.add(
                        ChipletSplitAxis.HEAD if can_tp else ChipletSplitAxis.TOKEN
                    )
                if len(axes) != 1:
                    raise RuntimeError(
                        f"RBLN NIXL (D2D): the ops reading the storage of a KV "
                        f"entry of shape {shape} cut it on "
                        f"{'/'.join(sorted(a.name for a in axes))}, and the "
                        f"frames they read it in are {[f for f, _ in frames]}."
                    )
                (axis,) = axes
                dim = token_dim if axis is ChipletSplitAxis.TOKEN else head_dim
                # A latent has no head dim, and that is the branch that forced
                # TOKEN above, so a head cut here always has one.
                assert dim is not None, "a head cut over a cache with no head dim"
                lengths = {frame[dim] for frame, _ in frames}
                if len(lengths) != 1:
                    raise RuntimeError(
                        f"RBLN NIXL (D2D): the frames {[f for f, _ in frames]} "
                        f"an op reads the storage of a KV entry of shape "
                        f"{shape} in hold {sorted(lengths)} on the cut dim "
                        f"{dim}, so one cut of it does not name the same bytes "
                        "in both."
                    )
                if shape[dim] % slices:
                    raise RuntimeError(
                        f"RBLN NIXL (D2D): a KV entry of shape {shape} is cut "
                        f"into {slices} piece(s) on dim {dim}, which does not "
                        f"tile its {shape[dim]} evenly."
                    )
                # The areas arrive in chiplet order, and a slice's copies sit
                # side by side: a shard's nodes are a contiguous block of ids,
                # which is the residency the compiler's mesh plan treats as
                # aligned. That is its convention, not a contract it offers.
                replicas = shards // slices
                slice_ids = tuple(i // replicas for i in range(shards))
            cuts[base_addr] = (
                axis,
                slices,
                tuple(zip((buffer.device_addr for buffer in buffers), slice_ids)),
            )
            areas += [(buffer.device_addr, area_size) for buffer in buffers]

        assert not {addr for addr, _ in self._swa_group_views} - cuts.keys(), (
            "a sliding-window group reads a storage no registered KV entry "
            "holds, although `canonical_kv_layers` keeps one layer per buffer"
        )
        geometries = []
        for entry, _, offset, block_len in regions:
            axis, slices, placed = cuts[entry.data_ptr()]
            geometries.append(
                RegionGeometry(
                    axis,
                    slices,
                    tuple(
                        AreaPlace(addr + offset // slices, block_len // slices, slice)
                        for addr, slice in placed
                    ),
                )
            )
        # One MR per chiplet area: an area is an allocation of its own, so no
        # MR can span two.
        return geometries, list(dict.fromkeys(areas))

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
        assert (
            self.use_mla or KVCacheLayout[self.kv_cache_layout].is_block_contiguous
        ), (
            "RBLN NIXL Connector needs [H, N, C] contiguous inside a block, "
            f"got {self.kv_cache_layout}"
        )
        first = next(iter(kv_caches.values()))
        kv_first = first.ndim > 1 and tuple(first.shape[:2]) == (2, self.num_blocks)
        if not self.use_mla and kv_first:
            # Upstream cuts regions by `cache.stride(0)`, which on a K/V-first
            # cache spans a whole half and divides into nothing. It raises with
            # the shapes and no remedy, so name the one here.
            raise RuntimeError(
                "RBLN NIXL: host staging cannot register a K/V-first KV cache "
                f"(got {tuple(first.shape)}), which is what the rbln_triton_ops "
                "kernels allocate. Upstream's `register_kv_caches`, which this "
                "path delegates to, partitions a blocks-first cache only. Use "
                "kv_buffer_device='rbln'."
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
        if self._has_mamba:
            raise RuntimeError(
                "RBLN NIXL: a Mamba/SSM KV-cache group is not supported over "
                "NIXL P/D. Its cache is one blocks-first region per layer, "
                "which the per-area region table this connector publishes "
                "cannot describe. Mixed full and sliding-window attention is a "
                "different thing and is supported."
            )
        # One region per buffer: sliding-window views alias a full-attention
        # layer's storage but count blocks differently, and KV-sharing layers
        # own neither a buffer nor a spec. Inspect the transfer specs only.
        canonical = canonical_kv_layers(
            self.kv_cache_config,
            {name: self._unwrapped_layer_spec(name) for name in self._layer_specs},
        )
        # The runner allocates from its own copy of the connector's config.
        missing = canonical - kv_caches.keys()
        assert not missing, f"Canonical layers missing from kv_caches: {missing}"
        # Which view the sliding-window kernel reads a block in, taken before
        # `canonical_kv_layers` drops the layer that names it. The registered
        # entry is another view of the same storage.
        self._swa_group_views = [
            (kv_caches[name].data_ptr(), tuple(kv_caches[name].shape))
            for group in self.kv_cache_config.transfer_groups
            if isinstance(group.kv_cache_spec, SlidingWindowSpec)
            for name in group.layer_names
        ]
        # Preserve the caller's layer order: iterating the canonical set varies
        # with PYTHONHASHSEED and can mismatch the P/D region-to-layer mapping.
        kv_caches = {name: kv for name, kv in kv_caches.items() if name in canonical}
        # `__init__` copied the count vllm estimated before the compile. A
        # dynamic-KV resize settles the real one after warm-up and registers
        # only then, so take what the worker last allocated for.
        num_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None
        self.num_blocks = num_gpu_blocks
        self._logical_num_blocks = num_gpu_blocks

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

            nixl_rbln.ensure_rbln_backend(
                self.nixl_wrapper, device_id=0, **self._backend_extra
            )
        page_sizes = self._layer_page_sizes(kv_caches)
        if len(page_sizes) > 1:
            # TODO(RBLN): lift once the head bands below are read per region.
            # Upstream stopped requiring one size a layer when it moved to
            # per-region lengths; what still requires it is this side.
            raise RuntimeError(
                "RBLN NIXL: host staging cannot register KV caches whose "
                f"per-layer size differs (got {sorted(page_sizes)}), which is "
                "what a speculative draft model with its own "
                "num_key_value_heads produces. The head bands this path "
                "derives below give every layer the same number of regions "
                "and the same band, so a differing layer would be addressed "
                "by another's geometry. Use kv_buffer_device='rbln'."
            )
        super().register_kv_caches(kv_caches)
        # Every layer contributes the same number of regions -- one where the
        # block holds K and V together -- so the count follows from the transfer
        # table upstream just filled. The D2D path collects the list while it
        # builds the regions instead.
        names = list(kv_caches.keys())
        per_layer, remainder = divmod(len(self.block_len_per_layer), len(names))
        assert remainder == 0, (
            f"{len(self.block_len_per_layer)} transfer region(s) do not divide "
            f"among {len(names)} layer(s)"
        )
        self._logical_region_kv_heads = self._logical_head_bands(
            [(name, per_layer) for name in names]
        )
        # A whole buffer a region: one area holding all of it.
        self._logical_region_geometry = [
            RegionGeometry(ChipletSplitAxis.HEAD, 1, (AreaPlace(base, length, 0),))
            for base, length in zip(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank],
                self.block_len_per_layer,
            )
        ]
        # Re-wrap upstream's published handshake metadata with this stage's PP
        # identity and chiplet geometry (no-op degrade for pp_size == 1).
        if self.xfer_handshake_metadata is not None:
            base_agent_metadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
                self.xfer_handshake_metadata.agent_metadata_bytes
            )
            self._publish_handshake_metadata(base_agent_metadata)

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
