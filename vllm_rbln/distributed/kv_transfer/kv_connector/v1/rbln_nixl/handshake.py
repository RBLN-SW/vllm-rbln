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

import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    ChipletSplitAxis,
    RblnNixlAgentMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    AreaRect,
    Piece,
    RblnNixlWorkerState,
    _grid_descs,
    every_local_link_down,
    pair_area,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

logger = init_logger(__name__)


class RblnNixlHandshakeMixin(RblnNixlWorkerState):
    """The handshake lifetime: querying a producer, refusing a pairing we cannot
    serve, and building the descriptors that address it.

    The head axis answers one question: for each of our chiplet areas, where in
    the peer do its heads live. A region is one chiplet area, so region i names
    different heads on two peers as soon as their TP degrees differ; an area may
    be a replica rather than a distinct slice, so a head width comes from slices
    while areas count regions.

    A peer answers per shard, one per (pp_rank, tp_rank), advertising the layer
    names it registered and the chiplet geometry of its regions. Each side
    derives its owned range from those names; neither sends it. A shard serving
    less than a whole engine gets its own descriptor lists; when nothing is
    narrowed, upstream's handle serves.
    """

    # Upstream owns this one, and `_regions_viewed_as` reorders it for the
    # duration of a peer-region view; declaring it keeps the reassignment from
    # leaving the type unresolvable here.
    block_len_per_layer: list[int]

    def _add_remote_agent_head_matched(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
    ) -> str:
        """Register a peer with a different TP degree by what each area holds.

        `registered_layer_names` narrows this to one pipeline stage's layers,
        `_fan_in_peer_areas` to the chiplet areas a finer peer owns.
        """
        engine_id = nixl_agent_meta.engine_id
        # Shards keep the flat rank because it collapses to the peer's TP rank
        # at pp_size 1, which is what upstream's int-keyed descriptor dicts get
        # from it; the agent map is keyed by the pair it decomposes into.
        agent_key = divmod(remote_tp_rank, remote_tp_size)
        if agent_key in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][agent_key]

        self._refuse_decode_context_parallel(remote_dcp_size)
        # Before the bands, which would refuse a latent cut unlike its attention
        # under a message about slice layouts.
        self._check_mla_constraints(nixl_agent_meta, remote_tp_size)
        areas_l, slices_l, _ = self._local_band(remote_tp_size)
        areas_r, slices_r = self._peer_band(nixl_agent_meta, remote_tp_size)
        # Ahead of the prelude, which the handshake checks run after: upstream
        # swallows a handshake exception, so a later raise leaves the peer
        # registered. The layout checks first, which the axes below index by.
        self._check_d2d_region_pairing(nixl_agent_meta, remote_tp_size)
        self._check_split_axis_constraints(nixl_agent_meta, remote_tp_size)
        head = ChipletSplitAxis.HEAD
        for logical_l, logical_r in self._logical_region_pairs(registered_layer_names):
            # A context cut's slices cut tokens, not heads.
            cuts_l = self.topo.tp_size * (
                slices_l if self._logical_region_geometry[logical_l].axis is head else 1
            )
            cuts_r = remote_tp_size * (
                slices_r if nixl_agent_meta.kv_split_axes[logical_r] is head else 1
            )
            if max(cuts_l, cuts_r) % min(cuts_l, cuts_r):
                raise RuntimeError(
                    f"RBLN NIXL D2D: for logical region {logical_l} this engine "
                    f"cuts KV heads {cuts_l} ways and the peer {cuts_r}, and "
                    "neither divides the other, so an area of one side would hold "
                    "part of one of the other's. The two sides must cut heads at "
                    "commensurate granularities."
                )
            # A peer rank has to hold whole areas of ours: its T / remote_tp
            # heads a multiple of an area's T / cuts_l. A coarser peer always
            # does, its TP dividing ours.
            if cuts_l % remote_tp_size:
                raise RuntimeError(
                    f"RBLN NIXL D2D: the peer runs TP {remote_tp_size}, which does "
                    f"not divide the {cuts_l} way(s) this engine cuts the KV "
                    f"heads of logical region {logical_l}, so one chiplet area "
                    "would straddle several peers. Run the peer at a "
                    f"tensor-parallel size dividing {cuts_l}, or use "
                    "kv_buffer_device='cpu'."
                )
            # A chunk range places a region's chunks by the token span its area
            # holds, and a head cut meeting the peer's context cut pairs in
            # pieces of the peer's spans instead.
            if (
                self._logical_region_geometry[logical_l].axis is head
                and nixl_agent_meta.kv_split_axes[logical_r] is ChipletSplitAxis.TOKEN
                and self._shape.cuts_a_block
            ):
                raise RuntimeError(
                    f"RBLN NIXL D2D: logical region {logical_l} is head-cut here "
                    f"and context-cut on the peer (TP {remote_tp_size}), so it "
                    "pairs in pieces of the peer's token ranges, which chunk_mode "
                    "cannot name. Turn chunk_mode off for this pairing, or pair "
                    "at equal TP."
                )
        self._register_remote_engine_prelude(
            nixl_agent_meta, remote_tp_size, remote_dcp_size
        )
        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        if engine_id not in self.dst_num_blocks:
            self._record_remote_region_ledger(engine_id, nixl_agent_meta)
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size
        )

        _, peer_tp_rank = agent_key
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
            areas_r,
            slices_r,
            self.topo.tp_size,
            areas_l,
            slices_l,
        )
        return remote_agent_name

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
        if not self.use_host_buffer or block_size != self.block_size:
            return None
        tp_ratio = self.topo.tp_ratio(remote_tp_size)
        plan = self.tp_mappings[engine_id]
        if not self._needs_split_local_xfer_handles(tp_ratio, plan):
            # Asking upstream rather than repeating its predicate: a replicated
            # MLA region is not split, so there is no handle to borrow and the
            # ratio alone does not say that.
            return None
        # vllm 0.30 keys that split by (ratio, the peer's block size), because
        # one ratio can now describe peers whose blocks differ. The guard above
        # already settled that this peer's block size is ours.
        handles = self.src_xfer_handles_by_tp_ratio[tp_ratio, block_size]
        assert region_ids == list(range(self.num_regions)), (
            "RBLN NIXL: borrowing upstream's split needs it to describe the "
            "same regions in the same order, but this peer narrows ours to "
            f"{region_ids} of {self.num_regions}"
        )
        assert len(handles) == len(plan.all_source_ranks), (
            f"RBLN NIXL: {len(handles)} handle(s) for "
            f"{len(plan.all_source_ranks)} source rank(s); the plan pairs "
            "them by position"
        )
        # A pipeline-parallel peer cannot reach here (rejected during the
        # handshake), so the rank we hold is the peer's TP rank as planned.
        return handles[plan.all_source_ranks.index(global_rank)]

    def _build_fa_remote(
        self,
        plan: TPMapping,
        nixl_agent_meta: NixlAgentMetadata,
        block_size_ratio: int,
    ) -> np.ndarray:
        # The one upstream loop that walks the peer's regions while reading ours.
        # Scoped to this call rather than to `add_remote_agent`, which also builds
        # local dlists from our own region ids and must not be translated.
        assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
        region_ids = self._peer_region_ids(nixl_agent_meta)
        with self._regions_viewed_as(region_ids):
            whole = super()._build_fa_remote(plan, nixl_agent_meta, block_size_ratio)
        # Upstream's builder is reached only by a peer this rank does not band
        # by head, so every region is one piece a block.
        grid = self._shard_chunk_grid(
            range(len(nixl_agent_meta.kv_caches_base_addr))
            if region_ids is None
            else region_ids,
            block_size=nixl_agent_meta.block_size,
        )
        if grid is None:
            return whole
        # The same addresses cut finer, appended after the whole-block range
        # rather than inside it, which is the order the local list takes --
        # a prepared transfer pairs the two by position. Upstream emits a
        # region's blocks consecutively, so a row's region is its quotient.
        num_blocks = nixl_agent_meta.num_blocks
        assert len(whole) == len(grid.regions) * num_blocks
        rows = whole.reshape(len(grid.regions), num_blocks, 3)
        return np.concatenate(
            [whole]
            + [
                _grid_descs(r[:, 0], r[:, 1], (runs, chunks), r[:, 2])
                for r, (_, runs, chunks) in zip(rows, grid.regions)
                if runs
            ]
        )

    def _head_matched_plan(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
    ) -> tuple[list[int], list[list[Piece]]]:
        """Our region ids for a peer of another TP degree, and per region the
        pieces it exchanges with that peer rank (`pair_area`).

        In LOCAL region order, so ``_compute_desc_ids``'s positional pairing
        keeps holding. Upstream instead walks the peer's list with one global
        ``rank_offset``, which assumes it holds our heads contiguously in a
        single region -- false once a region is one chiplet area and heads are
        area-major: P TP2 -> D TP4 reads local area 2 from remote area 1.

        ``registered_layer_names`` narrows this to one pipeline stage, its
        region list indexed by position within the stage; None means the peer
        owns every layer. ``peer_areas`` is `_fan_in_peer_areas`.
        """
        areas_l, _, _ = self._local_band(remote_tp_size)
        areas_iter = range(areas_l) if peer_areas is None else peer_areas
        region_ids: list[int] = []
        plans: list[list[Piece]] = []
        for logical_l, logical_r in self._logical_region_pairs(registered_layer_names):
            ours, peer = self._area_rects(
                nixl_agent_meta, logical_l, logical_r, remote_tp_rank, remote_tp_size
            )
            for area_l in areas_iter:
                region_id = logical_l * areas_l + area_l
                pieces = pair_area(
                    ours[area_l],
                    self.get_backend_aware_kv_block_len(
                        layer_idx=region_id, first_split=True, mamba_view=False
                    ),
                    peer,
                    kv_per_block=self._kv_per_block,
                    every_holder=self._writes_into_peer,
                    region_id=region_id,
                )
                # One chunk grid a region (`_shard_chunk_grid`), so its pieces
                # have to be cut alike: the handshake refuses incommensurate
                # head cuts, and each side cuts tokens into spans of one width.
                assert len({(x.length, x.rows) for x in pieces}) == 1, (
                    f"region {region_id} pairs in unlike pieces {pieces}"
                )
                region_ids.append(region_id)
                plans.append(pieces)
        return region_ids, plans

    def _area_rects(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        logical_l: int,
        logical_r: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[list[AreaRect], list[tuple[int, AreaRect, int]]]:
        """What each of our areas of one logical region holds, and per area of
        the peer's: its region id, what it holds, and its block length.

        A head cut gives a slice a band of the rank's heads and every token; a
        context cut gives it all of the rank's heads and a token range, which
        is `_slice_head_bounds` asked for one slice.
        """
        # A model-config count describes the target's layers, and on a draft's
        # it lands a plausible offset inside the wrong bytes. The peer's region
        # holds the same layer, so one count answers for both sides (the
        # handshake verifies it).
        total_heads = self._region_kv_heads(logical_l)
        # Ours too: a peer of another TP degree has our block size
        # (`_validate_head_matched_handshake`).
        block_tokens = nixl_agent_meta.block_size

        def rects(axis, slice_ids, rank, tp, side):
            slices = len(set(slice_ids))
            cut = 1 if axis is ChipletSplitAxis.TOKEN else slices
            first, per_cut = self._slice_head_bounds(
                rank, tp, total_heads, cut, side=side
            )
            if axis is ChipletSplitAxis.TOKEN:
                span = block_tokens // slices
                return [
                    AreaRect((first, first + per_cut), (sid * span, (sid + 1) * span))
                    for sid in slice_ids
                ]
            return [
                AreaRect(
                    (first + sid * per_cut, first + (sid + 1) * per_cut),
                    (0, block_tokens),
                )
                for sid in slice_ids
            ]

        geometry = self._logical_region_geometry[logical_l]
        ours = rects(
            geometry.axis,
            [place.slice_id for place in geometry.areas],
            self.tp_rank,
            self.topo.tp_size,
            "local",
        )
        ids_r = nixl_agent_meta.kv_area_slices[logical_r]
        theirs = rects(
            nixl_agent_meta.kv_split_axes[logical_r],
            ids_r,
            remote_tp_rank,
            remote_tp_size,
            "peer",
        )
        first_r = logical_r * len(ids_r)
        peer = [
            (first_r + a, rect, nixl_agent_meta.block_lens[first_r + a])
            for a, rect in enumerate(theirs)
        ]
        return ours, peer

    def _build_head_matched_remote(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
    ) -> np.ndarray:
        """Remote descriptors for a peer whose TP degree differs from ours.

        The plan is `_head_matched_plan`'s, and so is the order. The chunk
        range follows the whole-piece one where the geometry allows it, asked
        for here rather than passed in so that it is cut from the same plan.
        """
        region_ids, plans = self._head_matched_plan(
            nixl_agent_meta,
            remote_tp_rank,
            remote_tp_size,
            registered_layer_names=registered_layer_names,
            peer_areas=peer_areas,
        )
        chunk_grid = self._shard_chunk_grid(
            region_ids,
            block_size=nixl_agent_meta.block_size,
            rows=[plan[0].rows for plan in plans],
        )
        remote_bases = np.asarray(nixl_agent_meta.kv_caches_base_addr, dtype=np.uint64)
        remote_lens = np.asarray(nixl_agent_meta.block_lens, dtype=np.uint64)
        blocks = np.arange(nixl_agent_meta.num_blocks, dtype=np.uint64)[:, None]
        # Every region for one grid before the next, which is the order
        # `_register_shard_local_xfer_handler` builds its list in; within a
        # region, block-major then piece, as `_get_block_descs_ids_for_shard`
        # indexes it.
        passes = [[(1, 1)] * len(region_ids)]
        if chunk_grid is not None:
            passes.append([(runs, chunks) for _, runs, chunks in chunk_grid.regions])
        parts: list[np.ndarray] = []
        for grids in passes:
            for plan, grid in zip(plans, grids):
                if not grid[0]:
                    continue
                p = np.asarray(
                    [(x.remote_region, x.remote_offset, x.length) for x in plan],
                    dtype=np.uint64,
                )
                region = p[:, 0].astype(np.intp)
                starts = (
                    remote_bases[region] + blocks * remote_lens[region] + p[:, 1]
                ).ravel()
                lengths = np.broadcast_to(p[:, 2], (len(blocks), len(plan))).ravel()
                parts.append(
                    _grid_descs(starts, lengths, grid, nixl_agent_meta.device_id)
                )
        return np.concatenate(parts)

    def _check_d2d_region_pairing(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject D2D peers whose region lists cannot be paired.

        D2D publishes one region per chiplet area, so region i is the same head
        band only at equal TP; unequal TP pairs by head range instead
        (``_add_remote_agent_head_matched``), which a sliding window is not
        supported with. The other refusal divides the peer's region list by its
        layer count, so it has to run on what the peer published: a stage wider
        than our band reaches ``add_remote_agent`` already sliced to our own
        regions per layer, which would make that division an identity.

        Host-bounce has no per-area list, so none of it applies.
        """
        if self.use_host_buffer:
            return
        tp_ratio = self.topo.tp_ratio(remote_tp_size)
        if tp_ratio != 1 and self._shape.has_swa:
            raise RuntimeError(
                "RBLN NIXL D2D: sliding-window attention is not supported with "
                f"heterogeneous tensor parallelism (tp_ratio={tp_ratio})."
            )
        peer_kv_per_block = nixl_agent_meta.kv_per_block
        if peer_kv_per_block != self._kv_per_block:
            # Refused here rather than as a region-count mismatch further in:
            # that one names the count, not the cause an operator can act on.
            raise RuntimeError(
                f"RBLN NIXL D2D: a peer block holds {peer_kv_per_block} of K "
                f"and V and this worker's holds {self._kv_per_block}. The "
                "attention cache is packed for the rbln_custom_ops kernels and "
                "split for rbln_triton_ops, so both ends of a transfer must "
                "run under the same use_custom_kernel setting."
            )
        peer_swa_block = nixl_agent_meta.swa_kernel_block
        # Only a window range is cut by this number. Without one the lists are
        # whole blocks of the full-attention view, which is the same shape
        # under either geometry, so the pair is describable and stays allowed.
        if (
            self._window_grid_cut is not None
            and peer_swa_block
            and peer_swa_block not in self._swa_kernel_blocks
        ):
            raise RuntimeError(
                f"RBLN NIXL D2D: the peer's sliding-window kernel addresses "
                f"its cache in {peer_swa_block}-token blocks and this "
                f"worker's in {sorted(self._swa_kernel_blocks)}, so the two "
                "window ranges name different pieces of the same block. Both "
                "ends have to run under the same sub-block prefix caching "
                "setting."
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

    def _check_mla_constraints(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject the MLA topologies whose descriptor math is not established.

        MLA is REPLICATE and key-only, which upstream's positional pairing
        already expresses. What the first refusal below adds is head-band matching,
        where bands computed from the configured KV-head count give plausible
        wrong descriptors. A latent beside attention is the same on every rank,
        so on D2D a band misplaces it. Host staging does not record which
        regions are latents; it refuses an engine whose page sizes differ.
        """
        if self.use_mla and self._is_head_matched_peer(remote_tp_size):
            raise RuntimeError(
                "RBLN NIXL D2D: MLA is not supported with heterogeneous tensor "
                f"parallelism (peer TP {remote_tp_size}, local "
                f"{self.topo.tp_size})."
            )
        if (
            not self.use_mla
            and any(self._logical_region_latent)
            and self.topo.tp_ratio(remote_tp_size) != 1
        ):
            raise RuntimeError(
                "RBLN NIXL D2D: a key-only latent region (an MSA indexer or an "
                "MLA draft beside an attention target) is the "
                "same on every rank, and pairing it across tensor-parallel "
                f"degrees (peer TP {remote_tp_size}, local {self.topo.tp_size}) "
                "is not supported. Use equal TP on both sides."
            )

    def _check_split_axis_constraints(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject peers whose chiplet areas do not mean what the pairing assumes.

        Equal TP pairs area k with the peer's area k, so both sides have to lay
        each region's slices over their areas alike. Unequal TP pairs areas by
        what they hold (`pair_area`), so the axes may differ -- a head cut on
        one side, a context cut on the other -- but a context cut on both sides
        is a pairing this code has not opened.

        Paired by layer name, since a pipeline stage advertises its own layers
        only. Host-bounce advertises one area a region and is not compared.
        """
        if self.use_host_buffer:
            return
        head_matched = self._is_head_matched_peer(remote_tp_size)

        def cut(axes: list[ChipletSplitAxis], slice_ids: list[list[int]]) -> list:
            if head_matched:
                return [axis.name for axis in axes]
            return [(axis.name, ids) for axis, ids in zip(axes, slice_ids)]

        ours = cut(
            [g.axis for g in self._logical_region_geometry],
            [[p.slice_id for p in g.areas] for g in self._logical_region_geometry],
        )
        theirs = cut(nixl_agent_meta.kv_split_axes, nixl_agent_meta.kv_area_slices)
        names = nixl_agent_meta.registered_layer_names
        # A peer that names no layer is compared as one list, so the index the
        # mismatch is reported at counts regions rather than a layer's share.
        compared = [("the engine", theirs, ours)]
        if names:
            per_layer = self._logical_regions_per_layer()
            if len(theirs) != per_layer * len(names):
                raise RuntimeError(
                    f"RBLN NIXL D2D: peer advertises {len(theirs)} chiplet "
                    f"cut axes over {len(names)} layer(s), and this worker cuts "
                    f"{per_layer} logical region(s) a layer."
                )
            compared = [
                (
                    f"layer {names[peer_pos]}",
                    theirs[peer_pos * per_layer : (peer_pos + 1) * per_layer],
                    ours[local_idx * per_layer : (local_idx + 1) * per_layer],
                )
                for peer_pos, local_idx in self._layer_overlap(names)
            ]
        for what, peer_cut, our_cut in compared:
            if head_matched and len(peer_cut) == len(our_cut):
                for k, (peer_axis, our_axis) in enumerate(zip(peer_cut, our_cut)):
                    if peer_axis == our_axis == ChipletSplitAxis.TOKEN.name:
                        raise RuntimeError(
                            f"RBLN NIXL D2D: logical region {k} of {what} is "
                            "context-cut on both sides, and pairing two context "
                            "cuts across tensor-parallel degrees (peer TP "
                            f"{remote_tp_size}, local {self.topo.tp_size}) is not "
                            "supported. Use equal TP, or a TP on one side whose "
                            "share of KV heads keeps the head cut."
                        )
                continue
            if peer_cut == our_cut:
                continue
            k = next(
                (i for i, (p, o) in enumerate(zip(peer_cut, our_cut)) if p != o),
                min(len(peer_cut), len(our_cut)),
            )
            raise RuntimeError(
                f"RBLN NIXL D2D: peer cut logical region {k} of {what} of its KV "
                f"cache as {peer_cut[k : k + 1]} but this worker cut it as "
                f"{our_cut[k : k + 1]} ({len(peer_cut)} region(s) against "
                f"{len(our_cut)}). The area counts can agree while an area "
                "means something else on each side, so no pairing is "
                "meaningful."
            )
        # Both ranges that name less than a block cut it by a number derived
        # from OUR block size -- the chunk grid from it, the window range from
        # `_window_grid` -- and a streamed write places its blocks by counting
        # them on both sides. A peer whose block holds a different count is
        # then cut or counted in pieces that are not its own.
        if (self._shape.writes_part_of_a_request or self._own_engine_layout) and (
            nixl_agent_meta.block_size != self.block_size
        ):
            raise RuntimeError(
                "RBLN NIXL D2D: naming less than a whole block or request needs "
                "both sides to count a block the same way, but the peer's block "
                f"holds {nixl_agent_meta.block_size} tokens and this worker's holds "
                f"{self.block_size}."
            )

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
        for skey in [k for k in self._shard_chunk_grids if k[0] == engine_id]:
            del self._shard_chunk_grids[skey]
            del self._shard_region_spans[skey]
        self._remote_shard_layer_names.pop(engine_id, None)
        self._overlapping_ranks.pop(engine_id, None)
        self._remote_pp_size.pop(engine_id, None)
        super()._cleanup_remote_engine(engine_id, log_eviction=log_eviction)

    def _handle_dead_engine(
        self, engine_id: str, *, error: Exception | None = None
    ) -> None:
        """Fail this peer's in-flight reads and drop its handshake state.

        Both halves are needed once a connection is gone: the reads against it
        will never complete and are the only thing that can end the scheduler's
        wait, and the `_remote_agents` entry left behind makes `_ensure_handshake`
        skip the reconnect for the life of the process.

        In-flight handles are released and dropped rather than left to
        `_pop_done_transfers`, which would report the request a second time with
        its metadata already gone.
        """
        # Snapshotted: the handshake done-callback drops entries from the
        # executor thread, and reading the live view would raise here.
        for req_id, meta in list(self._recving_metadata.items()):
            if meta.remote is None or meta.remote.engine_id != engine_id:
                continue
            self._log_failure(
                failure_type="peer_unreachable",
                req_id=req_id,
                error=error,
                dst_engine_id=engine_id,
            )
            for handle in self._recving_transfers.pop(req_id, []):
                self.nixl_wrapper.release_xfer_handle(handle)
            self._handle_failed_transfer(req_id, None, self._recv_failures)

        try:
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        except Exception:
            # The teardown talks to the agent it is dropping, so a dead
            # connection can fail it halfway. The entry has to go regardless:
            # keeping it is what blocks the re-handshake.
            logger.warning(
                "Dropping the state of unreachable engine %s failed",
                engine_id,
                exc_info=True,
            )
            self._remote_agents.pop(engine_id, None)
        self._engines_to_rehandshake.discard(engine_id)

    def _send_heartbeats(self, metadata: "NixlConnectorMetadata") -> None:
        """Upstream's heartbeat, treating a send failure as the peer being gone.

        Upstream swallows it with a `logger.debug`, so a connection that died
        after the handshake is never noticed here -- which is the one place a
        producer that has stopped answering is touched every step.

        Mirrors `NixlBaseConnectorWorker._send_heartbeats` as of vllm 0.30.0,
        differing only in the `except` body.
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the **next** heartbeat for this remote can go through.
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    hb_info.dcp_size,
                    hb_info.pp_size,
                    self._hb_handshake_notif_only and hb_info.pp_size > 1,
                )
                is not None
            ):
                continue  # handshake is still pending

            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception as e:
                    logger.warning(
                        "Heartbeat to engine %s failed; treating it as gone",
                        engine_id,
                        exc_info=True,
                    )
                    # The remaining agents of a dead engine are gone with it,
                    # and the teardown drops them all.
                    self._handle_dead_engine(engine_id, error=e)
                    break

    def _ensure_handshake(
        self, engine_id: str, *args: Any, **kwargs: Any
    ) -> Future | None:
        # A dial from a dead local link leaves nixl a half-loaded remote that
        # fails every later handshake, so nothing is dialled until it is back.
        if self._link_down_since is None:
            return super()._ensure_handshake(engine_id, *args, **kwargs)
        # Counted where upstream would have dialled, short of its TTL sweep,
        # whose NIXL teardown this branch keeps off a dead link. Upstream counts
        # a failed dial there and drains the requests waiting on it uncounted.
        with self._handshake_lock:
            if (
                engine_id not in self._remote_agents
                and engine_id not in self._handshake_futures
            ):
                self.xfer_stats.record_failed_handshake()
        failed: Future = Future()
        failed.set_exception(RuntimeError("every local RDMA link is down"))
        return failed

    def get_transfer_results(self) -> KVConnectorTransferResults:
        now = time.monotonic()
        if not every_local_link_down():
            self._link_down_since = None
        elif self._link_down_since is None:
            self._link_down_since = now
            logger.warning("Every local RDMA link is down; KV transfers fail.")
        elif 0 < self._link_down_exit_s <= now - self._link_down_since:
            raise RuntimeError(
                f"Every local RDMA link down for {self._link_down_exit_s}s; "
                "exiting so this KV producer is recycled."
            )
        results = super().get_transfer_results()
        # A dead peer stays known -- every read refreshes its TTL on the way to
        # failing -- so upstream never re-dials it. Drop it once no handle of
        # its is in flight; a handle whose request is gone has no known owner.
        if self._engines_to_rehandshake:
            busy = {
                meta.remote.engine_id
                if (meta := self._recving_metadata.get(req_id)) and meta.remote
                else None
                for req_id, handles in self._recving_transfers.items()
                if handles
            }
            if None not in busy:
                for engine_id in self._engines_to_rehandshake - busy:
                    if engine_id in self._remote_agents:
                        logger.warning(
                            "Re-handshaking unreachable engine %s.", engine_id
                        )
                        self._handle_dead_engine(engine_id)
                self._engines_to_rehandshake &= busy
        return results

    def _fan_in_peer_areas(
        self, remote_tp_rank: int, remote_tp_size: int
    ) -> list[int] | None:
        """Which local chiplet areas live on this peer, when it has MORE TP.

        ``None`` when the peer holds all of them, so callers need no branch.

        Our band spreads over ``|tp_ratio|`` of its ranks, so a transfer to one
        must carry only the areas whose heads that rank owns -- otherwise every
        peer yields the same bytes and the last to land wins.
        """
        if self.use_host_buffer:
            # Host staging registers one logical full-shape buffer per layer,
            # so there are no chiplet areas to divide and nothing below applies.
            return None
        if self.topo.tp_ratio(remote_tp_size) > 0:
            return None
        # Model-level is right for every region here: both the area set and
        # the guard below reduce to cut-count ratios.
        total_heads = self.topo.total_num_kv_heads
        areas_l, slices_l, slice_ids_l = self._local_band(remote_tp_size)
        base_l, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.topo.tp_size,
            total_heads,
            slices_l,
            side="local",
        )
        heads_per_remote = total_heads // remote_tp_size
        assert heads_per_remote % per_slice_l == 0, (
            f"a peer rank's {heads_per_remote} heads split an area of "
            f"{per_slice_l}; the handshake refuses that before registering"
        )
        return [
            area_l
            for area_l in range(areas_l)
            if (base_l + slice_ids_l[area_l] * per_slice_l) // heads_per_remote
            == remote_tp_rank
        ]

    def _is_fan_in_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer has MORE TP ranks than us, so we gather from it
        together with its siblings."""
        if self.use_host_buffer:
            return False
        return self.topo.tp_ratio(remote_tp_size) < 0

    def _is_head_matched_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer is served by ``_build_head_matched_remote``.

        Any unequal TP degree, in either direction, on D2D where this engine
        does not own the whole-engine lists.
        """
        if self.use_host_buffer or self._own_engine_layout:
            return False
        return self.topo.tp_ratio(remote_tp_size) != 1

    def _logical_region_pairs(
        self, registered_layer_names: tuple[str, ...] | list[str] | None
    ) -> list[tuple[int, int]]:
        """(our logical region, its position in the peer's region list).

        A logical region is one K or V of one layer, before chiplet expansion.
        None means the peer owns every layer, so the two lists coincide;
        otherwise a peer position indexes ITS OWN list (see `_layer_overlap`).
        """
        if registered_layer_names is None:
            return [(i, i) for i in range(len(self._logical_region_geometry))]
        per_layer = self._logical_regions_per_layer()
        return [
            (layer_l * per_layer + c, peer_pos * per_layer + c)
            for peer_pos, layer_l in self._layer_overlap(registered_layer_names)
            for c in range(per_layer)
        ]

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dcp_size: int = 1,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> tuple[dict[tuple[int, int], str], float]:
        """Handshake with every shard of one peer engine.

        Runs off the main thread -- on upstream's handshake executor, or inline
        on the writer thread when a producer opens one -- and neither publishes
        anything of ours: both take the agents out of the return and assign them
        to ``_remote_agents[engine_id]`` under the handshake lock. A peer is
        reached only once that key exists, which is what makes the per-shard
        state written here visible. So every write has to land BEFORE the return
        -- state published after it would be read half-built.

        Upstream takes the peer's pipeline size from its caller and never
        asks, and no caller derives it: it is relayed from the producer
        through kv_transfer_params, while the read handshake and a producer's
        own reverse handshake both pass nothing and take the default of 1.
        Asking the peer is why this override exists, and it can go once a read
        path caller carries the peer's pp_size rather than defaulting it.
        """
        # Background thread needs a device context (see upstream _nixl_handshake).
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        p_remote_tp_ranks = self.topo.handshake_target_ranks(
            remote_tp_size, remote_dcp_size
        )
        path = make_zmq_path("tcp", host, port)
        remote_rank_to_agent_name: dict[tuple[int, int], str] = {}
        overlapping: list[int] = []

        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # ms; avoid hang on dead server

            # Bootstrap: the first shard (pp_rank 0) advertises pp_size.
            first_rank = p_remote_tp_ranks[0]
            first_meta, best_rtt, best_offset = self._query_agent_meta(
                sock, 0, first_rank, expected_engine_id
            )
            metas = {(0, first_rank): first_meta}
            pp_size = first_meta.pp_size
            if remote_pp_size > 1 and remote_pp_size != pp_size:
                raise RuntimeError(
                    f"RBLN NIXL: caller expects peer {expected_engine_id} to run "
                    f"pipeline size {remote_pp_size}, peer reports {pp_size}."
                )

            # Guard on either side's pipeline, not just the peer's: the peer
            # runs none in the reverse shape, where ours is the finer one.
            local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
            if pp_size > 1 or local_pp > 1:
                if self._shape.has_swa:
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
                tp_ratio = self.topo.tp_ratio(remote_tp_size)
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
                if notif_agents_only and pp_size > 1 and local_pp > 1:
                    # Upstream waits for pp_size * producers_per_consumer
                    # completions, a count with no term for our own stages, and
                    # refuses the shape itself only under kv_role=kv_consumer,
                    # which kv_both does not reach. A single producer stage
                    # writes every rank of ours, so that one is exact.
                    raise RuntimeError(
                        "RBLN NIXL: a pipelined producer pushing into a "
                        f"pipelined consumer (peer pp={pp_size}, local "
                        f"pp={local_pp}) is not supported; the completion count "
                        "the peer waits for cannot express this side's stages."
                    )
                if tp_ratio < 0 and pp_size > 1:
                    # The peer splits layers AND holds our heads across
                    # several of its ranks; host staging then borrows upstream's
                    # split (_base_fan_in_handle), which asserts a full region
                    # list a stage does not have.
                    raise RuntimeError(
                        "RBLN NIXL: a pipeline-parallel peer with a larger "
                        f"tensor-parallel size (peer {remote_tp_size} > local "
                        f"{self.topo.tp_size}) is not supported."
                    )

            for pp_rank in range(pp_size):
                for remote_tp_rank in p_remote_tp_ranks:
                    global_rank = pp_rank * remote_tp_size + remote_tp_rank
                    if (pp_rank, remote_tp_rank) in metas:
                        metadata = metas[(pp_rank, remote_tp_rank)]
                    else:
                        metadata, rtt, offset = self._query_agent_meta(
                            sock, pp_rank, remote_tp_rank, expected_engine_id
                        )
                        if rtt < best_rtt:
                            best_rtt, best_offset = rtt, offset
                    if notif_agents_only:
                        # The peer's memory is never addressed from here, so
                        # no descriptors are built for it.
                        remote_rank_to_agent_name[(pp_rank, remote_tp_rank)] = (
                            self._add_notif_only_remote_agent(
                                metadata, remote_tp_size, metadata.dcp_size
                            )
                        )
                        continue
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
                    fan_in = self._is_fan_in_peer(remote_tp_size)

                    head_matched = self._is_head_matched_peer(remote_tp_size)
                    if head_matched:
                        # Different TP degrees: pair by head range, over the
                        # layers we share.
                        remote_rank_to_agent_name[(pp_rank, remote_tp_rank)] = (
                            self._add_remote_agent_head_matched(
                                metadata,
                                global_rank,
                                remote_tp_size,
                                metadata.dcp_size,
                                registered_layer_names=names,
                            )
                        )
                    else:
                        # Equal TP delegates to upstream, which needs a wider
                        # stage trimmed to our band (_trim_agent_meta_to_layers).
                        # The pairing check divides the peer's regions by its
                        # layers and the trim makes that quotient our own, so it
                        # runs first, on what the peer published.
                        self._check_d2d_region_pairing(metadata, remote_tp_size)
                        agent = self.add_remote_agent(
                            self._trim_agent_meta_to_layers(metadata, overlap)
                            if partial
                            else metadata,
                            global_rank,
                            remote_tp_size,
                            metadata.dcp_size,
                        )
                        remote_rank_to_agent_name[(pp_rank, remote_tp_rank)] = agent

                    # After the agent: registering it is what validates the
                    # peer geometry the plan reads.
                    peer_areas = self._fan_in_peer_areas(remote_tp_rank, remote_tp_size)
                    plan = (
                        self._head_matched_plan(
                            metadata,
                            remote_tp_rank,
                            remote_tp_size,
                            registered_layer_names=names,
                            peer_areas=peer_areas,
                        )
                        if head_matched
                        else None
                    )
                    if not self._needs_own_descriptors(
                        pp_size=pp_size,
                        partial=partial,
                        fan_in=fan_in,
                        pieces=plan is not None
                        and any(len(pieces) > 1 for pieces in plan[1]),
                    ):
                        continue
                    self._register_shard_xfer_state(
                        expected_engine_id,
                        global_rank,
                        metadata.block_size,
                        names,
                        peer_areas=peer_areas,
                        plan=plan,
                        remote_tp_size=remote_tp_size,
                    )
                    overlapping.append(global_rank)
        # Published once, not accumulated: a handshake that raises partway
        # leaves no entry, so the retry that follows starts from empty instead
        # of appending its shards a second time and reading every block twice.
        self._overlapping_ranks[expected_engine_id] = overlapping
        self._remote_pp_size[expected_engine_id] = pp_size
        return remote_rank_to_agent_name, best_offset

    def _peer_region_ids(
        self, nixl_agent_meta: RblnNixlAgentMetadata
    ) -> list[int] | None:
        """Our region ids for a peer's regions, in the peer's order, or None.

        None where there is nothing to translate: a peer that advertises no layer
        names (nothing to match on), or one whose regions already line up with
        ours one for one from position 0. Refused where the two lists cannot
        describe the same regions, which upstream's positional pairing assumes.
        """
        names = nixl_agent_meta.registered_layer_names
        if not names or not self.local_seen_layer_names:
            return None
        region_ids = self._shard_local_region_ids(names)
        n_peer = len(nixl_agent_meta.kv_caches_base_addr)
        if len(region_ids) != n_peer:
            raise RuntimeError(
                f"RBLN NIXL: this rank owns {len(region_ids)} region(s) of the "
                f"{n_peer} the peer publishes over {len(names)} layer(s); "
                "upstream pairs remote region i with local region i, so the two "
                "lists have to describe the same regions. Regions per layer is "
                f"{self._regions_per_layer()} here."
            )
        if region_ids == list(range(n_peer)):
            return None
        return region_ids

    def _query_agent_meta(
        self,
        sock: "zmq.Socket",
        remote_pp_rank: int,
        remote_tp_rank: int,
        expected_engine_id: str,
    ) -> tuple[RblnNixlAgentMetadata, float, float]:
        """Returns the peer's metadata, the round trip time, and the clock
        offset that trip estimates. The counter-party is upstream's
        ``_nixl_handshake_listener``; the query arity and frame count are its.
        """
        start_time = time.perf_counter()
        sock.send(
            msgspec.msgpack.encode((GET_META_MSG, remote_pp_rank, remote_tp_rank))
        )
        payload_bytes, perf_bytes = sock.recv_multipart()
        recv_time = time.perf_counter()
        if not payload_bytes:
            raise RuntimeError(
                f"engine {expected_engine_id} reports every RDMA link down"
            )
        try:
            handshake_payload = msgspec.msgpack.Decoder(NixlHandshakePayload).decode(
                payload_bytes
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
        # Ahead of the geometry checks, so the refusal names the cause.
        peer_mem_types = set(metadata.region_mem_types or [self.nixl_memory_type])
        if peer_mem_types != {self.nixl_memory_type}:
            peer_devices = "/".join(
                sorted({"rbln" if mem == "VRAM" else "cpu" for mem in peer_mem_types})
            )
            raise RuntimeError(
                f"RBLN NIXL: the peer runs kv_buffer_device={peer_devices!r} and "
                f"this worker {self.kv_buffer_device!r} (it registered "
                f"{sorted(peer_mem_types)} against our {self.nixl_memory_type!r}). "
                "P and D must use the same one: 'rbln' registers device memory "
                "per chiplet area and 'cpu' a host staging buffer per layer, and "
                "the two region lists cannot be paired."
            )
        # Refused on arrival, not where it is first divided by: a region holds
        # whole slices, and is one axis over the same number of areas as every
        # other, its areas each a region of the list.
        geometry = metadata.kv_area_slices
        for r, ids in enumerate(geometry):
            slices = len(set(ids))
            if not ids or any(
                ids.count(j) != len(ids) // slices for j in range(slices)
            ):
                raise RuntimeError(
                    f"RBLN NIXL: the peer's logical region {r} reports its "
                    f"chiplet areas holding slices {ids}; a region is cut into "
                    "slices 0..n-1, each held by the same number of areas."
                )
        if (
            len(metadata.kv_split_axes) != len(geometry)
            or len({len(ids) for ids in geometry}) != 1
            or sum(map(len, geometry)) != len(metadata.kv_caches_base_addr)
        ):
            raise RuntimeError(
                f"RBLN NIXL: the peer advertises {len(metadata.kv_split_axes)} "
                f"cut axes, chiplet areas {[len(ids) for ids in geometry]} per "
                f"logical region and {len(metadata.kv_caches_base_addr)} "
                "regions; each logical region is one axis over the same number "
                "of areas, one region an area."
            )
        remote_perf = msgspec.msgpack.decode(perf_bytes)
        return (
            metadata,
            recv_time - start_time,
            remote_perf - (start_time + recv_time) / 2,
        )

    def _region_kv_heads(self, logical_region: int) -> int:
        """Model-wide KV heads of the layer one logical region belongs to."""
        heads = self._logical_region_kv_heads[logical_region]
        assert heads is not None, (
            f"logical region {logical_region} has no head band: its layer either "
            "has no head axis at all, or holds fewer KV heads than this rank's TP "
            "degree and has them replicated across ranks. Pairing by head range "
            "cannot describe either."
        )
        return heads

    @contextmanager
    def _regions_viewed_as(self, region_ids: list[int] | None):
        """Make our per-region arrays answer to a peer's region positions.

        Upstream's remote descriptor builder feeds the PEER's region position
        into `_is_region_replicated` and, since 0.30.0, straight into
        `block_len_per_layer` -- both of which index OUR arrays. That only means
        our region while our band starts at our region 0 -- true for a pipeline
        stage, false for a consumer holding every layer while the producer is
        pipelined, where **the length read belongs to a different layer than the
        address it is paired with**. So the lengths are reordered for the
        duration too, not only the positions the replication check reads.

        A plain attribute suffices: upstream reaches those two methods only from
        registration and from a handshake, handshakes run one at a time on a
        single-worker executor, and registration finishes during worker warm-up,
        before a request exists to start one. So no second view is ever live.
        """
        prev = self._viewed_region_ids
        prev_lens = self.block_len_per_layer
        self._viewed_region_ids = region_ids
        if region_ids is not None:
            self.block_len_per_layer = [prev_lens[r] for r in region_ids]
        try:
            yield
        finally:
            self._viewed_region_ids = prev
            self.block_len_per_layer = prev_lens

    def _register_remote_engine_prelude(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ) -> None:
        """Replicate upstream ``add_remote_agent``'s prelude.

        Upstream registers the remote engine in the TransferTopology and builds
        its TPMapping before any block_size_ratio / tp_ratio / get_engine_info
        lookup, which its callers and ``_validate_remote_agent_handshake``
        also make. A path that does not delegate to super() has to do this
        itself or get_engine_info() raises KeyError.
        """
        self.topo.register_remote_engine(
            nixl_agent_meta.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=nixl_agent_meta.block_size,
                remote_block_len=nixl_agent_meta.block_lens[0],
                remote_physical_blocks_per_logical=(
                    nixl_agent_meta.physical_blocks_per_logical_kv_block
                ),
                remote_dcp_size=remote_dcp_size,
            ),
        )
        self.tp_mappings[nixl_agent_meta.engine_id] = compute_tp_mapping(
            transfer_topology=self.topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
            remote_dcp_size=remote_dcp_size,
        )

    def _needs_own_descriptors(
        self,
        *,
        pp_size: int,
        partial: bool,
        fan_in: bool,
        pieces: bool,
    ) -> bool:
        """Whether this peer needs descriptors of ours rather than upstream's.

        Upstream registers one handle per engine, and its notification names a
        whole request. That describes a peer only while nothing about the
        transfer is narrower than the engine, so each way one can be is a
        reason here: a pipelined producer holds part of the layers, a partial
        overlap part of the names, a fan-in peer part of the heads, and
        `pieces` -- a region exchanged in more than one piece a block -- part
        of a region. That counts a peer replica and a K/V half as well as a
        head piece: upstream's handle names a block once, where our list names
        it once a piece.
        """
        # Trimming narrows a block rather than a peer, and the per-shard ids
        # are what can leave part of one out. Streaming narrows neither: its
        # own descriptors are what give a notification room to say which blocks
        # a write filled. Both give way to a sliding window, whose view leaves
        # the whole-engine list the only place such a range can sit.
        return (
            pp_size > 1
            or partial
            or fan_in
            or pieces
            or (self._shape.writes_part_of_a_request and not self._own_engine_layout)
        )

    def _register_shard_xfer_state(
        self,
        engine_id: str,
        global_rank: int,
        block_size: int,
        registered_layer_names: tuple[str, ...],
        peer_areas: list[int] | None = None,
        plan: tuple[list[int], list[list[Piece]]] | None = None,
        remote_tp_size: int = 1,
    ) -> None:
        """Register one peer shard's local list and what indexes into it.

        `plan` is `_head_matched_plan`'s for a peer of another TP degree; None
        names every region whole.
        """
        # Compute the local region ids once and reuse them for the handler
        # (SWA never reaches a pipelined peer: it is rejected earlier).
        if plan is None:
            region_ids = self._shard_local_region_ids(
                registered_layer_names, peer_areas=peer_areas
            )
            pieces = None
        else:
            region_ids, pieces = plan
        chunk_grid = self._shard_chunk_grid(
            region_ids,
            block_size=block_size,
            rows=None if pieces is None else [p[0].rows for p in pieces],
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
                region_ids=region_ids,
                pieces=pieces,
                chunk_grid=chunk_grid,
            )
        self.src_xfer_handles_by_remote[key] = handle
        n_groups = len(self.kv_cache_config.transfer_groups)
        assert n_groups == 1, (
            "RBLN NIXL per-shard transfers support a single KV-cache group, "
            f"got {n_groups}"
        )
        self._shard_region_group_ids[(engine_id, global_rank)] = (0,) * len(region_ids)
        self._shard_descs_per_block[(engine_id, global_rank)] = (
            (1,) * len(region_ids) if pieces is None else tuple(map(len, pieces))
        )
        self._shard_chunk_grids[(engine_id, global_rank)] = chunk_grid
        self._shard_region_spans[(engine_id, global_rank)] = tuple(
            map(self._region_span_index, region_ids)
        )

    def _local_band(self, remote_tp_size: int) -> tuple[int, int, tuple[int, ...]]:
        """(areas, slices, the slice each area holds) every head-banded region
        of this engine shares.

        The areas a finer peer holds are one filter for every region
        (`_fan_in_peer_areas`), so pairing by head band needs the banded regions
        to lay their slices alike; a speculative draft cut into another number
        of slices does not. Per peer, not at registration: symmetric TP does not
        band by head at all. Regions without a band are skipped.
        """
        bands = {
            tuple(place.slice_id for place in geometry.areas)
            for geometry, heads in zip(
                self._logical_region_geometry, self._logical_region_kv_heads
            )
            if heads is not None
        }
        if not bands:
            # Nothing bands by head (an MLA latent, refused by name at the entry,
            # or heads every rank holds); let the regions' own layout stand in.
            bands = {
                tuple(place.slice_id for place in geometry.areas)
                for geometry in self._logical_region_geometry
            }
        if len(bands) != 1:
            raise RuntimeError(
                "RBLN NIXL D2D: this engine's KV cache entries lay their slices "
                f"over the chiplet areas as {sorted(bands)}, so no one geometry "
                "bands every region. Head-band pairing with a peer at TP "
                f"{remote_tp_size} needs one."
            )
        (slice_ids,) = bands
        return len(slice_ids), len(set(slice_ids)), slice_ids

    @staticmethod
    def _peer_band(
        nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> tuple[int, int]:
        """(areas, slices) every region of a head-banded peer shares; see
        `_local_band`."""
        bands = {(len(ids), len(set(ids))) for ids in nixl_agent_meta.kv_area_slices}
        if len(bands) != 1:
            raise RuntimeError(
                "RBLN NIXL D2D: the peer's KV cache entries are cut into "
                f"(areas, slices) {sorted(bands)}, so no one geometry bands every "
                f"region. Head-band pairing with a peer at TP {remote_tp_size} "
                "needs one."
            )
        (band,) = bands
        return band

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
            "block_strides": nixl_agent_meta.block_strides[lo:hi],
        }
        # One entry a logical region rather than an area, so the same layers
        # are a shorter window of them.
        per_layer = len(nixl_agent_meta.kv_area_slices) // len(
            nixl_agent_meta.registered_layer_names
        )
        for name in ("kv_split_axes", "kv_area_slices"):
            trimmed[name] = getattr(nixl_agent_meta, name)[
                start * per_layer : end * per_layer
            ]
        # The rest arrived with 0.30.0 and are region-indexed the same way, so
        # they follow the window. Optional on the wire: a peer that sends none
        # keeps sending none rather than an empty list upstream would index.
        for field in (
            "region_group_ids",
            "region_names",
            "region_mem_types",
            "region_num_blocks",
        ):
            value = getattr(nixl_agent_meta, field)
            if value is not None:
                trimmed[field] = value[lo:hi]
        return replace(nixl_agent_meta, **trimmed)

    def _validate_head_matched_handshake(
        self,
        nixl_agent_meta: RblnNixlAgentMetadata,
        remote_tp_size: int,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """The byte invariant a peer with a different TP degree has to meet.

        Upstream scales a region by heads per RANK, which holds only for its
        one-region-per-layer model; after chiplet expansion a region is one area,
        so the per-area ratio governs. **The two agree at P TP1 -> D TP2 and
        diverge past it**, so the simplest asymmetric pair does not exercise this.

        Region 0 is always a target layer, so sampling it says nothing about a
        draft region -- exactly where the widths can disagree.

        Host staging reaches this through a pipelined peer and never calls
        upstream's own check, so this is its only per-head check.
        """
        speculative_config = self.vllm_config.speculative_config
        draft = (
            None
            if speculative_config is None
            else speculative_config.draft_parallel_config
        )
        target_tp = self.vllm_config.parallel_config.tensor_parallel_size
        if draft is not None and draft.tensor_parallel_size != target_tp:
            # The draft's regions then hold heads cut for another TP, which no
            # head band of this engine's names.
            raise RuntimeError(
                "RBLN NIXL: a peer of another TP degree is paired by head band, "
                f"and this engine runs its draft at TP {draft.tensor_parallel_size} "
                f"beside a target at TP {target_tp}. Run the draft at the "
                "target's TP, or pair with a peer of the same TP."
            )
        block_size_ratio = self.topo.block_size_ratio(nixl_agent_meta.block_size)
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
        areas_l, _, _ = self._local_band(remote_tp_size)
        for logical_l, logical_r in self._logical_region_pairs(registered_layer_names):
            # A cell is one token of one head; an area holds heads x tokens of
            # them, so the two sides agree on a cell's bytes or on nothing.
            ours, peer = self._area_rects(
                nixl_agent_meta, logical_l, logical_r, 0, remote_tp_size
            )
            (lh0, lh1), (lt0, lt1) = ours[0]
            _, ((rh0, rh1), (rt0, rt1)), remote_len = peer[0]
            cells_l, cells_r = (lh1 - lh0) * (lt1 - lt0), (rh1 - rh0) * (rt1 - rt0)
            local_len = self.block_len_per_layer[logical_l * areas_l]
            if local_len * cells_r != remote_len * cells_l:
                raise RuntimeError(
                    f"RBLN NIXL: for logical region {logical_l} (the peer's "
                    f"{logical_r}) a KV head occupies "
                    f"{local_len / cells_l:.0f}B a token here but "
                    f"{remote_len / cells_r:.0f}B on the peer "
                    f"(local {local_len}B over {lh1 - lh0} head(s) x "
                    f"{lt1 - lt0} token(s), remote {remote_len}B over "
                    f"{rh1 - rh0} x {rt1 - rt0}). Block size, head_dim, dtype "
                    "and the layer's KV head count must match across P and D."
                )

    def _add_notif_only_remote_agent(
        self,
        metadata: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ) -> str:
        """Upstream's notif-only registration, with the DCP refusal in front.

        Upstream builds no descriptors here and so never calls
        `_validate_remote_agent_handshake` -- which is where every other route
        refuses a decode-context-parallel peer. Without this the push decode
        side accepts one and accounts its blocks.
        """
        self._refuse_decode_context_parallel(remote_dcp_size)
        return super()._add_notif_only_remote_agent(
            metadata, remote_tp_size, remote_dcp_size
        )

    @staticmethod
    def _refuse_decode_context_parallel(remote_dcp_size: int) -> None:
        """Refuse a decode-context-parallel peer before anything is recorded.

        The RBLN descriptor arithmetic addresses a peer rank's whole region; a
        DCP peer holds a slice of one, which no pairing here expresses. It has
        to come first: a raise after the prelude leaves the peer registered
        with the topology, and a failed handshake never reaches the teardown.
        """
        if remote_dcp_size != 1:
            raise RuntimeError(
                "RBLN NIXL does not support a decode-context-parallel peer: "
                f"peer reports dcp_size={remote_dcp_size}."
            )

    def _record_remote_region_ledger(
        self, engine_id: str, nixl_agent_meta: NixlAgentMetadata
    ) -> None:
        """The five entries upstream writes for a peer the first time it pairs.

        Its transfer paths index all five unguarded, and the two RBLN routes
        that register a peer themselves wrote only the block count -- which was
        the whole of it before vllm 0.30 added the region ledger.
        """
        num_remote_regions = len(nixl_agent_meta.kv_caches_base_addr)
        self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks
        self.dst_region_num_blocks[engine_id] = (
            nixl_agent_meta.region_num_blocks
            or [nixl_agent_meta.num_blocks] * num_remote_regions
        )
        self.dst_region_group_ids[engine_id] = nixl_agent_meta.region_group_ids or (
            self.region_group_ids
            if len(self.region_group_ids) == num_remote_regions
            else [0] * num_remote_regions
        )
        self.dst_uses_region_group_mapping[engine_id] = (
            len(set(self.dst_region_group_ids[engine_id])) > 1
        )
        self.dst_region_mem_types[engine_id] = (
            nixl_agent_meta.region_mem_types
            or [self.nixl_memory_type] * num_remote_regions
        )

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ) -> None:
        assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
        # The RBLN descriptor arithmetic addresses a peer rank's whole region;
        # a decode-context-parallel peer holds a slice of one, which no pairing
        # here expresses. Refuse it rather than pair the wrong bytes.
        if remote_dcp_size != 1:
            raise RuntimeError(
                "RBLN NIXL does not support a decode-context-parallel peer: "
                f"peer reports dcp_size={remote_dcp_size}."
            )
        self._check_d2d_region_pairing(nixl_agent_meta, remote_tp_size)
        self._check_split_axis_constraints(nixl_agent_meta, remote_tp_size)
        self._check_mla_constraints(nixl_agent_meta, remote_tp_size)
        if nixl_agent_meta.pp_size <= 1:
            if self._is_head_matched_peer(remote_tp_size):
                self._validate_head_matched_handshake(
                    nixl_agent_meta,
                    remote_tp_size,
                    nixl_agent_meta.registered_layer_names or None,
                )
                return
            # Upstream cuts a block into `block_size_ratio` equal pieces and
            # pairs each with one peer block. A block holding both K and V is
            # cut on their boundary instead of between token ranges, and its
            # descriptors carry no K/V axis to say which half is which.
            if self._kv_per_block > 1 and (
                self.topo.block_size_ratio(nixl_agent_meta.block_size) != 1
            ):
                raise RuntimeError(
                    "RBLN NIXL: a KV cache that packs K and V into one block "
                    "requires equal P/D block sizes."
                )
            super()._validate_remote_agent_handshake(
                nixl_agent_meta, remote_tp_size, remote_dcp_size
            )
            return

        remote_engine_id = nixl_agent_meta.engine_id
        remote_info = self.topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size, (
            f"RBLN NIXL: {remote_engine_id} is registered at TP "
            f"{remote_info.remote_tp_size} and handshaking at {remote_tp_size}"
        )
        # A producer with FEWER TP ranks is matched per head band; the other
        # direction never reaches here, rejected during the handshake.
        pp_tp_ratio = self.topo.tp_ratio(remote_tp_size)
        assert pp_tp_ratio > 0, (
            "PP over NIXL P/D does not support a peer with a larger TP size."
        )
        if pp_tp_ratio != 1:
            # The layer axis does not change what a head costs per block.
            self._validate_head_matched_handshake(
                nixl_agent_meta,
                remote_tp_size,
                nixl_agent_meta.registered_layer_names or None,
            )
        assert self.topo.block_size_ratio(nixl_agent_meta.block_size) == 1, (
            "PP over NIXL P/D requires equal P/D block sizes."
        )
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks, (
            f"RBLN NIXL: {remote_engine_id} advertised "
            f"{nixl_agent_meta.num_blocks} block(s) after registering "
            f"{self.dst_num_blocks[remote_engine_id]}; a descriptor id is "
            "region * blocks + block"
        )
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

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_dcp_size: int = 1,
    ) -> str:
        if not self._own_engine_layout:
            if self._is_head_matched_peer(remote_tp_size):
                # Different TP degrees, either direction: pair by head range
                # instead of by position (_build_head_matched_remote).
                assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
                return self._add_remote_agent_head_matched(
                    nixl_agent_meta, remote_tp_rank, remote_tp_size, remote_dcp_size
                )
            # Equal TP, or host staging's one region per layer: local region i
            # IS remote region i, which upstream's descriptor math assumes.
            return super().add_remote_agent(
                nixl_agent_meta, remote_tp_rank, remote_tp_size, remote_dcp_size
            )
        engine_id = nixl_agent_meta.engine_id
        agent_key = divmod(remote_tp_rank, remote_tp_size)
        if agent_key in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank %s already "
                "exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][agent_key]

        self._refuse_decode_context_parallel(remote_dcp_size)
        self._register_remote_engine_prelude(
            nixl_agent_meta, remote_tp_size, remote_dcp_size
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba."

        block_size_ratio = self.topo.block_size_ratio(nixl_agent_meta.block_size)

        if engine_id not in self.dst_num_blocks:
            self._record_remote_region_ledger(engine_id, nixl_agent_meta)

        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size
        )

        tp_ratio = self.topo.tp_ratio(remote_tp_size)
        indexes_into_remote = not self.topo.is_kv_replicated(engine_id) and tp_ratio > 0

        # The whole-engine lists never meet fan-in: unequal TP is head-matched,
        # and _check_d2d_region_pairing rejects SWA with any of it.
        assert tp_ratio >= 0, (
            "RBLN NIXL whole-engine lists do not support remote TP > local TP "
            f"(tp_ratio={tp_ratio})."
        )

        num_blocks = nixl_agent_meta.num_blocks

        pieces: list[tuple[int, int, int, int]] = []
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            local_block_len = self.get_backend_aware_kv_block_len(
                layer_idx=i, first_split=True, mamba_view=False
            )
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                local_block_len = remote_kv_block_len
            rank_offset = (
                self.tp_rank % tp_ratio * remote_kv_block_len
                if indexes_into_remote
                else 0
            )
            page_size = nixl_agent_meta.block_lens[i]
            pieces.append(
                (
                    base_addr + rank_offset,
                    local_block_len,
                    page_size,
                    nixl_agent_meta.device_id,
                )
            )

        grids = [[(1, 1)] * len(pieces)]
        # Derived here rather than passed between the lists: a range one side
        # carries and the other does not pairs its descriptors against whole
        # blocks, and the length check ahead of a transfer compares how many
        # indices each side named, not how far they reach.
        window = self._window_grid_cut
        if window is not None:
            grids.append([window] * len(pieces))
        grid = self._shard_chunk_grid(
            range(len(pieces)), block_size=nixl_agent_meta.block_size
        )
        if grid is not None:
            grids.append([(runs, chunks) for _, runs, chunks in grid.regions])
        blocks_data = np.concatenate(
            [
                self._chunk_range_descs(pieces, num_blocks=num_blocks, grids=g)
                for g in grids
            ]
        )

        logger.info(
            "RBLN NIXL: %d remote descriptor(s) for engine %s rank %d: whole, "
            "%s, and %s.",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            f"a window range of {window[0]} run(s) x {window[1]} granule(s)"
            if window is not None
            else "no window range",
            f"a chunk range of {grid.chunks_per_block} chunk(s) a block"
            if grid is not None
            else "no chunk range",
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
