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

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Literal

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
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
    KVSplitAxis,
    RblnNixlAgentMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    RblnNixlWorkerState,
)
from vllm_rbln.logger import init_logger

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

        self._reject_uneven_region_slices(remote_tp_size)
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
            self.topo.tp_size,
            self._kv_areas,
            self._kv_slices,
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

    def _build_fa_remote(
        self,
        plan: TPMapping,
        nixl_agent_meta: NixlAgentMetadata,
        block_size_ratio: int,
    ) -> list[tuple[int, int, int]]:
        # The one upstream loop that walks the peer's regions while reading ours.
        # Scoped to this call rather than to `add_remote_agent`, which also builds
        # local dlists from our own region ids and must not be translated.
        assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
        with self._regions_viewed_as(self._peer_region_ids(nixl_agent_meta)):
            return super()._build_fa_remote(plan, nixl_agent_meta, block_size_ratio)

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
        areas_l, slices_l = self._kv_areas, self._kv_slices
        areas_r = nixl_agent_meta.kv_areas
        slices_r = nixl_agent_meta.kv_slices

        split = self._head_split(
            self.topo.tp_size * slices_l, remote_tp_size * slices_r
        )

        replicas_l = areas_l // slices_l
        replicas_r = areas_r // slices_r
        num_blocks = nixl_agent_meta.num_blocks
        remote_bases = nixl_agent_meta.kv_caches_base_addr
        remote_lens = nixl_agent_meta.block_lens

        logical_pairs = self._logical_region_pairs(registered_layer_names)

        # Which of our areas this peer holds (see _fan_in_peer_areas).
        areas_iter = range(areas_l) if peer_areas is None else peer_areas

        out: list[tuple[int, int, int]] = []
        # Axis order is `_shard_local_region_ids`'; within a region, block-major
        # to match _compute_desc_ids' region_id * num_blocks + b.
        for logical_l, logical_r in logical_pairs:
            # A model-config count describes the target's layers, and on a
            # draft's it lands a plausible offset inside the wrong bytes. The
            # peer's region holds the same layer, so one count answers for both
            # sides (the handshake verifies it).
            total_heads = self._region_kv_heads(logical_l)
            base_l, per_slice_l = self._slice_head_bounds(
                self.tp_rank,
                self.topo.tp_size,
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
                        split=split,
                    )
                )
        # Per block, a region becomes one descriptor per head piece per peer
        # copy -- the same count `_register_shard_local_xfer_handler` builds
        # locally and `_shard_descs_per_block` records.
        fanout = replicas_r if self._writes_into_peer else 1
        assert len(out) == (
            len(logical_pairs) * len(areas_iter) * num_blocks * split * fanout
        )
        return out

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
        tp_ratio = self.topo.tp_ratio(remote_tp_size)
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
        if self._is_head_matched_peer(remote_tp_size):
            raise RuntimeError(
                "RBLN NIXL D2D: MLA is not supported with heterogeneous tensor "
                f"parallelism (peer TP {remote_tp_size}, local "
                f"{self.topo.tp_size})."
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

    def _check_split_axis_constraints(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Reject peers whose chiplet areas do not mean what head bands assume.

        Head bands only make sense over a ``HEAD`` cut; against a context cut a
        band names a range no area holds, while the byte counts stay right and
        the handshake passes. Equal TP is exempt: bands are not consulted, area
        k pairs with area k, and both sides cut the same way.

        Host-bounce has no areas, so its permanent HEAD default is correct.
        """
        if self.use_host_buffer:
            return
        peer_axis = nixl_agent_meta.kv_split_axis
        if peer_axis != self._kv_split_axis:
            raise RuntimeError(
                f"RBLN NIXL D2D: peer cut its KV cache on the {peer_axis.name} "
                f"axis but this worker cut it on {self._kv_split_axis.name}. "
                "The area counts can agree while an area means something else "
                "on each side, so no pairing is meaningful."
            )
        if self._kv_split_axis is KVSplitAxis.NON_HEAD and self._is_head_matched_peer(
            remote_tp_size
        ):
            raise RuntimeError(
                "RBLN NIXL D2D: a context-cut KV cache is not supported with "
                f"heterogeneous tensor parallelism (peer TP {remote_tp_size}, "
                f"local {self.topo.tp_size}). Bands computed from the "
                "configured KV-head count would give plausible wrong "
                "descriptors. Use equal TP on both sides, or "
                "kv_buffer_device='cpu'."
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
        self._remote_shard_layer_names.pop(engine_id, None)
        self._overlapping_ranks.pop(engine_id, None)
        self._remote_pp_size.pop(engine_id, None)
        super()._cleanup_remote_engine(engine_id, log_eviction=log_eviction)

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
        # the guard below reduce to cut-count ratios (`_head_split`).
        total_heads = self.topo.total_num_kv_heads
        base_l, per_slice_l = self._slice_head_bounds(
            self.tp_rank,
            self.topo.tp_size,
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
        split: int,
    ) -> list[tuple[int, int, int]]:
        """Descriptors for one local region: ``split`` (`_head_split`) per block.

        The split is the caller's, one value for the whole transfer; the head
        bands in ``geom``/``peer`` are this region's.

        Order is block-major, piece-minor, to match the local list
        `_register_shard_local_xfer_handler` builds.
        """
        base_l, per_slice_l, replicas_l = geom
        base_r, per_slice_r, replicas_r, slices_r = peer

        head = base_l + (area_l // replicas_l) * per_slice_l
        desc_len = self.get_backend_aware_kv_block_len(
            layer_idx=region_id, first_split=True, mamba_view=False
        )
        per_piece = per_slice_l // split
        sub_len = desc_len // split

        # Replicas of a slice hold identical bytes, so reading any one of them
        # answers -- but writing only one leaves the peer's other chiplets on
        # stale KV. Reading takes the first; writing takes them all.
        fanout = replicas_r if self._writes_into_peer else 1

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
            for k in range(fanout):
                remote_region = logical_r * areas_r + slice_r * replicas_r + k
                page = remote_lens[remote_region]
                if page % per_slice_r != 0:
                    raise RuntimeError(
                        f"RBLN NIXL D2D: peer region {remote_region} block "
                        f"length {page}B does not split into {per_slice_r} "
                        "heads."
                    )
                # Heads are contiguous inside a block, so skipping
                # `head_within` of them is a plain byte offset.
                head_offset = head_within * (page // per_slice_r)
                if sub_len + head_offset > page:
                    raise RuntimeError(
                        f"RBLN NIXL D2D: local region {region_id} piece {j} "
                        f"wants {sub_len}B at +{head_offset}B, past the end of "
                        f"the peer's {page}B region {remote_region}."
                    )
                pieces.append((remote_bases[remote_region] + head_offset, page))

        for block_id in range(num_blocks):
            for base, page in pieces:
                out.append((base + block_id * page, sub_len, device_id))
        return out

    @staticmethod
    def _head_split(cuts_l: int, cuts_r: int) -> int:
        """How many pieces one of our regions is read in.

        A descriptor names one contiguous range per side, so an area coarser than
        the peer's slice needs as many pieces as the peer spreads its heads over.

        Cuts (TP degree x slices per shard), not heads per slice: a slice holds
        `T / cuts`, so the width ratio is `cuts_r / cuts_l` for any `T`. **One
        split therefore serves every region even when they hold different
        numbers of heads.**
        """
        if cuts_r <= cuts_l:
            return 1
        if cuts_r % cuts_l:
            raise RuntimeError(
                f"RBLN NIXL D2D: the peer cuts KV heads {cuts_r} ways and this "
                f"rank {cuts_l}, which does not divide it; the two sides must "
                "cut heads at commensurate granularities."
            )
        return cuts_r // cuts_l

    def _is_fan_in_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer has MORE TP ranks than us, so we gather from it
        together with its siblings."""
        if self.use_host_buffer:
            return False
        return self.topo.tp_ratio(remote_tp_size) < 0

    def _is_head_matched_peer(self, remote_tp_size: int) -> bool:
        """Whether this peer is served by ``_build_head_matched_remote``.

        Any unequal TP degree, in either direction, on D2D without SWA
        view-opt. ``tp_ratio`` is pure arithmetic on the two TP sizes, so this
        is safe to ask before the engine is registered.
        """
        if self.use_host_buffer or self._sw_ratio is not None:
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
        areas = self._kv_areas
        if registered_layer_names is None:
            return [(i, i) for i in range(len(self.block_len_per_layer) // areas)]
        per_layer = self._regions_per_layer() // areas
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

        p_remote_tp_ranks = self.topo.handshake_target_ranks(remote_tp_size)
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
                    fanout = self._peer_replica_fanout(metadata, remote_tp_size)
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
                        replica_fanout=fanout,
                    )
                    overlapping.append(global_rank)
        # Published once, not accumulated: a handshake that raises partway
        # leaves no entry, so the retry that follows starts from empty instead
        # of appending its shards a second time and reading every block twice.
        self._overlapping_ranks[expected_engine_id] = overlapping
        self._remote_pp_size[expected_engine_id] = pp_size
        return remote_rank_to_agent_name

    def _peer_head_split(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> int:
        """`_head_split` for a peer, from its advertised chiplet geometry.

        Reads no head count, so it runs before any per-region band exists -- a
        transfer is set up ahead of the handshake check on a layer's width.
        """
        if not self._is_head_matched_peer(remote_tp_size):
            return 1
        return self._head_split(
            self.topo.tp_size * self._kv_slices,
            remote_tp_size * nixl_agent_meta.kv_slices,
        )

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

    def _peer_replica_fanout(
        self, nixl_agent_meta: RblnNixlAgentMetadata, remote_tp_size: int
    ) -> int:
        """How many of the peer's copies of a slice one transfer must touch.

        The compiler duplicates a KV head across chiplet areas when a shard
        owns fewer heads than the device has chiplets. Reading is free to pick
        one; writing has to fill them all, or the peer's other chiplets keep
        serving what was there before.
        """
        if not self._writes_into_peer:
            return 1
        if not self._is_head_matched_peer(remote_tp_size):
            return 1
        return max(1, nixl_agent_meta.kv_areas // nixl_agent_meta.kv_slices)

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
        # Refused on arrival, not where it is first divided by.
        if metadata.kv_slices <= 0:
            raise RuntimeError(
                f"RBLN NIXL: the peer advertises {metadata.kv_slices} logical "
                "slice(s) per shard; the compiler cuts a shard's heads into at "
                "least one, and the descriptor arithmetic divides by this count."
            )
        return metadata

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

        Upstream's remote descriptor builder feeds the PEER's region position into
        `get_backend_aware_kv_block_len` and `_is_region_replicated`, which index
        OUR arrays. That only means our region while our band starts at our
        region 0 -- true for a pipeline stage, false for a consumer holding every
        layer while the producer is pipelined, where **the length read belongs to
        a different layer than the address it is paired with**.

        A plain attribute suffices: upstream reaches those two methods only from
        registration and from a handshake, and runs handshakes one at a time on a
        single-worker executor. Registration cannot overlap one either:
        `finalize_kv_cache_registration` runs during worker warm-up, before a
        request exists to start a handshake. So no second view is ever live.
        """
        prev = self._viewed_region_ids
        self._viewed_region_ids = region_ids
        try:
            yield
        finally:
            self._viewed_region_ids = prev

    def _register_remote_engine_prelude(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
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
            ),
        )
        self.tp_mappings[nixl_agent_meta.engine_id] = compute_tp_mapping(
            transfer_topology=self.topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

    def _register_shard_xfer_state(
        self,
        engine_id: str,
        global_rank: int,
        block_size: int,
        registered_layer_names: tuple[str, ...],
        peer_areas: list[int] | None = None,
        split: int = 1,
        remote_tp_size: int = 1,
        replica_fanout: int = 1,
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
                replica_fanout=replica_fanout,
            )
        self.src_xfer_handles_by_remote[key] = handle
        n_groups = len(self.kv_cache_config.kv_cache_groups)
        assert n_groups == 1, (
            "RBLN NIXL per-shard transfers support a single KV-cache group, "
            f"got {n_groups}"
        )
        self._shard_region_group_ids[(engine_id, global_rank)] = (0,) * len(region_ids)
        self._shard_descs_per_block[(engine_id, global_rank)] = split * replica_fanout

    def _reject_uneven_region_slices(self, remote_tp_size: int) -> None:
        """Refuse a head-banded peer whose regions disagree on their slice count.

        `_build_head_matched_remote` bands every region with the engine-wide
        `_kv_slices`, which describes the LAST region -- a draft's, named past the
        target's depth. **Divisibility does not catch that**: a slice count divides
        the chiplet count, so the smaller divides the larger side's heads per rank
        and every check in `_slice_head_bounds` passes.

        Per peer, not at registration: symmetric TP does not band by head at all,
        so refusing the same engine there would reject deployments that work.
        Regions without a band are skipped -- one has its own refusal further in.
        """
        distinct = {
            slices
            for slices, heads in zip(
                self._logical_region_slices, self._logical_region_kv_heads
            )
            if heads is not None
        }
        if len(distinct) <= 1:
            return
        raise RuntimeError(
            "RBLN NIXL D2D: this engine's KV cache entries are cut into "
            f"different numbers of chiplet slices {sorted(distinct)}, so no one "
            "count bands every region. Head-band pairing with a peer at TP "
            f"{remote_tp_size} needs one."
        )

    @staticmethod
    def _slice_head_bounds(
        tp_rank: int,
        tp_size: int,
        total_kv_heads: int,
        areas: int,
        slices: int,
        *,
        side: Literal["local", "peer"],
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
        for logical_l, logical_r in self._logical_region_pairs(registered_layer_names):
            total_heads = self._region_kv_heads(logical_l)
            _, per_slice_l = self._slice_head_bounds(
                self.tp_rank,
                self.topo.tp_size,
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
            local_len = self.block_len_per_layer[logical_l * self._kv_areas]
            remote_len = nixl_agent_meta.block_lens[
                logical_r * nixl_agent_meta.kv_areas
            ]
            if local_len * per_slice_r != remote_len * per_slice_l:
                raise RuntimeError(
                    f"RBLN NIXL: for logical region {logical_l} (the peer's "
                    f"{logical_r}) a KV head occupies "
                    f"{local_len / per_slice_l:.0f}B per block here but "
                    f"{remote_len / per_slice_r:.0f}B on the peer "
                    f"(local {local_len}B over {per_slice_l} head(s), remote "
                    f"{remote_len}B over {per_slice_r}). Block size, head_dim, "
                    "dtype and the layer's KV head count must match across P "
                    "and D."
                )

    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        assert isinstance(nixl_agent_meta, RblnNixlAgentMetadata)
        self._check_split_axis_constraints(nixl_agent_meta, remote_tp_size)
        self._check_mla_constraints(nixl_agent_meta, remote_tp_size)
        self._check_d2d_region_pairing(nixl_agent_meta, remote_tp_size)
        if nixl_agent_meta.pp_size <= 1:
            if self._is_head_matched_peer(remote_tp_size):
                self._validate_head_matched_handshake(
                    nixl_agent_meta,
                    remote_tp_size,
                    nixl_agent_meta.registered_layer_names or None,
                )
                return
            super()._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)
            return

        remote_engine_id = nixl_agent_meta.engine_id
        remote_info = self.topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size
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

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        if self._sw_ratio is None:
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

        assert not self.topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout."
        )

        block_size_ratio = self.topo.block_size_ratio(nixl_agent_meta.block_size)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        tp_ratio = self.topo.tp_ratio(remote_tp_size)
        indexes_into_remote = not self.topo.is_kv_replicated(engine_id) and tp_ratio > 0

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
