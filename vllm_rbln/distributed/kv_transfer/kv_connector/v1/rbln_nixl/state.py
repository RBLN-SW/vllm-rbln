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
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    ChipletSplitAxis,
    TransferShape,
    connector_option,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_transfer_topology import (
    RblnTransferTopology,
)
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

#: What a transfer parks for `_compute_desc_ids`, whose signature is
#: upstream's: the request's final token count and block count, and the chunk
#: ranges of a block a streamed batch named on the side that call is for.
RequestTail = tuple[int | None, int | None, tuple[tuple[int, tuple[int, int]], ...]]


class ChunkGrid(NamedTuple):
    """How a chunk range cuts one descriptor list's regions.

    One chunk width serves the whole engine, so a block is `chunks_per_block`
    chunks in every region however the compiler spread its tokens: an area of
    a context cut holds a window of them, a region of a head cut all of them.
    Regions still differ in how many runs a chunk is, which is why the range
    is indexed by a running sum rather than a stride.
    """

    chunks_per_block: int
    #: Per region of the list, in its order: (the first chunk of a block it
    #: holds, byte runs one chunk of it is, chunks it holds). Runs are 0 where
    #: the region holds one chunk or part of one: its whole-block descriptor
    #: names that, so the range carries nothing for it.
    regions: tuple[tuple[int, int, int], ...]


class AreaPlace(NamedTuple):
    """Where one logical region sits in one chiplet area of its entry."""

    #: The region's first byte in the area, and one block of it there.
    device_addr: int
    block_len: int
    #: Which slice of the region the area holds; replicas of one share it. On
    #: a token cut, slice k is the k-th token range of every block.
    slice_id: int


class RegionGeometry(NamedTuple):
    """One logical region over its entry's chiplet areas, as the layout
    derived for the entry lays it."""

    #: HEAD stands in where the entry is not cut at all and every area holds
    #: it whole; `ChipletSplitAxis` owns what the two mean.
    axis: ChipletSplitAxis
    slices: int
    areas: tuple[AreaPlace, ...]


def kv_chunk_tokens(
    *,
    block_tokens: int,
    span_tokens: Sequence[int],
    prefill_step_tokens: int,
    chunk_tokens: int = 0,
) -> int:
    """Tokens one chunk of a block holds, ``block_tokens`` for the whole block.

    One prefill step is the floor, and the size when nobody names one. A step
    closes at most one chunk, which is what lets a chunk be the unit a coverage
    range counts in; a smaller chunk asks for something the write path cannot
    count in.

    In tokens rather than bytes because a byte target has to be divided by
    what a token costs, and a rank reads that off the regions it registered --
    which a pipeline stage holding a layer its siblings do not makes local to
    the rank. What is left is the request's span and the scheduler's step.

    Args:
        block_tokens: tokens one block holds.
        span_tokens: the token range one descriptor of each region covers -- a
            block, or one chiplet area's range where a context cut gives it one.
        prefill_step_tokens: ``max_num_batched_tokens``. Named for the step
            rather than the chunk it is, because the chunk this returns is the
            other one and the two are compared here.
        chunk_tokens: the size itself; 0 takes the floor.
    """
    if chunk_tokens and chunk_tokens < prefill_step_tokens:
        raise RuntimeError(
            f"RBLN NIXL: chunk_tokens={chunk_tokens} is under the "
            f"{prefill_step_tokens}-token prefill step, so a step would close "
            "several chunks; raise it or leave it unset for the step itself."
        )
    want = max(chunk_tokens, prefill_step_tokens)
    # A chunk either cuts every span or takes several whole: one reaching into
    # a span it does not cover whole names bytes no descriptor holds. So the
    # answer is the smallest size at or above the floor that does either. The
    # block itself always qualifies, which is where "no chunk" comes from.
    return next(
        size
        for size in range(min(want, block_tokens), block_tokens + 1)
        if block_tokens % size == 0
        and all(span % size == 0 or size % span == 0 for span in span_tokens)
    )


_SYS_CLASS_NET = Path("/sys/class/net")
_LINK_POLL_S = 1.0
# (read at, every link down), shared by every caller in the process.
_link_poll: tuple[float, bool] = (-_LINK_POLL_S, False)


def every_local_link_down() -> bool:
    """Every physical link in this namespace is down; a veth has no `device`."""
    global _link_poll
    now = time.monotonic()
    read_at, all_down = _link_poll
    if now - read_at < _LINK_POLL_S:
        return all_down
    states = []
    for dev in _SYS_CLASS_NET.iterdir():
        if not (dev / "device").exists():
            continue
        try:
            states.append((dev / "operstate").read_text().strip())
        except FileNotFoundError:  # unregistered since the listing
            continue
    all_down = bool(states) and all(s == "down" for s in states)
    _link_poll = (now, all_down)
    return all_down


def _grid_descs(
    starts: np.ndarray,
    length: np.ndarray | int,
    grid: tuple[int, int],
    device_id: np.ndarray | int,
) -> np.ndarray:
    """`runs * chunks` descriptors under each of `starts`, run-major.

    `starts` is already in descriptor order; `length` (the span a start's grid
    cuts) and `device_id` are per start or one for all.
    """
    runs, chunks = grid
    starts = np.asarray(starts, dtype=np.uint64)
    run_span = np.asarray(length, dtype=np.uint64)[..., None] // np.uint64(runs)
    desc_len = run_span // np.uint64(chunks)
    r = np.repeat(np.arange(runs, dtype=np.uint64), chunks)
    c = np.tile(np.arange(chunks, dtype=np.uint64), runs)
    addrs = starts[:, None] + r * run_span + c * desc_len
    lens = np.broadcast_to(desc_len, addrs.shape)
    devs = np.broadcast_to(
        np.asarray(device_id, dtype=np.uint64)[..., None], addrs.shape
    )
    return np.stack([addrs.ravel(), lens.ravel(), devs.ravel()], axis=1)


class RblnNixlWorkerState(NixlBaseConnectorWorker):
    """What more than one class here declares, assigns or reads -- plus the
    private helper of one such member, since a mixin cannot see a helper parked
    elsewhere either.

    A lifetime is a mixin whose only base is this class, so anything parked in
    one of them is untyped where anything else reads it -- and an untyped read
    is not an error, which is what makes the loss silent.
    """

    #: Whether this side originates the bytes into the peer's memory. Two things
    #: turn on it: a chiplet replica is a valid source to read from but not a
    #: valid sole destination to write to (`_head_matched_desc`), and a peer
    #: moving bytes the other way must not pass the handshake (`rbln_compat_hash`).
    _writes_into_peer: ClassVar[bool] = False

    # While registering one peer, the local region ids that peer's regions
    # correspond to, in ITS order -- see `_regions_viewed_as`.
    _viewed_region_ids: list[int] | None = None

    _group_specs: list[Any]
    #: What the knobs and the groups settled, before any geometry. Both sides
    #: derive it from the same inputs (`transfer_shape`), so neither can hold
    #: an answer of its own.
    _shape: TransferShape
    #: Tokens one block holds in the view the sliding-window kernel reads, per
    #: sliding-window group, which is not `block_size` where that kernel takes
    #: a window a block. Read off the device at registration, not derived.
    #: Empty where this rank holds no such group.
    _swa_kernel_blocks: set[int]
    #: Per sliding-window transfer group, `(storage address, view shape)` of the
    #: layer the group hands over, kept before the canonical filter drops it.
    #: Registration names the device frame with it.
    _swa_group_views: list[tuple[int, tuple[int, ...]]]

    #: The chiplet areas every KV entry is laid over, so transfer region
    #: `logical * _kv_areas + area` is one area of one logical region.
    _kv_areas: int
    #: Per logical region, how it lies over its entry's chiplet areas. A
    #: speculative draft keeps its own head count under the target's TP, so
    #: one engine can hold both axes.
    _logical_region_geometry: list[RegionGeometry]
    #: How many of K and V one region's block holds. Every lifetime reads it:
    #: registration derives it, the handshake pairs and advertises on it, and
    #: both a shard list's descriptors and a chunk range's runs are cut by it.
    _kv_per_block: int
    _logical_region_kv_heads: list[int | None]
    #: This RANK's share of them, which is what a token range of an area is
    #: one byte run per. `_logical_region_kv_heads` is a band across ranks and
    #: names no run.
    _logical_region_local_heads: list[int | None]
    local_seen_layer_names: list[str]
    compat_hash: str | None

    _remote_shard_layer_names: defaultdict[str, dict[int, tuple[str, ...]]]
    _remote_pp_size: dict[str, int]
    _overlapping_ranks: defaultdict[str, list[int]]
    src_xfer_handles_by_remote: dict[tuple[str, int, int], int]
    _borrowed_src_handles: set[tuple[str, int, int]]
    _shard_region_group_ids: dict[tuple[str, int], tuple[int, ...]]
    _shard_descs_per_block: dict[tuple[str, int], int]
    _shard_chunk_grids: dict[tuple[str, int], ChunkGrid | None]
    #: Per peer shard, which token range of a block each region of its list
    #: holds (`_region_span_index`).
    _shard_region_spans: dict[tuple[str, int], tuple[int, ...]]
    _chunk_grid: ChunkGrid | None
    #: `_window_grid()` for this engine, parked so a transfer does
    #: not recompute the head band per request.
    _window_grid_cut: tuple[int, int] | None
    _request_tail: "RequestTail | None"

    @property
    def _own_engine_layout(self) -> bool:
        """Whether the whole-engine lists carry a range upstream has no room for.

        Upstream names a region's block once. A window range cuts that block
        again; naming part of a block needs a list that can name both KV
        groups, which a per-shard list cannot since each names one. So a hybrid
        belongs here whether a knob or a streamed write cuts that block, while
        a single-group engine's chunk range rides the shard lists. A window as
        wide as the block adds no range at all.
        """
        return self._shape.owns_engine_lists

    def _region_span_index(self, region_id: int) -> int:
        """Which token range of a block a transfer region holds, by the slice
        its area holds; 0 where the block is one."""
        logical, area = divmod(region_id, self._kv_areas)
        geometry = self._logical_region_geometry[logical]
        if geometry.axis is not ChipletSplitAxis.TOKEN:
            return 0
        return geometry.areas[area].slice_id

    @property
    def _spans_per_block(self) -> int:
        """Token ranges a block is cut into, where no chunk range is there to
        reconcile them.

        Without one a block is cut only where the areas cut it: a context cut
        gives each slice a token range of every block, a head cut gives every
        area every token of some heads. Regions that cut it into different
        spans share no piece finer than the block, so that is one; so is a
        block nothing spreads, which is host staging.
        """
        spans = {
            g.slices if g.axis is ChipletSplitAxis.TOKEN else 1
            for g in self._logical_region_geometry
        }
        return spans.pop() if len(spans) == 1 else 1

    def _chunks_per_block(self, chunk_grid: ChunkGrid | None) -> int:
        """Chunks a whole block is cut into.

        Every count that names part of a block -- what `_tail_chunks` returns,
        the coverage unit, the window's own high-water mark -- is in this one.
        Without a grid a span is the finest piece there is.
        """
        if chunk_grid is None:
            return self._spans_per_block
        return chunk_grid.chunks_per_block

    _engines_to_rehandshake: set[str]
    _link_down_since: float | None
    _link_down_exit_s: int

    @property
    def topo(self) -> RblnTransferTopology:
        """The transfer topology, which registration produces.

        Upstream types it optional because it does not exist until the KV caches
        are registered, and on the D2D path that is deferred past warm-up. The
        RBLN type is the one every reader here needs: 0.30 dropped
        `cross_layers_blocks`, `virtually_split_kv_in_blocks` and
        `get_transfer_cache_regions` from upstream's, and this connector reads
        all three off it.
        """
        assert self.transfer_topo is not None, (
            "the transfer topology is read before the KV caches are registered"
        )
        return self.transfer_topo

    def _handle_failed_transfer(
        self,
        req_id: str,
        handle: int | None,
        failed_req_ids: set[str] | None = None,
        record_failed_transfer: bool = True,
    ) -> bool:
        """Upstream's failure report, minus the reads nothing is waiting on.

        A read with no local block moved nothing: `load_kv_async` is returned
        only with a positive external token count, so such a request was never
        put in WAITING_FOR_REMOTE_KVS, and reporting it trips the scheduler's
        `assert req_id in self.requests`.

        The entry stays: the handshake done-callback runs on the executor
        thread, so it and the heartbeat can reach one request, and dropping it
        here would leave whichever came second without the list to judge by --
        reporting the request after all.
        """
        meta = self._recving_metadata.get(req_id)
        if meta is not None and not meta.local_block_ids:
            assert handle is None
            return True
        if meta is not None and meta.remote is not None:
            self._engines_to_rehandshake.add(meta.remote.engine_id)
        return super()._handle_failed_transfer(
            req_id,
            handle,
            failed_req_ids,
            record_failed_transfer=record_failed_transfer,
        )

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

    def _logical_regions_per_layer(self) -> int:
        return len(self._logical_region_geometry) // len(self.local_seen_layer_names)

    def _regions_per_layer(self) -> int:
        num_layers = len(self.local_seen_layer_names)
        assert num_layers > 0 and self.num_regions % num_layers == 0, (
            f"num_regions={self.num_regions} not divisible by num_layers={num_layers}"
        )
        return self.num_regions // num_layers

    def _viewed_region(self, position: int) -> int:
        """Our region id for a position in the peer's region list."""
        ids = self._viewed_region_ids
        return position if ids is None else ids[position]

    def _shard_local_region_ids(
        self,
        registered_layer_names: tuple[str, ...] | list[str],
        peer_areas: list[int] | None = None,
    ) -> list[int]:
        """Our region ids that take part in a transfer with one peer.

        One filter per axis a peer can be narrower on: its layers, and the
        chiplet areas whose heads it owns.

        The order here IS the descriptor order -- the local dlist and
        `_build_head_matched_remote` walk these axes in the same nesting.
        """
        rpl = self._regions_per_layer()
        layer_indices = [
            local for _, local in self._layer_overlap(registered_layer_names)
        ]
        region_ids = [
            layer_idx * rpl + k for layer_idx in layer_indices for k in range(rpl)
        ]
        if peer_areas is None:
            return region_ids
        wanted = set(peer_areas)
        return [r for r in region_ids if r % self._kv_areas in wanted]

    def get_backend_aware_kv_block_len(
        self, layer_idx: int, first_split: bool = True, mamba_view: bool = False
    ) -> int:
        """One K/V element's block length, read through the peer's view.

        K and V sit in separate regions here, so an attention block's element
        is the whole block; a Mamba block instead holds a conv and an ssm part
        of different sizes, which `virtually_split_kv_in_blocks` indexes apart.
        vllm 0.30.0 removed the base method along with that split, so the
        subclass that kept the split carries the body too.
        """
        if self.topo.virtually_split_kv_in_blocks and mamba_view:
            return self._mamba_ssm_size[not first_split]
        # No translation here: `_regions_viewed_as` reorders the list itself for
        # the duration of a peer view, so the index is already the peer's
        # position. Translating again would apply the permutation twice.
        return self.block_len_per_layer[layer_idx]

    @staticmethod
    def _chunk_range_descs(
        pieces: list[tuple[int, int, int, int]],
        *,
        num_blocks: int,
        grids: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        """Every block of every region, cut into token chunks; `(1, 1)` per
        region is the whole-block range itself.

        `pieces` is `(base address, whole length, block stride, device id)` per
        region and `grids` its `(runs, chunks)`; the two sides differ in where
        the pieces come from and in nothing else. Region-major, then block,
        then run, then chunk, which is the order `_chunk_desc_ids` reads back.

        A run is one contiguous stretch of a block's bytes. Heads are the outer
        axis inside a block, so a token range is one run per head the region
        holds -- naming it as a single prefix would move the first head's
        tokens and leave the rest stale, and nothing would report it.
        """
        assert len(pieces) == len(grids)
        blocks = np.arange(num_blocks, dtype=np.uint64)
        parts = [
            _grid_descs(
                np.uint64(base) + blocks * np.uint64(stride), whole_len, grid, device_id
            )
            for (base, whole_len, stride, device_id), grid in zip(pieces, grids)
            if grid[0]
        ]
        if not parts:
            return np.empty((0, 3), dtype=np.uint64)
        return np.concatenate(parts)

    def _shard_chunk_grid(
        self,
        region_ids: Sequence[int],
        *,
        block_size: int,
        split: int,
        kv_runs: int = 1,
    ) -> ChunkGrid | None:
        """The chunk range over a list of `region_ids`, in that order.

        The runs are `_block_runs`; the chunk is what a prefill step leaves of
        the block, at a width that cuts every region's span or takes several
        whole. None where every region's span is one chunk -- what a step at
        or above one asks for -- so neither list grows. A side that writes a
        request in pieces asks for a grid as the knob does.
        """
        if not self._shape.writes_part_of_a_block:
            return None
        cuts = self._block_runs(split=split, kv_runs=kv_runs)
        if cuts is None:
            return None
        for spans, _ in cuts:
            assert block_size % spans == 0, (
                f"RBLN NIXL: {spans} span(s) do not tile a {block_size}-token block"
            )
        chunk_tokens = kv_chunk_tokens(
            block_tokens=block_size,
            span_tokens=sorted({block_size // spans for spans, _ in cuts}),
            prefill_step_tokens=(
                self.vllm_config.scheduler_config.max_num_batched_tokens
            ),
            chunk_tokens=connector_option(self.vllm_config, "chunk_tokens", 0),
        )
        chunks_per_block = block_size // chunk_tokens
        if all(spans == chunks_per_block for spans, _ in cuts):
            return None
        regions = []
        for region_id in region_ids:
            spans, runs = cuts[region_id // self._kv_areas]
            span = self._region_span_index(region_id)
            if spans >= chunks_per_block:
                regions.append((span * chunks_per_block // spans, 0, 1))
                continue
            chunks = chunks_per_block // spans
            regions.append((span * chunks, runs, chunks))
        return ChunkGrid(chunks_per_block, tuple(regions))

    def _block_runs(self, *, split: int, kv_runs: int) -> list[tuple[int, int]] | None:
        """Per logical region, `(token spans a block is cut into, byte runs one
        span is)`.

        Both ranges that name less than a block read this. The axis says where
        a token range sits: slice k of a context cut holds the in-block range
        [k * span, (k + 1) * span) of every head the rank holds, and the heads
        are the outer axis, so a range of it is one run per head; a head cut
        gives every area every token of its slice of them, so the span is the
        block and a range is one run per head of the slice. None where a piece
        is narrower than a head.

        A packed block holds a token's K beside its V, so a range is one run in
        each, unless the peer pairing already read the two apart (`kv_runs`).
        """
        assert self._logical_region_local_heads, (
            "head counts per region are the direct path's, and so is every "
            "range that reads this"
        )
        cuts = []
        for region, heads in enumerate(self._logical_region_local_heads):
            assert heads is not None, f"logical region {region} has no head axis"
            geometry = self._logical_region_geometry[region]
            token_cut = geometry.axis is ChipletSplitAxis.TOKEN
            spans = geometry.slices if token_cut else 1
            slices = 1 if token_cut else geometry.slices
            assert heads % slices == 0, (
                f"logical region {region} holds {heads} KV head(s) over "
                f"{slices} slice(s)"
            )
            heads_per_span = heads // slices
            if heads_per_span % split:
                return None
            cuts.append(
                (spans, self._kv_per_block // kv_runs * heads_per_span // split)
            )
        return cuts

    def _window_grid(self) -> tuple[int, int] | None:
        """`(byte runs one window-wide piece is, pieces a block is cut into)`.

        None where window mode registered no range. Where the kernel addresses
        the cache in window-wide blocks, a piece IS one of them and so one run.
        Where it addresses whole blocks, a piece is a token range of one, which
        the head cut spreads exactly as it spreads a chunk.
        """
        ratio = self._shape.window_ratio
        if ratio is None:
            return None
        window = self.block_size // ratio
        if self._swa_kernel_blocks != {window} and self._swa_kernel_blocks != {
            self.block_size
        }:
            # One number cuts the range, and only these two are a granule or a
            # whole block of it. A third would be cut into pieces that are
            # neither, with the descriptor count unchanged.
            raise RuntimeError(
                "RBLN NIXL: a window range needs every sliding-window group "
                f"addressed in blocks of {window} or {self.block_size} tokens, "
                f"and this engine's are {sorted(self._swa_kernel_blocks)}."
            )
        if self._swa_kernel_blocks == {window}:
            return 1, ratio
        cuts = self._block_runs(split=1, kv_runs=1)
        assert cuts is not None, "a split of one is never narrower than a head"
        if any(spans != 1 for spans, _ in cuts):
            raise RuntimeError(
                "RBLN NIXL: a window range over a block-wide kernel view needs "
                "a head cut, which leaves one token range a block. This cache "
                "is cut on the axes "
                f"{[g.axis.name for g in self._logical_region_geometry]} into "
                f"{self._kv_areas} area(s)."
            )
        bands = {runs for _, runs in cuts}
        if len(bands) != 1:
            raise RuntimeError(
                "RBLN NIXL: a window range over a block-wide kernel view names "
                "a token range of a block, which needs one head band across "
                "every region. This engine's regions hold KV head counts "
                f"{self._logical_region_local_heads}."
            )
        return bands.pop(), ratio

    @staticmethod
    def _slice_head_bounds(
        tp_rank: int,
        tp_size: int,
        total_kv_heads: int,
        slices: int,
        *,
        side: Literal["local", "peer"],
    ) -> tuple[int, int]:
        """(first head this shard owns, heads per logical slice).

        The compiler cuts a shard's heads into ``slices`` pieces, replicated
        over the chiplet areas when a shard owns fewer heads than the device has
        chiplets. Which area holds which slice is `RegionGeometry`'s answer.

        Callers pass a peer's advertised geometry as well as this rank's, so
        the two below refuse a pairing rather than assert an invariant;
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
        if heads_per_rank % slices:
            raise RuntimeError(
                f"RBLN NIXL: the {side} shard owns {heads_per_rank} KV heads cut "
                f"into {slices} logical slice(s), which does not divide them; the "
                "compiler gives every slice the same head count."
            )
        return tp_rank * heads_per_rank, heads_per_rank // slices

    def register_local_xfer_handler(
        self,
        block_size: int,
        *,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
        split: int = 1,
        region_ids: list[int] | None = None,
        replica_fanout: int = 1,
        kv_runs: int = 1,
        chunk_grid: ChunkGrid | None = None,
    ) -> tuple[int, np.ndarray]:
        """This engine's local descriptors, and the views of a block they name.

        A sliding window shortens the RDMA descriptor, not the tensor and not
        the host copy: regions stay Full-sized whatever the groups are. So the
        list carries whole blocks first, then the ranges the knobs asked for
        over the same addresses -- a window range, a chunk range, or both --
        and a transfer picks the one its group needs (`_compute_desc_ids`).
        The window range names the granules the window sits in rather than a
        prefix of the block, which is why `_window_grid` reads the request's
        token count to find them.
        """
        if not self._own_engine_layout:
            if (
                registered_layer_names is None
                and peer_areas is None
                and split == 1
                and replica_fanout == 1
                and kv_runs == 1
            ):
                # No SWA window mode, whole-engine peer: upstream's Full-only
                # layout, one handle covering every region.
                return super().register_local_xfer_handler(block_size)
            # Per-peer shard: only the regions this peer serves, each cut into
            # `split` pieces (_shard_local_region_ids, _head_split) and into
            # `kv_runs` halves (_kv_runs).
            return self._register_shard_local_xfer_handler(
                block_size,
                registered_layer_names or self.local_seen_layer_names,
                peer_areas=peer_areas,
                split=split,
                region_ids=region_ids,
                replica_fanout=replica_fanout,
                kv_runs=kv_runs,
                chunk_grid=chunk_grid,
            )
        assert (
            registered_layer_names is None
            and peer_areas is None
            and split == 1
            and replica_fanout == 1
        ), (
            "RBLN NIXL: the whole-engine descriptor ranges are not supported "
            "with pipeline parallelism or heterogeneous tensor parallelism"
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba layers."

        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        num_blocks = self.num_blocks * block_size_ratio

        # A whole block is one range whatever it packs, since K and V are
        # adjacent in it.
        pieces: list[tuple[int, int, int, int]] = []
        for i, base_addr in enumerate(local_base_addresses):
            kv_block_len = (
                self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=True, mamba_view=False
                )
                // block_size_ratio
            )
            stride = self.block_len_per_layer[i] // block_size_ratio
            pieces.append((base_addr, kv_block_len, stride, self.device_id))

        grids = [[(1, 1)] * len(pieces)]
        # Both further ranges cut the same block, and differ only in the piece:
        # a window's is `sliding_window` tokens, a chunk's a prefill step.
        window = self._window_grid_cut
        if window is not None:
            grids.append([window] * len(pieces))
        # Asked for here rather than handed in, so that the block size it is
        # derived from is the block size this list was built with.
        grid = self._shard_chunk_grid(
            range(len(pieces)), block_size=block_size, split=1
        )
        if grid is not None:
            grids.append([(runs, chunks) for _, runs, chunks in grid.regions])
        descs_data = np.concatenate(
            [
                self._chunk_range_descs(pieces, num_blocks=num_blocks, grids=g)
                for g in grids
            ]
        )

        logger.info(
            "RBLN NIXL: %d local descriptor(s) for this engine over %d region(s) "
            "x %d block(s): whole, %s, and %s.",
            len(descs_data),
            len(local_base_addresses),
            num_blocks,
            f"a window range of {window[0]} run(s) x {window[1]} granule(s)"
            if window is not None
            else "no window range",
            f"a chunk range of {grid.chunks_per_block} chunk(s) a block"
            if grid is not None
            else "no chunk range",
        )

        descs = self.nixl_wrapper.get_xfer_descs(descs_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            descs_data,
        )

    def _register_shard_local_xfer_handler(
        self,
        block_size: int,
        registered_layer_names: tuple[str, ...] | list[str],
        peer_areas: list[int] | None = None,
        split: int = 1,
        region_ids: list[int] | None = None,
        replica_fanout: int = 1,
        kv_runs: int = 1,
        chunk_grid: ChunkGrid | None = None,
    ) -> tuple[int, np.ndarray]:
        """Prepare this shard's local descriptors, optionally twice over.

        `chunk_grid` says, per region, how many byte runs one piece is spread
        over and how many token chunks each run is cut into. Given, a second
        range of descriptors follows the first over the same addresses, naming
        those chunks -- the shape `swa_window_mode` already uses for a shorter
        view of the same blocks. A transfer picks one range or the other by
        index, so both can be selected in one call.

        A token range is one run per axis the block spreads it over, so it is
        `runs * chunks` descriptors where the whole piece is 1.
        """
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba."

        block_size_ratio = self.block_size // block_size
        num_blocks = self.num_blocks * block_size_ratio
        all_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        if region_ids is None:
            region_ids = self._shard_local_region_ids(
                registered_layer_names, peer_areas=peer_areas
            )

        parts: list[np.ndarray] = []
        # The whole piece first, then its chunks. `(1, 1)` names the piece
        # itself, so the first pass is what this built before `chunk_grid`.
        passes = [[(1, 1)] * len(region_ids)]
        if chunk_grid is not None:
            assert len(chunk_grid.regions) == len(region_ids)
            passes.append([(runs, chunks) for _, runs, chunks in chunk_grid.regions])
        block_ids = np.arange(num_blocks, dtype=np.uint64)[:, None, None, None]
        js = np.arange(split, dtype=np.uint64)[None, :, None, None]
        kvs = np.arange(kv_runs, dtype=np.uint64)[None, None, None, :]
        for grids in passes:
            for region_id, grid in zip(region_ids, grids):
                if not grid[0]:
                    continue
                base_addr = all_base_addrs[region_id]
                kv_block_len = (
                    self.get_backend_aware_kv_block_len(
                        layer_idx=region_id, first_split=True, mamba_view=False
                    )
                    // block_size_ratio
                )
                stride = self.block_len_per_layer[region_id] // block_size_ratio
                # Head pieces are consecutive byte ranges within K, and within
                # V -- it is the REMOTE side that scatters. The two sit
                # `kv_stride` apart, so a piece that names a head band names it
                # in each (see _head_matched_desc), and a chunk of that piece
                # stays inside the one it started in.
                kv_stride = kv_block_len // kv_runs
                sub_len = kv_stride // split
                # Axes are block, piece, peer copy, K/V. One entry per peer copy
                # this piece goes to: the same bytes reach every replica (see
                # _head_matched_desc), so the source repeats while the
                # destination advances.
                starts = np.broadcast_to(
                    base_addr + block_ids * stride + kvs * kv_stride + js * sub_len,
                    (num_blocks, split, replica_fanout, kv_runs),
                )
                parts.append(_grid_descs(starts.ravel(), sub_len, grid, self.device_id))
        descs_data = np.concatenate(parts)

        if chunk_grid is not None:
            whole = len(region_ids) * num_blocks * split * replica_fanout * kv_runs
            logger.info(
                "RBLN NIXL: %d local descriptor(s) for this shard, %d of them a "
                "chunk range of %d chunk(s) a block over (runs, chunks) %s.",
                len(descs_data),
                len(descs_data) - whole,
                chunk_grid.chunks_per_block,
                sorted({(runs, chunks) for _, runs, chunks in chunk_grid.regions}),
            )
        descs = self.nixl_wrapper.get_xfer_descs(descs_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            descs_data,
        )
