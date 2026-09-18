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
from typing import Any, ClassVar, Literal

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    connector_option,
)
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# What the byte target is when nobody names one: large enough that the fixed
# cost of a descriptor is small beside what it carries.
DEFAULT_CHUNK_BYTES = 256 * 1024


def kv_chunk_tokens(
    *,
    span_tokens: int,
    bytes_per_token: int,
    prefill_step_tokens: int,
    chunk_bytes: int = 0,
    chunk_tokens: int = 0,
) -> int:
    """Tokens one chunk of a span holds, ``span_tokens`` for the whole span.

    The byte knob is a floor, not a size: under the transfer size a fabric
    moves efficiently a descriptor pays more fixed cost than it carries, so a
    target landing between two sizes that tile the span takes the larger.

    One prefill step is a second floor. A step closes at most one chunk, which
    is what lets a chunk be the unit a coverage range counts in, and the token
    knob has to obey it as well -- given a smaller one, an operator has asked
    for something the write path cannot count in.

    Args:
        span_tokens: tokens one whole descriptor covers -- a block, or one
            chiplet area's token range where a context cut gives it one.
        bytes_per_token: what one token of one span costs.
        prefill_step_tokens: ``max_num_batched_tokens``. Named for the step
            rather than the chunk it is, because the chunk this returns is the
            other one and the two are compared here.
        chunk_bytes: the byte target; 0 where nobody named one.
        chunk_tokens: the size itself, named instead of the target; 0 unset.
    """
    if chunk_tokens and chunk_tokens < prefill_step_tokens:
        raise RuntimeError(
            f"RBLN NIXL: chunk_tokens={chunk_tokens} is under the "
            f"{prefill_step_tokens}-token prefill step, so a step would close "
            "several chunks; raise it or leave it unset for the byte target."
        )
    if chunk_tokens and chunk_bytes:
        raise RuntimeError(
            f"RBLN NIXL: chunk_tokens={chunk_tokens} and chunk_bytes="
            f"{chunk_bytes} both name a chunk size; set one."
        )
    want = max(
        chunk_tokens or (chunk_bytes or DEFAULT_CHUNK_BYTES) // bytes_per_token,
        prefill_step_tokens,
    )
    # A chunk that does not tile the span leaves a descriptor reaching into the
    # next one, so the answer is the smallest size at or above the floor that
    # does. The span itself always qualifies, which is where "no chunk" comes
    # from.
    return next(
        size
        for size in range(min(want, span_tokens), span_tokens + 1)
        if span_tokens % size == 0
    )


def _as_descs(blocks_data: list[tuple[int, int, int]]) -> np.ndarray:
    """Columns are (addr, len, device), in upstream's own descriptor shape.

    ``src_blocks_data`` is read back by upstream's hetero-TP split, which calls
    ``.tolist()`` on it.
    """
    return np.asarray(blocks_data, dtype=np.uint64).reshape(-1, 3)


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
    _has_swa: bool
    _sw_ratio: int | None

    _kv_areas: int
    _kv_slices: int
    _kv_split_axis: KVSplitAxis
    #: How many of K and V one region's block holds. Every lifetime reads it:
    #: registration derives it, the handshake pairs and advertises on it, the
    #: window and chunk ranges of a descriptor list are cut by it.
    _kv_per_block: int
    _chunk_mode: bool
    _logical_region_kv_heads: list[int | None]
    _logical_region_slices: list[int]
    local_seen_layer_names: list[str]
    compat_hash: str | None

    _remote_shard_layer_names: defaultdict[str, dict[int, tuple[str, ...]]]
    _remote_pp_size: dict[str, int]
    _overlapping_ranks: defaultdict[str, list[int]]
    src_xfer_handles_by_remote: dict[tuple[str, int, int], int]
    _borrowed_src_handles: set[tuple[str, int, int]]
    _shard_region_group_ids: dict[tuple[str, int], tuple[int, ...]]
    _shard_descs_per_block: dict[tuple[str, int], int]
    _shard_chunk_grids: dict[tuple[str, int], tuple[int, int] | None]
    _chunk_grid: tuple[int, int] | None
    _request_tail: tuple[int | None, int] | None

    @property
    def _spans_per_block(self) -> int:
        """Descriptors a block's token axis is spread over.

        A context cut gives each chiplet area a token range of every block, so
        the block's tokens are `_kv_areas` descriptors. A head cut gives every
        area every token of some heads, so they are one.
        """
        return self._kv_areas if self._kv_split_axis is KVSplitAxis.NON_HEAD else 1

    @property
    def topo(self) -> TransferTopology:
        """The transfer topology, which registration produces.

        Upstream types it optional because it does not exist until the KV caches
        are registered, and on the D2D path that is deferred past warm-up.
        """
        assert self.transfer_topo is not None, (
            "the transfer topology is read before the KV caches are registered"
        )
        return self.transfer_topo

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

    def get_backend_aware_kv_block_len(
        self, layer_idx: int, first_split: bool = True, mamba_view: bool = False
    ) -> int:
        return super().get_backend_aware_kv_block_len(
            layer_idx=self._viewed_region(layer_idx),
            first_split=first_split,
            mamba_view=mamba_view,
        )

    @staticmethod
    def _chunk_range_descs(
        pieces: list[tuple[int, int, int, int]],
        *,
        num_blocks: int,
        grid: tuple[int, int],
    ) -> list[tuple[int, int, int]]:
        """A third range: every block of every region, cut into token chunks.

        `pieces` is `(base address, whole length, block stride, device id)` per
        region; the two sides differ in where those come from and in nothing
        else. Region-major, then block, then run, then chunk, which is the
        order `_chunk_desc_ids` reads back.

        A run is one contiguous stretch of a block's bytes. Heads are the outer
        axis inside a block, so a token range is one run per head the region
        holds -- naming it as a single prefix would move the first head's
        tokens and leave the rest stale, and nothing would report it.
        """
        runs, chunks = grid
        out: list[tuple[int, int, int]] = []
        for base, whole_len, stride, device_id in pieces:
            run_span = whole_len // runs
            desc_len = run_span // chunks
            for block_id in range(num_blocks):
                start = base + block_id * stride
                for r in range(runs):
                    for c in range(chunks):
                        out.append(
                            (start + r * run_span + c * desc_len, desc_len, device_id)
                        )
        return out

    def _shard_chunk_grid(
        self, *, block_size: int, split: int, kv_runs: int = 1
    ) -> tuple[int, int] | None:
        """`(byte runs a chunk is spread over, chunks each run is cut into)`.

        None where a chunk comes out the whole span, so neither list grows and
        a transfer keeps naming whole spans -- which is what a block too small
        for the byte target, or a prefill chunk as wide as a span, asks for.

        The axis says where a chunk sits. A context cut gives an area the
        in-block token range [a * span, (a + 1) * span), so a chunk of it is
        one run of bytes. A head cut gives every area every token of some
        heads, so the span is the block and a chunk is one run per head -- and
        that band has to be one number, since the two lists carry one grid:
        regions that disagree, a draft's named past the target's, get no grid
        rather than one region's band standing for the rest. A piece narrower
        than a head gets none for the same reason.

        A packed block holds a token's K beside its V, so a token range is one
        run in each -- unless the peer pairing already read the two apart,
        which is what `kv_runs` reports.
        """
        if not self._chunk_mode:
            return None
        if self._kv_split_axis is KVSplitAxis.NON_HEAD:
            spans, heads_per_span = self._kv_areas, 1
        else:
            bands = {
                self._slice_head_bounds(
                    self.tp_rank,
                    self.topo.tp_size,
                    heads,
                    self._kv_areas,
                    self._kv_slices,
                    side="local",
                )[1]
                for heads in self._logical_region_kv_heads
                if heads is not None
            }
            if len(bands) != 1:
                return None
            spans, heads_per_span = 1, bands.pop()
        if heads_per_span % split:
            return None
        parts = self._kv_per_block // kv_runs
        span_tokens = block_size // spans
        # The widest region rather than the first: one layer may register
        # several of very different sizes -- a latent, its indexer, its scale
        # -- and the byte target belongs on the one carrying the bytes. Off
        # the first, registration order picks the grid, and a narrow region
        # there sizes a chunk past the span and drops the range entirely.
        widest = max(self.block_len_per_layer)
        bytes_per_token = widest // (span_tokens * heads_per_span * parts)
        assert bytes_per_token > 0, (
            f"RBLN NIXL: the widest region holds {widest}B per block, which is "
            f"under one byte per token for {span_tokens} token(s) of "
            f"{heads_per_span} head(s) in {parts} K/V part(s)"
        )
        chunk_tokens = kv_chunk_tokens(
            span_tokens=span_tokens,
            # A region holds one band of one span in one part, so this divides
            # out all three.
            bytes_per_token=bytes_per_token,
            prefill_step_tokens=(
                self.vllm_config.scheduler_config.max_num_batched_tokens
            ),
            chunk_bytes=connector_option(self.vllm_config, "chunk_bytes", 0),
            chunk_tokens=connector_option(self.vllm_config, "chunk_tokens", 0),
        )
        chunks = span_tokens // chunk_tokens
        # What keeps the arithmetic downstream free of the axis: a block is a
        # whole number of chunks however its spans are cut.
        assert chunk_tokens * chunks * spans == block_size
        if chunks == 1:
            return None
        return parts * heads_per_span // split, chunks

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
        chunk_grid: tuple[int, int] | None = None,
    ) -> tuple[int, np.ndarray]:
        """This engine's local descriptors, and the views of a block they name.

        A sliding window shortens the RDMA descriptor, not the tensor and not
        the host copy: regions stay Full-sized whatever the groups are. The
        list therefore carries whole blocks and then a `sliding_window`-length
        view over the same addresses, and a transfer picks the range its group
        needs (`_compute_desc_ids`). The prefix is enough because the tail a
        window writes back is never read, and the two groups draw block ids
        from disjoint pools.
        """
        if self._sw_ratio is None:
            if (
                registered_layer_names is None
                and peer_areas is None
                and split == 1
                and replica_fanout == 1
                and kv_runs == 1
            ):
                # No SWA view opt, whole-engine peer: upstream's Full-only
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
            "RBLN NIXL: SWA view-opt is not supported with pipeline "
            "parallelism or heterogeneous tensor parallelism"
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba layers."

        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        num_blocks = self.num_blocks * block_size_ratio
        t0 = time.perf_counter()
        blocks_data: list[tuple[int, int, int]] = []

        # A whole block is one range whatever it packs, since K and V are
        # adjacent in it. A window is a prefix and takes one inside each --
        # a single prefix runs past K's end and never reaches V.
        kv_per_block = self._kv_per_block
        length_divisors = [1, self._sw_ratio]
        pieces: list[tuple[int, int, int, int]] = []
        for divisor in length_divisors:
            for i, base_addr in enumerate(local_base_addresses):
                kv_block_len = (
                    self.get_backend_aware_kv_block_len(
                        layer_idx=i, first_split=True, mamba_view=False
                    )
                    // block_size_ratio
                )
                kv_runs = 1 if divisor == 1 else kv_per_block
                kv_stride = kv_block_len // kv_per_block
                desc_len = kv_block_len // kv_runs // divisor
                stride = self.block_len_per_layer[i] // block_size_ratio
                if divisor == 1:
                    pieces.append((base_addr, kv_block_len, stride, self.device_id))
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * stride
                    for kv in range(kv_runs):
                        blocks_data.append(
                            (addr + kv * kv_stride, desc_len, self.device_id)
                        )

        # Asked for here rather than handed in, so that the block size it is
        # derived from is the block size this list was built with.
        grid = self._shard_chunk_grid(block_size=block_size, split=1)
        if grid is not None:
            blocks_data += self._chunk_range_descs(
                pieces, num_blocks=num_blocks, grid=grid
            )

        logger.info(
            "RBLN NIXL: %d local descriptor(s) for this engine over %d region(s) "
            "x %d block(s): whole, a 1/%d sliding-window view, and %s. Built in "
            "%.1fms.",
            len(blocks_data),
            len(local_base_addresses),
            num_blocks,
            self._sw_ratio,
            f"a chunk range of {grid[0]} run(s) x {grid[1]} chunk(s)"
            if grid is not None
            else "no chunk range",
            (time.perf_counter() - t0) * 1000.0,
        )

        descs_data = _as_descs(blocks_data)
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
        chunk_grid: tuple[int, int] | None = None,
    ) -> tuple[int, np.ndarray]:
        """Prepare this shard's local descriptors, optionally twice over.

        `chunk_grid` is `(runs, chunks)`: how many byte runs one piece is
        spread over and how many token chunks each run is cut into. Given, a
        second range of descriptors follows the first over the same addresses,
        naming those chunks -- the shape `swa_view_opt` already
        uses for a shorter view of the same blocks. A transfer picks one range
        or the other by index, so both can be selected in one call.

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

        t0 = time.perf_counter()
        blocks_data: list[tuple[int, int, int]] = []
        # The whole piece first, then its chunks. `(1, 1)` names the piece
        # itself, so the first pass is what this built before `chunk_grid`.
        grids = [(1, 1)] if chunk_grid is None else [(1, 1), chunk_grid]
        for runs, chunks in grids:
            for region_id in region_ids:
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
                run_span = sub_len // runs
                desc_len = run_span // chunks
                for block_id in range(num_blocks):
                    for j in range(split):
                        # One entry per peer copy this piece goes to: the same
                        # bytes reach every replica (see _head_matched_desc),
                        # so the source repeats while the destination advances.
                        for _ in range(replica_fanout):
                            for kv in range(kv_runs):
                                piece = (
                                    base_addr
                                    + block_id * stride
                                    + kv * kv_stride
                                    + j * sub_len
                                )
                                for r in range(runs):
                                    for c in range(chunks):
                                        blocks_data.append(
                                            (
                                                piece + r * run_span + c * desc_len,
                                                desc_len,
                                                self.device_id,
                                            )
                                        )

        if chunk_grid is not None:
            runs, chunks = chunk_grid
            per_grid = 1 + runs * chunks
            logger.info(
                "RBLN NIXL: %d local descriptor(s) for this shard, %d of them a "
                "chunk range of %d run(s) x %d chunk(s) carrying %dB each; a "
                "whole piece carries %dB. Built in %.1fms.",
                len(blocks_data),
                len(blocks_data) // per_grid * (per_grid - 1),
                runs,
                chunks,
                blocks_data[-1][1],
                blocks_data[0][1],
                (time.perf_counter() - t0) * 1000.0,
            )
        descs_data = _as_descs(blocks_data)
        descs = self.nixl_wrapper.get_xfer_descs(descs_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            descs_data,
        )
