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
from typing import Any, ClassVar

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
)
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


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

    # ------------------------------------------------------------------
    # Hybrid Full + SWA desc layout (RDMA payload only)
    # ------------------------------------------------------------------
    #
    # Regions are Full-sized. With swa_view_opt and an SWA group,
    # two desc ranges share the base addrs: [0, N) Full-length, [N, 2N)
    # sliding_window-length. SWA groups read only the prefix, so RDMA moves less
    # while the host copy still moves whole blocks. _compute_desc_ids routes each
    # group to its range; _sw_ratio None collapses to Full-only. Safe because the
    # tail SWA writes back is never read and the Full/SWA block-id pools are
    # disjoint.

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

    def register_local_xfer_handler(
        self,
        block_size: int,
        *,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
        peer_areas: list[int] | None = None,
        split: int = 1,
        region_ids: list[int] | None = None,
        replica_fanout: int = 1,
        chunk_grid: tuple[int, int] | None = None,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        if self._sw_ratio is None:
            if (
                registered_layer_names is None
                and peer_areas is None
                and split == 1
                and replica_fanout == 1
            ):
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
                replica_fanout=replica_fanout,
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
        assert not self.topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout (K and V in "
            "separate regions), not FlashInfer."
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba layers."

        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        num_blocks = self.num_blocks * block_size_ratio
        t0 = time.perf_counter()
        blocks_data: list[tuple[int, int, int]] = []

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses but `sliding_window`-sized.
        # _sw_ratio is not None here (the None case returned early above).
        length_divisors = [1, self._sw_ratio]
        pieces: list[tuple[int, int, int, int]] = []
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
                if divisor == 1:
                    pieces.append((base_addr, kv_block_len, stride, self.device_id))
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * stride
                    blocks_data.append((addr, kv_block_len, self.device_id))

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

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )

    def _register_shard_local_xfer_handler(
        self,
        block_size: int,
        registered_layer_names: tuple[str, ...] | list[str],
        peer_areas: list[int] | None = None,
        split: int = 1,
        region_ids: list[int] | None = None,
        replica_fanout: int = 1,
        chunk_grid: tuple[int, int] | None = None,
    ) -> tuple[int, list[tuple[int, int, int]]]:
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
        assert not self.topo.is_kv_layout_blocks_first, (
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
                # The pieces are consecutive byte ranges on this side -- it is
                # the REMOTE side that scatters.
                sub_len = kv_block_len // split
                run_span = sub_len // runs
                desc_len = run_span // chunks
                for block_id in range(num_blocks):
                    for j in range(split):
                        # One entry per peer copy this piece goes to: the same
                        # bytes reach every replica (see _head_matched_desc),
                        # so the source repeats while the destination advances.
                        for _ in range(replica_fanout):
                            piece = base_addr + block_id * stride + j * sub_len
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
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )
