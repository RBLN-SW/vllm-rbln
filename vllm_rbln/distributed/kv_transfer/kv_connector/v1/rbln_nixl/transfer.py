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

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    SlidingWindowSpec,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    RblnNixlWorkerState,
)


def _chunk_desc_ids(
    *,
    start: int,
    positions: np.ndarray,
    num_blocks: int,
    block_id: int,
    per_block: int,
    grid: tuple[int, int],
    chunk_span: tuple[int, int],
) -> np.ndarray:
    """Descriptor ids for chunks `[lo, hi)` of one block, in `positions`.

    `start` is where the chunk range begins: one whole-block range before it on
    a shard's lists, two on the whole-engine lists where a sliding window's
    view sits in between. That offset is the only thing the two callers differ
    in, so the arithmetic lives here -- an index computed one way and a
    descriptor emitted the other lands on bytes nothing reports.
    """
    runs, chunks = grid
    lo, hi = chunk_span
    within = (
        np.arange(per_block, dtype=np.int64)[:, None, None] * (runs * chunks)
        + np.arange(runs, dtype=np.int64)[None, :, None] * chunks
        + np.arange(lo, hi, dtype=np.int64)[None, None, :]
    ).ravel()
    starts = start + (positions * num_blocks + block_id) * per_block * runs * chunks
    return (starts[:, None] + within[None, :]).ravel()


class RblnNixlTransferMixin(RblnNixlWorkerState):
    """Issuing one transfer: which descriptors a request needs, and how a
    completion is named.

    The transfer lifetime, and the only one that runs per request: registration
    produces the region table and the handshake produces the pairing, both once,
    and this reads what they left.
    """

    @contextmanager
    def _tail_viewed_as(self, valid_tokens: int | None, prompt_blocks: int):
        """Park how far a request's last block is filled, for upstream's call.

        `_compute_desc_ids` is what selects the descriptors, and it takes block
        ids and nothing else -- upstream's signature, with no room for a token
        count. A plain attribute suffices: one transfer reaches it twice, both
        synchronously, and a worker moves one request at a time on one thread
        (the read path on the worker's, the write path on the single writer's).
        """
        prev = self._request_tail
        self._request_tail = (valid_tokens, prompt_blocks)
        try:
            yield
        finally:
            self._request_tail = prev

    def _prompt_blocks(self, block_ids: BlockIds) -> int:
        """How many blocks the request holds, read off the group a chunk cuts.

        A sliding-window group's list is clipped to its own window, so summing
        the groups describes no request. Chunk mode registers against exactly
        one full-attention group, which is the one the token count describes.
        """
        return next(
            len(group)
            for g, group in enumerate(block_ids)
            if not isinstance(self._group_specs[g], SlidingWindowSpec)
        )

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
        # One number for the whole request, whichever list this call is for.
        tail = self._request_tail
        needed = (
            None
            if tail is None or self._chunk_grid is None
            else self._tail_chunks(
                tail[1], tail[0], chunks_per_span=self._chunk_grid[1]
            )
        )
        all_descs: list[np.ndarray] = []
        for g, group in enumerate(block_ids):
            if not group:
                continue
            is_sw = isinstance(self._group_specs[g], SlidingWindowSpec)
            group_arr = np.asarray(group)[None, :]
            if is_sw:
                # A window's group keeps every block whole: its descriptor is
                # the window, so there is no unwritten tail inside it.
                all_descs.append(
                    (region_ids * num_blocks + group_arr + num_full_descs).flatten()
                )
                continue
            if needed is None:
                all_descs.append((region_ids * num_blocks + group_arr).flatten())
                continue
            # The last block leaves the whole-block range and comes back as the
            # chunks that hold tokens, from the third range.
            assert self._chunk_grid is not None
            all_descs.append((region_ids * num_blocks + group_arr[:, :-1]).flatten())
            all_descs.append(
                _chunk_desc_ids(
                    start=2 * num_full_descs,
                    positions=np.arange(self.num_regions, dtype=np.int64),
                    num_blocks=num_blocks,
                    block_id=group[-1],
                    per_block=1,
                    grid=self._chunk_grid,
                    chunk_span=(0, needed),
                )
            )
        return np.concatenate(all_descs) if all_descs else np.empty(0, dtype=int)

    def _get_block_descs_ids_for_shard(
        self,
        engine_id: str,
        global_rank: int,
        num_blocks: int,
        block_ids: BlockIds,
        keep_spans: int | None = None,
    ) -> np.ndarray:
        region_group_ids = self._shard_region_group_ids[(engine_id, global_rank)]
        per_block = self._shard_descs_per_block[(engine_id, global_rank)]
        # Zero drops the last block from every region, whatever a position
        # names, which is what a head cut asks for -- there the token axis is
        # not spread over the regions at all.
        assert not keep_spans or per_block == 1
        # Converted once, not once per region: this runs per request, and every
        # region of a layer names the same group.
        group_arrays = [np.asarray(g, dtype=np.int64) for g in block_ids]
        desc_ids: list[np.ndarray] = []
        for region_id, group_id in enumerate(region_group_ids):
            group_arr = group_arrays[group_id]
            # Regions run area-minor within a layer, and a shard that leaves
            # part of a block out keeps every area (asserted where it
            # registers), so this position names the span whose token range the
            # request's last block does not reach.
            if keep_spans is not None and region_id % self._kv_areas >= keep_spans:
                group_arr = group_arr[:-1]
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

    def _chunk_descs_ids_for_shard(
        self,
        engine_id: str,
        global_rank: int,
        num_blocks: int,
        block_id: int,
        chunk_span: tuple[int, int],
        span_ix: int | None = None,
    ) -> np.ndarray:
        """Descriptors for chunks `[lo, hi)` of one block, in every region.

        `span_ix` narrows that to the regions holding one span of the block,
        which is what a context cut needs: an area IS a token range there, so
        the chunks of a partly-filled one belong to that area alone. None
        takes every region, which is a head cut, where every region holds the
        same token range of a different head band.

        Indexes the second range the shard's lists carry (see
        `_register_shard_local_xfer_handler`), which follows the whole-block
        range and holds `runs * chunks` entries where that one holds 1. Both
        lists are laid out the same way, so one index array serves either side.

        Empty where this shard has no such range, which is every peer the
        chunk grid was not derived for.
        """
        grid = self._shard_chunk_grids.get((engine_id, global_rank))
        lo, hi = chunk_span
        if grid is None or hi <= lo:
            return np.empty(0, dtype=np.int64)
        runs, chunks = grid
        if not 0 <= lo < hi <= chunks:
            raise RuntimeError(
                f"RBLN NIXL: chunk range [{lo}, {hi}) is outside the "
                f"{chunks} chunk(s) a block holds"
            )
        region_group_ids = self._shard_region_group_ids[(engine_id, global_rank)]
        per_block = self._shard_descs_per_block.get((engine_id, global_rank), 1)
        positions = np.arange(len(region_group_ids), dtype=np.int64)
        if span_ix is not None:
            positions = positions[positions % self._kv_areas == span_ix]
        return _chunk_desc_ids(
            # Where the chunk range starts: the whole-block range covers every
            # region, block and piece once.
            start=len(region_group_ids) * num_blocks * per_block,
            positions=positions,
            num_blocks=num_blocks,
            block_id=block_id,
            per_block=per_block,
            grid=grid,
            chunk_span=chunk_span,
        )

    def _tail_chunks(
        self,
        num_blocks: int,
        num_valid_tokens: int | None,
        *,
        chunks_per_span: int,
    ) -> int | None:
        """How many chunks of a request's last block hold its tokens.

        A block is `spans * chunks_per_span` chunks of equal token width
        however its spans are cut, so a last block filled to `rem` tokens has
        nothing above cdiv(rem, chunk). None keeps the whole block: that is
        what a full last block wants, and what one needing every chunk wants
        -- the same bytes in more descriptors is a loss.

        Rounds up, because every token counted has to reach the peer.
        """
        if not self._chunk_mode or not num_valid_tokens or num_blocks <= 0:
            return None
        rem = num_valid_tokens - (num_blocks - 1) * self.block_size
        if not 1 <= rem <= self.block_size:
            raise RuntimeError(
                f"RBLN NIXL D2D: a request holding {num_blocks} block(s) of "
                f"{self.block_size} reports {num_valid_tokens} token(s); its "
                "block list and its token count describe different KV."
            )
        chunks_per_block = self._spans_per_block * chunks_per_span
        needed = cdiv(rem, self.block_size // chunks_per_block)
        return needed if needed < chunks_per_block else None

    def _shard_descs_for_tokens(
        self,
        engine_id: str,
        global_rank: int,
        num_blocks: int,
        block_ids: BlockIds,
        *,
        num_valid_tokens: int | None,
        num_prompt_blocks: int,
    ) -> np.ndarray:
        """This shard's descriptors for `block_ids`, cut to the tokens held.

        The last block goes out as the chunks that hold tokens: the spans below
        the one its last token falls in from the block range, and that span's
        chunks from the chunk range. Both sides build this the same way, so the
        two lists still pair by position.

        `num_prompt_blocks` is the request's own block count, which is what
        the token count describes -- `block_ids` may have lost its prefix to a
        cache hit.
        """
        grid = self._shard_chunk_grids.get((engine_id, global_rank))
        chunks_per_span = grid[1] if grid is not None else 1
        needed = self._tail_chunks(
            num_prompt_blocks, num_valid_tokens, chunks_per_span=chunks_per_span
        )
        keep_spans = None if needed is None else needed // chunks_per_span
        descs = self._get_block_descs_ids_for_shard(
            engine_id, global_rank, num_blocks, block_ids, keep_spans=keep_spans
        )
        part = 0 if needed is None else needed % chunks_per_span
        if not part or not block_ids[0]:
            return descs
        # A head cut spreads no span axis over the regions, so every one of
        # them holds this block's chunks.
        span_ix = keep_spans if self._spans_per_block > 1 else None
        return np.concatenate(
            (
                descs,
                self._chunk_descs_ids_for_shard(
                    engine_id,
                    global_rank,
                    num_blocks,
                    block_ids[0][-1],
                    (0, part),
                    span_ix=span_ix,
                ),
            )
        )

    def _xfer_notif_id(
        self, engine_id: str, remote_request_id: str, remote_tp_size: int
    ) -> bytes:
        """Notification carrying how many of our ranks pair with one peer rank.

        NOTE(RBLN): upstream sends its own tensor-parallel size, which the peer
        divides by its own to learn how many of us to hear from before settling
        the request -- freeing its blocks on the read path, declaring them
        written on the write path. A finer pipeline on our side multiplies that,
        each stage pairing with the same peer rank for its own layers, so send
        the count in the unit the peer already divides by: ours times the peer's
        TP. The two agree whenever the pipelines match, which is every shape
        upstream assumes.

        Direction-free -- it only asks how much finer we are cut than the peer.
        """
        remote_pp = self._remote_pp_size.get(engine_id, 1)
        local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
        peers = max(1, self.world_size // remote_tp_size) * max(
            1, local_pp // remote_pp
        )
        return f"{remote_request_id}:{peers * remote_tp_size}".encode()
