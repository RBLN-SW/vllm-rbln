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
    a shard's lists, and on the whole-engine lists a window range as well,
    which holds the granules that tile a block where the whole-block range
    holds the block. That offset is the only thing the two callers differ in, so the
    arithmetic lives here -- an index computed one way and a descriptor emitted
    the other lands on bytes nothing reports.
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
    def _tail_viewed_as(
        self,
        valid_tokens: int | None,
        prompt_blocks: int | None,
        pieces: tuple[tuple[int, int | None, tuple[int, int]], ...] = (),
    ):
        """Park which part of a request's blocks this transfer names.

        `_compute_desc_ids` is what selects the descriptors, and it takes block
        ids and nothing else -- upstream's signature, with no room for a token
        count. A plain attribute suffices: one transfer reaches it twice, both
        synchronously, and a worker moves one request at a time on one thread
        (the read path on the worker's, the write path on the single writer's).

        A write naming the whole request parks how far its last block is filled
        and lets the tail be derived. One naming part of it -- a streamed batch
        -- parks the chunk ranges it decided on instead. They arrive by the same
        door because the block ids alone cannot say which of the two this is.

        The pieces are one side's, so a caller with two lists parks each
        around its own call.
        """
        prev = self._request_tail
        self._request_tail = (valid_tokens, prompt_blocks, pieces)
        try:
            yield
        finally:
            self._request_tail = prev

    def _counted_group(self, block_ids: BlockIds) -> int | None:
        """The group a token count, a chunk and a coverage range are counted in.

        A sliding-window group holds one block whatever the prompt length, so
        summing the groups -- or taking the first -- describes no request.
        Chunk mode and streaming both register against exactly one
        full-attention group, which is the one those counts belong to. With a
        single group there is nothing to select, and the specs need not be read
        to know it; an engine without such a group has nothing to count in.
        """
        if len(block_ids) == 1:
            return 0
        return next(
            (
                g
                for g, spec in enumerate(self._group_specs)
                if not isinstance(spec, SlidingWindowSpec)
            ),
            None,
        )

    def _prompt_blocks(self, block_ids: BlockIds) -> int | None:
        """How many blocks the request holds, read off the group a chunk cuts."""
        counted = self._counted_group(block_ids)
        return None if counted is None else len(block_ids[counted])

    def _window_granules(
        self, blocks: list[int], valid_tokens: int, sw_ratio: int
    ) -> list[tuple[int, int]]:
        """(block, the granule in it) the request's window lands in.

        A window is the last `sliding_window` tokens, so the group's last block
        holds the newest one and the token count says where inside. It spans one
        granule, or two when it straddles a boundary -- and the second of those
        sits in the block before, so a list too short to reach it names one and
        says nothing.

        What keeps the two ends agreeing is upstream's `get_sw_clipped_blocks`,
        which cuts an SWA group to `blocks_per_sw` = `cdiv(sliding_window,
        block_size) + 1` from the tail. Both ends of a read run it -- the
        producer in `request_finished` before it publishes the ids, the consumer
        in `update_state_after_alloc` on its own -- so neither is clipped here
        and a short list means the peer was short too.
        """
        sw = self.block_size // sw_ratio
        newest = (valid_tokens - 1) // sw
        oldest = max(0, valid_tokens - sw) // sw
        # The group's list ends at the block holding the newest token, so a
        # position in it is a logical block counted back from there.
        last_block = newest // sw_ratio
        return [
            (block_id, granule % sw_ratio)
            for i, block_id in enumerate(blocks)
            for granule in range(oldest, newest + 1)
            if granule // sw_ratio == last_block - (len(blocks) - 1 - i)
        ]

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        if not self._own_engine_layout:
            # Upstream's Full/SSM desc layout applies.
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

        # Both lists run region-major then block. A whole block is one
        # descriptor, and the window range that follows cuts that same block
        # into `runs x granules` (`_window_grid`). Without window mode that
        # range is absent, so the chunk range starts one range earlier.
        window = self._window_grid_cut
        window_units = window[0] * window[1] if window is not None else 0
        region_ids = np.arange(self.num_regions)[:, None]
        num_whole_descs = self.num_regions * num_blocks
        # One decision for the whole request, whichever list this call is for:
        # the chunk ranges a streamed batch named, or how many chunks of the
        # last block a whole-request write still owes.
        tail = self._request_tail
        pieces = tail[2] if tail is not None else ()
        if tail is None or pieces or self._chunk_grid is None:
            needed = None
        else:
            # A chunk range exists only where a full-attention group does, and
            # that group is what `_counted_group` selects.
            prompt_blocks = tail[1]
            assert prompt_blocks is not None
            needed = self._tail_chunks(
                prompt_blocks, tail[0], chunks_per_span=self._chunk_grid[1]
            )

        def whole(blocks: list[int]) -> np.ndarray:
            return (region_ids * num_blocks + np.asarray(blocks)[None, :]).flatten()

        def chunks_of(
            block_id: int, span_ix: int | None, chunk_span: tuple[int, int]
        ) -> np.ndarray:
            assert self._chunk_grid is not None
            # A context cut never reaches these lists: registration refuses one
            # beside a window range, so no piece parked here names a span.
            assert span_ix is None
            return _chunk_desc_ids(
                start=num_whole_descs * (1 + window_units),
                positions=np.arange(self.num_regions, dtype=np.int64),
                num_blocks=num_blocks,
                block_id=block_id,
                per_block=1,
                grid=self._chunk_grid,
                chunk_span=chunk_span,
            )

        all_descs: list[np.ndarray] = []
        for g, group in enumerate(block_ids):
            is_sw = isinstance(self._group_specs[g], SlidingWindowSpec)
            if is_sw:
                if window is None:
                    # Window mode registered no range: this group's blocks go
                    # whole, and the chunk range cuts the full-attention
                    # group's last block only.
                    if group:
                        all_descs.append(whole(group))
                    continue
                runs, granules_per_block = window
                # Nothing having said how many tokens the request holds leaves
                # nothing to say where its window is, so every granule goes --
                # which is the block itself.
                if group:
                    picked = (
                        [(b, gran) for b in group for gran in range(granules_per_block)]
                        if tail is None or tail[0] is None
                        else self._window_granules(group, tail[0], granules_per_block)
                    )
                    ids = (
                        region_ids * num_blocks
                        + np.asarray([b for b, _ in picked])[None, :]
                    )
                    granules = np.asarray([gran for _, gran in picked], dtype=np.int64)
                    # A granule is one run of bytes only where the kernel
                    # addresses the cache in window-wide blocks; otherwise the
                    # head cut spreads it, and every run of it goes.
                    all_descs.append(
                        (
                            ids[:, :, None] * window_units
                            + granules[None, :, None]
                            + np.arange(runs, dtype=np.int64)[None, None, :]
                            * granules_per_block
                            + num_whole_descs
                        ).ravel()
                    )
                continue
            # The full-attention group, which is the one a chunk cuts. Its
            # whole blocks may be empty while its pieces are not: a batch can
            # owe nothing but the rest of a block it half-wrote.
            keep = group if needed is None else group[:-1]
            if keep:
                all_descs.append(whole(keep))
            if needed is not None and group:
                # The last block leaves the whole-block range and comes back as
                # the chunks that hold tokens, from the third range.
                all_descs.append(chunks_of(group[-1], None, (0, needed)))
            all_descs += [
                chunks_of(block_id, span_ix, span) for block_id, span_ix, span in pieces
            ]
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
            # part of a block out keeps every area: the peer that would not is
            # refused in `_check_split_axis_constraints`. So this position names
            # the span whose token range the request's last block does not reach.
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
        grid = self._shard_chunk_grids[(engine_id, global_rank)]
        lo, hi = chunk_span
        if grid is None or hi <= lo:
            return np.empty(0, dtype=np.int64)
        runs, chunks = grid
        if not 0 <= lo < hi <= chunks:
            raise RuntimeError(
                f"RBLN NIXL: chunk range [{lo}, {hi}) is outside the "
                f"{chunks} chunk(s) a span holds"
            )
        region_group_ids = self._shard_region_group_ids[(engine_id, global_rank)]
        per_block = self._shard_descs_per_block[(engine_id, global_rank)]
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
        grid = self._shard_chunk_grids[(engine_id, global_rank)]
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
        self,
        engine_id: str,
        remote_request_id: str,
        remote_tp_size: int,
        *,
        count_stages: bool = True,
    ) -> bytes:
        """Notification carrying how many of our ranks pair with one peer rank.

        NOTE(RBLN): upstream sends its own tensor-parallel size, which the peer
        divides by its own to learn how many of us to hear from before settling
        the request. A finer pipeline on our side multiplies that, each stage
        pairing with the same peer rank for its own layers, so the read path
        sends the count in the unit the peer divides by: ours times the peer's
        TP. The write path must not -- 0.26's writer accounting multiplies OUR
        `pp_size` back in, taking it from the producer's own kv_transfer_params,
        so the stages would be counted twice.
        """
        peers = max(1, self.world_size // remote_tp_size)
        if count_stages:
            remote_pp = self._remote_pp_size.get(engine_id, 1)
            local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
            peers *= max(1, local_pp // remote_pp)
        return f"{remote_request_id}:{peers * remote_tp_size}".encode()
