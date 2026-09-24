# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Unit coverage: which descriptors one request needs -- the ids a block list
# turns into, over a region table registration already produced.

from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from tests.vllm.distributed.kv_connector.utils import (
    build_worker,
    sliding_window_spec,
    window_mode,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)


class TestComputeDescIds:
    # Routes block ids into the Full range (offset 0) or the SWA range (offset
    # num_full_descs) by group spec, expanded across regions.
    def test_none_ratio_delegates_to_super(self, monkeypatch):
        worker = build_worker(monkeypatch)  # _sw_ratio is None
        captured = []

        def super_impl(self, block_ids, dst, ratio, phys):
            captured.append((block_ids, dst, ratio, phys))
            return "super"

        monkeypatch.setattr(NixlBaseConnectorWorker, "_compute_desc_ids", super_impl)
        out = worker._compute_desc_ids([[0]], 4, None, 1)
        assert out == "super"
        assert captured == [([[0]], 4, None, 1)]

    def test_sw_group_shifted_by_full_desc_count_across_regions(self, monkeypatch):
        # Full group -> offset 0; SWA group -> offset num_full_descs. Each id is
        # also expanded across regions as region_id * num_blocks + id.
        worker = build_worker(monkeypatch)
        window_mode(worker, 2)
        worker.num_regions = 2
        full_spec = MagicMock()  # not a SlidingWindowSpec
        worker._group_specs = [
            full_spec,
            sliding_window_spec(block_size=64, sliding_window=32),
        ]

        # dst_num_blocks=4 -> num_full_descs = num_regions(2) * 4 = 8.
        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        # Full ids [0,1] -> r*4 + id: 0,1 then 4,5. SWA id [2] -> the granules
        # of that block, (r*4 + 2) * sw_ratio + {0,1} + 8.
        assert list(out) == [0, 1, 4, 5, 12, 13, 20, 21]

    def test_block_size_ratio_scales_block_span(self, monkeypatch):
        # A block_size_ratio widens the per-region block span (num_blocks *= ratio),
        # shifting both the region stride and the SWA offset.
        worker = build_worker(monkeypatch)
        window_mode(worker, 2)
        worker.num_regions = 1
        worker._group_specs = [sliding_window_spec(block_size=64, sliding_window=32)]

        # dst_num_blocks=2, ratio=2 -> num_blocks=4, num_full_descs = 1*4 = 4.
        out = worker._compute_desc_ids([[1]], 2, 2.0, 1)
        # single region: (0*4 + 1) * sw_ratio + {0,1} + offset(4) = 6, 7.
        assert list(out) == [6, 7]

    def test_rejects_multi_physical_blocks_per_logical(self, monkeypatch):
        # The SWA desc formula indexes physical blocks directly; the connector
        # pins one physical block per logical, so >1 is rejected.
        worker = build_worker(monkeypatch)
        window_mode(worker, 2)
        worker.num_regions = 1
        worker._group_specs = [sliding_window_spec(block_size=64, sliding_window=32)]
        with pytest.raises(AssertionError, match="physical_blocks_per_logical"):
            worker._compute_desc_ids([[0]], 4, None, 2)

    def test_empty_groups_yield_empty(self, monkeypatch):
        worker = build_worker(monkeypatch)
        window_mode(worker, 2)
        worker.num_regions = 1
        worker._group_specs = [MagicMock()]
        out = worker._compute_desc_ids([[]], 4, None, 1)
        assert out.size == 0

    @staticmethod
    def _hybrid_worker(monkeypatch, *, grid=(2, 2), tail=None):
        # Two groups over two regions, four blocks: num_full_descs = 8, so the
        # SWA range is [8, 24) at sw_ratio 2 a block, and chunks start at 24.
        worker = build_worker(monkeypatch, block_size=64)
        window_mode(worker, 2)
        worker.num_regions = 2
        worker._chunk_mode = True
        worker._kv_areas = 1
        worker._kv_split_axis = KVSplitAxis.HEAD
        worker._chunk_grid = grid
        worker._request_tail = tail
        worker._group_specs = [
            MagicMock(),  # full attention
            sliding_window_spec(block_size=64, sliding_window=32),
        ]
        return worker

    def test_a_partly_filled_last_block_goes_out_as_its_chunks(self, monkeypatch):
        # 2 chunks a block, a last block holding 1 of 64 tokens -> one chunk.
        # The block leaves the whole-block range and comes back from the third.
        worker = self._hybrid_worker(monkeypatch, tail=(65, 2))

        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        # Full group keeps block 0 (ids 0, 4) and drops block 1; block 1's
        # first chunk arrives at 24 + (r*4 + 1)*runs*chunks + run*chunks:
        # region 0 -> 24+4, 24+6; region 1 -> 24+20, 24+22. The window takes
        # the one granule 65 tokens put it in.
        assert list(out) == [0, 4, 28, 30, 44, 46, 12, 20]

    def test_the_windowed_group_takes_no_chunk_range(self, monkeypatch):
        # A granule holds no unwritten tail -- it is the window, not a block a
        # request stopped partway into -- so a windowed group's ids stay in the
        # window range and never reach the chunk range beyond it.
        worker = self._hybrid_worker(monkeypatch, tail=(65, 2))

        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        whole = worker.num_regions * 4
        assert list(out)[-2:] == [12, 20]
        assert all(whole <= i < whole * (1 + worker._sw_ratio) for i in out[-2:])

    def test_a_last_block_needing_every_chunk_is_left_whole(self, monkeypatch):
        # The benefit test: the same bytes in more descriptors is a loss, so
        # the ids are the ones this returned before the third range existed.
        worker = self._hybrid_worker(monkeypatch, tail=(128, 2))

        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        assert list(out) == [0, 1, 4, 5, 13, 21]

    def test_no_parked_tail_is_todays_ids(self, monkeypatch):
        # Nothing said how far the request fills its last block -- every block
        # goes whole, which is what every engine without chunk mode gets.
        worker = self._hybrid_worker(monkeypatch, tail=None)

        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        assert list(out) == [0, 1, 4, 5, 12, 13, 20, 21]

    def test_chunk_mode_alone_sends_the_windowed_group_whole(self, monkeypatch):
        # The knobs are separate now. With no window range the sliding-window
        # group's blocks go whole, and the chunk range sits one range earlier
        # -- right after the whole-block range, not after a window range that
        # was never built.
        worker = self._hybrid_worker(monkeypatch, tail=(65, 2))
        window_mode(worker, None)
        worker._has_swa = True

        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        whole = worker.num_regions * 4
        # Full group: block 0 whole (0, 4), block 1 as its first chunk from a
        # range that now starts at `whole`; SWA group: block 2 whole (2, 6).
        assert list(out) == [0, 4, whole + 4, whole + 6, whole + 20, whole + 22, 2, 6]

    def test_a_block_wide_kernel_spreads_a_granule_over_its_runs(self, monkeypatch):
        # Same spec, other geometry: a granule is a token range of a block, so
        # every run of it goes and the range is `runs` times as long.
        worker = self._hybrid_worker(monkeypatch, grid=None, tail=None)
        window_mode(worker, 2, runs=3)

        out = worker._compute_desc_ids([[0], [2]], 4, None, 1)

        whole = worker.num_regions * 4
        # SWA block 2, region r: whole + (r*4 + 2)*(3*2) + granule + run*2.
        assert list(out) == [0, 4] + [
            whole + (r * 4 + 2) * 6 + gran + run * 2
            for r in range(2)
            for gran in range(2)
            for run in range(3)
        ]

    def test_no_chunk_grid_is_todays_ids(self, monkeypatch):
        # A geometry whose span a chunk cannot cut registered no third range,
        # so no index may reach past the second.
        worker = self._hybrid_worker(monkeypatch, grid=None, tail=(65, 2))

        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        assert list(out) == [0, 1, 4, 5, 12, 20]


# What each layout puts in one block.
PACKED = 2  # rbln_custom_ops: (num_blocks, 2, H, 1, S, D)
SPLIT = 1  # rbln_triton_ops: (2, num_blocks, H, 1, S, D)


class TestDescIdsForAPackedBlock:
    """`_compute_desc_ids` indexes the lists the class above builds."""

    @staticmethod
    def _worker(kv_per_block, spec):
        w = object.__new__(RblnNixlPullConnectorWorker)
        window_mode(w, 2)
        w._kv_per_block = kv_per_block
        w.num_regions = 2
        w._chunk_grid = None
        w._request_tail = None
        w._group_specs = [spec]
        return w

    def _ids(self, kv_per_block, spec):
        return self._worker(kv_per_block, spec)._compute_desc_ids(
            [[1]],
            dst_num_blocks=4,
            block_size_ratio=None,
            physical_blocks_per_logical=1,
        )

    def test_a_whole_block_is_named_once_however_it_is_packed(self):
        full = MagicMock()
        # Region 0 block 1 -> id 1, region 1 block 1 -> id 5: the descriptor
        # is the block, so the packing does not enter the arithmetic.
        assert sorted(self._ids(PACKED, full)) == [1, 5]
        assert sorted(self._ids(SPLIT, full)) == [1, 5]

    def test_the_sliding_window_range_starts_past_every_whole_desc(self):
        sw = MagicMock(spec=SlidingWindowSpec)
        # num_regions(2) * num_blocks(4) whole descs come first; the window
        # range then names the `sw_ratio` kernel blocks that tile each block,
        # and one of those is a run of bytes however K and V sit in it.
        assert sorted(self._ids(PACKED, sw)) == [10, 11, 18, 19]
        assert sorted(self._ids(SPLIT, sw)) == [10, 11, 18, 19]


class TestTailChunks:
    # `_tail_chunks` over a 64-token block cut into 4 areas by a context cut,
    # so a chunk is 16 tokens at one per area and the boundaries the table
    # walks fall at 16 and its multiples; two per area makes it 8.

    @staticmethod
    def _worker(monkeypatch, *, chunked=True, axis=KVSplitAxis.NON_HEAD):
        w = build_worker(monkeypatch, block_size=64)
        w._kv_areas = 4
        w._kv_split_axis = axis
        w._chunk_mode = chunked
        return w

    @pytest.mark.parametrize(
        ("chunks_per_span", "num_valid_tokens", "expected"),
        [
            # One chunk per area, which is what every deployment whose spans a
            # chunk cannot cut keeps getting.
            (1, 129, 1),
            # 16 tokens exactly fills the first chunk; 17 reaches the second.
            (1, 144, 1),
            (1, 145, 2),
            (1, 176, 3),
            # A full last block has no tail to drop.
            (1, 192, None),
            # No count means no claim about the tail.
            (1, None, None),
            (1, 0, None),
            # Twice as many chunks: half an area is addressable now, so the
            # same one-token last block sends half of what it did.
            (2, 129, 1),
            (2, 136, 1),
            (2, 137, 2),
            (2, 144, 2),
            (2, 145, 3),
            # One token short of full still needs every chunk, and the same
            # bytes in more descriptors is a loss.
            (2, 191, None),
        ],
    )
    def test_the_tail_is_counted_in_chunks(
        self, monkeypatch, chunks_per_span, num_valid_tokens, expected
    ):
        w = self._worker(monkeypatch)
        assert (
            w._tail_chunks(3, num_valid_tokens, chunks_per_span=chunks_per_span)
            == expected
        )

    @pytest.mark.parametrize(
        ("chunks_per_span", "num_valid_tokens", "expected"),
        [(4, 129, 1), (4, 144, 1), (4, 145, 2), (4, 192, None)],
    )
    def test_a_head_cut_counts_the_whole_block_in_chunks(
        self, monkeypatch, chunks_per_span, num_valid_tokens, expected
    ):
        # A head cut gives every area every token, so the span a chunk cuts is
        # the block itself -- the same four chunks of 16 tokens, reached
        # without the area count.
        w = self._worker(monkeypatch, axis=KVSplitAxis.HEAD)

        assert (
            w._tail_chunks(3, num_valid_tokens, chunks_per_span=chunks_per_span)
            == expected
        )

    def test_chunk_mode_being_off_keeps_every_chunk(self, monkeypatch):
        # The same input the table answers 1 for.
        w = self._worker(monkeypatch, chunked=False)
        assert w._tail_chunks(3, 129, chunks_per_span=1) is None

    @pytest.mark.parametrize("num_valid_tokens", [128, 300])
    def test_a_count_that_does_not_fit_the_block_list_raises(
        self, monkeypatch, num_valid_tokens
    ):
        # 3 blocks of 64 hold between 129 and 192 tokens. Below or above that,
        # the list and the count describe different KV, and no descriptor
        # derived from either is safe.
        w = self._worker(monkeypatch)
        with pytest.raises(RuntimeError, match="different KV"):
            w._tail_chunks(3, num_valid_tokens, chunks_per_span=1)


class TestTheWindowsOwnGranules:
    """Which granules of a block a window's group names.

    `sw_ratio` 2 over a 64-token block: a granule is 32 tokens, so a block
    holds two and a window is one of them or one either side of a boundary.
    One region keeps the ids readable -- the window range starts at 4, and a
    granule is `4 + block * 2 + granule`.
    """

    @staticmethod
    def _ids(valid_tokens, blocks):
        w = object.__new__(RblnNixlPullConnectorWorker)
        window_mode(w, 2)
        w.block_size = 64
        w.num_regions = 1
        w._chunk_grid = None
        w._request_tail = (valid_tokens, None)
        w._group_specs = [MagicMock(spec=SlidingWindowSpec)]
        return list(
            w._compute_desc_ids(
                [blocks],
                dst_num_blocks=4,
                block_size_ratio=None,
                physical_blocks_per_logical=1,
            )
        )

    def test_a_window_inside_one_granule_names_that_one(self):
        # 64 tokens end a granule, so the last 32 are exactly the second one.
        assert self._ids(64, [3]) == [11]

    def test_a_window_across_a_granule_boundary_names_both(self):
        # 48 tokens put the window over tokens 16..47, which is the back of the
        # first granule and the front of the second -- both inside the last
        # block, so the block named before it contributes nothing.
        assert self._ids(48, [2, 3]) == [10, 11]

    def test_a_window_across_a_block_boundary_names_one_in_each(self):
        # 65 tokens put it over 33..64: the last granule of the earlier block
        # and the first of the later one. The two are not adjacent addresses,
        # which is the case a single block-wide descriptor cannot express.
        assert self._ids(65, [2, 3]) == [9, 10]

    def test_a_list_that_cannot_reach_the_earlier_granule_names_one(self):
        # Guard: a list shape the other cases do not build. The straddle wants
        # the block before and a one-block list lacks it -- and upstream clips
        # the producer's list and the consumer's alike, so a peer is short
        # there too and names one.
        assert self._ids(65, [3]) == [10]

    def test_a_straddle_is_what_makes_it_two_granules(self):
        # The straddle is what the count decides; the list decides only
        # whether the earlier granule is reachable, and every path equalises
        # the two lists from the tail before this runs.
        for tokens in (32, 64, 96):
            assert len(self._ids(tokens, [2, 3])) == 1
        for tokens in (33, 48, 65, 80):
            assert len(self._ids(tokens, [2, 3])) == 2
