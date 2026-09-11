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

# Unit coverage: the read path -- which producer shards this rank reads from,
# and the descriptor ids it reads with. The pairing those ids index is covered by
# test_rbln_nixl_handshake.py, and the ids themselves by test_rbln_nixl_transfer.py.

import queue
import threading
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from tests.native.distributed.kv_connector.utils import (
    mock_vllm_config,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    RblnNixlConnectorMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)


def _sliding_window_spec():
    spec = MagicMock(spec=SlidingWindowSpec)
    spec.block_size, spec.sliding_window = 16, 8
    return spec


class TestShardReadPath:
    # Per-stage read loop + shard descriptor ids.

    def test_get_block_descs_ids_for_shard(self):
        # 0-based descs over the shard's regions; group_id picks the block
        # group. Single group -> region_id * num_blocks + block_ids[0].
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._shard_region_group_ids = {("eng", 1): (0, 0, 0, 0)}  # 4 regions, group 0
        # Written together with the group ids by _register_shard_xfer_state, so
        # a shard the read path can reach always has both.
        w._shard_descs_per_block = {("eng", 1): 1}
        w._shard_chunk_grids = {}
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5]]
        )
        # region r contributes r*10 + [2,5]: [2,5, 12,15, 22,25, 32,35]
        assert list(descs) == [2, 5, 12, 15, 22, 25, 32, 35]

    def test_get_block_descs_ids_for_shard_with_split(self):
        """split=2: each block becomes two consecutive descriptors, because
        both dlists are laid out region-major, then block, then piece."""
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._shard_region_group_ids = {("eng", 1): (0, 0)}  # 2 regions
        w._shard_descs_per_block = {("eng", 1): 2}
        w._shard_chunk_grids = {}
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5]]
        )
        # region 0: blocks 2,5 -> descs (2*2,+1) and (5*2,+1); region 1 adds 10*2.
        assert list(descs) == [4, 5, 10, 11, 24, 25, 30, 31]

    @staticmethod
    def _trim_worker():
        # 4 regions over 2 chiplet areas: positions 0,2 are area 0 and 1,3 are
        # area 1, which is the relation the trim indexes by.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._shard_region_group_ids = {("eng", 1): (0, 0, 0, 0)}
        w._shard_descs_per_block = {("eng", 1): 1}
        w._shard_chunk_grids = {}
        w._kv_areas = 2
        w._kv_split_axis = KVSplitAxis.NON_HEAD
        return w

    def test_the_tail_block_leaves_the_areas_above_it_out(self):
        w = self._trim_worker()
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5, 7]], keep_spans=1
        )
        # Area 0 keeps all three blocks; area 1 drops the last, and the list
        # stays region-major so the two sides still pair by position.
        assert list(descs) == [2, 5, 7, 12, 15, 22, 25, 27, 32, 35]

    def test_no_tail_is_todays_list(self):
        # Same input, and the descriptors a full last block still needs.
        w = self._trim_worker()
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5, 7]], keep_spans=None
        )
        assert list(descs) == [2, 5, 7, 12, 15, 17, 22, 25, 27, 32, 35, 37]

    def test_a_single_block_drops_the_higher_areas_entirely(self):
        # The whole request is that one block, so an area above its last token
        # contributes no descriptor at all rather than an empty range.
        w = self._trim_worker()
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[4]], keep_spans=1
        )
        assert list(descs) == [4, 24]

    @classmethod
    def _chunk_worker(cls, grid):
        # The same 4 regions over 2 areas, now with a chunk range: a 64-token
        # block is two 32-token areas, each cut into `grid[1]` chunks.
        w = cls._trim_worker()
        w.block_size = 64
        w._chunk_mode = True
        w._shard_chunk_grids = {("eng", 1): grid}
        return w

    def test_the_last_block_goes_out_as_the_chunks_that_hold_tokens(self):
        # 129 tokens over 3 blocks: one token of the last, which is one 16-token
        # chunk of area 0. So the block range drops that block everywhere and
        # the chunk range names its first chunk -- in area 0's regions only,
        # since an area IS a token range here.
        w = self._chunk_worker((1, 2))

        descs = w._shard_descs_for_tokens(
            "eng",
            1,
            10,
            [[2, 5, 7]],
            num_valid_tokens=129,
            num_prompt_blocks=3,
        )

        # 40 whole descriptors come first (4 regions x 10 blocks), then two
        # chunks per block: 40 + (0*10 + 7)*2 for region 0 and 40 + (2*10 +
        # 7)*2 for region 2.
        assert list(descs) == [2, 5, 12, 15, 22, 25, 32, 35, 54, 94]

    def test_a_tail_that_lands_on_a_span_boundary_needs_no_chunk(self):
        # 145 tokens: 17 of the last block, which needs both 16-token chunks of
        # area 0 and none of area 1. Whole spans come from the block range, so
        # this is the trim on its own -- the chunk range names nothing.
        w = self._chunk_worker((1, 2))

        descs = w._shard_descs_for_tokens(
            "eng", 1, 10, [[2, 5, 7]], num_valid_tokens=145, num_prompt_blocks=3
        )

        assert list(descs) == [2, 5, 7, 12, 15, 22, 25, 27, 32, 35]

    def test_a_last_block_needing_every_chunk_goes_whole(self):
        # A full last block: the chunk range would cost the same bytes in more
        # descriptors.
        w = self._chunk_worker((1, 2))

        descs = w._shard_descs_for_tokens(
            "eng", 1, 10, [[2, 5, 7]], num_valid_tokens=192, num_prompt_blocks=3
        )

        assert list(descs) == [2, 5, 7, 12, 15, 17, 22, 25, 27, 32, 35, 37]

    def test_a_head_cut_sends_the_last_blocks_chunks_in_every_region(self):
        # A head cut gives every region every token of some heads, so the
        # chunks of a partly-filled block are in all of them and the block
        # range drops that block everywhere. Two chunks a block here, so one
        # token of the last block is its first chunk in each of two runs.
        w = self._chunk_worker((2, 2))
        w._kv_split_axis = KVSplitAxis.HEAD

        descs = w._shard_descs_for_tokens(
            "eng", 1, 10, [[2, 5, 7]], num_valid_tokens=129, num_prompt_blocks=3
        )

        # Four chunk descriptors a block, so region r's block 7 starts at
        # 40 + (r*10 + 7)*4 and its two runs are two apart.
        assert list(descs) == [2, 5, 12, 15, 22, 25, 32, 35] + [
            68,
            70,
            108,
            110,
            148,
            150,
            188,
            190,
        ]

    @pytest.mark.parametrize("grid", [(1, 2), (2, 4), (3, 8)])
    def test_a_head_cut_never_drops_the_last_block_from_a_region(self, grid):
        """Every region carries the last block, whatever it is filled to.

        A head cut spreads no span axis over the regions, so `keep_spans` has
        to stay 0 and the block range has to drop the last block from all of
        them or none. `_tail_chunks` is what holds it there: it answers None
        once every chunk is needed, and `needed // chunks_per_span` is 0 for
        every answer below that. Raise that cut by one and `keep_spans`
        becomes 1, whose position predicate is the context-cut shape -- the
        regions above area 0 drop the block while `part` is 0, so no chunk
        descriptor replaces it and that KV never leaves. The pre-transfer
        length check compares index counts, so nothing fails: the peer settles
        on a request whose last block is partly missing.
        """
        w = self._chunk_worker(grid)
        w._kv_split_axis = KVSplitAxis.HEAD
        runs, chunks = grid
        num_blocks, last_block = 10, 7
        regions = len(w._shard_region_group_ids[("eng", 1)])
        chunk_range = regions * num_blocks

        for rem in range(1, w.block_size + 1):
            descs = set(
                w._shard_descs_for_tokens(
                    "eng",
                    1,
                    num_blocks,
                    [[2, 5, last_block]],
                    num_valid_tokens=2 * w.block_size + rem,
                    num_prompt_blocks=3,
                ).tolist()
            )
            for region in range(regions):
                block_ix = region * num_blocks + last_block
                base = chunk_range + block_ix * runs * chunks
                carried = ({block_ix} | set(range(base, base + runs * chunks))) & descs
                assert carried, (
                    f"{rem} token(s) in the last block: region {region} sends "
                    f"none of it (grid {grid})"
                )

    def test_a_full_last_block_is_one_descriptor_a_region_under_a_head_cut(self):
        """The collapse point of the head-cut sweep, named on its own.

        A last block needing every chunk goes whole, so each region spends one
        descriptor on it rather than `runs * chunks`. This is the value the
        cut is asserting, and the case a raised cut turns into silent loss.
        """
        w = self._chunk_worker((2, 4))
        w._kv_split_axis = KVSplitAxis.HEAD

        descs = w._shard_descs_for_tokens(
            "eng",
            1,
            10,
            [[2, 5, 7]],
            num_valid_tokens=3 * w.block_size,
            num_prompt_blocks=3,
        )

        assert list(descs) == [2, 5, 7, 12, 15, 17, 22, 25, 27, 32, 35, 37]

    def test_a_peer_without_a_chunk_range_still_drops_whole_areas(self):
        # What a deployment whose spans a chunk cannot cut keeps getting: the
        # area is the unit, and no index reaches a range its lists lack.
        w = self._chunk_worker(None)

        descs = w._shard_descs_for_tokens(
            "eng", 1, 10, [[2, 5, 7]], num_valid_tokens=129, num_prompt_blocks=3
        )

        assert list(descs) == [2, 5, 7, 12, 15, 22, 25, 27, 32, 35]

    def test_a_head_band_split_cannot_be_trimmed(self):
        # With more than one descriptor per block the position no longer names
        # an area, so the two would index different things.
        w = self._trim_worker()
        w._shard_descs_per_block = {("eng", 1): 2}
        w._shard_chunk_grids = {}
        with pytest.raises(AssertionError):
            w._get_block_descs_ids_for_shard(
                "eng", 1, num_blocks=10, block_ids=[[2, 5]], keep_spans=1
            )

    def test_get_block_descs_ids_for_shard_empty_group(self):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._shard_region_group_ids = {("eng", 0): (0, 0)}
        w._shard_descs_per_block = {("eng", 0): 1}
        w._shard_chunk_grids = {}
        descs = w._get_block_descs_ids_for_shard("eng", 0, num_blocks=4, block_ids=[[]])
        assert descs.size == 0

    @staticmethod
    def _read_worker(pp_size):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._remote_pp_size = {"eng": pp_size}
        # Full-model decode: every stage overlaps. Decode-PP subsets this (see
        # test_reads_only_overlapping_stages).
        w._overlapping_ranks = {"eng": list(range(pp_size))}
        # The read notification carries how many of us read each producer rank,
        # which counts our pipeline ranks too (see _xfer_notif_id).
        w.vllm_config = mock_vllm_config()
        w.vllm_config.parallel_config.pipeline_parallel_size = 1
        w._has_mamba = False  # non-Mamba scope: _apply_prefix_caching end-trims
        w.world_size = 1
        w.num_blocks = 8
        w.dst_num_blocks = {"eng": 8, "local": 8}
        w._recving_transfers = defaultdict(list)
        w._engine_last_active = {}
        # What upstream's failure path reads: it logs with the engine id, looks the
        # request's metadata up, queues the failure and the invalidated blocks.
        w.engine_id = "local"
        w._recving_metadata = {}
        w._invalid_block_ids = queue.Queue()
        w._failed_recv_reqs = queue.Queue()
        w._is_hma_required = False
        w.xfer_stats = MagicMock()
        # single group, 2 regions per shard
        w.kv_cache_config = MagicMock(kv_cache_groups=[0])
        w._shard_region_group_ids = {("eng", r): (0, 0) for r in range(pp_size)}
        w._shard_descs_per_block = {("eng", r): 1 for r in range(pp_size)}
        # Off, as the default is; the chunk tests turn it on.
        w._shard_chunk_grids = {}
        w._chunk_mode = False
        w._recv_valid_tokens = {}
        w.src_xfer_handles_by_remote = {("eng", r, 16): 100 + r for r in range(pp_size)}
        w.dst_xfer_side_handles = {"eng": {r: 200 + r for r in range(pp_size)}}
        w._remote_agents = {"eng": {r: f"agent{r}" for r in range(pp_size)}}
        topo = MagicMock()
        topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1, remote_block_size=16, remote_physical_blocks_per_logical=1
        )
        topo.tp_ratio.return_value = 1
        topo.block_size_ratio.return_value = 1
        w.transfer_topo = topo
        w._logical_to_remote_kernel_block_ids = lambda ids, _n: ids
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.make_prepped_xfer.side_effect = lambda *a, **k: object()
        return w

    @staticmethod
    def _meta(local_ids, remote_ids):
        remote = MagicMock()
        remote.engine_id = "eng"
        remote.request_id = "r0"
        remote.block_ids = remote_ids
        meta = MagicMock()
        meta.remote = remote
        meta.local_physical_block_ids = local_ids
        return meta

    def test_reads_every_stage(self):
        # pp_size=2 -> one prepped READ per stage, each with its own shard
        # handles; the request accrues one transfer handle per stage.
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # stage 0 uses local handle 100 / remote 200; stage 1 -> 101 / 201.
        calls = w.nixl_wrapper.make_prepped_xfer.call_args_list
        assert calls[0].args[1] == 100 and calls[0].args[3] == 200
        assert calls[1].args[1] == 101 and calls[1].args[3] == 201
        # completion: one handle per stage -> req done only when both finish.
        assert len(w._recving_transfers["r0"]) == 2

    def test_a_failed_stage_leaves_no_handles_behind(self):
        # A failed stage takes the request with it, so nothing may stay in
        # flight: get_finished drops the metadata, and a leftover handle
        # completing later would report that request against metadata now gone.
        w = self._read_worker(pp_size=3)
        first = object()
        w.nixl_wrapper.make_prepped_xfer.side_effect = [first, RuntimeError("boom")]
        w._log_failure = MagicMock()

        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))

        assert w._recving_transfers["r0"] == []
        # The mock is installed to keep the log quiet; assert on it too, or a
        # failure that reports nothing to an operator reads as a clean abort.
        assert (
            w._log_failure.call_args.kwargs["failure_type"] == "transfer_setup_failed"
        )
        # The stage that already submitted is released, and the third is never
        # submitted -- the request is failed, not partially read.
        w.nixl_wrapper.release_xfer_handle.assert_called_once_with(first)
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # Reported failed exactly once, which is what the engine counts.
        assert w._failed_recv_reqs.qsize() == 1
        assert w._failed_recv_reqs.get_nowait() == "r0"

    def test_prefix_hit_notifies_each_stage_no_read(self):
        # Full prefix hit (empty local list): no read, one notif per stage.
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert w.nixl_wrapper.send_notif.call_count == 2
        assert len(w._recving_transfers["r0"]) == 0

    def test_a_dropped_notification_still_reaches_the_other_stages(self):
        # One notif per stage is new here -- upstream sends one -- so a stage
        # that throws must not take the rest with it. The peer whose notif was
        # lost keeps its blocks until the lease expires; the others are freed.
        w = self._read_worker(pp_size=3)
        w._log_failure = MagicMock()
        w.nixl_wrapper.send_notif.side_effect = [RuntimeError("dropped"), None, None]

        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))

        assert w.nixl_wrapper.send_notif.call_count == 3
        assert w._log_failure.call_args.kwargs["remote_pp_rank"] == 0
        w.xfer_stats.record_failed_notification.assert_called_once()

    def test_partial_prefix_hit_trims_remote_per_stage(self):
        # Partial hit: D allocated only the uncached suffix (1 block) while the
        # remote prompt is 3 blocks, so the remote is end-trimmed to the last
        # block -> local/remote desc counts match per stage and the read fires.
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([[7]], [[3, 4, 7]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        for c in w.nixl_wrapper.make_prepped_xfer.call_args_list:
            local_descs, remote_descs = c.args[2], c.args[4]
            assert len(local_descs) == len(remote_descs)
            # 2 regions per shard x 1 (trimmed) block = 2 descriptors.
            assert len(remote_descs) == 2
        assert len(w._recving_transfers["r0"]) == 2

    def test_the_producers_token_count_shortens_the_read(self):
        # The same read with the producer's count in hand: its
        # last block holds one token, so only the first of the two areas is
        # read and each stage issues half the descriptors.
        w = self._read_worker(pp_size=2)
        w._chunk_mode = True
        w._kv_areas = 2
        w._kv_split_axis = KVSplitAxis.NON_HEAD
        w.block_size = 16
        w._recv_valid_tokens = {"r0": 33}

        w._read_blocks_for_req("r0", self._meta([[7]], [[3, 4, 7]]))

        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        for c in w.nixl_wrapper.make_prepped_xfer.call_args_list:
            local_descs, remote_descs = c.args[2], c.args[4]
            assert len(local_descs) == len(remote_descs)
            assert len(remote_descs) == 1
        # Consumed, so a later step cannot read it against another block list.
        assert w._recv_valid_tokens == {}

    def test_a_count_arriving_before_the_read_is_kept(self):
        # A request whose handshake is still running is read on a later step,
        # whose metadata no longer lists it -- so the counts accumulate.
        w = self._read_worker(pp_size=1)
        w._recv_valid_tokens = {"earlier": 5}
        meta = RblnNixlConnectorMetadata()
        meta.valid_tokens = {"r0": 33}

        with patch.object(NixlPullConnectorWorker, "start_load_kv"):
            w.start_load_kv(meta)

        assert w._recv_valid_tokens == {"earlier": 5, "r0": 33}

    def test_single_stage_read_delegates_to_upstream(self):
        # A producer that advertised pp_size 1 reads through the upstream path:
        # the per-stage loop would key handles the non-PP registration never
        # wrote. Liveness is stamped first so TTL eviction sees it as active.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._engine_last_active = {}
        w._remote_pp_size = {}  # unknown engine defaults to a single stage
        w._overlapping_ranks = {}  # nothing narrowed -> upstream's handle covers it
        w._chunk_mode = False
        w._recv_valid_tokens = {}
        w.transfer_topo = MagicMock()
        meta = MagicMock()
        meta.remote.engine_id = "eng"

        with patch.object(NixlPullConnectorWorker, "_read_blocks_for_req") as base_read:
            w._read_blocks_for_req("r0", meta)

        base_read.assert_called_once_with("r0", meta)
        assert "eng" in w._engine_last_active

    def test_a_windowed_engine_reads_chunked_through_upstreams_route(self):
        # Chunk mode normally means per-shard descriptors, and reaching this
        # route without them is a bug. A sliding window is the exception: its
        # view gave the whole-engine list the extra range, and the per-shard
        # lists cannot name its two KV groups at all.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._engine_last_active = {}
        w._remote_pp_size = {}
        w._overlapping_ranks = {}
        w._chunk_mode = True
        w._sw_ratio = 8
        w._chunk_grid = None
        w._request_tail = None
        w._group_specs = [MagicMock()]  # one full-attention group
        w._recv_valid_tokens = {"r0": 17}
        w.transfer_topo = MagicMock()
        meta = MagicMock()
        meta.remote.engine_id = "eng"
        meta.remote.block_ids = [[1, 2]]

        seen = []
        with patch.object(
            NixlPullConnectorWorker,
            "_read_blocks_for_req",
            lambda self, req_id, m: seen.append(self._request_tail),
        ):
            w._read_blocks_for_req("r0", meta)

        # The token count and the request's own block count, parked for the
        # length of that call: upstream's `_compute_desc_ids` is what selects
        # the descriptors and its signature has no room for either.
        assert seen == [(17, 2)]
        assert w._request_tail is None

    def test_a_chunked_engine_without_a_window_may_not_reach_it(self):
        # The other side of the same rule: nothing else leaves a chunked
        # engine on a list that cannot leave part of a block out.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._engine_last_active = {}
        w._remote_pp_size = {}
        w._overlapping_ranks = {}
        w._chunk_mode = True
        w._sw_ratio = None
        w._recv_valid_tokens = {}
        w.transfer_topo = MagicMock()
        meta = MagicMock()
        meta.remote.engine_id = "eng"

        with pytest.raises(AssertionError):
            w._read_blocks_for_req("r0", meta)

    @pytest.mark.parametrize(
        ("local_tp", "remote_tp", "local_pp", "remote_pp", "expected_readers"),
        [
            (1, 1, 1, 1, 1),  # one to one
            (4, 1, 1, 1, 4),  # TP fan-out: 4 of us read the one producer rank
            (1, 4, 1, 1, 1),  # TP fan-in: we alone read each producer rank
            (1, 1, 4, 1, 4),  # PP fan-out: our 4 stages read the one rank
            (1, 4, 4, 1, 4),  # both: fan-in leaves 1 per rank, times 4 stages
            (4, 1, 4, 1, 16),  # both fanning out
            (1, 1, 4, 2, 2),  # finer pipeline by a factor of two
        ],
    )
    def test_read_notif_counts_every_reader_of_a_producer_rank(
        self, local_tp, remote_tp, local_pp, remote_pp, expected_readers
    ):
        # The producer divides the number we send by its own tensor-parallel
        # size to get the count it waits for, so send it in that unit.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.world_size = local_tp
        w._remote_pp_size = {"eng": remote_pp}
        w.vllm_config = mock_vllm_config()
        w.vllm_config.parallel_config.pipeline_parallel_size = local_pp

        notif = w._xfer_notif_id("eng", "req-1", remote_tp).decode()
        req_id, sent = notif.rsplit(":", 1)
        assert req_id == "req-1"
        assert int(sent) % remote_tp == 0
        assert int(sent) // remote_tp == expected_readers

    def test_delegates_when_the_handshake_narrowed_nothing(self):
        # A peer serving our whole band with our own head split: upstream's
        # whole-engine handle covers it, so the read path must delegate.
        w = self._read_worker(pp_size=1)
        w._overlapping_ranks = {}
        with patch.object(NixlPullConnectorWorker, "_read_blocks_for_req") as base_read:
            w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        base_read.assert_called_once()
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0

    def test_reverse_pipeline_uses_the_shard_path(self):
        # A producer without pipeline parallelism still serves only part of our
        # band when ours is the finer pipeline, so the per-shard lists apply
        # even though its pp_size is 1.
        w = self._read_worker(pp_size=1)
        w._overlapping_ranks = {"eng": [0]}
        with patch.object(NixlPullConnectorWorker, "_read_blocks_for_req") as base_read:
            w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        base_read.assert_not_called()
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 1

    def test_reads_only_overlapping_stages(self):
        # Decode-PP (m=4, n=2): this rank owns 2 of the 4 producer stages, so
        # it READs only its overlapping stages; the other two are neither read
        # nor notified (another decode rank owns and notifies them).
        w = self._read_worker(pp_size=4)
        w._overlapping_ranks = {"eng": [0, 1]}  # this rank's band = stages 0,1
        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # only stages 0,1 local handles used (100, 101); never 102/103.
        used = {c.args[1] for c in w.nixl_wrapper.make_prepped_xfer.call_args_list}
        assert used == {100, 101}
        assert w.nixl_wrapper.send_notif.call_count == 0
        assert len(w._recving_transfers["r0"]) == 2

    def test_prefix_hit_notifies_only_overlapping_stages(self):
        # Full prefix hit under decode-PP: notify only this rank's stages.
        w = self._read_worker(pp_size=4)
        w._overlapping_ranks = {"eng": [0, 1]}
        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert w.nixl_wrapper.send_notif.call_count == 2  # only stages 0,1
        notified = {c.args[0] for c in w.nixl_wrapper.send_notif.call_args_list}
        assert notified == {"agent0", "agent1"}


class TestUpstreamReachesTheOverride:
    # Each case enters through the upstream method that calls our override, so a
    # rename surfaces here rather than leaving the direct-call tests above green.

    def test_start_load_kv_reaches_the_per_shard_read(self):
        # Upstream's start_load_kv calls _read_blocks_for_req; if that call
        # moves, every transfer falls back to the whole-engine path.
        w = TestShardReadPath._read_worker(pp_size=2)
        w._logical_to_kernel_block_ids = lambda ids: ids
        w._handshake_lock = threading.RLock()
        w._ready_requests = queue.Queue()
        meta = TestShardReadPath._meta([[1, 2]], [[3, 4]])
        meta.local_block_ids = [[1, 2]]

        NixlPullConnectorWorker.start_load_kv(w, MagicMock(reqs_to_recv={"r0": meta}))

        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        assert len(w._recving_transfers["r0"]) == 2

    def test_upstream_read_path_reaches_our_desc_ids(self):
        # Upstream's _read_blocks calls _compute_desc_ids, and our single-stage
        # delegation runs it; if that call moves, SWA groups read Full-length
        # descriptors.
        w = TestShardReadPath._read_worker(pp_size=1)
        w._overlapping_ranks = {}  # nothing narrowed -> delegate to upstream
        w._sw_ratio = 2
        w._chunk_grid = None  # no chunk range: the two ranges as before
        w._request_tail = None
        w._group_specs = [_sliding_window_spec()]
        w.num_regions = 2
        w._physical_blocks_per_logical_kv_block = 1
        w.engine_id = "local"
        w.src_xfer_handles_by_block_size = {16: 900}
        w.block_size = 16
        w.src_xfer_handles_by_tp_ratio = {}
        w.tp_rank = 0
        w.use_mla = False
        w.tp_mappings = {
            "eng": TPMapping(
                source_ranks_per_group=((0,),),
                all_source_ranks=(0,),
                rank_to_attention_slot={0: 0},
                rank_offset_factor=0,
            )
        }

        w._read_blocks_for_req("r0", TestShardReadPath._meta([[1]], [[1]]))

        # The SWA descs live in the second range, which starts past every id
        # upstream's own formula can produce.
        ids = w.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        assert min(ids) >= w.num_regions * w.dst_num_blocks["eng"]


class TestReadMarksTheEngineActive:
    # TTL eviction (base, engine_ttl=3600s by default) tears a remote engine's
    # state down once its timestamp goes stale, so reading from it has to
    # refresh the timestamp -- on the notif-only path too.

    def test_read_marks_remote_active(self):
        w = TestShardReadPath._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", TestShardReadPath._meta([[1, 2]], [[3, 4]]))
        assert "eng" in w._engine_last_active

    def test_prefix_hit_read_marks_remote_active(self):
        # A full prefix hit sends notifs only -- still activity.
        w = TestShardReadPath._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", TestShardReadPath._meta([], [[3, 4]]))
        assert "eng" in w._engine_last_active
