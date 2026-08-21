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
# and the descriptor ids it reads with. The pairing those ids come from lives in
# base_worker and is covered by test_rbln_nixl_handshake.py.

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
        w._shard_descs_per_block = {}
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
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5]]
        )
        # region 0: blocks 2,5 -> descs (2*2,+1) and (5*2,+1); region 1 adds 10*2.
        assert list(descs) == [4, 5, 10, 11, 24, 25, 30, 31]

    def test_get_block_descs_ids_for_shard_empty_group(self):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._shard_region_group_ids = {("eng", 0): (0, 0)}
        w._shard_descs_per_block = {}
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
        w.vllm_config = MagicMock()
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
        w._shard_descs_per_block = {}
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

        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))

        assert w._recving_transfers["r0"] == []
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

    def test_single_stage_read_delegates_to_upstream(self):
        # A producer that advertised pp_size 1 reads through the upstream path:
        # the per-stage loop would key handles the non-PP registration never
        # wrote. Liveness is stamped first so TTL eviction sees it as active.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._engine_last_active = {}
        w._remote_pp_size = {}  # unknown engine defaults to a single stage
        w._overlapping_ranks = {}  # nothing narrowed -> upstream's handle covers it
        w.transfer_topo = MagicMock()
        meta = MagicMock()
        meta.remote.engine_id = "eng"

        with patch.object(NixlPullConnectorWorker, "_read_blocks_for_req") as base_read:
            w._read_blocks_for_req("r0", meta)

        base_read.assert_called_once_with("r0", meta)
        assert "eng" in w._engine_last_active

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
        w.vllm_config = MagicMock()
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
