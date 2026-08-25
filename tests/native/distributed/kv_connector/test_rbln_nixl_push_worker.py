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

# The write path: that it inherits the shared machinery next to the upstream
# push classes, that the writer thread survives the D2D registration deferral,
# that a request is settled only once every writer of it has reported, and that
# a peer holding part of what we do is written per shard.

import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnectorWorker,
    NixlPushConnectorWorker,
)

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker as pw
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl import (
    RblnNixlPullConnectorWorker,
    RblnNixlPushConnectorWorker,
    RblnNixlWorkerBase,
)


class _FakeThread:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True


@pytest.fixture
def fake_thread(monkeypatch):
    """Capture the writer thread instead of running it."""
    created = []

    def factory(**kwargs):
        created.append(_FakeThread(**kwargs))
        return created[-1]

    monkeypatch.setattr(pw.threading, "Thread", factory)
    return created


def _push_worker():
    w = object.__new__(RblnNixlPushConnectorWorker)
    w._push_writer_thread = None
    w.tp_rank = 0
    # The direct path with no sliding-window view: what the shared guards read
    # before an engine is registered.
    w.use_host_buffer = False
    w._sw_ratio = None
    # __init__ never ran, so the writer state shutdown() reaches through
    # __del__ is absent; silence it rather than leak an unraisable at GC.
    w.shutdown = lambda: None
    return w


class TestInheritance:
    def test_shared_machinery_sits_next_to_the_upstream_push_classes(self):
        # The pairing layer must come first so its overrides win, and the
        # upstream push class must follow so the write path is the inherited one.
        assert RblnNixlPushConnectorWorker.__mro__[1:4] == (
            RblnNixlWorkerBase,
            NixlPushConnectorWorker,
            NixlBaseConnectorWorker,
        )

    def test_teardown_reaches_the_writer_thread(self):
        # Nothing of ours overrides shutdown, so the upstream push one runs and
        # joins the writer; an override that forgot to chain would strand it.
        assert RblnNixlPushConnectorWorker.shutdown is NixlPushConnectorWorker.shutdown

    def test_the_read_path_does_not_leak_in(self):
        # The shared layer is direction-free, so nothing of the read path may
        # arrive through it.
        assert not hasattr(RblnNixlPushConnectorWorker, "_read_blocks_for_req")

    def test_the_submission_itself_stays_the_upstream_one(self):
        # We choose the peers and the descriptors; posting the transfer and
        # its failure handling remain the base's.
        assert (
            RblnNixlPushConnectorWorker._xfer_blocks
            is NixlPushConnectorWorker._xfer_blocks
        )


class TestMlaOnTheWritePath:
    # MLA lives in the shared layer, so the write path gets it by inheritance.
    # What that has to mean in practice is that the guards fire here too: a
    # peer at a different TP degree would otherwise have its bands derived from
    # the configured KV-head count -- which these models report as their
    # attention head count -- and the descriptors would be plausible and wrong.
    def test_unequal_tp_is_refused_when_writing_a_latent_cache(self):
        worker = _push_worker()
        worker.use_mla = True
        worker.transfer_topo = SimpleNamespace(tp_ratio=lambda peer: 2, tp_size=1)

        with pytest.raises(RuntimeError, match="heterogeneous tensor"):
            worker._check_mla_constraints(SimpleNamespace(), remote_tp_size=2)

    def test_a_peer_whose_chiplet_geometry_differs_is_refused(self):
        # Positional pairing needs both sides to expand a logical region the
        # same way; each derives it from its own device buffers, so a mismatch
        # moves the wrong bytes without failing.
        worker = _push_worker()
        worker.use_mla = True
        worker.transfer_topo = SimpleNamespace(tp_ratio=lambda peer: 1, tp_size=1)
        worker._kv_areas, worker._kv_slices = 4, 1

        with pytest.raises(RuntimeError, match="chiplet geometry"):
            worker._check_mla_constraints(
                SimpleNamespace(kv_areas=2, kv_slices=1), remote_tp_size=1
            )

    def test_a_matching_peer_passes(self):
        worker = _push_worker()
        worker.use_mla = True
        worker.transfer_topo = SimpleNamespace(tp_ratio=lambda peer: 1, tp_size=1)
        worker._kv_areas, worker._kv_slices = 4, 1

        worker._check_mla_constraints(
            SimpleNamespace(kv_areas=4, kv_slices=1), remote_tp_size=1
        )


class TestPushWriterStart:
    def test_finalize_starts_the_writer_after_the_deferred_registration(
        self, monkeypatch, fake_thread
    ):
        # D2D registers at finalize, so the start upstream hangs off
        # register_kv_caches has to happen here instead -- and only after the
        # memory it writes into exists.
        order = []
        monkeypatch.setattr(
            RblnNixlWorkerBase,
            "_register_kv_caches_impl",
            lambda self, pending: order.append("register"),
        )
        worker = _push_worker()
        worker._pending_kv_caches = {"layer.0": object()}

        worker.finalize_kv_cache_registration()

        order.append("start" if fake_thread[0].started else "not-started")
        assert order == ["register", "start"]
        assert fake_thread[0].kwargs["name"] == "nixl-push-writer"

    def test_start_is_skipped_when_the_upstream_path_already_ran(self, fake_thread):
        # Host staging reaches the upstream register_kv_caches, so the thread is
        # already up by the time finalize no-ops; replacing it would orphan one.
        worker = _push_worker()
        worker._pending_kv_caches = None
        existing = object()
        worker._push_writer_thread = existing

        worker.finalize_kv_cache_registration()

        assert worker._push_writer_thread is existing
        assert fake_thread == []


class TestPerShardWrite:
    """One WRITE per peer rank this one pairs with, each over that peer's own
    narrowed descriptors -- the mirror of the read path's per-shard loop."""

    @staticmethod
    def _writing_worker(ranks):
        w = _push_worker()
        w._remote_pp_size = {"eng": 1}
        w._overlapping_ranks = {"eng": list(range(ranks))}
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = 1
        w.world_size = 1
        w.num_blocks = 8
        w.dst_num_blocks = {"eng": 8}
        w._engine_last_active = {}
        w.kv_cache_config = MagicMock(kv_cache_groups=[0])
        # single group, 2 regions per shard
        w._shard_region_group_ids = {("eng", r): (0, 0) for r in range(ranks)}
        w._shard_descs_per_block = {}
        w.src_xfer_handles_by_remote = {("eng", r, 16): 100 + r for r in range(ranks)}
        w.dst_xfer_side_handles = {"eng": {r: 200 + r for r in range(ranks)}}
        w._sending_transfers = defaultdict(list)
        w._sending_transfers_lock = threading.Lock()
        topo = MagicMock()
        topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
        )
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

    def test_one_write_per_paired_rank_with_that_rank_s_handles(self):
        worker = self._writing_worker(ranks=2)

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        calls = worker.nixl_wrapper.make_prepped_xfer.call_args_list
        assert len(calls) == 2
        assert [c.args[0] for c in calls] == ["WRITE", "WRITE"]
        assert (calls[0].args[1], calls[0].args[3]) == (100, 200)
        assert (calls[1].args[1], calls[1].args[3]) == (101, 201)
        # The request settles only once every WRITE it issued has completed.
        assert len(worker._sending_transfers["r0"]) == 2

    def test_a_peer_a_whole_engine_handle_describes_is_delegated(self, monkeypatch):
        # Nothing narrowed for this engine: the base's own route covers it.
        delegated = []
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, meta: delegated.append(req_id),
        )
        worker = self._writing_worker(ranks=2)
        worker._overlapping_ranks = {}

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        assert delegated == ["r0"]
        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 0

    def test_every_write_carries_the_same_writer_count(self):
        worker = self._writing_worker(ranks=2)

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        # world_size 1 against a TP-1 peer: one writer, so the count divides
        # out to one on the far side.
        assert notifs == {b"r0:1"}

    def test_a_partial_prefix_hit_keeps_our_matching_tail(self):
        # The consumer registered only the last block of a three-block prompt,
        # so this side has to send its LAST block, not its first.
        worker = self._writing_worker(ranks=1)

        worker._xfer_blocks_for_req("r0", self._meta(([5, 6, 7],), ([9],)))

        call = worker.nixl_wrapper.make_prepped_xfer.call_args_list[0]
        local_descs, remote_descs = call.args[2], call.args[4]
        assert len(local_descs) == len(remote_descs)
        # 2 regions x 1 block, and block 7 is the one that maps to descs 7 / 15
        # of an 8-block, 2-region shard.
        assert sorted(local_descs) == [7, 15]

    def test_a_failed_submission_releases_the_handle(self):
        worker = self._writing_worker(ranks=1)
        worker.nixl_wrapper.make_prepped_xfer.side_effect = RuntimeError("nope")
        worker._log_failure = MagicMock()
        worker.xfer_stats = MagicMock()

        worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

        # Nothing to invalidate on this side, so the handle is all there is to
        # let go of, and the request must not be left holding one.
        assert worker._sending_transfers["r0"] == []
        worker.xfer_stats.record_failed_transfer.assert_called_once()


class TestTrimToConsumerBlocks:
    def test_more_blocks_than_the_consumer_registered_trims_the_head(self):
        trimmed = RblnNixlPushConnectorWorker._trim_to_consumer_blocks(
            ([1, 2, 3, 4],), ([7, 8],)
        )
        assert trimmed == ([3, 4],)

    def test_a_consumer_asking_for_more_than_we_hold_is_an_error(self):
        with pytest.raises(AssertionError, match="cannot be aligned"):
            RblnNixlPushConnectorWorker._trim_to_consumer_blocks(([1],), ([7, 8],))


class TestWriterCountAccounting:
    """The consumer must not settle a request while some writer is still going.

    The count rides in the notification scaled by this side's TP, so the same
    field also has to keep working for a transfer the upstream path submitted.
    """

    def test_the_counter_survives_a_request_it_has_not_seen(self, monkeypatch):
        # The one place __init__ runs: every other test here builds the worker
        # with object.__new__ and hands the counter in already made, so a plain
        # dict would pass all of them and KeyError on the first notification.
        monkeypatch.setattr(RblnNixlWorkerBase, "__init__", lambda self, *a: None)
        worker = RblnNixlPushConnectorWorker(MagicMock(), "eng", MagicMock())
        worker.shutdown = lambda: None  # the writer state __del__ reaches is absent
        worker._writer_counts_by_req["r0"] += 1
        assert worker._writer_counts_by_req == {"r0": 1}

    @staticmethod
    def _receiving_worker(world_size):
        w = _push_worker()
        w.world_size = world_size
        w._writer_counts_by_req = defaultdict(int)
        w._reqs_to_send = {}
        w._reqs_to_process = set()
        w._recving_metadata = {"r0": object()}
        w._pending_completion_notifs = queue.Queue()
        return w

    @pytest.fixture
    def handed_through(self, monkeypatch):
        """What reaches the base, in order."""
        seen = []

        def fake_base(self):
            while True:
                try:
                    seen.append(self._pending_completion_notifs.get_nowait())
                except queue.Empty:
                    return set()

        monkeypatch.setattr(NixlPushConnectorWorker, "_get_new_notifs", fake_base)
        return seen

    def test_only_the_last_of_four_writers_reaches_the_base(self, handed_through):
        # Four peer ranks writing into this one: the base settles on whatever it
        # sees, so it may see exactly one notification.
        worker = self._receiving_worker(world_size=1)
        for _ in range(4):
            worker._pending_completion_notifs.put(b"r0:4")

        worker._get_new_notifs()

        assert handed_through == [b"r0:4"]

    def test_the_count_carries_across_steps(self, handed_through):
        # Writers finish whenever they finish; the tally has to survive the
        # steps in between.
        worker = self._receiving_worker(world_size=1)
        for _ in range(3):
            worker._pending_completion_notifs.put(b"r0:4")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"r0:4")
        worker._get_new_notifs()
        assert handed_through == [b"r0:4"]

    def test_a_single_writer_is_passed_straight_through(self, handed_through):
        # Equal TP with no pipeline: the upstream path submits it and puts its
        # own tensor-parallel size in the notification, which divides out to one.
        worker = self._receiving_worker(world_size=4)
        worker._pending_completion_notifs.put(b"r0:4")

        worker._get_new_notifs()

        assert handed_through == [b"r0:4"]

    @pytest.mark.parametrize(
        "notif, reason",
        [
            (b"HB:engine-1", "heartbeat"),
            (b"other:4", "not a request we are receiving"),
        ],
    )
    def test_notifications_the_base_owns_are_untouched(
        self, handed_through, notif, reason
    ):
        worker = self._receiving_worker(world_size=1)
        worker._pending_completion_notifs.put(notif)

        worker._get_new_notifs()

        assert handed_through == [notif], reason
        assert worker._writer_counts_by_req == {}

    def test_our_own_outbound_request_is_left_to_the_base(self, handed_through):
        # A request this rank is sending is the base's own accounting, even
        # though the notification looks identical.
        worker = self._receiving_worker(world_size=1)
        worker._reqs_to_process = {"r0"}
        worker._pending_completion_notifs.put(b"r0:4")

        worker._get_new_notifs()

        assert handed_through == [b"r0:4"]
        assert worker._writer_counts_by_req == {}

    def test_a_finished_request_drops_its_tally(self, monkeypatch):
        # A retry reuses the request id, so a leftover partial count would
        # settle it early.
        worker = self._receiving_worker(world_size=1)
        worker._writer_counts_by_req["r0"] = 2
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_finished",
            lambda self: (set(), {"r0"}),
        )

        worker.get_finished()

        assert worker._writer_counts_by_req == {}


class TestSaveBeforeWriteInvariant:
    """The host copy for a step runs after this call, so a request may not be
    staged for it and handed to the writer at the same time."""

    @staticmethod
    def _worker(use_host_buffer):
        w = _push_worker()
        w.use_host_buffer = use_host_buffer
        return w

    @staticmethod
    def _meta(saves, pushes):
        return SimpleNamespace(
            reqs_to_save=dict.fromkeys(saves, object()),
            push_finished_blocks=dict.fromkeys(pushes, ([1],)),
        )

    def test_the_same_request_in_both_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        worker = self._worker(use_host_buffer=True)
        with pytest.raises(AssertionError, match="unfilled buffer"):
            worker.start_load_kv(self._meta(saves=["r0"], pushes=["r0"]))

    def test_a_step_apart_is_the_normal_case(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "start_load_kv",
            lambda self, metadata: seen.append(metadata),
        )
        worker = self._worker(use_host_buffer=True)
        meta = self._meta(saves=["r0"], pushes=["r1"])

        worker.start_load_kv(meta)

        assert seen == [meta]

    def test_direct_transfer_skips_the_check(self, monkeypatch):
        # No staging buffer to be caught half-filled; the device KV is settled
        # by the time a request finishes.
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        worker = self._worker(use_host_buffer=False)
        worker.start_load_kv(self._meta(saves=["r0"], pushes=["r0"]))


class TestHandlesArePublishedAtOnce:
    """The engine thread settles a request once every handle it can see is
    done. A half-published set lets it settle early, and the peers landing
    afterwards become a second completion for a request already forgotten."""

    def test_nothing_is_visible_until_every_peer_is_submitted(self):
        worker = TestPerShardWrite._writing_worker(ranks=4)
        seen_midway = []

        def submit(*args, **kwargs):
            # Stand in for the engine thread polling between submissions.
            seen_midway.append(len(worker._sending_transfers.get("r0", [])))
            return object()

        worker.nixl_wrapper.make_prepped_xfer.side_effect = submit

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

        # Nothing was published while the four were being prepped, and all
        # four appeared together.
        assert seen_midway == [0, 0, 0, 0]
        assert len(worker._sending_transfers["r0"]) == 4

    def test_a_failed_peer_does_not_hide_the_ones_that_went_out(self):
        worker = TestPerShardWrite._writing_worker(ranks=4)
        worker._log_failure = MagicMock()
        worker.xfer_stats = MagicMock()
        calls = {"n": 0}

        def submit(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("nope")
            return object()

        worker.nixl_wrapper.make_prepped_xfer.side_effect = submit

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

        # The three that were transferred still have to be waited on.
        assert len(worker._sending_transfers["r0"]) == 3
        worker.xfer_stats.record_failed_transfer.assert_called_once()


class TestReplicaFanOut:
    """A chiplet replica is a fine thing to read from and a bad thing to be the
    only destination written to: the peer's other chiplets would keep serving
    what was there before. Reading takes one copy, writing takes them all."""

    @staticmethod
    def _worker(cls, *, areas=4, slices=4):
        w = object.__new__(cls)
        w.shutdown = lambda: None
        w._kv_areas, w._kv_slices = areas, slices
        w._sw_ratio = None
        w.use_host_buffer = False
        w.device_id = 0
        w.block_len_per_layer = [4096] * 2
        topo = MagicMock()
        topo.total_num_kv_heads = 8
        topo.tp_size = 1
        topo.tp_rank = 0
        topo.tp_ratio.return_value = -4  # the peer is cut four ways
        w.transfer_topo = topo
        w.tp_rank = 0
        w.get_backend_aware_kv_block_len = lambda **kw: 1024
        return w

    @staticmethod
    def _peer_meta(areas=4, slices=2):
        # A TP4 rank of an 8-head model: 2 heads, each duplicated across two of
        # the four chiplet areas -> two copies of every slice.
        meta = MagicMock()
        meta.kv_areas, meta.kv_slices = areas, slices
        meta.num_blocks = 1
        meta.kv_caches_base_addr = [1000 * (i + 1) for i in range(areas)]
        meta.block_lens = [1024] * areas
        return meta

    def test_the_read_path_names_one_copy(self):
        worker = self._worker(RblnNixlPullConnectorWorker)
        assert worker._peer_replica_fanout(self._peer_meta(), 4) == 1

    def test_the_write_path_names_every_copy(self):
        worker = self._worker(RblnNixlPushConnectorWorker)
        assert worker._peer_replica_fanout(self._peer_meta(), 4) == 2

    def test_a_peer_without_copies_is_the_same_either_way(self):
        # areas == slices: nothing is duplicated, so there is only one place to
        # put the bytes and the two directions agree.
        for cls in (RblnNixlPullConnectorWorker, RblnNixlPushConnectorWorker):
            worker = self._worker(cls)
            assert worker._peer_replica_fanout(self._peer_meta(slices=4), 4) == 1

    def test_host_staging_has_no_copies_to_fan_out_to(self):
        worker = self._worker(RblnNixlPushConnectorWorker)
        worker.use_host_buffer = True
        assert worker._peer_replica_fanout(self._peer_meta(), 4) == 1

    @pytest.mark.parametrize(
        "cls, expected_regions",
        [
            # Peer regions are (logical * areas + slice * replicas + copy).
            # Reading takes copy 0 of the slice; writing takes 0 and 1.
            (RblnNixlPullConnectorWorker, [1000]),
            (RblnNixlPushConnectorWorker, [1000, 2000]),
        ],
    )
    def test_descriptors_land_on_those_copies(self, cls, expected_regions):
        worker = self._worker(cls)
        fanout = worker._peer_replica_fanout(self._peer_meta(), 4)
        meta = self._peer_meta()

        descs = worker._head_matched_desc(
            region_id=0,
            logical_r=0,
            area_l=0,
            geom=(0, 2, 1),  # our area carries heads 0-1, no duplication
            peer=(0, 2, 2, 2),  # peer: heads from 0, 2 per slice, 2 copies, 2 slices
            areas_r=4,
            remote_bases=meta.kv_caches_base_addr,
            remote_lens=meta.block_lens,
            device_id=0,
            num_blocks=1,
        )

        assert [addr for addr, _len, _dev in descs] == expected_regions
        assert len(descs) == fanout


class TestTheThreeListsAgree:
    """remote dlist, local dlist and the descriptor ids index one another, so
    they have to expand a block by the same factor. Checking the fan-out in
    isolation missed that the remote builder's own tally still assumed the
    head pieces alone."""

    @staticmethod
    def _worker(cls):
        # One layer, K and V, each expanded across four chiplet areas.
        regions = 8
        w = object.__new__(cls)
        w.shutdown = lambda: None
        w._kv_areas, w._kv_slices = 4, 4
        w._sw_ratio = None
        w.use_host_buffer = False
        w.device_id = 0
        w.engine_id = "eng-local"
        w.tp_rank = 0
        w.num_blocks = 2
        w.block_size = 16
        w._has_mamba = False
        w.num_regions = regions
        w.block_len_per_layer = [512] * regions
        w.local_seen_layer_names = ["l0"]
        w.kv_caches_base_addr = {
            "eng-local": {0: [10_000 * (i + 1) for i in range(regions)]}
        }
        topo = MagicMock()
        topo.total_num_kv_heads = 8
        topo.tp_size = 1
        topo.is_kv_layout_blocks_first = False
        topo.tp_ratio.return_value = -4
        w.transfer_topo = topo
        w.get_backend_aware_kv_block_len = lambda **kw: 512
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.get_xfer_descs.side_effect = lambda data, _t: data
        w.nixl_wrapper.prep_xfer_dlist.return_value = 7
        w.nixl_memory_type = "VRAM"
        return w

    @staticmethod
    def _peer():
        # A TP4 rank: 2 heads, each duplicated across two of its four areas.
        meta = MagicMock()
        meta.kv_areas, meta.kv_slices = 4, 2
        meta.num_blocks = 2
        meta.device_id = 0
        # Same shape as ours: K and V, each across its four areas.
        meta.kv_caches_base_addr = [100_000 * (i + 1) for i in range(8)]
        meta.block_lens = [512] * 8
        return meta

    @pytest.mark.parametrize(
        "cls", [RblnNixlPullConnectorWorker, RblnNixlPushConnectorWorker]
    )
    def test_both_sides_describe_the_same_number_of_pieces(self, cls):
        worker = self._worker(cls)
        peer = self._peer()
        fanout = worker._peer_replica_fanout(peer, 4)
        split = worker._peer_head_split(peer, 4)
        areas = worker._fan_in_peer_areas(0, 4)

        remote = worker._build_head_matched_remote(peer, 0, 4, peer_areas=areas)
        _handle, local = worker._register_shard_local_xfer_handler(
            worker.block_size,
            ("l0",),
            peer_areas=areas,
            split=split,
            replica_fanout=fanout,
        )

        assert len(remote) == len(local)
        # And that length is the fan-out times what one copy would have needed.
        assert len(remote) % fanout == 0
        assert (cls is RblnNixlPushConnectorWorker) == (fanout == 2)


class TestDelegatedRouteAlignment:
    """A peer a whole-engine handle already describes is written by the base,
    whose alignment truncates the longer block list and keeps its head. Under a
    partial prefix hit the consumer registered its tail, so the head is wrong."""

    @staticmethod
    def _worker(*, blocks_per_logical=1):
        w = TestPerShardWrite._writing_worker(ranks=1)
        w._overlapping_ranks = {}  # nothing narrowed: the base's route
        w.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1,
            remote_block_size=16,
            remote_physical_blocks_per_logical=blocks_per_logical,
        )
        return w

    @staticmethod
    def _local_reaching_base(monkeypatch, worker, meta):
        seen: dict[str, tuple] = {}
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, meta: seen.update(local=meta.local_physical_block_ids),
        )
        worker._xfer_blocks_for_req("r0", meta)
        return seen["local"]

    def test_the_producer_tail_is_what_reaches_the_base(self, monkeypatch):
        # The consumer kept one block: its cache covered everything before it.
        local = self._local_reaching_base(
            monkeypatch, self._worker(), TestPerShardWrite._meta(([5, 6, 7],), ([9],))
        )

        assert local == ([7],)

    def test_an_expanded_remote_list_is_left_to_the_base(self, monkeypatch):
        # Guard: past the identity expansion the two lengths count different
        # things, so trimming one against the other would be arithmetic on
        # unlike units.
        local = self._local_reaching_base(
            monkeypatch,
            self._worker(blocks_per_logical=2),
            TestPerShardWrite._meta(([5, 6, 7],), ([9],)),
        )

        assert local == ([5, 6, 7],)

    def test_without_a_prefix_hit_nothing_moves(self, monkeypatch):
        # Guard: equal lists are the ordinary case and must pass through.
        local = self._local_reaching_base(
            monkeypatch,
            self._worker(),
            TestPerShardWrite._meta(([5, 6, 7],), ([1, 2, 3],)),
        )

        assert local == ([5, 6, 7],)
