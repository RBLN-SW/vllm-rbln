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
from unittest.mock import ANY, MagicMock

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
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
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
    # What __init__ leaves on a D2D worker with the SWA view-opt off, which is
    # the shape the pairing predicates read before an engine is registered.
    w.use_host_buffer = False
    w._sw_ratio = None
    # Off, as the environment variable is; the early-write tests turn it on.
    w._early_push_enabled = False
    w._early_sends = set()
    w._empty_receives = set()
    w._early_transfers = defaultdict(list)
    w._recving_transfers = {}
    w._batches_queued = defaultdict(int)
    w._batches_done = defaultdict(int)
    w._batches_expected = {}
    w._stream_total = {}
    w._issued_hwm = defaultdict(int)
    w._physical_blocks_per_logical_kv_block = 1
    w._recving_metadata = {}
    w._writer_counts_by_req = defaultdict(int)
    w._coverage_by_req = defaultdict(lambda: defaultdict(list))
    w._reqs_to_send = {}
    w._reqs_to_process = set()
    w.consumer_notification_counts_by_req = {}
    w._pending_early_offers = {}
    w._sending_transfers = defaultdict(list)
    w._sending_transfers_lock = threading.Lock()
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

    def test_only_the_write_direction_carries_the_direction_flag(self):
        # The flag decides whether descriptors fan out over a peer's replicas and
        # it joins the compatibility hash, so the shared layer holding it would
        # make the read path claim both.
        assert RblnNixlPushConnectorWorker._writes_into_peer is True
        assert RblnNixlWorkerBase._writes_into_peer is False

    def test_teardown_releases_the_parked_handles_and_reaches_the_writer_thread(
        self, monkeypatch
    ):
        # Handles held back from the completion accounting are on no other
        # path out, and the upstream shutdown -- which joins the writer -- only
        # runs if this override chains to it.
        chained = []
        monkeypatch.setattr(
            NixlPushConnectorWorker, "shutdown", lambda self: chained.append(True)
        )
        worker = _push_worker()
        worker.nixl_wrapper = MagicMock()
        worker._early_sends = {"r0"}
        worker._early_transfers = defaultdict(list, {"r0": [[7, 8]]})

        RblnNixlPushConnectorWorker.shutdown(worker)

        assert worker.nixl_wrapper.release_xfer_handle.call_count == 2
        assert worker._early_transfers == {}
        assert chained == [True]

    def test_the_read_path_does_not_leak_in(self):
        # The shared layer is direction-free, so nothing of the read path may
        # arrive through it.
        assert not hasattr(RblnNixlPushConnectorWorker, "_read_blocks_for_req")

    def test_the_submission_itself_stays_the_upstream_one(self):
        # We choose the peers and the descriptors; posting the transfer and
        # its failure handling remain upstream's.
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

    def test_upstreams_own_submission_path_reaches_our_override(self):
        # Calling the override directly keeps passing if upstream renames the
        # hook it dispatches to, so one case has to arrive through the caller.
        worker = self._writing_worker(ranks=1)
        worker._ensure_d_handshake = lambda *_args: True
        worker._physical_blocks_per_logical_kv_block = 1

        NixlPushConnectorWorker._do_start_push_kv(
            worker,
            request_id="r0",
            local_block_ids=[1, 2],
            registration_data={
                "decode_engine_id": "eng",
                "local_block_ids": [3, 4],
                "decode_host": "",
                "decode_port": 0,
                "request_id": "r0",
                "decode_tp_size": 1,
            },
        )

        assert worker.nixl_wrapper.make_prepped_xfer.call_args.args[0] == "WRITE"

    def test_a_peer_a_whole_engine_handle_describes_is_delegated(self, monkeypatch):
        # Nothing narrowed for this engine: upstream's own route covers it.
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
        # out to one on the far side. Ahead of it, the range of consumer
        # blocks this write filled -- both of the two it registered.
        assert notifs == {b"RBLNS:0:0:2:r0:1"}

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

    def test_a_failed_submission_leaves_no_handle_on_the_request(self):
        # Failing before a handle exists: there is nothing to release, and the
        # request must not be left holding one.
        worker = self._writing_worker(ranks=1)
        worker.nixl_wrapper.make_prepped_xfer.side_effect = RuntimeError("nope")
        worker._log_failure = MagicMock()
        worker.xfer_stats = MagicMock()

        worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

        assert worker._sending_transfers["r0"] == []
        worker.xfer_stats.record_failed_transfer.assert_called_once()
        worker.nixl_wrapper.release_xfer_handle.assert_not_called()

    def test_a_failure_after_the_handle_exists_releases_it(self):
        # The other half of the same branch: the submission succeeded, so a
        # handle is live and only this releases it. Outbound has no local
        # metadata to invalidate, so a leak here is silent.
        worker = self._writing_worker(ranks=1)
        handle = object()
        worker.nixl_wrapper.make_prepped_xfer.side_effect = lambda *a, **k: handle
        worker.nixl_wrapper.transfer.side_effect = RuntimeError("nope")
        worker._log_failure = MagicMock()
        worker.xfer_stats = MagicMock()

        worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(handle)
        assert worker._sending_transfers["r0"] == []

    def test_the_engine_is_kept_off_the_staleness_sweep(self):
        # The sweep drops a quiet engine's state, and this path reads it. The
        # read path asserts the same touch.
        worker = self._writing_worker(ranks=1)

        worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

        assert "eng" in worker._engine_last_active

    def test_unequal_block_sizes_are_refused_on_the_per_shard_route(self):
        # Descriptor ids are counted in blocks, so a peer whose blocks are a
        # different size makes both sides agree on a count that means two
        # different spans.
        worker = self._writing_worker(ranks=1)
        worker.transfer_topo.block_size_ratio.return_value = 2

        with pytest.raises(AssertionError, match="equal P/D block sizes"):
            worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

    def test_a_consumer_that_had_everything_cached_is_not_written_to(self):
        # A full prefix hit trims our side to nothing. Posting a zero-descriptor
        # WRITE would be the alternative, and it is the peer that would then
        # wait on a notification for a transfer that carries no bytes.
        worker = self._writing_worker(ranks=1)
        worker._trim_to_consumer_blocks = lambda local, remote: ((),)

        worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

        worker.nixl_wrapper.make_prepped_xfer.assert_not_called()
        assert worker._sending_transfers["r0"] == []


class TestTrimToConsumerBlocks:
    def test_more_blocks_than_the_consumer_registered_trims_the_head(self):
        trimmed = RblnNixlPushConnectorWorker._trim_to_consumer_blocks(
            ([1, 2, 3, 4],), ([7, 8],)
        )
        assert trimmed == ([3, 4],)

    def test_a_consumer_asking_for_more_than_we_hold_is_an_error(self):
        with pytest.raises(AssertionError, match="cannot be aligned"):
            RblnNixlPushConnectorWorker._trim_to_consumer_blocks(([1],), ([7, 8],))


@pytest.fixture
def handed_through(monkeypatch):
    """What reaches upstream, in order."""
    seen = []

    def fake_base(self):
        while True:
            try:
                seen.append(self._pending_completion_notifs.get_nowait())
            except queue.Empty:
                return set()

    monkeypatch.setattr(NixlPushConnectorWorker, "_get_new_notifs", fake_base)
    return seen


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
        w._coverage_by_req = defaultdict(lambda: defaultdict(list))
        w._reqs_to_send = {}
        w._reqs_to_process = set()
        # Three blocks registered: what a writer's ranges have to cover.
        w._recving_metadata = {
            "r0": SimpleNamespace(local_physical_block_ids=([4, 5, 6],))
        }
        w._pending_completion_notifs = queue.Queue()
        return w

    def test_only_the_last_of_four_writers_reaches_the_base(self, handed_through):
        # Four peer ranks writing into this one: upstream settles on whatever it
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
        # A request this rank is sending is upstream's own accounting, even
        # though the notification looks identical.
        worker = self._receiving_worker(world_size=1)
        worker._reqs_to_process = {"r0"}
        worker._pending_completion_notifs.put(b"r0:4")

        worker._get_new_notifs()

        assert handed_through == [b"r0:4"]
        assert worker._writer_counts_by_req == {}

    def test_upstreams_own_get_finished_is_what_reaches_the_filter(self):
        # Same reason as the submission path's entry-point case: a direct call
        # would survive upstream renaming the hook.
        worker = self._receiving_worker(world_size=1)
        worker.transfer_topo = MagicMock()
        worker._recving_transfers = {}
        worker._failed_recv_reqs = queue.Queue()
        worker._pop_done_transfers = lambda _transfers: set()
        worker._pending_completion_notifs.put(b"r0:2")

        done_sending, _ = NixlBaseConnectorWorker.get_finished(worker)

        # Two writers, one report: held back, and the tally advanced.
        assert done_sending == set()
        assert worker._writer_counts_by_req["r0"] == 1

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
            reqs_to_recv={},
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
    """A peer a whole-engine handle already describes is written by upstream,
    whose alignment truncates the longer block list and keeps its head. Under a
    partial prefix hit the consumer registered its tail, so the head is wrong."""

    @staticmethod
    def _worker(*, blocks_per_logical=1):
        w = TestPerShardWrite._writing_worker(ranks=1)
        w._overlapping_ranks = {}  # nothing narrowed: upstream's route
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


class TestEarlySend:
    """A stage offers its layers at the chunk that closes the prefill, which is
    before the engine says the request finished. Two things follow: the write
    has to stay invisible to the completion accounting until the handover, and
    it cannot go out in the step that produced the KV -- that forward completes
    asynchronously, so the offer waits for this rank's next step."""

    @staticmethod
    def _worker(*, enabled=True, use_host_buffer=False):
        w = TestPerShardWrite._writing_worker(ranks=2)
        w._early_push_enabled = enabled
        w.use_host_buffer = use_host_buffer
        w._finished_blocks_inbox = queue.Queue()
        w._evict_finished_inbox = queue.Queue()
        w._push_writer_wake = threading.Event()
        return w

    @staticmethod
    def _meta(saves=(), pushes=(), recvs=None, totals=None):
        """The real metadata object, built the way the scheduler builds it.

        Hand-rolling it here would leave the one thing the two sides have to
        agree on -- the shape `add_new_req_to_save` stores and the total the
        window is measured from -- pinned on neither side.
        """
        meta = RblnNixlConnectorMetadata()
        for r in saves:
            meta.add_new_req_to_save(
                request_id=r, local_block_ids=([1, 2],), kv_transfer_params={}
            )
        meta.push_finished_blocks = dict.fromkeys(pushes, ([1, 2],))
        for r, blocks in (recvs or {}).items():
            meta.add_new_req_to_recv(
                request_id=r,
                local_block_ids=blocks,
                kv_transfer_params={
                    "remote_block_ids": [0],
                    "remote_engine_id": "eng",
                    "remote_request_id": r,
                    "remote_host": "h",
                    "remote_port": 1,
                },
            )
        meta.push_stream_total.update(totals or {})
        return meta

    def test_a_closed_prefill_is_held_rather_than_handed_over(self):
        # Handing it over in this step would let the writer read KV the
        # forward that just returned may still be writing.
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"]))

        assert worker._pending_early_offers == {"r0": ([1, 2],)}
        assert worker._finished_blocks_inbox.empty()
        assert worker._early_sends == set()

    def test_the_total_the_window_is_measured_from_comes_with_the_offer(self):
        # The prefix offered mid-stream is shorter than the request's final
        # block list, so where the consumer's window begins can only be found
        # from the total the scheduler sends alongside it.
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"], totals={"r0": 7}))

        assert worker._stream_total == {"r0": 7}

    def test_a_step_with_nothing_held_releases_nothing(self):
        # Guard: release runs on every step, including the ones that close no
        # block, and must not wake the writer for an empty handover.
        worker = self._worker()

        worker.release_early_offers()

        assert worker._finished_blocks_inbox.empty()
        assert not worker._push_writer_wake.is_set()

    def test_the_next_step_hands_it_over(self):
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))

        worker.release_early_offers()

        assert worker._finished_blocks_inbox.get_nowait() == ("r0", ([1, 2],))
        assert worker._early_sends == {"r0"}
        assert worker._pending_early_offers == {}
        assert worker._push_writer_wake.is_set()

    def test_a_step_that_closes_nothing_still_releases_what_is_held(self):
        # The release cannot wait for another closing chunk: steps that close
        # one are not every step, and the offer would sit until one came.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))

        worker.start_early_push(self._meta())  # a step with nothing to offer
        worker.release_early_offers()

        assert worker._finished_blocks_inbox.get_nowait() == ("r0", ([1, 2],))

    def test_host_staging_is_never_written_early(self):
        # The copy that fills the staging buffer runs after this, so a write
        # issued now would ship a buffer still being filled.
        worker = self._worker(use_host_buffer=True)

        worker.start_early_push(self._meta(saves=["r0"]))

        assert worker._pending_early_offers == {}

    def test_the_gate_off_holds_nothing(self):
        # Guard: the direct path had no offer before this change either, so
        # what this pins is that the flag is what turns it on.
        worker = self._worker(enabled=False)

        worker.start_early_push(self._meta(saves=["r0"]))

        assert worker._pending_early_offers == {}

    def test_an_early_write_is_kept_out_of_the_completion_accounting(self):
        # `_sending_transfers` is what upstream reports completions from, and
        # the scheduler frees a request's blocks on that report -- this one is
        # still prefilling.
        worker = self._worker()
        worker._early_sends = {"r0"}

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

        assert "r0" not in worker._sending_transfers
        # One batch, one handle per overlapping peer rank.
        assert worker._early_transfers["r0"] == [[ANY, ANY]]

    def test_the_handover_is_the_last_batch_and_seals_the_count(self, monkeypatch):
        # The request appearing here IS the engine saying it is over. It is one
        # more batch, not a duplicate: what was streamed is the prefix a
        # prefill closed, and a prompt's last block is closed by nothing.
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        worker = self._worker()
        worker._early_sends = {"r0"}
        worker._batches_queued["r0"] = 2
        meta = self._meta(pushes=["r0"])

        worker.start_load_kv(meta)

        assert worker._batches_expected == {"r0": 3}
        assert meta.push_finished_blocks == {"r0": ([1, 2],)}

    def test_the_handover_does_not_publish_what_upstream_could_report(
        self, monkeypatch
    ):
        # Guard: a batch handed over on this same step has not been issued
        # yet, and upstream reports a request the moment the handles it can
        # see have landed. What it cannot see cannot be reported early.
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        worker = self._worker()
        worker._early_sends = {"r0"}
        worker._early_transfers["r0"] = [[7, 8]]

        worker.start_load_kv(self._meta(pushes=["r0"]))

        assert "r0" not in worker._sending_transfers
        assert worker._early_transfers["r0"] == [[7, 8]]

    def test_a_request_written_only_at_the_handover_is_left_alone(self, monkeypatch):
        # Guard: the suppression must reach exactly the requests written
        # early. Dropping the membership test would silence every handover and
        # nothing would ever be sent.
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        worker = self._worker()
        meta = self._meta(pushes=["r1"])

        worker.start_load_kv(meta)

        assert meta.push_finished_blocks == {"r1": ([1, 2],)}

    def test_the_writer_count_is_the_one_the_handover_would_have_sent(self):
        # Guard: an early write is the same one write the handover would have
        # made, so the count the consumer settles on must not move.
        worker = self._worker()
        worker._early_sends = {"r0"}

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        assert notifs == {b"RBLNS:0:0:1:r0:1"}

    def test_the_delegated_route_never_carries_an_early_write(self, monkeypatch):
        # That route's notification has no room for the range a write filled,
        # so a prefix over it would settle nothing. The handshake is what keeps
        # this unreachable -- see `_writes_less_than_a_request`.
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, meta: None,
        )
        worker = self._worker()
        worker._overlapping_ranks = {}
        worker._early_sends = {"r0"}

        with pytest.raises(AssertionError, match="whole-engine handle"):
            worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

    @pytest.mark.parametrize(
        ("enabled", "host_buffer", "expected"),
        [(True, False, True), (False, False, False), (True, True, False)],
    )
    def test_streaming_asks_every_peer_for_its_own_descriptors(
        self, enabled, host_buffer, expected
    ):
        # The handshake reads this to decide whether a peer that narrows
        # nothing still needs per-shard state. Host staging moves a request at
        # a time, so it wants the ordinary route.
        worker = self._worker(enabled=enabled, use_host_buffer=host_buffer)

        assert worker._writes_less_than_a_request() is expected

    def test_the_read_path_never_asks_for_its_own_descriptors(self):
        # The predicate lives on the shared layer and the read path inherits
        # it; answering yes there would register shards nothing reads.
        assert RblnNixlWorkerBase._writes_less_than_a_request(object()) is False


class TestFlushEarlySends:
    """Blocks a write is reading can go back to the allocator without the lease
    that normally protects them. The bytes on their way are correct -- the
    prefill did finish -- so they are waited for, not cancelled."""

    @staticmethod
    def _worker(states):
        w = TestEarlySend._worker()
        w._early_sends = {"r0"}
        w._early_transfers["r0"] = [[7]]
        w.nixl_wrapper.check_xfer_state.side_effect = states
        return w

    def test_a_held_offer_is_dropped_rather_than_sent_later(self):
        # Its blocks go back to the allocator now; releasing the offer at the
        # next step would write into whatever took them.
        worker = self._worker(["DONE"])
        worker._pending_early_offers = {"r0": ([1, 2],)}

        worker.flush_early_sends({"r0"})

        assert worker._pending_early_offers == {}

    def test_an_in_flight_write_is_waited_for_and_released(self):
        worker = self._worker(["PROC", "DONE"])

        worker.flush_early_sends({"r0"})

        assert worker.nixl_wrapper.check_xfer_state.call_count == 2
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
        assert worker._early_sends == set()
        assert "r0" not in worker._early_transfers
        # The writer holds state for a request it may never see finish.
        assert worker._evict_finished_inbox.get_nowait() == "r0"

    def test_a_wedged_write_does_not_take_the_engine_with_it(self, monkeypatch):
        # This runs on the engine main thread, ahead of the forward.
        monkeypatch.setattr(pw, "_EARLY_FLUSH_DRAIN_TIMEOUT_S", 0.0)
        worker = self._worker(lambda handle: "PROC")

        worker.flush_early_sends({"r0"})

        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)


class TestOutboundFailure:
    """The write path runs upstream's completion check over its own outbound
    handles, and upstream's failure handler is written for the read direction.
    A failed WRITE queued as a failed receive kills the engine a step later:
    `get_finished` asserts every request it reports as received has receive
    metadata, and a request this rank was sending has none."""

    @staticmethod
    def _worker(*, receiving):
        w = _push_worker()
        w.nixl_wrapper = MagicMock()
        w.xfer_stats = MagicMock()
        w._failed_recv_reqs = queue.Queue()
        w._invalid_block_ids = queue.Queue()
        w._is_hma_required = False
        # Upstream's structured failure log names this rank's own engine.
        w.engine_id = "local"
        w._recving_metadata = (
            {"r0": MagicMock(local_block_ids=([1, 2],))} if receiving else {}
        )
        return w

    def test_the_completion_check_is_what_routes_a_failed_write_here(self):
        # The two above call the handler directly, which says nothing about
        # upstream still calling it. It reaches this handler from the state
        # check over outbound handles, and if that call site moves the engine
        # goes back to dying a step later.
        worker = self._worker(receiving=False)
        worker.nixl_wrapper.check_xfer_state.return_value = "ERR"
        sending = {"r0": [7]}

        worker._pop_done_transfers(sending)

        assert worker._failed_recv_reqs.empty()
        assert worker._invalid_block_ids.empty()
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)

    def test_a_failed_write_is_not_queued_as_a_failed_receive(self):
        worker = self._worker(receiving=False)

        worker._handle_failed_transfer("r0", 7)

        assert worker._failed_recv_reqs.empty()
        assert worker._invalid_block_ids.empty()
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
        worker.xfer_stats.record_failed_transfer.assert_called_once()

    def test_a_failed_read_still_reaches_upstream(self):
        # Guard: this engine receives as well, and that direction is the one
        # upstream's handler was written for.
        worker = self._worker(receiving=True)

        worker._handle_failed_transfer("r0", 7)

        assert worker._failed_recv_reqs.get_nowait() == "r0"
        assert worker._invalid_block_ids.get_nowait() == {1, 2}


class TestCoverageNotif:
    """A completion notification names the half-open range of consumer blocks
    the write filled, ahead of the message upstream builds. Nothing settles on
    the range yet; what has to hold now is that the producer states it and the
    consumer hands upstream exactly what it was handed before."""

    def test_the_range_covers_every_block_the_consumer_registered(self):
        worker = TestPerShardWrite._writing_worker(ranks=1)

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([1, 2, 3],), ([7, 8, 9],))
        )

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        assert notifs == {b"RBLNS:0:0:3:r0:1"}

    def test_a_pipeline_stage_names_itself(self, monkeypatch):
        # The writer id is the flat rank the per-shard route already pairs by,
        # so a consumer can tell one stage's ranges from another's.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker.vllm_config.parallel_config.pipeline_parallel_size = 4
        worker.world_size = 2
        worker.tp_rank = 1
        monkeypatch.setattr(
            "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl"
            ".push_worker.get_pp_group",
            lambda: SimpleNamespace(rank_in_group=3),
        )

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([7],)))

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        # pp_rank 3 of a 2-wide tensor split, tp_rank 1 -> flat rank 7.
        assert {n.split(b":")[1] for n in notifs} == {b"7"}

    def test_several_kv_groups_leave_the_range_off(self):
        # Guard: the groups go out together but carry their own lengths, so no
        # single range describes the write. The consumer has to cope with the
        # prefix being absent, which is also how the route upstream drives
        # stays on the old wire.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker._shard_region_group_ids = {("eng", 0): (0, 0)}

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([1], [2, 3]), ([7], [8, 9]))
        )

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        assert notifs == {b"r0:1"}

    def test_upstream_is_handed_the_message_it_had_before(self):
        # Upstream reads a notification as `req_id:count` with rsplit, so a
        # prefix left on would make it look up a request that does not exist.
        assert RblnNixlPushConnectorWorker._split_coverage(b"RBLNS:2:0:5:r0:4") == (
            2,
            (0, 5),
            b"r0:4",
        )

    def test_a_message_without_a_range_passes_through(self):
        assert RblnNixlPushConnectorWorker._split_coverage(b"r0:4") == (
            None,
            None,
            b"r0:4",
        )


class TestSettleOnCoverage:
    """A writer that names the ranges it filled is settled on those ranges, not
    on how many notifications it sent. That is what lets one writer's KV arrive
    in pieces: the request waits until every writer has spanned every block
    this rank registered."""

    def test_a_writer_that_has_sent_part_does_not_settle_the_request(
        self, handed_through
    ):
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        # Three blocks registered; this writer has filled the first two.
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:r0:1")

        worker._get_new_notifs()

        assert handed_through == []

    def test_the_pieces_together_settle_it(self, handed_through):
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:r0:1")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"RBLNS:0:2:3:r0:1")
        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_every_writer_has_to_cover_it(self, handed_through):
        # Two writers: one of them spanning the whole list says nothing about
        # the other, whose layers are just as missing.
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:r0:2")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"RBLNS:1:0:3:r0:2")
        worker._get_new_notifs()

        assert handed_through == [b"r0:2"]

    def test_one_write_for_the_whole_request_settles_at_once(self, handed_through):
        # Guard: this is every request today, and it has to settle exactly
        # where the count would have settled it.
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:r0:1")

        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_a_resend_does_not_stand_in_for_the_gap_it_skips(self, handed_through):
        # A preempted request is rescheduled from the start of its block list,
        # so a writer re-sends what it already sent. Counting blocks would
        # reach three here with the third never written.
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        for span in (b"0:1", b"0:1", b"0:1"):
            worker._pending_completion_notifs.put(b"RBLNS:0:" + span + b":r0:1")

        worker._get_new_notifs()

        assert handed_through == []

    def test_a_later_range_does_not_close_an_earlier_gap(self, handed_through):
        # The ranges reach the last block while the middle one was never
        # written. Whether a range starts past what has been covered is the
        # only thing separating this from a complete request.
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        worker._pending_completion_notifs.put(b"RBLNS:0:0:1:r0:1")
        worker._pending_completion_notifs.put(b"RBLNS:0:2:3:r0:1")

        worker._get_new_notifs()

        assert handed_through == []

    def test_filling_the_gap_settles_it(self, handed_through):
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        worker._pending_completion_notifs.put(b"RBLNS:0:0:1:r0:1")
        worker._pending_completion_notifs.put(b"RBLNS:0:2:3:r0:1")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"RBLNS:0:1:2:r0:1")
        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_a_finished_request_drops_its_ranges(self, monkeypatch):
        # A retry reuses the request id, so leftover ranges would settle it
        # before the retry had written anything.
        worker = TestWriterCountAccounting._receiving_worker(world_size=1)
        worker._coverage_by_req["r0"][0].append((0, 3))
        monkeypatch.setattr(
            NixlPushConnectorWorker, "get_finished", lambda self: (set(), {"r0"})
        )

        worker.get_finished()

        assert worker._coverage_by_req == {}


class TestABatchThatSendsNothing:
    """A streamed request is reported when its landed batches reach the sealed
    count, so a batch that issues no write still has to raise the count --
    otherwise the request is one short of its seal forever and never
    finishes."""

    @staticmethod
    def _worker():
        w = TestPerShardWrite._writing_worker(ranks=2)
        w._early_sends = {"r0"}
        w.xfer_stats = MagicMock()
        # The failure path names this rank's own engine in its log line.
        w.engine_id = "local"
        return w

    def test_a_write_with_no_blocks_left_still_counts(self):
        # The window can come up empty when the offer adds nothing past what
        # the high-water mark already covers.
        worker = self._worker()

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([],), ([],)))

        assert worker._batches_done["r0"] == 1
        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 0

    def test_every_peer_failing_still_counts(self):
        worker = self._worker()
        worker.nixl_wrapper.make_prepped_xfer.side_effect = RuntimeError("boom")

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1, 2],), ([3, 4],)))

        assert worker._batches_done["r0"] == 1
        assert worker._early_transfers == {}

    def test_a_request_not_streamed_counts_nothing(self):
        # Guard: the count only exists for requests this side seals.
        worker = self._worker()
        worker._early_sends = set()

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([],), ([],)))

        assert worker._batches_done == {}


class TestSealedCompletion:
    """A request written before the engine finished it is reported by this
    side, not by upstream: it reports what it can see, and a batch parked
    here is what keeps it from seeing a request whose next batch has not been
    issued. So the report waits for the seal and for every batch to land."""

    @staticmethod
    def _worker(states):
        w = _push_worker()
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.check_xfer_state.side_effect = states
        w.xfer_stats = MagicMock()
        w._recving_metadata = {}
        w._evict_finished_inbox = queue.Queue()
        w._push_writer_wake = threading.Event()
        w._writer_counts_by_req = defaultdict(int)
        w._coverage_by_req = defaultdict(lambda: defaultdict(list))
        w._early_sends = {"r0"}
        return w

    @staticmethod
    def _upstream_reports_nothing(monkeypatch):
        monkeypatch.setattr(
            NixlPushConnectorWorker, "get_finished", lambda self: (set(), set())
        )

    def test_a_landed_batch_is_not_reported_before_the_seal(self, monkeypatch):
        # The engine has not finished the request, so its blocks must not be
        # freed however much of its KV is already across.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        worker._early_transfers["r0"] = [[7]]
        worker._batches_queued["r0"] = 1

        done_sending, _ = worker.get_finished()

        assert done_sending == set()

    def test_the_seal_completes_a_request_whose_batches_already_landed(
        self, monkeypatch
    ):
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        worker._early_transfers["r0"] = [[7]]
        worker._batches_queued["r0"] = 1
        worker.get_finished()
        worker._batches_expected["r0"] = 1

        done_sending, _ = worker.get_finished()

        assert done_sending == {"r0"}

    def test_a_sealed_request_waits_for_its_last_batch(self, monkeypatch):
        # Two batches handed over, one still going: the seal alone does not
        # finish it.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE", "PROC", "DONE"])
        worker._early_transfers["r0"] = [[7], [8]]
        worker._batches_queued["r0"] = 2
        worker._batches_expected["r0"] = 2

        assert worker.get_finished()[0] == set()

        assert worker.get_finished()[0] == {"r0"}

    def test_a_finished_request_leaves_nothing_behind(self, monkeypatch):
        # A retry reuses the request id, so a leftover count would report the
        # retry finished before it had written anything.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        worker._early_transfers["r0"] = [[7]]
        worker._batches_queued["r0"] = 1
        worker._batches_expected["r0"] = 1
        worker._reqs_to_send = {"r0": 1.0}
        worker._reqs_to_process = {"r0"}
        worker.consumer_notification_counts_by_req = {"r0": 1}

        worker.get_finished()

        assert worker._batches_expected == {}
        assert worker._batches_queued == {}
        assert worker._early_sends == set()
        assert worker._reqs_to_send == {}
        assert worker._reqs_to_process == set()
        assert worker.consumer_notification_counts_by_req == {}

    def test_a_request_upstream_owns_is_left_to_it(self, monkeypatch):
        # Guard: with the flag off nothing is written early, so none of this
        # runs and upstream reports every request as it did before.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker([])
        worker._early_sends = set()

        assert worker.get_finished()[0] == set()
        assert worker.nixl_wrapper.check_xfer_state.call_count == 0


class TestEmptyReceive:
    """A request turned away before it was scheduled is registered as a receive
    of no blocks, so the producer stops holding what it pinned. Nothing is ever
    written into no blocks, so the notification that settles a receive never
    comes and the entry outlives the request.

    This side settles it. Upstream cannot: what it does with a receive it
    reports starts by asking which engine the blocks came from, and a request
    turned away never made this rank handshake with one."""

    @staticmethod
    def _worker(monkeypatch):
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        monkeypatch.setattr(
            NixlPushConnectorWorker, "get_finished", lambda self: (set(), set())
        )
        w = _push_worker()
        w.use_host_buffer = False
        return w

    def test_a_receive_of_nothing_is_finished_on_arrival(self, monkeypatch):
        worker = self._worker(monkeypatch)
        worker._recving_metadata["r0"] = MagicMock()

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ()}))

        assert worker.get_finished()[1] == {"r0"}
        assert worker._recving_metadata == {}

    def test_an_empty_group_counts_as_nothing(self, monkeypatch):
        # The block ids arrive per KV cache group, so "no blocks" can be a
        # group carrying none rather than no groups at all.
        worker = self._worker(monkeypatch)

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ([],)}))

        assert worker.get_finished()[1] == {"r0"}

    def test_upstream_is_never_asked_to_finish_it(self, monkeypatch):
        # Reported through upstream's transfer table, the request reaches the
        # post-processing that looks up an engine this rank never met.
        worker = self._worker(monkeypatch)

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ()}))

        assert worker._recving_transfers == {}

    def test_it_is_settled_once(self, monkeypatch):
        # Its receive metadata is gone after the first report, so a second one
        # names a request the scheduler has already released.
        worker = self._worker(monkeypatch)

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ()}))
        worker.get_finished()

        assert worker.get_finished()[1] == set()

    def test_a_receive_with_blocks_is_left_to_the_write_that_fills_it(
        self, monkeypatch
    ):
        # Guard: settling this one would free blocks the producer is writing.
        worker = self._worker(monkeypatch)

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ([1, 2],)}))

        assert worker.get_finished()[1] == set()
        assert worker._recving_transfers == {}


class TestRegistrationOutlivesOneBatch:
    """The consumer registers once and is written to until the request is over.
    Taking the registration out on the first batch leaves every later one with
    nothing to match, parked against a registration that already arrived."""

    @staticmethod
    def _worker():
        w = _push_worker()
        w._pending_d_registrations = {"r0": {"decode_engine_id": "eng"}}
        return w

    def test_a_second_batch_still_finds_it(self):
        worker = self._worker()

        first = worker._pop_matching_registration("r0")
        second = worker._pop_matching_registration("r0")

        assert first is not None
        assert second is first

    def test_a_retried_request_matches_on_the_id_without_its_suffix(self):
        # A retry carries a fresh random suffix; the registration is the one
        # the consumer sent before it.
        worker = self._worker()
        worker._pending_d_registrations = {"r0-aabbccdd": {"decode_engine_id": "eng"}}

        assert worker._pop_matching_registration("r0-11223344") is not None
        assert "r0-aabbccdd" in worker._pending_d_registrations

    def test_a_request_with_no_registration_is_unmatched(self):
        worker = self._worker()

        assert worker._pop_matching_registration("other") is None


class TestStreamWindow:
    """A streamed offer is a growing prefix of the producer's blocks, and the
    consumer registered the tail of the prompt. So each batch writes the part
    of the consumer's list it is the first to reach, and the total is what
    places that list inside ours."""

    @staticmethod
    def _worker(total=None):
        w = TestPerShardWrite._writing_worker(ranks=1)
        w._physical_blocks_per_logical_kv_block = 1
        w._recving_metadata = {}
        if total is not None:
            w._stream_total["r0"] = total
        return w

    @staticmethod
    def _sent(worker):
        """The local block ids each WRITE was built from.

        Read back off the descriptor ids: an 8-block shard with 2 regions maps
        block b to descs b and b+8, so the low half names the blocks.
        """
        return [
            sorted(d for d in c.args[2] if d < worker.num_blocks)
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        ]

    def test_the_first_batch_starts_where_the_consumer_window_does(self):
        # Four blocks in the prompt, the consumer registered the last two: its
        # window begins at our block 2. Two closed so far, so exactly one of
        # them is inside it.
        worker = self._worker(total=4)
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )

        assert self._sent(worker) == [[2]]
        assert worker._issued_hwm["r0"] == 1

    def test_the_next_batch_starts_where_the_last_one_stopped(self):
        worker = self._worker(total=4)
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        )

        assert self._sent(worker) == [[2], [3]]
        assert worker._issued_hwm["r0"] == 2

    def test_a_prefix_short_of_the_window_writes_nothing(self):
        # The consumer's cache covered everything this prefix has reached.
        worker = self._worker(total=4)

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([0, 1],), ([4, 5],)))

        assert self._sent(worker) == []
        assert worker._issued_hwm["r0"] == 0

    def test_the_range_says_which_blocks_the_batch_filled(self):
        worker = self._worker(total=4)
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        )

        notifs = [
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        ]
        assert notifs == [b"RBLNS:0:0:1:r0:1", b"RBLNS:0:1:2:r0:1"]

    def test_a_request_that_is_not_streamed_is_left_to_the_trim(self):
        # Guard: with the flag off no total ever arrives, and this is every
        # request today -- one write covering the consumer's whole list.
        worker = self._worker()

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )

        assert self._sent(worker) == [[1, 2]]

    def test_an_unequal_expansion_is_left_to_the_trim(self):
        # Guard: past the identity the two lengths this subtracts count
        # different things.
        worker = self._worker(total=4)
        worker.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1,
            remote_block_size=16,
            remote_physical_blocks_per_logical=2,
        )
        worker._recving_metadata = {
            "r0": SimpleNamespace(remote=SimpleNamespace(engine_id="eng"))
        }

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )

        assert self._sent(worker) == [[1, 2]]


class TestRegistrationSurvivesEitherArrival:
    """A registration reaches the writer two ways: it arrives and finds the
    blocks already parked, or it is already held when they arrive. Only the
    second stored it, so a request whose registration came late was written
    once and then never again -- every batch after it parked against a
    registration that had come and gone."""

    @staticmethod
    def _worker(monkeypatch):
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_do_start_push_kv",
            lambda self, req_id, blocks, reg: None,
        )
        w = _push_worker()
        w._pending_d_registrations = {}
        return w

    def test_a_write_leaves_the_registration_behind(self, monkeypatch):
        worker = self._worker(monkeypatch)
        reg = {"request_id": "r0", "decode_engine_id": "eng"}

        worker._do_start_push_kv("r0", ([1],), reg)

        assert worker._pending_d_registrations == {"r0": reg}
        assert worker._pop_matching_registration("r0") is reg

    def test_the_one_already_held_is_not_replaced(self, monkeypatch):
        # Guard: upstream stores it on the other path, and that entry is the
        # one the writer has been matching against.
        worker = self._worker(monkeypatch)
        held = {"request_id": "r0", "decode_engine_id": "eng"}
        worker._pending_d_registrations["r0"] = held

        worker._do_start_push_kv("r0", ([1],), dict(held))

        assert worker._pending_d_registrations["r0"] is held
