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

import dataclasses
import logging
import queue
import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnectorWorker,
    NixlPushConnectorWorker,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker as pw
from tests.vllm.distributed.kv_connector.utils import (
    mock_vllm_config,
    set_mock_connector_options,
    set_shape,
    setattr_in_package,
    shape,
    window_mode,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl import (
    RblnNixlPullConnectorWorker,
    RblnNixlPushConnectorWorker,
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
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


def _send(worker, req_id="r0", **fields):
    """Seed the record a request being pushed in pieces would have.

    The names are checked because `_StreamedSend` is a plain dataclass: a
    misspelled one would become a new attribute and the test would pin nothing.
    """
    send = worker._streamed.setdefault(req_id, pw._StreamedSend())
    known = {f.name for f in dataclasses.fields(pw._StreamedSend)}
    unknown = set(fields) - known
    assert not unknown, f"_StreamedSend has no field(s) {sorted(unknown)}"
    for name, value in fields.items():
        setattr(send, name, value)
    return send


@pytest.fixture(autouse=True)
def _pp_rank_zero(monkeypatch):
    """A pipeline rank for workers built by hand, which have no process group.

    The shared helper gives every write test a pipelined producer, so the
    writer id a coverage range carries reaches for the group this lane never
    initialises. A test that cares which stage it is says so itself.
    """
    setattr_in_package(
        monkeypatch, get_pp_group=lambda: SimpleNamespace(rank_in_group=0)
    )


def _push_worker():
    w = object.__new__(RblnNixlPushConnectorWorker)
    w._push_writer_thread = None
    w.tp_rank = 0
    # What __init__ leaves on a D2D worker with the SWA window mode off, which is
    # the shape the pairing predicates read before an engine is registered.
    w.use_host_buffer = False
    window_mode(w, None, streams_prefix=False)
    w._streamed = {}
    w._recving_transfers = {}
    # Off, as the connector option is; the trim tests turn it on.
    set_shape(w, chunk_mode=False)
    # The geometry `__init__` leaves until registration reports one, which is
    # also the permanent value on host staging. `_spans_per_block` reads both.
    w._kv_areas = 1
    w._kv_split_axis = KVSplitAxis.HEAD
    w._valid_tokens = {}
    w._physical_blocks_per_logical_kv_block = 1
    w._recving_metadata = {}
    w._coverage_by_req = defaultdict(lambda: defaultdict(list))
    w._coverage_units_by_req = defaultdict(dict)
    w._reqs_to_send = {}
    w._reqs_to_process = set()
    w.consumer_notification_counts_by_req = {}
    w._sending_transfers = defaultdict(list)
    w._sending_transfers_lock = threading.Lock()
    # __init__ never ran, so the writer state shutdown() reaches through
    # __del__ is absent; silence it rather than leak an unraisable at GC.
    w.shutdown = lambda: None
    return w


class TestInheritance:
    def test_shared_machinery_sits_next_to_the_upstream_push_classes(self):
        # The pairing layer must come before the upstream push class so its
        # overrides win, and that one before the upstream base so the write
        # path is the inherited one. Asserted as an order rather than the exact
        # tuple: what matters is which class wins, not how many sit between.
        mro = RblnNixlPushConnectorWorker.__mro__
        assert (
            mro.index(RblnNixlWorkerBase)
            < mro.index(NixlPushConnectorWorker)
            < mro.index(NixlBaseConnectorWorker)
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
        _send(worker, released=True, transfers=[[7, 8]])

        RblnNixlPushConnectorWorker.shutdown(worker)

        assert worker.nixl_wrapper.release_xfer_handle.call_count == 2
        assert worker._streamed == {}
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


class TestBuiltForReal:
    # Built through the real __init__ and registered, so the direction facts are
    # asserted against a worker that reached them rather than against attributes
    # set by hand.

    def test_the_two_directions_advertise_different_hashes(self, make_worker):
        # A writer must not present the hash a reader would accept: the peer's
        # handshake gate is what refuses a transfer aimed the wrong way.
        from tests.vllm.distributed.kv_connector.utils import KvGeometry

        geo = KvGeometry(layers=("l0",))
        reader = make_worker(kv_cache=geo)
        writer = make_worker(kv_cache=geo, direction="push")

        assert (reader._writes_into_peer, writer._writes_into_peer) == (False, True)
        assert reader.compat_hash != writer.compat_hash


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
        worker._kv_per_block = 1

        with pytest.raises(RuntimeError, match="chiplet geometry"):
            worker._check_mla_constraints(
                SimpleNamespace(kv_areas=2, kv_slices=1), remote_tp_size=1
            )

    def test_a_matching_peer_passes(self):
        worker = _push_worker()
        worker.use_mla = True
        worker.transfer_topo = SimpleNamespace(tp_ratio=lambda peer: 1, tp_size=1)
        worker._kv_areas, worker._kv_slices = 4, 1
        worker._kv_per_block = 1

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
        w.vllm_config = mock_vllm_config()
        # A pipelined producer (PP4 -> TP1 push), so a write path that counted
        # its stages would say so on the wire.
        w.vllm_config.parallel_config.pipeline_parallel_size = 4
        w.world_size = 1
        w.num_blocks = 8
        w.block_size = 16
        w.dst_num_blocks = {"eng": 8}
        w._engine_last_active = {}
        # 0.30.0 state the inherited write path reads. One memory type and one
        # group here, so neither mapping ever kicks in.
        w.dst_region_num_blocks = {"eng": [8, 8]}
        w.dst_region_group_ids = {"eng": [0, 0]}
        w.region_group_ids = [0, 0]
        w._mixed_mem_types = False
        w._uses_region_group_mapping = False
        w.dst_uses_region_group_mapping = {"eng": False}
        w._engine_ttl = 0.0
        w._recving_transfers = defaultdict(list)
        w.dcp_size = 1
        w.pcp_size = 1
        w.dcp_rank = 0
        w.pcp_rank = 0
        w.kv_cache_config = MagicMock(kv_cache_groups=[0])
        # single group, 2 regions per shard
        w._shard_region_group_ids = {("eng", r): (0, 0) for r in range(ranks)}
        # Written together with the group ids by _register_shard_xfer_state, so a
        # shard the write path can reach always has both.
        w._shard_descs_per_block = {("eng", r): 1 for r in range(ranks)}
        w._shard_chunk_grids = {("eng", r): None for r in range(ranks)}
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
        w._logical_to_kernel_block_ids = lambda ids, _n: ids
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
        # The logical list is where a released offer carries its token count,
        # so a mock in its place answers every question with a truthy Mock.
        meta.local_block_ids = local_ids
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

    @classmethod
    def _trimming_worker(cls):
        # 2 regions over 2 areas, 16-token blocks: area 0 holds the first 8
        # in-block positions and area 1 the rest.
        w = cls._writing_worker(ranks=1)
        set_shape(w, chunk_mode=True)
        w._kv_areas = 2
        w._kv_split_axis = KVSplitAxis.NON_HEAD
        w.block_size = 16
        w._valid_tokens = {"r0": 17}
        return w

    @pytest.mark.parametrize("chunk_mode", [False, True])
    def test_an_offer_locates_the_last_block_by_the_request_not_its_prefix(
        self, chunk_mode
    ):
        # A released offer is a PREFIX of the request's blocks. Where the
        # request's last block sits is read off how many blocks the REQUEST
        # holds, and taking the prefix's length instead puts it one block
        # early: the request's token count then overshoots the list and the
        # tail reader refuses it, which costs the write.
        #
        # Both knobs open this path, so both ask the question -- streaming on
        # its own is the shape the defect was found in.
        worker = self._trimming_worker()
        set_shape(worker, streams_prefix=True, chunk_mode=chunk_mode)
        # The request spans two blocks; the offer carries only the first.
        worker._streamed["r0"] = pw._StreamedSend(released=True, serial=1, total=2)
        meta = self._meta(([1],), ([3, 4],))
        meta.local_block_ids = pw.OfferedBlocks(([1],), 16, 1)

        worker._xfer_blocks_for_req("r0", meta)

        assert worker.nixl_wrapper.make_prepped_xfer.called

    def test_the_handover_leaves_the_empty_area_of_the_last_block_out(self):
        # 17 tokens over two blocks: the second holds one, which is area 0's.
        worker = self._trimming_worker()

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        call = worker.nixl_wrapper.make_prepped_xfer.call_args
        local_descs, remote_descs = call.args[2], call.args[4]
        assert len(local_descs) == len(remote_descs) == 3

    def test_the_count_reaches_the_writer_through_the_metadata(self, monkeypatch):
        # The write reads worker state, and only start_load_kv puts the
        # scheduler's count there -- so the trim has to survive that hop.
        worker = self._trimming_worker()
        worker._valid_tokens = {}
        worker._seal_at_handover = lambda metadata: None
        worker._drop_empty_receives = lambda metadata: None
        meta = RblnNixlConnectorMetadata()
        meta.valid_tokens = {"r0": 17}
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )

        worker.start_load_kv(meta)
        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        assert len(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4]) == 3

    def test_a_count_from_an_earlier_step_survives_the_next(self, monkeypatch):
        # A request handed over in one step is written in a later one, whose
        # metadata no longer lists it -- so the counts accumulate rather than
        # replace. Replaced, that request writes its last block whole.
        worker = self._trimming_worker()
        worker._valid_tokens = {"earlier": 5}
        meta = RblnNixlConnectorMetadata()
        meta.valid_tokens = {"r0": 17}
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )

        worker.start_load_kv(meta)

        assert worker._valid_tokens == {"earlier": 5, "r0": 17}

    def test_a_consumer_that_had_the_front_still_sizes_the_last_block(self):
        # The consumer registered only the suffix it was missing, so our list
        # is trimmed before the write. The token count describes the request's
        # OWN blocks, so counting after the trim would read 17 tokens as one
        # block's worth and refuse the write.
        worker = self._trimming_worker()

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([4],)))

        call = worker.nixl_wrapper.make_prepped_xfer.call_args
        local_descs, remote_descs = call.args[2], call.args[4]
        # One block left to write, and the trim on the last block still applies.
        assert len(local_descs) == len(remote_descs) == 1

    def test_a_full_last_block_writes_every_area(self):
        # The same write with a count that fills both blocks: nothing to leave
        # out, and the descriptor list is the one this path always sent.
        worker = self._trimming_worker()
        worker._valid_tokens = {"r0": 32}

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        assert len(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4]) == 4

    @pytest.mark.parametrize(
        ("stream_total", "expected_descs"),
        # The total places the window the real `_stream_window` returns: at 3
        # this batch is one block of a longer prompt and a trim would leave 1;
        # at 2 it is the whole prompt and reaches the consumer's end.
        [(3, 2), (2, 3)],
    )
    def test_only_the_batch_reaching_the_consumers_end_trims(
        self, stream_total, expected_descs
    ):
        # A streamed batch carries a closed prefix, which never holds the
        # request's last block.
        worker = self._trimming_worker()
        worker._physical_blocks_per_logical_kv_block = 1
        worker._recving_metadata = {}
        _send(worker, total=stream_total)

        worker._xfer_blocks_for_req("r0", self._meta(([1, 2],), ([3, 4],)))

        descs = worker.nixl_wrapper.make_prepped_xfer.call_args.args[4]
        assert len(descs) == expected_descs

    def test_upstreams_own_submission_path_reaches_our_override(self):
        # Calling the override directly keeps passing if upstream renames the
        # hook it dispatches to, so one case has to arrive through the caller.
        worker = self._writing_worker(ranks=1)
        # 0.30.0 folded push's blocking _ensure_d_handshake into the base
        # _ensure_handshake, which returns None once the peer is known.
        worker._ensure_handshake = lambda *_args, **_kw: None
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
        # out to one on the far side -- and our stages stay out of it, because
        # that far side multiplies them back in itself. Ahead of it, the range
        # of consumer blocks this write filled -- both of the two it registered.
        assert notifs == {b"RBLNS:0:0:2:1:r0:1"}

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
        # The mock is installed to keep the log quiet; assert on it too, or the
        # operator-facing half of the report can be deleted unnoticed -- the stat
        # above is the other half.
        assert (
            worker._log_failure.call_args.kwargs["failure_type"]
            == "transfer_setup_failed"
        )

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
        worker._trim_to_consumer_blocks = lambda *_: ((),)

        worker._xfer_blocks_for_req("r0", self._meta(([1],), ([3],)))

        worker.nixl_wrapper.make_prepped_xfer.assert_not_called()
        assert worker._sending_transfers["r0"] == []


class TestTrimToConsumerBlocks:
    def test_more_blocks_than_the_consumer_registered_trims_the_head(self):
        trimmed = RblnNixlPushConnectorWorker._trim_to_consumer_blocks(
            ([1, 2, 3, 4],), ([7, 8],), "eng0", "req0"
        )
        assert trimmed == ([3, 4],)

    def test_a_consumer_asking_for_more_than_we_hold_is_refused(self):
        # A peer's advertised length, so it is refused rather than asserted --
        # and the message has to name the pair an operator has to go look at.
        with pytest.raises(RuntimeError) as excinfo:
            RblnNixlPushConnectorWorker._trim_to_consumer_blocks(
                ([1],), ([7, 8],), "eng0", "req0"
            )
        msg = str(excinfo.value)
        assert "eng0" in msg and "req0" in msg
        assert "registered 2 block(s)" in msg and "the 1 this producer holds" in msg


class TestWriterCompletionAccounting:
    """0.26 counts a pushed request's writers itself, so this side must not.

    Upstream settles on ``pp_size * producers_per_consumer`` notifications and
    reads the producer's TP out of the notification. A count of our own on top
    of that held all but one back, and a pipeline factor in the payload was
    multiplied by the peer's ``pp_size`` a second time -- either one leaves the
    request waiting for notifications that never come.
    """

    @staticmethod
    def _receiving_worker(world_size, pp_size):
        w = _push_worker()
        w.world_size = world_size
        w._reqs_to_send = {}
        w._reqs_to_process = set()
        w._recving_metadata = {"r0": SimpleNamespace(pp_size=pp_size)}
        w._recving_transfers = {}
        w.consumer_notification_counts_by_req = defaultdict(int)
        w._pending_completion_notifs = queue.Queue()
        w.transfer_topo = MagicMock()
        return w

    @staticmethod
    def _writer_notif(*, producer_tp, consumer_tp):
        """What a producer of that shape writes, from the write path itself.

        A literal here would let the two halves drift apart again -- that they
        cannot disagree unseen is the point.
        """
        w = _push_worker()
        w.world_size = producer_tp
        w.vllm_config = MagicMock()
        return w._xfer_notif_id("eng", "r0", consumer_tp, count_stages=False)

    @staticmethod
    def _feed(worker, notif):
        """One notification in, upstream's own done-report out.

        0.30.0 returns (done, failed) here; these cases never fail a transfer,
        so only the done half is compared.
        """
        worker._pending_completion_notifs.put(notif)
        worker._get_new_notifs()
        done, failed = worker._pop_done_transfers(worker._recving_transfers)
        assert not failed
        return done

    def test_a_pipelined_producer_settles_on_its_last_stage(self):
        # Four stages write into this one rank; upstream expects pp_size * 1.
        worker = self._receiving_worker(world_size=1, pp_size=4)
        notif = self._writer_notif(producer_tp=1, consumer_tp=1)

        assert [self._feed(worker, notif) for _ in range(4)] == [
            set(),
            set(),
            set(),
            {"r0"},
        ]

    def test_a_wider_producer_settles_on_its_last_rank(self):
        # Four producer ranks per consumer rank, no pipeline: pp_size 1 * 4.
        worker = self._receiving_worker(world_size=1, pp_size=1)
        notif = self._writer_notif(producer_tp=4, consumer_tp=1)

        assert [self._feed(worker, notif) for _ in range(4)] == [
            set(),
            set(),
            set(),
            {"r0"},
        ]


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


class TestNotifIdStageFactor:
    """The read path counts our stages into the payload, the write path cannot.

    Upstream's read side takes the payload as a plain consumer count; its write
    side multiplies the peer's own ``pp_size`` back in.
    """

    @staticmethod
    def _worker(local_pp, remote_pp, world_size=1):
        w = _push_worker()
        w.world_size = world_size
        w._remote_pp_size = {"eng": remote_pp}
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = local_pp
        return w

    def test_the_write_path_leaves_the_stages_to_upstream(self):
        worker = self._worker(local_pp=4, remote_pp=1)

        assert worker._xfer_notif_id("eng", "r0", 1, count_stages=False) == b"r0:1"

    def test_the_read_path_still_carries_them(self):
        worker = self._worker(local_pp=4, remote_pp=1)

        assert worker._xfer_notif_id("eng", "r0", 1) == b"r0:4"

    def test_a_wider_local_tp_is_scaled_either_way(self):
        worker = self._worker(local_pp=1, remote_pp=1, world_size=4)

        assert worker._xfer_notif_id("eng", "r0", 1, count_stages=False) == b"r0:4"


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
        # The real type, not a stand-in: start_load_kv reads a field only the
        # promoted metadata carries.
        meta = RblnNixlConnectorMetadata()
        meta.reqs_to_save = dict.fromkeys(saves, object())
        meta.push_finished_blocks = {r: ([1],) for r in pushes}
        return meta

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
        w._kv_per_block = 1
        window_mode(w, None)
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
            # Both sides cut heads the same number of ways here, so one piece.
            split=1,
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
        w._kv_per_block = 1
        window_mode(w, None)
        w.use_host_buffer = False
        w.device_id = 0
        w.engine_id = "eng-local"
        w.tp_rank = 0
        w.num_blocks = 2
        w.block_size = 16
        w._has_mamba = False
        w.num_regions = regions
        w.block_len_per_layer = [512] * regions
        # Two logical regions (K and V of the one layer), both the target's width.
        w._logical_region_kv_heads = [8, 8]
        w.local_seen_layer_names = ["l0"]
        w.kv_caches_base_addr = {
            "eng-local": {0: [10_000 * (i + 1) for i in range(regions)]}
        }
        topo = MagicMock()
        topo.total_num_kv_heads = 8
        topo.tp_size = 1
        topo.tp_ratio.return_value = -4
        w.transfer_topo = topo
        w.get_backend_aware_kv_block_len = lambda **kw: 512
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.get_xfer_descs.side_effect = lambda data, _t: data
        w.nixl_wrapper.prep_xfer_dlist.return_value = 7
        w.nixl_memory_type = "VRAM"
        # A block of 16 tokens whose 2-head region costs 16B a token: a 128B
        # descriptor target is 8 tokens, so a block is two chunks.
        set_shape(w, chunk_mode=True)
        w._kv_split_axis = KVSplitAxis.HEAD
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = 8
        return w

    @staticmethod
    def _peer():
        # A TP4 rank: 2 heads, each duplicated across two of its four areas.
        meta = MagicMock()
        meta.kv_areas, meta.kv_slices = 4, 2
        meta.num_blocks = 2
        meta.block_size = 16
        meta.device_id = 0
        # Same shape as ours: K and V, each across its four areas.
        meta.kv_caches_base_addr = [100_000 * (i + 1) for i in range(8)]
        meta.block_lens = [512] * 8
        return meta

    @pytest.mark.parametrize(
        "cls", [RblnNixlPullConnectorWorker, RblnNixlPushConnectorWorker]
    )
    def test_both_sides_describe_the_same_number_of_pieces(self, cls, monkeypatch):
        worker = self._worker(cls)
        set_mock_connector_options(worker.vllm_config, chunk_tokens=8)
        peer = self._peer()
        fanout = worker._peer_replica_fanout(peer, 4)
        split = worker._peer_head_split(peer, 4)
        areas = worker._fan_in_peer_areas(0, 4)

        # The two builders derive the grid rather than take it, so this is
        # what `_register_shard_xfer_state` passes the local one, not a knob.
        grid = worker._shard_chunk_grid(block_size=worker.block_size, split=split)

        remote = worker._build_head_matched_remote(peer, 0, 4, peer_areas=areas)
        _handle, local = worker._register_shard_local_xfer_handler(
            worker.block_size,
            ("l0",),
            chunk_grid=grid,
            peer_areas=areas,
            split=split,
            replica_fanout=fanout,
        )

        assert len(remote) == len(local)
        # And that length is the fan-out times what one copy would have needed.
        assert len(remote) % fanout == 0
        assert (cls is RblnNixlPushConnectorWorker) == (fanout == 2)
        # This shape has to be one that carries a chunk range -- without it
        # the two lists agree trivially and say nothing about the range.
        assert grid == (1, 2)


class TestTheCompletionHookUpstreamActuallyCalls:
    """vllm 0.30 inverted the completion API.

    `get_transfer_results` is the entry point and `get_finished` a wrapper over
    it, so a body hung off `get_finished` runs for nobody. Everything this
    connector reports itself -- the seal, the empty-receive drain, the coverage
    teardown -- lives in that body.
    """

    def test_the_override_sits_on_the_hook_the_model_runner_calls(self):
        # `KVConnectorModelRunnerMixin` calls `get_transfer_results`, and
        # `NixlConnector` overrides it, so upstream's base never routes a call
        # back through `get_finished`.
        assert "get_transfer_results" in RblnNixlPushConnectorWorker.__dict__

    def test_the_streaming_completion_runs_from_it(self, monkeypatch):
        # A sealed request is reported nowhere else: upstream cannot see a
        # batch parked on this side.
        worker = _push_worker()
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_transfer_results",
            lambda self: KVConnectorTransferResults(),
        )
        monkeypatch.setattr(
            RblnNixlPushConnectorWorker,
            "_finish_sealed_requests",
            lambda self: {"r-sealed"},
        )
        worker._evict_finished_inbox = SimpleNamespace(put=lambda r: None)
        worker._push_writer_wake = SimpleNamespace(set=lambda: None)

        results = worker.get_transfer_results()

        assert "r-sealed" in results.finished_sending


class TestFailedTransferFollowsTheWidenedContract:
    """0.30 widened `_handle_failed_transfer` and gave it an answer.

    It takes the caller's failure set and returns whether the handle was dealt
    with; three upstream call sites pass three arguments, and one of them sits
    outside the `try`, so a narrow override kills the engine step.
    """

    def test_it_takes_the_callers_failure_set(self):
        worker = _push_worker()
        worker._streamed = {"r0": object()}
        worker.nixl_wrapper = MagicMock()
        worker.xfer_stats = MagicMock()

        # Three positional arguments, the way upstream calls it.
        dealt_with = worker._handle_failed_transfer("r0", 7, set())

        # The handle was released here, so retaining it would double-release.
        assert dealt_with is True
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)


class TestStreamedEngineHandleWrite:
    """A hybrid streams over the whole-engine handle, because that is the only
    descriptor list able to name two KV cache groups. So this side issues the
    batch itself: the notification upstream builds has no room for a range."""

    @staticmethod
    def _worker(*, total=4, grid=(2, 2), kv_per_block=1):
        w = TestPerShardWrite._writing_worker(ranks=1)
        w._overlapping_ranks = {}  # nothing narrowed: the engine handle
        window_mode(w, 2, chunk_mode=True)
        w._chunk_grid = grid
        w._request_tail = None
        w.num_regions = 2
        w._kv_per_block = kv_per_block
        w._kv_areas = 1
        w._kv_split_axis = KVSplitAxis.HEAD
        w._physical_blocks_per_logical_kv_block = 1
        w._recving_metadata = {}
        w._group_specs = [
            MagicMock(),  # full attention
            MagicMock(spec=SlidingWindowSpec),
        ]
        w.engine_id = "self"
        w.dst_num_blocks["self"] = 8
        w.src_xfer_handles_by_block_size = {16: 300}
        w.tp_mappings = {"eng": MagicMock(all_source_ranks=[0])}
        _send(w, total=total)
        w._streamed["r0"].released = True
        return w

    @staticmethod
    def _offer(worker, local_groups, remote_groups, tokens):
        """An offer built for `worker`'s record, so the write is not refused."""
        meta = TestPerShardWrite._meta(local_groups, remote_groups)
        meta.local_block_ids = pw.OfferedBlocks(
            local_groups, tokens, worker._streamed["r0"].serial
        )
        return meta

    @staticmethod
    def _descs(worker):
        call = worker.nixl_wrapper.make_prepped_xfer.call_args
        return list(call.args[2]), call.kwargs["notif_msg"]

    def test_a_write_with_no_range_carries_the_bare_notification(self):
        # A request written in one transfer has no range to name, and the
        # consumer settles it on upstream's own count -- so what reaches it is
        # the message upstream would have built.
        worker = self._worker()

        assert worker._with_coverage(b"r0:1", None, 1) == b"r0:1"

    def test_a_peer_that_expands_a_block_differently_is_refused(self):
        # The two block lists then count different things, and the handshake
        # does not refuse the pairing. Raised rather than asserted: upstream's
        # writer loop logs what it catches and goes on, so an assertion here
        # leaves the request stalled until its lease expires.
        worker = self._worker()
        worker.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1,
            remote_block_size=16,
            remote_physical_blocks_per_logical=2,
        )
        meta = self._offer(worker, ([0, 1, 2, 3], []), ([4, 5], [6]), 3 * 16)

        with pytest.raises(RuntimeError, match="expands a logical block"):
            worker._xfer_blocks_for_req("r0", meta)

    def test_a_streamed_batch_names_its_range_and_leaves_the_window_alone(self):
        # Three blocks closed of four and the consumer registered the last
        # two, so its block 0 is what this batch is the first to fill. The
        # window's group is not in the offer -- its one block holds the live
        # window, which the kernel is still overwriting -- and the consumer's
        # registration of it is trimmed away rather than written.
        worker = self._worker()
        meta = self._offer(worker, ([0, 1, 2, 3], []), ([4, 5], [6]), 3 * 16)

        worker._xfer_blocks_for_req("r0", meta)

        descs, notif = self._descs(worker)
        assert notif.startswith(b"RBLNS:0:0:2:2:")  # one block, counted in chunks
        # 2 regions x 8 blocks = 16 whole descriptors, so the window's range
        # is [16, 32) and no descriptor of this write may fall in it.
        assert descs == [2, 10]
        remote_descs = list(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4])
        assert remote_descs == [4, 12]

    def test_the_handover_carries_the_window_with_the_last_block(self):
        # The prompt's last block is half full, so it goes as the one chunk
        # that holds tokens -- and the window's block rides the same write,
        # which is the only one that may carry it.
        worker = self._worker()
        worker._valid_tokens = {"r0": 3 * 16 + 8}
        meta = self._offer(worker, ([0, 1, 2, 3], [7]), ([4, 5], [6]), 0)

        worker._xfer_blocks_for_req("r0", meta)

        _descs, notif = self._descs(worker)
        assert notif.startswith(b"RBLNS:0:0:4:2:")
        remote = list(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4])
        # The block that closed whole, one chunk of the last block from the
        # third range, and the granule the window sits in from the second.
        assert remote[:2] == [4, 12]
        # 16 whole descriptors, then 16 x sw_ratio window ones, so the chunk
        # range is at 48 -- which a batch indexes into for every block it
        # names rather than only the last.
        assert remote[2:6] == [68, 70, 100, 102]
        assert remote[6:] == [28, 44]

    def test_a_handover_after_a_partial_block_stops_at_its_tokens(self):
        # The batch before this one sent the half of the last block that held
        # tokens. The handover owes the rest of that block only as far as the
        # count reaches -- which is nowhere, so the chunk range is untouched
        # and what is left to carry is the window's granule.
        worker = self._worker()
        worker._valid_tokens = {"r0": 3 * 16 + 8}
        worker._xfer_blocks_for_req(
            "r0", self._offer(worker, ([0, 1, 2, 3], []), ([4, 5], [6]), 3 * 16 + 8)
        )
        first = list(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4])
        first_local = [
            int(x) for x in worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        ]

        worker._xfer_blocks_for_req(
            "r0", self._offer(worker, ([0, 1, 2, 3], [7]), ([4, 5], [6]), 0)
        )
        second = list(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4])
        second_local = [
            int(x) for x in worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        ]

        # The first batch took the block that closed and the one chunk of the
        # next that holds tokens; the second repeats neither.
        assert first == [4, 12, 68, 70, 100, 102]
        assert second == [28, 44]
        # The same two batches off this side's own block ids. Pinned here
        # because a batch carrying chunks is where the two lists part: the
        # pieces hold both, and a local view taking the peer's numbers still
        # pairs by length -- it writes whatever KV sits at those block ids.
        assert first_local == [2, 10, 60, 62, 92, 94]
        assert second_local == [30, 46]

    def test_a_hybrid_whose_window_comes_first_counts_the_other_group(self):
        # The prompt's block count comes off the full-attention group, and
        # every other case here has that group first. A window's group holds
        # one block whatever the prompt is, so counting it would size the last
        # block and the coverage total off a length that describes no request.
        worker = self._worker()
        worker._group_specs = [
            MagicMock(spec=SlidingWindowSpec),
            MagicMock(),  # full attention, and the second group at that
        ]
        set_shape(worker, counted_group=1)
        worker._valid_tokens = {"r0": 3 * 16 + 8}
        meta = self._offer(worker, ([7], [0, 1, 2, 3]), ([6], [4, 5]), 0)

        worker._xfer_blocks_for_req("r0", meta)

        _descs, notif = self._descs(worker)
        assert notif == b"RBLNS:0:0:4:2:r0:4"

    def test_a_window_that_straddles_names_a_granule_either_side(self):
        # 57 tokens leave the window over two granules of the block -- a
        # count that does not end on a granule boundary, which is where
        # rounding it down and rounding it up stop agreeing.
        worker = self._worker()
        worker._valid_tokens = {"r0": 3 * 16 + 9}
        meta = self._offer(worker, ([0, 1, 2, 3], [7]), ([4, 5], [6]), 0)

        worker._xfer_blocks_for_req("r0", meta)

        remote = list(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4])
        assert remote[-4:] == [28, 29, 44, 45]

    def test_the_packing_does_not_move_what_the_batch_indexes(self):
        # A whole block is one descriptor whatever it packs, and the window
        # range is counted in kernel blocks either way, so neither the offsets
        # a batch indexes into nor the granules it picks move with the packing.
        offer = (([0, 1, 2, 3], [7]), ([4, 5], [6]), 0)
        remotes = []
        for kv_per_block in (2, 1):
            worker = self._worker(kv_per_block=kv_per_block)
            worker._valid_tokens = {"r0": 3 * 16 + 8}
            worker._xfer_blocks_for_req("r0", self._offer(worker, *offer))
            remotes.append(
                list(worker.nixl_wrapper.make_prepped_xfer.call_args.args[4])
            )

        assert remotes[0] == remotes[1]

    def test_a_batch_that_writes_nothing_still_raises_the_landed_count(self):
        # The per-shard route has the same guard: a batch issuing no write
        # still raises the landed count, or `done` stays short of `issued`
        # forever. This route reaches it when the offer adds nothing past the
        # high-water mark and the window's block is not in the offer.
        worker = self._worker()
        worker._streamed["r0"].issued_hwm = 2
        meta = self._offer(worker, ([0, 1, 2, 3], []), ([4, 5], [6]), 2 * 16)

        worker._xfer_blocks_for_req("r0", meta)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert worker._streamed["r0"].done == 1

    def test_a_handover_with_nothing_left_still_writes_the_window(self):
        # The prefill ended on a chunk boundary, so streaming already sent
        # every chunk the prompt backs and the handover owes no full-attention
        # bytes at all. It still owes the window: nothing else may send it.
        worker = self._worker()
        worker._valid_tokens = {"r0": 4 * 16}
        worker._streamed["r0"].issued_hwm = 2
        meta = self._offer(worker, ([0, 1, 2, 3], [7]), ([100, 101], [50]), 0)

        worker._xfer_blocks_for_req("r0", meta)

        descs, _notif = self._descs(worker)
        # The granule the window sits in, and only it.
        assert descs == [31, 47]


class TestTheDelegatingRouteHandsTheListOver:
    """A peer a whole-engine handle already describes is written by upstream.

    This route used to trim the producer's blocks to the consumer's suffix
    first, because 0.26 aligned by truncating the longer list and keeping its
    head. vllm 0.30 end-trims in `_xfer_blocks` itself, so trimming here would
    apply the same cut twice on the way in.
    """

    def test_the_block_ids_reach_the_base_untouched(self, monkeypatch):
        # A partial prefix hit: the consumer kept one block, we hold three.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker._overlapping_ranks = {}  # nothing narrowed: upstream's route
        worker.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1, remote_block_size=16, remote_physical_blocks_per_logical=1
        )
        meta = TestPerShardWrite._meta(([5, 6, 7],), ([9],))
        seen: dict = {}
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, meta: seen.update(local=meta.local_physical_block_ids),
        )

        worker._xfer_blocks_for_req("r0", meta)

        assert seen["local"] == ([5, 6, 7],)

    @pytest.mark.parametrize("sw_ratio, raises", [(8, False), (None, True)])
    def test_a_chunked_engine_reaches_this_route_only_with_a_window(
        self, monkeypatch, sw_ratio, raises
    ):
        # Chunk mode asks every peer for per-shard state, so arriving here
        # without it is a bug -- except where a sliding window's view already
        # put the extra range on the whole-engine list.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker._overlapping_ranks = {}
        worker.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1, remote_block_size=16, remote_physical_blocks_per_logical=1
        )
        window_mode(worker, sw_ratio, chunk_mode=True)
        worker._chunk_grid = None
        worker._request_tail = None
        worker._group_specs = [MagicMock()]  # one full-attention group
        worker._valid_tokens = {"r0": 17}
        meta = TestPerShardWrite._meta(([5, 6, 7],), ([9],))
        seen: list = []
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, m: seen.append(self._request_tail),
        )

        if raises:
            with pytest.raises(AssertionError):
                worker._xfer_blocks_for_req("r0", meta)
            return

        worker._xfer_blocks_for_req("r0", meta)

        # The token count and the request's own block count, counted before
        # the trim and parked for the length of upstream's call.
        assert seen == [(17, 3, ())]
        assert worker._request_tail is None

    def test_the_window_knob_alone_parks_the_count(self, monkeypatch):
        # The write side of the same rule: the window range rides the
        # whole-engine list without chunk mode, so the count it is cut from
        # has to be parked here too.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker._overlapping_ranks = {}
        worker.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1, remote_block_size=16, remote_physical_blocks_per_logical=1
        )
        window_mode(worker, 8)
        worker._chunk_grid = None
        worker._request_tail = None
        worker._group_specs = [MagicMock()]  # one full-attention group
        worker._valid_tokens = {"r0": 17}
        meta = TestPerShardWrite._meta(([5, 6, 7],), ([9],))

        seen: list = []
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, m: seen.append(self._request_tail),
        )
        worker._xfer_blocks_for_req("r0", meta)

        assert seen == [(17, 3, ())]


class TestEarlySend:
    """A stage offers its layers at the chunk that closes the prefill, which is
    before the engine says the request finished. Two things follow: the write
    has to stay invisible to the completion accounting until the handover, and
    it cannot go out in the step that produced the KV -- that forward completes
    asynchronously, so the offer waits for this rank's next step."""

    @pytest.fixture(autouse=True)
    def _off_the_device(self, monkeypatch):
        # Recording a `torch.rbln.Event` opens the NPU, and `conftest.py` keeps
        # the parent process off it. The tests that assert on the event patch it
        # with a recorder of their own; this is for the rest, which reach
        # `start_early_push` for other reasons and would otherwise pin a device
        # they never use.
        monkeypatch.setattr(
            pw.torch.rbln,
            "Event",
            lambda: SimpleNamespace(record=lambda _s: None, synchronize=lambda: None),
        )

    @staticmethod
    def _worker(*, enabled=True, use_host_buffer=False):
        w = TestPerShardWrite._writing_worker(ranks=2)
        # Derived rather than assigned: host staging is folded into the gate,
        # so "streaming on with a host buffer" is a state __init__ cannot
        # produce and a test must not invent.
        w._shape = shape(
            streams_prefix=enabled and not use_host_buffer,
            use_host_buffer=use_host_buffer,
        )
        w.use_host_buffer = w._shape.use_host_buffer
        w._finished_blocks_inbox = queue.Queue()
        w._evict_finished_inbox = queue.Queue()
        w._push_writer_wake = threading.Event()
        w._pending_d_registrations = {}
        w._push_finished_blocks = {}
        return w

    @staticmethod
    def _meta(saves=(), pushes=(), recvs=None, totals=None, offered=None):
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
        meta.push_finished_blocks = {r: ([1, 2],) for r in pushes}
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
        meta.push_stream_tokens.update(offered or {})
        return meta

    def test_a_closed_prefill_is_held_rather_than_handed_over(self):
        # Handing it over in this step would let the writer read KV the
        # forward that just returned may still be writing.
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"]))

        assert worker._streamed["r0"].pending_offer == ([1, 2],)
        assert worker._finished_blocks_inbox.empty()
        assert not worker._streamed["r0"].released

    def test_the_total_the_window_is_measured_from_comes_with_the_offer(self):
        # The prefix offered mid-stream is shorter than the request's final
        # block list, so where the consumer's window begins can only be found
        # from the total the scheduler sends alongside it.
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"], totals={"r0": 7}))

        assert worker._streamed["r0"].total == 7

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
        assert worker._streamed["r0"].released
        assert worker._streamed["r0"].pending_offer is None
        assert worker._push_writer_wake.is_set()

    def test_a_released_offer_carries_the_tokens_it_was_built_for(self):
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"], offered={"r0": 1500}))
        worker.release_early_offers()

        _, blocks = worker._finished_blocks_inbox.get_nowait()
        assert blocks.offered_tokens == 1500

    def test_a_later_offer_does_not_rewrite_an_earlier_one_s_tokens(self):
        # The writer drains on its own thread, so a step can hold and release
        # the next offer before the writer has picked the previous one up.
        # Reading the count off the request would then claim tokens this
        # rank has not written -- silently, into the consumer's blocks.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"], offered={"r0": 1500}))
        worker.release_early_offers()
        _, first = worker._finished_blocks_inbox.get_nowait()

        worker.start_early_push(self._meta(saves=["r0"], offered={"r0": 3800}))
        worker.release_early_offers()

        assert first.offered_tokens == 1500

    @staticmethod
    def _writer_matches_the_inbox(worker):
        """Upstream's writer step that pairs an offer with a D registration.

        Copied from `nixl/push_worker.py` rather than driven through the real
        loop: that loop is a thread with its own wake protocol, and the one
        line this reproduces is the parking, which is a plain assignment into
        a dict keyed by request id. Only the unmatched branch is here -- what
        a matched list goes on to do is `_xfer_blocks_for_req`, which
        `TestStreamWindow` drives with a worker built for a write.
        """
        while True:
            try:
                rid, blocks = worker._finished_blocks_inbox.get_nowait()
            except queue.Empty:
                return
            if worker._pop_matching_registration(rid) is None:
                worker._push_finished_blocks[rid] = blocks

    def _two_offers_parked(self):
        """A request whose second offer reached the writer before the first was
        matched, which is the state the parking folds into one entry."""
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"]))
        worker.release_early_offers()
        worker.start_early_push(self._meta(saves=["r0"]))
        worker.release_early_offers()
        self._writer_matches_the_inbox(worker)
        return worker

    def test_the_handover_mark_is_what_survives_a_fold(self):
        # The shorter path to the same fold, and the common one: the handover
        # goes into the writer's inbox as well, so one offer ahead of it is
        # enough. What the parking keeps is the one put last, which is why the
        # mark rides it out and a count does not.
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"]))
        worker.release_early_offers()
        handover = self._meta(pushes=["r0"])
        worker._seal_at_handover(handover)
        for req_id, blocks in handover.push_finished_blocks.items():
            worker._finished_blocks_inbox.put((req_id, blocks))
        self._writer_matches_the_inbox(worker)

        assert worker._push_finished_blocks["r0"].handover

    def test_a_record_not_released_is_not_marked(self):
        # Marking says the writer has no more batches to make, which is only
        # true once the offers are with it. Unmarked, the batch upstream makes
        # ends nothing and the request waits for its own handover.
        worker = self._worker()
        _send(worker)  # `released` left False
        meta = self._meta(pushes=["r0"])

        worker._seal_at_handover(meta)

        assert not isinstance(meta.push_finished_blocks["r0"], pw.OfferedBlocks)

    def test_a_released_offer_is_not_marked_as_the_handover(self):
        # The mark is what ends a request, so an offer that carried it would
        # end one on a batch written before the handover was even queued --
        # handing its blocks back while that write is still reading them.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))

        worker.release_early_offers()

        _, blocks = worker._finished_blocks_inbox.get_nowait()
        assert not blocks.handover

    def test_a_request_cannot_hand_over_and_be_flushed_on_one_step(self):
        # The guard refuses a list no live record claims, and the handover is
        # the one list the engine cannot number once it has dropped the
        # record. So the two sets have to stay apart, and the scheduler is what
        # keeps them apart -- this stops where it does not.
        worker = self._worker()
        _send(worker, released=True)
        meta = self._meta(pushes=["r0"])
        meta.push_early_flush = {"r0"}

        with pytest.raises(AssertionError):
            worker._seal_at_handover(meta)

    def test_the_handover_carries_the_number_of_the_record_it_ends(self):
        # The guard refuses a list whose number no live record claims, so a
        # handover stamped with anything else is dropped -- and the request it
        # was the last list of never ends.
        worker = self._worker()
        _send(worker, released=True)
        meta = self._meta(pushes=["r0"])

        worker._seal_at_handover(meta)

        assert meta.push_finished_blocks["r0"].serial == worker._streamed["r0"].serial

    def test_the_handover_offer_carries_no_token_count(self):
        # `_stream_window` reads the length of a list that carries no count,
        # and a count instead would shorten the window the handover exists to
        # cover -- leaving the consumer short of the prompt's tail.
        worker = self._worker()
        meta = self._meta(pushes=["r0"])
        _send(worker, released=True)

        worker._seal_at_handover(meta)

        assert meta.push_finished_blocks["r0"].offered_tokens == 0

    def test_a_list_from_before_a_re_prefill_is_refused_whole(self):
        # The record is back, so "is there a record" says yes. Only the number
        # says this list belongs to the incarnation whose blocks are gone. And
        # a refusal must leave no trace: a count raised here would make the
        # live incarnation wait for a batch nobody will ever record.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))
        worker.release_early_offers()
        _, stale = worker._finished_blocks_inbox.get_nowait()
        worker.flush_early_sends({"r0"})
        worker.start_early_push(self._meta(saves=["r0"]))
        # The new incarnation offers too, or `released` stays False and the
        # count below would read 0 whatever the guard did.
        worker.release_early_offers()
        meta = TestPerShardWrite._meta(([1, 2],), ([3],))
        meta.local_block_ids = stale

        worker._xfer_blocks_for_req("r0", meta)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert worker._streamed["r0"].issued == 0

    def test_a_flat_block_list_is_refused_where_it_is_built(self):
        # Upstream rebuilds a flat list as a plain tuple of one group, and the
        # number rides on the subclass -- so a lost number reads as "not ours"
        # and the writer's guard waves it through. Refused at construction,
        # where it is loud. The empty group below is the other side: a sliding
        # window offers `[]` for the group it does not count.
        pw.OfferedBlocks(([], [4, 5]), 0, 7)

        with pytest.raises(RuntimeError, match="grouped per KV cache group"):
            pw.OfferedBlocks([1, 2, 3], 0, 7)

    def test_a_batch_is_filed_on_the_record_it_was_authorised_against(self):
        # The descriptors are built with the lock down, so a flush can drop the
        # record and a re-prefill install another before the handles are filed.
        # Filed on whatever sits under the id then, the dead incarnation's
        # batch would land on the live one and could end it before its own
        # writes do.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))
        worker.release_early_offers()
        _, live = worker._finished_blocks_inbox.get_nowait()
        meta = TestPerShardWrite._meta(([1, 2],), ([3],))
        meta.local_block_ids = live

        def _re_prefill_mid_batch(*args, **kwargs):
            worker._streamed["r0"] = pw._StreamedSend(released=True)
            return 7

        worker.nixl_wrapper.make_prepped_xfer.side_effect = _re_prefill_mid_batch
        worker._xfer_blocks_for_req("r0", meta)

        assert worker._streamed["r0"].transfers == []
        # One handle per paired peer, and both belong to the dead incarnation.
        assert worker._sending_transfers["r0"] == [7, 7]

    def test_a_list_the_engine_still_owns_is_not_refused(self):
        # The guard's other side: refusing too much is as wrong as refusing
        # too little. The count rising is what says the guard let it past.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))
        worker.release_early_offers()
        _, live = worker._finished_blocks_inbox.get_nowait()
        meta = TestPerShardWrite._meta(([1, 2],), ([3],))
        meta.local_block_ids = live

        worker._xfer_blocks_for_req("r0", meta)

        assert worker._streamed["r0"].issued == 1

    def test_a_parked_list_writes_nothing_once_its_request_is_flushed(self):
        # A parked list becomes a batch when the consumer's registration
        # arrives, and the eviction the flush queues only reaches the writer on
        # its next turn -- the registration can win that race. The list has read
        # no block yet, so it is refused where bytes leave rather than waited for.
        worker = self._two_offers_parked()
        parked = worker._push_finished_blocks["r0"]
        worker.flush_early_sends({"r0"})

        meta = TestPerShardWrite._meta(([1, 2],), ([3],))
        meta.local_block_ids = parked
        worker._xfer_blocks_for_req("r0", meta)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 0

    def test_a_flushed_request_drops_its_token_count(self):
        # The flush is driven by the scheduler's own list, so these requests
        # are not in the step's completion set -- the trim that runs there
        # never reaches them, and a count kept here outlives its request.
        worker = self._two_offers_parked()
        worker._valid_tokens = {"r0": 33, "other": 9}

        worker.flush_early_sends({"r0"})

        assert worker._valid_tokens == {"other": 9}

    def test_the_early_flush_settles_a_folded_offer_without_waiting(
        self, monkeypatch, caplog
    ):
        # An offer that folded never became a batch, so the writer has
        # nothing to record for it. Counting the lists instead would end the
        # wait only at the deadline, whose warning says a batch was lost.
        worker = self._two_offers_parked()
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=0.0)

        with caplog.at_level(logging.WARNING):
            worker.flush_early_sends({"r0"})

        assert "without recording it" not in caplog.text

    def test_the_offer_carries_the_device_work_that_wrote_it(self, monkeypatch):
        # Nothing else names when those blocks stop being written: the runtime's
        # transfer wait covers pending transfers, not compute.
        recorded: list = []
        monkeypatch.setattr(
            pw.torch.rbln,
            "Event",
            lambda: SimpleNamespace(record=recorded.append, synchronize=lambda: None),
        )
        worker = self._worker()

        worker.start_early_push(self._meta(saves=["r0"]))

        assert worker._streamed["r0"].pending_offer_done is not None
        assert recorded == [pw.torch.rbln.current_stream()]

    def test_a_step_with_nothing_to_offer_records_no_event(self, monkeypatch):
        # Recording one costs a device call, and `wait_for_save` runs every
        # step -- including the decode steps, which offer nothing.
        recorded: list = []
        monkeypatch.setattr(
            pw.torch.rbln,
            "Event",
            lambda: SimpleNamespace(record=recorded.append, synchronize=lambda: None),
        )
        worker = self._worker()

        worker.start_early_push(self._meta())

        assert recorded == []

    def test_the_offer_waits_for_that_work_before_the_writer_sees_it(self, monkeypatch):
        # The writer reads the blocks straight out of device memory and reports
        # no error when they are still being written, so the wait is the only
        # thing standing between a half-written block and a silent bad transfer.
        order = []
        monkeypatch.setattr(
            pw.torch.rbln,
            "Event",
            lambda: SimpleNamespace(
                record=lambda _stream: None,
                synchronize=lambda: order.append("wait"),
            ),
        )
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))
        monkeypatch.setattr(
            worker._finished_blocks_inbox, "put", lambda item: order.append("hand")
        )

        worker.release_early_offers()

        assert order == ["wait", "hand"]
        assert worker._streamed["r0"].pending_offer_done is None

    def test_a_step_that_closes_nothing_still_releases_what_is_held(self):
        # The release cannot wait for another closing chunk: steps that close
        # one are not every step, and the offer would sit until one came.
        worker = self._worker()
        worker.start_early_push(self._meta(saves=["r0"]))

        worker.start_early_push(self._meta())  # a step with nothing to offer
        worker.release_early_offers()

        assert worker._finished_blocks_inbox.get_nowait() == ("r0", ([1, 2],))

    def test_the_gate_off_holds_nothing(self):
        # Guard: the direct path had no offer before this change either, so
        # what this pins is that the flag is what turns it on.
        worker = self._worker(enabled=False)

        worker.start_early_push(self._meta(saves=["r0"]))

        assert worker._streamed == {}

    def test_an_early_write_is_kept_out_of_the_completion_accounting(self):
        # `_sending_transfers` is what upstream reports completions from, and
        # the scheduler frees a request's blocks on that report -- this one is
        # still prefilling.
        worker = self._worker()
        _send(worker, released=True)

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

        assert "r0" not in worker._sending_transfers
        # One batch, one handle per overlapping peer rank.
        assert worker._streamed["r0"].transfers == [[ANY, ANY]]

    def test_the_handover_is_marked_on_the_list_upstream_will_park(self, monkeypatch):
        # The request appearing here IS the engine saying it is over. It is one
        # more batch, not a duplicate: what was streamed is the prefix a
        # prefill closed, and a prompt's last block is closed by nothing. The
        # mark goes on the list because that is what upstream parks, and the
        # blocks have to read back unchanged for its own readers.
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        worker = self._worker()
        _send(worker, released=True)
        meta = self._meta(pushes=["r0"])

        worker.start_load_kv(meta)

        assert meta.push_finished_blocks["r0"].handover
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
        _send(worker, released=True, transfers=[[7, 8]])

        worker.start_load_kv(self._meta(pushes=["r0"]))

        assert "r0" not in worker._sending_transfers
        assert worker._streamed["r0"].transfers == [[7, 8]]

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

        assert not isinstance(meta.push_finished_blocks["r1"], pw.OfferedBlocks)

    def test_the_writer_count_is_the_one_the_handover_would_have_sent(self):
        # Guard: an early write is the same one write the handover would have
        # made, so the count the consumer settles on must not move.
        worker = self._worker()
        _send(worker, released=True)

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        assert notifs == {b"RBLNS:0:0:1:1:r0:1"}

    def test_the_delegated_route_never_carries_an_early_write(self, monkeypatch):
        # That route's notification has no room for the range a write filled,
        # so a prefix over it would settle nothing. The handshake is what keeps
        # this unreachable -- see `TransferShape.streams_prefix`.
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "_xfer_blocks_for_req",
            lambda self, req_id, meta: None,
        )
        worker = self._worker()
        worker._overlapping_ranks = {}
        _send(worker, released=True)

        with pytest.raises(AssertionError, match="whole-engine handle"):
            worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([3],)))


class TestFlushEarlySends:
    """Blocks a write is reading can go back to the allocator without the lease
    that normally protects them. The bytes on their way are correct -- the
    prefill did finish -- so they are waited for, not cancelled."""

    @staticmethod
    def _worker(states):
        w = TestEarlySend._worker()
        _send(w, released=True, transfers=[[7]])
        w.nixl_wrapper.check_xfer_state.side_effect = states
        return w

    def test_taking_waits_out_the_whole_budget_not_a_share_of_it(
        self, monkeypatch, caplog
    ):
        # Giving up drops the record, and a batch the writer records after
        # that is one nobody drains -- it goes to upstream, which reports a
        # request this side held back. So a take stops at the latest the
        # budget allows, however many requests are sharing it.
        worker = self._worker(None)
        bound = 0.04
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=bound)
        # Past a quarter of the budget, which is all a share of four would be,
        # and well inside the whole of it -- the widest band on either side.
        settles_at = int(0.6 * bound / pw._EARLY_FLUSH_POLL_INTERVAL_S)
        polls = 0

        class _SettlesLate(dict):
            def get(self, key, default=None):
                nonlocal polls
                send = super().get(key, default)
                if key == "r0":
                    polls += 1
                    if polls >= settles_at and send is not None:
                        send.transfers = [[7]]
                return send

        _send(worker, "r0", released=True, issued=1, transfers=[])
        # Settled already, so only `r0` decides what is logged and released.
        for req_id, handle in (("r1", 8), ("r2", 9), ("r3", 10)):
            _send(worker, req_id, released=True, issued=1, transfers=[[handle]])
        worker._streamed = _SettlesLate(worker._streamed)

        with caplog.at_level(logging.WARNING):
            worker.flush_early_sends({"r0", "r1", "r2", "r3"})

        # On half the budget `r0` is dropped before it settles, and warns.
        assert "without recording it" not in caplog.text

    def test_draining_shares_the_budget_out_and_spends_it_once(self, monkeypatch):
        # Draining is where the budget is divided: a request holding many
        # handles must not leave the rest of the call releasing theirs on one
        # probe, and the call must still cost one budget rather than one each.
        worker = self._worker(None)
        worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
        for req_id, handle in (("r0", 7), ("r1", 8), ("r2", 9)):
            _send(worker, req_id, released=True, transfers=[[handle]])
        bound = 0.03
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=bound)

        worker.flush_early_sends({"r0", "r1", "r2"})

        probed = [
            c.args[0] for c in worker.nixl_wrapper.check_xfer_state.call_args_list
        ]
        counts = [probed.count(h) for h in (7, 8, 9)]
        # Taken first-come, one handle gets them all and the rest one probe.
        assert max(counts) - min(counts) <= max(counts) // 2
        # A budget each would cost three times this many probes.
        assert len(probed) < 2 * bound / pw._EARLY_FLUSH_POLL_INTERVAL_S

    def test_one_flush_spends_one_budget_however_many_handles(self, monkeypatch):
        # The bound belongs to the call, not to each handle. A request holds a
        # handle per batch per peer, and what brings a flush here is what makes
        # every one of them wait the whole bound -- so per handle the engine
        # thread pays it multiplied by a number nothing here bounds.
        worker = self._worker(None)
        worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
        _send(worker, transfers=[[7, 8, 9]])
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=0.02)

        worker.flush_early_sends({"r0"})

        # The first handle spends it; the rest are released on one probe.
        probed = [
            call.args[0] for call in worker.nixl_wrapper.check_xfer_state.call_args_list
        ]
        assert probed.count(8) == 1
        assert probed.count(9) == 1

    def test_a_held_offer_is_dropped_rather_than_sent_later(self):
        # Its blocks go back to the allocator now; releasing the offer at the
        # next step would write into whatever took them.
        worker = self._worker(["DONE"])
        _send(worker, pending_offer=([1, 2],))

        worker.flush_early_sends({"r0"})

        assert worker._streamed == {}

    def test_an_in_flight_write_is_waited_for_and_released(self):
        worker = self._worker(["PROC", "DONE"])

        worker.flush_early_sends({"r0"})

        assert worker.nixl_wrapper.check_xfer_state.call_count == 2
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
        assert worker._streamed == {}
        # The writer holds state for a request it may never see finish.
        assert worker._evict_finished_inbox.get_nowait() == "r0"

    def test_a_batch_issued_but_not_yet_recorded_is_waited_for(self):
        # The writer issues a batch before it records it, so a send read
        # between the two looks idle. `issued` is what says otherwise.
        worker = self._worker(["DONE", "DONE"])
        send = _send(worker, issued=2)

        def record_late():
            time.sleep(pw._EARLY_FLUSH_POLL_INTERVAL_S * 2)
            with worker._sending_transfers_lock:
                send.transfers.append([9])

        writer = threading.Thread(target=record_late)
        writer.start()
        worker.flush_early_sends({"r0"})
        writer.join()

        # Both handles drained: the one already recorded and the late one.
        assert worker.nixl_wrapper.release_xfer_handle.call_count == 2
        assert worker._streamed == {}

    def test_a_batch_the_writer_finished_counts_toward_what_it_issued(
        self, monkeypatch, caplog
    ):
        # The writer moves a batch to `done` where it left no handle in flight,
        # so a send whose batches split between `transfers` and `done` has
        # nothing outstanding. Counting `transfers` alone waits the bound out
        # and then says the blocks are going back under a live write.
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=0.05)
        worker = self._worker(["DONE"])
        _send(worker, issued=2, done=1)

        with caplog.at_level(logging.WARNING):
            worker.flush_early_sends({"r0"})

        assert "without recording it" not in caplog.text
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
        assert worker._streamed == {}

    def test_a_writer_that_never_records_does_not_hold_the_engine(self, monkeypatch):
        # Same bound as a wedged write, and it says the blocks are going back
        # with a write still reading them.
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=0.0)
        worker = self._worker(["DONE"])
        _send(worker, issued=2)

        worker.flush_early_sends({"r0"})

        assert worker._streamed == {}

    def test_a_wedged_write_does_not_take_the_engine_with_it(self, monkeypatch):
        # This runs on the engine main thread, ahead of the forward.
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=0.0)
        worker = self._worker(lambda handle: "PROC")

        worker.flush_early_sends({"r0"})

        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)


class TestAFailedSendIsNotAFailedReceive:
    """Upstream owns the split, and this connector now depends on it.

    A push worker polls its own outbound handles through the same call the
    read side uses, so a failed WRITE reaches upstream's failure handler. In
    0.26 that handler queued the request as a failed receive, and reporting it
    tripped the assertion that every reported receive carries receive metadata
    -- which a request this rank was sending never has. This side carried an
    override for exactly that. 0.30 removed the reason: the handler only
    records the failure in the caller's set, and the drain over that set drops
    any request with no receive metadata before it can be reported. The
    override went with it, so what has to hold is upstream's behaviour."""

    @staticmethod
    def _worker():
        w = _push_worker()
        w.nixl_wrapper = MagicMock()
        w.xfer_stats = MagicMock()
        w._failed_recv_reqs = queue.Queue()
        w._invalid_block_ids = queue.Queue()
        w._is_hma_required = False
        # Upstream's structured failure log names this rank's own engine.
        w.engine_id = "local"
        w._recving_metadata = {}
        return w

    def test_a_failed_send_hands_the_failure_to_the_caller(self):
        # Called through upstream's own handler: it records the failure in the
        # set the poll passes down and releases the handle, and it queues
        # nothing. A queue entry here is what used to kill the engine.
        worker = self._worker()
        failed: set[str] = set()

        assert worker._handle_failed_transfer("r0", 7, failed)

        assert failed == {"r0"}
        assert worker._failed_recv_reqs.empty()
        assert worker._invalid_block_ids.empty()
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)

    def test_the_completion_check_is_what_routes_a_failed_write_there(self):
        # Calling the handler directly says nothing about the poll still
        # reaching it. A failed outbound handle has to arrive through the
        # state check, and be named in the set that check returns.
        worker = self._worker()
        worker.nixl_wrapper.check_xfer_state.return_value = "ERR"
        sending = {"r0": [7]}

        _done, failed = worker._pop_done_transfers(sending)

        assert failed == {"r0"}
        assert worker._failed_recv_reqs.empty()
        assert worker._invalid_block_ids.empty()


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
        assert notifs == {b"RBLNS:0:0:3:1:r0:1"}

    def test_a_pipeline_stage_names_itself(self, monkeypatch):
        # The writer id is the flat rank the per-shard route already pairs by,
        # so a consumer can tell one stage's ranges from another's.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker.vllm_config.parallel_config.pipeline_parallel_size = 4
        worker.world_size = 2
        worker.tp_rank = 1
        setattr_in_package(
            monkeypatch, get_pp_group=lambda: SimpleNamespace(rank_in_group=3)
        )

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1],), ([7],)))

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        # pp_rank 3 of a 2-wide tensor split, tp_rank 1 -> flat rank 7.
        assert {n.split(b":")[1] for n in notifs} == {b"7"}

    def test_several_kv_groups_name_the_range_of_the_counted_one(self):
        # The groups go out together and carry their own lengths, and the one
        # a coverage range is counted in is the full-attention group -- a
        # window's group holds one block whatever the prompt length, so its
        # length describes no request.
        worker = TestPerShardWrite._writing_worker(ranks=1)
        worker._shard_region_group_ids = {("eng", 0): (0, 0)}
        worker._group_specs = [
            MagicMock(spec=SlidingWindowSpec),
            MagicMock(),  # full attention, and the second group at that
        ]
        set_shape(worker, counted_group=1)

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([1], [2, 3]), ([7], [8, 9]))
        )

        notifs = {
            c.kwargs["notif_msg"]
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        assert notifs == {b"RBLNS:0:0:2:1:r0:1"}

    def test_upstream_is_handed_the_message_it_had_before(self):
        # Upstream reads a notification as `req_id:count` with rsplit, so a
        # prefix left on would make it look up a request that does not exist.
        assert RblnNixlPushConnectorWorker._split_coverage(
            b"RBLNS:2:0:5:4:r0:4", {"r0"}
        ) == (2, (0, 5), 4, b"r0:4")

    def test_a_message_without_a_range_passes_through(self):
        # One unit per block: what a range means when nobody named a unit is
        # what every range meant before the field existed.
        assert RblnNixlPushConnectorWorker._split_coverage(b"r0:4", {"r0"}) == (
            None,
            None,
            1,
            b"r0:4",
        )

    def test_a_request_named_like_a_prefix_is_read_as_the_bare_message_it_is(self):
        # A request id is the caller's on an OpenAI-compatible server. One that
        # spells a range leaves five fields with four digits in them, so the
        # shape alone cannot tell it from a real prefix -- what it cannot do is
        # name a request this rank is receiving.
        forged = b"RBLNS:0:0:5:1:x:4"

        assert RblnNixlPushConnectorWorker._split_coverage(
            forged, {"RBLNS:0:0:5:1:x"}
        ) == (
            None,
            None,
            1,
            forged,
        )

    def test_a_prefix_with_too_few_fields_is_left_alone(self):
        # The caller's request id can be the prefix itself, which leaves a
        # bare `id:count` whose fields all parse as numbers. Unpacked without
        # counting them first this raises, on the engine's main thread inside
        # upstream's notification drain, which catches nothing.
        short = b"RBLNS:12:7"

        assert RblnNixlPushConnectorWorker._split_coverage(short, {"RBLNS:12"}) == (
            None,
            None,
            1,
            short,
        )

    def test_a_prefix_whose_fields_do_not_parse_is_left_alone(self):
        # The prefix is four numbers and a request id. A message that opens
        # with it but does not hold them is not ours to strip: doing so would
        # hand upstream a request id this rank invented.
        malformed = b"RBLNS:0:0:5:x:r0:4"

        assert RblnNixlPushConnectorWorker._split_coverage(malformed, {"r0"}) == (
            None,
            None,
            1,
            malformed,
        )


class TestSettleOnCoverage:
    """A writer that names the ranges it filled is settled on those ranges, not
    on how many notifications it sent. That is what lets one writer's KV arrive
    in pieces: its reports are dropped here until its ranges span every block
    this rank registered, leaving upstream the one notification per writing
    rank that its own count is written for."""

    @staticmethod
    def _receiving_worker():
        w = _push_worker()
        w.world_size = 1
        # Three blocks registered: what a writer's ranges have to cover.
        w._recving_metadata = {
            "r0": SimpleNamespace(local_physical_block_ids=([4, 5, 6],))
        }
        w._pending_completion_notifs = pw._CoverageNotifQueue(w)
        w.transfer_topo = MagicMock()
        return w

    @pytest.mark.parametrize(
        "notif, reason",
        [
            (b"HB:engine-1", "heartbeat"),
            (b"r0:1", "names no range"),
            (b"rx:1", "a request this rank is not receiving"),
        ],
    )
    def test_what_upstream_owns_is_passed_straight_through(
        self, handed_through, notif, reason
    ):
        # A heartbeat is not a completion at all, a request written in one
        # transfer names no range, and a request this rank never registered for
        # has no blocks here to span -- upstream counts all three itself.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(notif)

        worker._get_new_notifs()

        assert handed_through == [notif], reason

    def test_our_own_outbound_request_is_left_to_upstream(self, handed_through):
        # A request this rank is sending is upstream's own accounting. The
        # notification carries a range, so nothing earlier in the function can
        # answer for it -- being outbound is the only reason it passes.
        worker = self._receiving_worker()
        worker._reqs_to_process = {"r0"}
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:1:r0:4")

        worker._get_new_notifs()

        assert handed_through == [b"r0:4"]
        assert worker._coverage_by_req == {}

    def test_a_request_this_rank_is_still_sending_is_left_to_upstream(
        self, handed_through
    ):
        # The sibling term. One id can be in both maps on a chained hop, where
        # this rank receives a request it also pushes on; the outbound half is
        # upstream's to count either way.
        worker = self._receiving_worker()
        worker._reqs_to_send = {"r0": object()}
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:1:r0:4")

        worker._get_new_notifs()

        assert handed_through == [b"r0:4"]
        assert worker._coverage_by_req == {}

    def test_an_id_holding_a_colon_is_read_from_the_right_end(self, handed_through):
        # A request id is caller-supplied, so it can hold the separator the
        # count is split off by. Read from the wrong end the id comes back
        # short, no metadata matches it, and the range is released unheld --
        # the request then settles on upstream's count with KV still missing.
        worker = self._receiving_worker()
        worker._recving_metadata = {
            "a:b": SimpleNamespace(local_physical_block_ids=([4, 5, 6],))
        }
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:1:a:b:4")

        worker._get_new_notifs()

        assert handed_through == []
        assert worker._coverage_by_req == {"a:b": {0: [(0, 2)]}}

    def test_a_notification_queued_mid_drain_still_arrives_stripped(self):
        """The writer thread puts into this queue throughout the step.

        Stripping in a pass of its own left a window: a notification arriving
        after that pass and before upstream's drain reached upstream still
        prefixed, and upstream reads the prefix as part of the request id. No
        request has that id, so the range is dropped -- and a request missing a
        range never settles, which makes the loss silent until it hangs.
        """
        worker = self._receiving_worker()
        # Both are requests this rank is receiving, which is what separates a
        # prefix from a request id that spells one.
        worker._recving_metadata["r1"] = SimpleNamespace(
            local_physical_block_ids=([7, 8, 9],)
        )
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:1:r0:1")
        seen = []

        def fake_base(self):
            """Upstream's drain, with one more arriving part way through it."""
            while True:
                try:
                    seen.append(self._pending_completion_notifs.get_nowait())
                except queue.Empty:
                    return set()
                if len(seen) == 1:
                    self._pending_completion_notifs.put(b"RBLNS:0:0:3:1:r1:1")

        with patch.object(NixlPushConnectorWorker, "_get_new_notifs", fake_base):
            worker._get_new_notifs()

        assert seen == [b"r0:1", b"r1:1"]

    def test_the_ranges_survive_a_request_the_worker_has_not_seen(self, monkeypatch):
        # Built through __init__ rather than by the helper, which hands the map
        # in already made: a plain dict passes there and KeyErrors on the first
        # notification.
        def stub_base(self, *a):
            # The three the gate reads; the real base derives them from the
            # config.
            self._is_hma_required = False
            self.use_host_buffer = False
            self._group_specs = []

        monkeypatch.setattr(RblnNixlWorkerBase, "__init__", stub_base)
        worker = RblnNixlPushConnectorWorker(mock_vllm_config(), "eng", MagicMock())
        worker.shutdown = lambda: None  # the writer state __del__ reaches is absent
        worker._coverage_by_req["r0"][0].append((0, 1))
        assert worker._coverage_by_req == {"r0": {0: [(0, 1)]}}

    def test_a_writer_that_has_sent_part_does_not_settle_the_request(
        self, handed_through
    ):
        worker = self._receiving_worker()
        # Three blocks registered; this writer has filled the first two.
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:1:r0:1")

        worker._get_new_notifs()

        assert handed_through == []

    def test_the_pieces_together_settle_it(self, handed_through):
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:1:r0:1")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"RBLNS:0:2:3:1:r0:1")
        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_each_writer_is_released_on_its_own_ranges(self, handed_through):
        # One writer spanning the whole list says nothing about the other,
        # whose layers are just as missing -- but it is upstream that waits for
        # the second, counting one notification per writing rank. Handing it
        # both is what lets that count reach its total.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:1:r0:2")
        worker._get_new_notifs()
        assert handed_through == [b"r0:2"]

        worker._pending_completion_notifs.put(b"RBLNS:1:0:3:1:r0:2")
        worker._get_new_notifs()

        assert handed_through == [b"r0:2", b"r0:2"]

    def test_a_finer_unit_needs_that_many_more_to_settle(self, handed_through):
        # Three blocks at four units each: spanning three units is one block,
        # not the request. Read in the wrong unit this settles at once.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:4:r0:1")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"RBLNS:0:3:12:4:r0:1")
        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_each_writer_is_measured_in_its_own_unit(self, handed_through):
        # One producer can serve two peers at different units, so a consumer
        # cannot hold every writer to one of them.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:1:r0:2")
        worker._get_new_notifs()
        assert handed_through == [b"r0:2"]

        worker._pending_completion_notifs.put(b"RBLNS:1:0:6:2:r0:2")
        worker._get_new_notifs()

        assert handed_through == [b"r0:2", b"r0:2"]

    def test_a_writer_that_changes_its_unit_is_refused(self, handed_through):
        # Ranges already counted in one unit cannot be compared against a
        # range counted in another; taking the second would reach the total
        # early and settle a request whose KV is incomplete.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:2:1:r0:1")
        worker._get_new_notifs()

        worker._pending_completion_notifs.put(b"RBLNS:0:2:12:4:r0:1")
        with pytest.raises(RuntimeError, match="changed the coverage unit"):
            worker._get_new_notifs()

    def test_a_unit_below_one_is_refused(self, handed_through):
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:0:r0:1")

        with pytest.raises(RuntimeError, match="units per block"):
            worker._get_new_notifs()

    def test_one_writer_s_ranges_do_not_settle_another(self, handed_through):
        # Held per writer, not per request: read as one bucket, a writer that
        # has covered the list settles a writer that has filled one block of it,
        # and the request goes on with that rank's KV missing.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:1:r0:2")
        worker._get_new_notifs()
        assert handed_through == [b"r0:2"]

        worker._pending_completion_notifs.put(b"RBLNS:1:0:1:1:r0:2")
        worker._get_new_notifs()

        assert handed_through == [b"r0:2"]

    def test_a_resent_range_does_not_undo_what_it_overlaps(self, handed_through):
        # A preempted request re-sends a range already inside one that landed,
        # and a later batch continues past it. Taking each range's end as the
        # reach rather than the furthest seen walks the coverage backwards, and
        # the request never settles.
        worker = self._receiving_worker()
        worker._recving_metadata = {
            "r0": SimpleNamespace(local_physical_block_ids=([1, 2, 3, 4, 5, 6],))
        }
        for span in (b"0:3", b"1:2", b"3:6"):
            worker._pending_completion_notifs.put(b"RBLNS:0:" + span + b":1:r0:1")

        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_one_write_for_the_whole_request_settles_at_once(self, handed_through):
        # Guard: this is every request today, and it has to settle exactly
        # where the count would have settled it.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:3:1:r0:1")

        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    def test_a_resend_does_not_stand_in_for_the_gap_it_skips(self, handed_through):
        # A preempted request is rescheduled from the start of its block list,
        # so a writer re-sends what it already sent. Counting blocks would
        # reach three here with the third never written.
        worker = self._receiving_worker()
        for span in (b"0:1", b"0:1", b"0:1"):
            worker._pending_completion_notifs.put(b"RBLNS:0:" + span + b":1:r0:1")

        worker._get_new_notifs()

        assert handed_through == []

    def test_a_later_range_does_not_close_an_earlier_gap(self, handed_through):
        # The ranges reach the last block while the middle one was never
        # written. Whether a range starts past what has been covered is the
        # only thing separating this from a complete request.
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:1:1:r0:1")
        worker._pending_completion_notifs.put(b"RBLNS:0:2:3:1:r0:1")

        worker._get_new_notifs()

        assert handed_through == []

    def test_filling_the_gap_settles_it(self, handed_through):
        worker = self._receiving_worker()
        worker._pending_completion_notifs.put(b"RBLNS:0:0:1:1:r0:1")
        worker._pending_completion_notifs.put(b"RBLNS:0:2:3:1:r0:1")
        worker._get_new_notifs()
        assert handed_through == []

        worker._pending_completion_notifs.put(b"RBLNS:0:1:2:1:r0:1")
        worker._get_new_notifs()

        assert handed_through == [b"r0:1"]

    @pytest.mark.parametrize("spans", [[], [(0, 0)], [(1, 2)]])
    def test_a_request_of_no_blocks_is_covered_by_whatever_arrives(self, spans):
        # Vacuously true, and the answer cannot depend on where the ranges sit:
        # read as a gap, the writer is held and the receive never settles.
        assert pw.RblnNixlPushConnectorWorker._covers(spans, 0) is True

    def test_a_finished_request_drops_its_ranges(self, monkeypatch):
        # A retry reuses the request id, so leftover ranges would settle it
        # before the retry had written anything.
        worker = self._receiving_worker()
        worker._coverage_by_req["r0"][0].append((0, 3))
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_transfer_results",
            lambda self: KVConnectorTransferResults(finished_recving={"r0"}),
        )

        worker.get_finished()

        assert worker._coverage_by_req == {}

    def test_a_request_written_in_one_transfer_drops_its_token_count(self, monkeypatch):
        # `_forget_send` only runs for a request written in batches, so an
        # ordinary push would keep its count for the life of the process.
        worker = self._receiving_worker()
        worker._valid_tokens = {"r0": 33, "other": 9}
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_transfer_results",
            lambda self: KVConnectorTransferResults(finished_sending={"r0"}),
        )

        worker.get_finished()

        assert worker._valid_tokens == {"other": 9}


class TestABatchThatSendsNothing:
    """A streamed request is reported once `done` reaches `issued`, so a batch
    that issues no write still has to raise `done` -- otherwise `done` stays
    one short of `issued` forever and the request never finishes."""

    @staticmethod
    def _worker():
        w = TestPerShardWrite._writing_worker(ranks=2)
        _send(w, released=True)
        w.xfer_stats = MagicMock()
        # The failure path names this rank's own engine in its log line.
        w.engine_id = "local"
        return w

    def test_a_write_with_no_blocks_left_still_counts(self):
        # The window can come up empty when the offer adds nothing past what
        # the high-water mark already covers.
        worker = self._worker()

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([],), ([],)))

        assert worker._streamed["r0"].done == worker._streamed["r0"].issued == 1
        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 0

    def test_every_peer_failing_still_counts(self):
        worker = self._worker()
        worker.nixl_wrapper.make_prepped_xfer.side_effect = RuntimeError("boom")

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1, 2],), ([3, 4],)))

        assert worker._streamed["r0"].done == 1
        assert worker._streamed["r0"].transfers == []

    def test_a_request_not_streamed_counts_nothing(self):
        # Guard: the count only exists for requests this side seals.
        worker = self._worker()
        worker._streamed.clear()

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([],), ([],)))

        assert worker._streamed == {}


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
        w._coverage_by_req = defaultdict(lambda: defaultdict(list))
        w._coverage_units_by_req = defaultdict(dict)
        _send(w, released=True)
        return w

    @staticmethod
    def _upstream_reports_nothing(monkeypatch):
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_transfer_results",
            lambda self: KVConnectorTransferResults(),
        )

    def test_a_request_upstream_reported_leaves_no_record_and_no_handle(
        self, monkeypatch
    ):
        # The lease is timed, so it falls due whatever the writer is doing,
        # and a batch can be in flight when the record goes. Its handles go
        # with it, and teardown reclaims what a record still holds.
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_transfer_results",
            lambda self: KVConnectorTransferResults(finished_sending={"r0"}),
        )
        setattr_in_package(monkeypatch, _EARLY_FLUSH_DRAIN_TIMEOUT_S=0.0)
        # Still writing: a landed batch would be released by the seal scan
        # anyway, which would let this pass without the drop.
        worker = self._worker(None)
        worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
        _send(worker, transfers=[[7]], issued=1)

        worker.get_finished()

        assert "r0" not in worker._streamed
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)

    def test_a_landed_batch_is_not_reported_before_the_seal(self, monkeypatch):
        # The engine has not finished the request, so its blocks must not be
        # freed however much of its KV is already across.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        _send(worker, transfers=[[7]], issued=1)

        done_sending, _ = worker.get_finished()

        assert done_sending == set()

    def test_the_seal_completes_a_request_whose_batches_already_landed(
        self, monkeypatch
    ):
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        _send(worker, transfers=[[7]], issued=1)
        worker.get_finished()
        _send(worker, handover_written=True)

        done_sending, _ = worker.get_finished()

        assert done_sending == {"r0"}

    def test_a_sealed_report_tells_the_writer_to_drop_its_state(self, monkeypatch):
        # Upstream drops the writer's state for what it reports itself, and it
        # has already gone past this request for this step. Nothing else takes
        # the writer's copy out, so a report that does not say so leaves the
        # finished blocks and the pending registration held until the lease.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        _send(worker, transfers=[[7]], issued=1)
        worker.get_finished()
        _send(worker, handover_written=True)

        done_sending, _ = worker.get_finished()

        assert done_sending == {"r0"}
        assert worker._evict_finished_inbox.get_nowait() == "r0"
        assert worker._push_writer_wake.is_set()

    def test_a_sealed_request_waits_for_its_last_batch(self, monkeypatch):
        # Two batches handed over, one still going: the seal alone does not
        # finish it.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE", "PROC", "DONE"])
        _send(
            worker,
            transfers=[[7], [8]],
            issued=2,
            handover_written=True,
        )

        assert worker.get_finished()[0] == set()

        assert worker.get_finished()[0] == {"r0"}

    def test_a_finished_request_leaves_nothing_behind(self, monkeypatch):
        # A retry reuses the request id, so a leftover count would report the
        # retry finished before it had written anything.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker(["DONE"])
        _send(worker, transfers=[[7]], issued=1, handover_written=True)
        worker._reqs_to_send = {"r0": 1.0}
        worker._reqs_to_process = {"r0"}
        worker.consumer_notification_counts_by_req = {"r0": 1}

        worker.get_finished()

        assert worker._streamed == {}
        assert worker._reqs_to_send == {}
        assert worker._reqs_to_process == set()
        assert worker.consumer_notification_counts_by_req == {}

    def test_a_request_upstream_owns_is_left_to_it(self, monkeypatch):
        # Guard: with the flag off nothing is written early, so none of this
        # runs and upstream reports every request as it did before.
        self._upstream_reports_nothing(monkeypatch)
        worker = self._worker([])
        worker._streamed.clear()

        assert worker.get_finished()[0] == set()
        assert worker.nixl_wrapper.check_xfer_state.call_count == 0


class TestAStreamedSendThatLostAWrite:
    """A batch raises `issued` whether or not its writes went out, so the
    counts alone report a request whose KV never moved as sent -- freeing the
    producer's blocks while the consumer is still waiting for them. The lease
    owns such a request instead."""

    def test_a_peer_that_did_not_submit_marks_the_send(self):
        worker = TestABatchThatSendsNothing._worker()
        worker.nixl_wrapper.make_prepped_xfer.side_effect = [RuntimeError("boom"), 7]

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1, 2],), ([3, 4],)))

        assert worker._streamed["r0"].transfers == [[7]]
        assert worker._streamed["r0"].failed

    def test_a_later_batch_does_not_clear_an_earlier_failure(self):
        # The flag is the request's, not the batch's: the hole a dropped batch
        # left is still there when the next one lands whole.
        worker = TestABatchThatSendsNothing._worker()
        _send(worker, failed=True)
        worker.nixl_wrapper.make_prepped_xfer.side_effect = [7, 8]

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([1, 2],), ([3, 4],)))

        assert worker._streamed["r0"].failed

    def test_a_transfer_that_came_back_failed_is_one_too(self, monkeypatch):
        # The submission went out and the transfer itself failed. Nothing is
        # left in flight, so the batch counts as done and the request would
        # read as sent.
        TestSealedCompletion._upstream_reports_nothing(monkeypatch)
        worker = TestSealedCompletion._worker(["ERR"])
        # The failure path names this rank's own engine in its log line.
        worker.engine_id = "local"
        _send(worker, transfers=[[7]], issued=1, handover_written=True)
        worker._reqs_to_send = {"r0": 1.0}

        done_sending, _ = worker.get_finished()

        assert done_sending == set()
        assert worker._reqs_to_send == {"r0": 1.0}

    def test_a_failed_send_is_left_to_its_lease(self, monkeypatch):
        TestSealedCompletion._upstream_reports_nothing(monkeypatch)
        worker = TestSealedCompletion._worker(["DONE"])
        _send(
            worker,
            transfers=[[7]],
            issued=1,
            handover_written=True,
            failed=True,
        )
        worker._reqs_to_send = {"r0": 1.0}

        done_sending, _ = worker.get_finished()

        assert done_sending == set()
        assert worker._reqs_to_send == {"r0": 1.0}
        assert worker._streamed == {}


class TestEmptyReceive:
    """A request turned away before it was scheduled is registered as a receive
    of no blocks. Nothing is ever written into no blocks, so the notification
    that settles a receive never comes and the entry outlives the request.

    Dropped, not reported: the engine deleted that request in the same pass
    that registered the receive, so naming it in a completion asks the
    scheduler about a request it no longer has."""

    @staticmethod
    def _worker(monkeypatch):
        monkeypatch.setattr(
            NixlPushConnectorWorker, "start_load_kv", lambda self, metadata: None
        )
        monkeypatch.setattr(
            NixlPushConnectorWorker,
            "get_transfer_results",
            lambda self: KVConnectorTransferResults(),
        )
        w = _push_worker()
        w.use_host_buffer = False
        return w

    def test_a_receive_of_nothing_is_dropped_and_not_reported(self, monkeypatch):
        # Reported, it would reach `assert req_id in self.requests` in the
        # scheduler for a request freed on the pass that registered this.
        worker = self._worker(monkeypatch)
        worker._recving_metadata["r0"] = MagicMock()

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ()}))

        assert worker._recving_metadata == {}
        assert worker.get_finished()[1] == set()

    def test_an_empty_group_counts_as_nothing(self, monkeypatch):
        # The block ids arrive per KV cache group, so "no blocks" can be a
        # group carrying none rather than no groups at all.
        worker = self._worker(monkeypatch)
        worker._recving_metadata["r0"] = MagicMock()

        worker.start_load_kv(TestEarlySend._meta(recvs={"r0": ([],)}))

        assert worker._recving_metadata == {}

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
            _send(w, total=total)
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

    def _chunked(
        self,
        *,
        total,
        offered,
        have,
        registered=2,
        gpb=2,
        hwm=0,
        chunks=0,
        valid=None,
        areas=1,
        expand=1,
    ):
        """Call the window directly with a grid. The offer's block list is
        `have` long and holds `offered` tokens; the consumer registered
        `registered` blocks of the prompt's `total`. `valid` is the request's
        final token count, which only the handover carries. `areas` above one
        is a context cut, where an area IS a token range of the block and
        `gpb` counts the chunks of one area rather than of the block."""
        w = self._worker(total=total)
        w.block_size = 16
        w._physical_blocks_per_logical_kv_block = expand
        w._kv_areas = areas
        w._kv_split_axis = KVSplitAxis.NON_HEAD if areas > 1 else KVSplitAxis.HEAD
        send = w._streamed["r0"]
        send.issued_hwm = hwm
        send.issued_chunks = chunks
        tail = None
        if valid is not None:
            set_shape(w, chunk_mode=True)
            tail = w._tail_chunks(total, valid, chunks_per_span=gpb)
        return w, w._stream_window(
            "r0",
            (list(range(have)),),
            (list(range(100, 100 + registered)),),
            chunk_grid=(2, gpb),
            offered_tokens=offered,
            tail=tail,
            remote_expand=expand,
        )

    def test_an_expanded_block_places_the_window_by_the_expanded_count(self):
        # Both sides count in kernel blocks, so our window has to be placed in
        # that unit too. Off the logical total it opens `total` blocks early:
        # every batch writes the wrong local blocks into correctly addressed
        # remote ones, and both lists stay the right length, so nothing
        # downstream notices.
        w, out = self._chunked(total=2, offered=4 * 16, have=4, registered=2, expand=2)
        local, remote, _span, _pieces = out

        # 4 kernel blocks of ours, 2 registered: the window opens at our 2.
        assert local == ([2, 3],)
        assert remote == ([100, 101],)
        assert w._physical_blocks_per_logical_kv_block == 2

    def test_a_block_short_of_the_last_owes_its_whole_rest(self):
        # The tail count sizes the request's LAST block, and here streaming
        # stopped two blocks before it: block 1 owes every chunk it has left,
        # not the one the last block's tokens reach. Sized by the tail, the
        # rest of block 1 is already covered and the chunk never leaves.
        _w, out = self._chunked(
            total=4, offered=0, have=4, registered=4, hwm=1, chunks=1, valid=3 * 16 + 4
        )
        _local, _remote, _span, pieces = out

        assert pieces == ((1, 101, None, (1, 2)),)

    def test_an_offer_past_the_whole_prompt_is_refused(self):
        # Upstream truncates a P-side request so no decode step is scheduled,
        # and the allocator leases it the prompt's blocks and no more. An offer
        # naming more closed blocks than that cannot be placed inside the
        # consumer's list, and writing it anyway addresses past what the peer
        # registered.
        with pytest.raises(RuntimeError, match="closed block"):
            self._chunked(total=2, offered=3 * 16, have=3, registered=2)

    def test_an_offer_short_of_the_consumer_window_takes_no_chunks(self):
        # The consumer's cache held the prompt's head, so it registered only
        # the tail and our block 0 is before its window. An offer that has not
        # reached the window yet holds nothing the consumer wants -- and the
        # block the chunks would come from is past the end of the offer.
        w, out = self._chunked(total=2, offered=8, have=1, registered=1)
        local, remote, span, pieces = out

        assert local == ([],) and remote == ([],)
        assert pieces == ()
        assert span == (0, 0)
        assert w._streamed["r0"].issued_chunks == 0

    def test_a_context_cut_counts_the_chunks_of_the_block_not_of_one_area(self):
        # Two areas, two chunks each, so the block is four chunks of four
        # tokens. Eight tokens of it are computed: that is the whole of area 0,
        # and the piece has to say which area it is -- the chunk range is
        # indexed inside one, and every area carries the same indices.
        w, out = self._chunked(total=4, offered=3 * 16 + 8, have=4, gpb=2, areas=2)
        _, _, span, pieces = out

        assert pieces == ((3, 101, 0, (0, 2)),)
        assert span == (0, 1 * 4 + 2)
        assert w._streamed["r0"].issued_chunks == 2

    def test_a_context_cut_splits_a_piece_that_crosses_an_area(self):
        # Twelve of sixteen tokens is three of the block's four chunks, which
        # is area 0 whole and the first chunk of area 1. One range cannot name
        # both, since each is indexed inside its own area.
        _, out = self._chunked(total=4, offered=3 * 16 + 12, have=4, gpb=2, areas=2)
        _, _, _, pieces = out

        assert pieces == ((3, 101, 0, (0, 2)), (3, 101, 1, (0, 1)))

    def test_a_partial_last_block_leaves_as_chunks(self):
        # Four blocks held, three closed and the fourth holding 8 of 16 tokens.
        # The consumer registered two, so its block 0 goes whole and its block
        # 1 goes half -- and the range says so in chunks, not blocks.
        w, out = self._chunked(total=4, offered=3 * 16 + 8, have=4)
        local, remote, span, pieces = out

        assert local == ([2],) and remote == ([100],)
        assert span == (0, 1 * 2 + 1)
        assert pieces == ((3, 101, None, (0, 1)),)  # block 3, their 101, chunk 0
        assert w._streamed["r0"].issued_chunks == 1

    def test_the_next_batch_only_advances_the_chunks(self):
        # Four chunks a block, so a block can fill in steps without closing:
        # 12 of 16 tokens is three chunks, one past what already went. Nothing
        # goes whole and the range moves by that one chunk.
        w, out = self._chunked(
            total=4, offered=3 * 16 + 12, have=4, gpb=4, hwm=1, chunks=2
        )
        local, remote, span, pieces = out

        assert local == ([],) and remote == ([],)
        assert span == (1 * 4 + 2, 1 * 4 + 3)
        # Picks up where the last batch stopped.
        assert pieces == ((3, 101, None, (2, 3)),)

    def test_a_block_that_closes_sends_only_the_chunks_still_missing(self):
        # Half of the consumer's block 1 went out as a chunk last batch. It has
        # closed now, and writing it whole would repeat that half -- so the
        # write takes the rest of it in chunks and the whole-block range starts
        # after it. Here that leaves the whole-block range empty.
        w, out = self._chunked(total=4, offered=4 * 16, have=4, hwm=1, chunks=1)
        local, remote, span, pieces = out

        assert local == ([],) and remote == ([],)
        assert pieces == ((3, 101, None, (1, 2)),)  # chunk 1 only, not the block
        assert span == (1 * 2 + 1, 2 * 2)
        assert w._streamed["r0"].issued_hwm == 2
        assert w._streamed["r0"].issued_chunks == 0

    def test_a_step_that_closes_no_chunk_writes_nothing(self):
        w, out = self._chunked(total=4, offered=3 * 16 + 8, have=4, hwm=1, chunks=1)
        local, remote, span, pieces = out

        # Every group keeps its slot -- the window empties one, it does not
        # drop the rest -- so a hybrid's window block still rides the write
        # this batch would have carried.
        assert local == ([],) and remote == ([],) and pieces == ()
        assert span == (3, 3)

    def test_the_handover_stops_at_the_tokens_the_last_block_holds(self):
        # The last block half-filled and half of that already streamed. The
        # handover owes the chunk that holds tokens, not every chunk left in
        # the block: the tokens past the prompt are what chunk mode exists not
        # to send, and this is the write the consumer is waiting on.
        w, out = self._chunked(
            total=4, offered=0, have=4, gpb=4, hwm=1, chunks=1, valid=3 * 16 + 8
        )
        _local, _remote, span, pieces = out

        assert pieces == ((3, 101, None, (1, 2)),)  # chunk 1 only, not chunks 1..4
        # The range still names the whole request: it says what the write is
        # responsible for, not which descriptors went out.
        assert span == (1 * 4 + 1, 2 * 4)

    def test_a_peer_without_chunks_leaves_the_partial_block_alone(self):
        # The offer holds the block being filled, but this peer's lists have
        # no chunk range. It must write the blocks that closed and stop --
        # writing the partial one whole would ship KV not computed yet.
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._shard_chunk_grids = {("eng", 0): None}
        meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )

        worker._xfer_blocks_for_req("r0", meta)

        assert self._sent(worker) == [[2]]  # our block 2, not 2 and 3
        assert worker._streamed["r0"].issued_hwm == 1

    def test_a_batch_with_a_tail_writes_both_ranges_in_one_transfer(self):
        # The chunks come from the second range of the same lists, so they
        # ride the batch rather than costing it a transfer of its own -- which
        # would be a second notification for one offer.
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._shard_chunk_grids = {("eng", 0): (2, 2)}
        meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )

        worker._xfer_blocks_for_req("r0", meta)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 1
        local = worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        # 2 regions x (1 whole block + 2 heads x 1 chunk).
        assert len(local) == 2 * 3
        whole = 2 * worker.num_blocks
        assert sum(1 for d in local if d >= whole) == 2 * 2
        # And the range is counted in chunks, not blocks.
        notif = worker.nixl_wrapper.make_prepped_xfer.call_args.kwargs["notif_msg"]
        assert notif.startswith(b"RBLNS:0:0:3:2:")

    def test_a_context_cut_writes_the_chunks_of_one_area_only(self):
        # Two areas, so a position IS a token range: area 0 holds the block's
        # first half and area 1 the second. Eight of sixteen tokens are
        # computed, which is area 0 whole -- and area 1 holds nothing yet, so
        # its descriptors must not go out.
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._kv_areas = 2
        worker._kv_split_axis = KVSplitAxis.NON_HEAD
        worker._shard_chunk_grids = {("eng", 0): (1, 2)}
        meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )

        worker._xfer_blocks_for_req("r0", meta)

        local = worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        whole = 2 * worker.num_blocks
        # Our block 2 whole in both regions, then area 0's two chunks of our
        # block 3 -- not the four that naming every area would give.
        assert sorted(d for d in local if d < whole) == [2, 10]
        assert sorted(d for d in local if d >= whole) == [22, 23]

    def test_a_context_cut_writes_the_second_area_at_its_own_positions(self):
        # Twelve of sixteen tokens: area 0 whole and the first chunk of area 1.
        # The two pieces land in different regions, which is the whole point of
        # the span index -- every area holds chunk 0 of this block.
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._kv_areas = 2
        worker._kv_split_axis = KVSplitAxis.NON_HEAD
        worker._shard_chunk_grids = {("eng", 0): (1, 2)}
        meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 3 * 16 + 12, worker._streamed["r0"].serial
        )

        worker._xfer_blocks_for_req("r0", meta)

        local = worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        whole = 2 * worker.num_blocks
        # Area 0's chunks 0 and 1 at region 0, area 1's chunk 0 at region 1.
        assert sorted(d for d in local if d >= whole) == [22, 23, 38]

    def test_the_next_batch_does_not_repeat_a_chunk_it_already_sent(self):
        """A block that closes after part of it went out is written from where
        that part stopped. Writing it whole instead moves those bytes a second
        time, and every block passes through the partly-filled state whenever a
        prefill chunk is narrower than a block -- so it would be every block."""
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._shard_chunk_grids = {("eng", 0): (2, 2)}

        first = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        first.local_block_ids = pw.OfferedBlocks(
            first.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", first)

        second = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        second.local_block_ids = pw.OfferedBlocks(
            second.local_block_ids, 4 * 16, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", second)

        # Nothing goes whole in the second write: the only block it still owes
        # is the one it half-sent, and it owes just the other half.
        assert self._sent(worker) == [[2], []]
        descs = worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        whole = 2 * worker.num_blocks
        assert all(d >= whole for d in descs)
        assert len(descs) == 2 * 2  # 2 regions x 2 heads x 1 chunk

    def test_the_closing_batch_stops_at_the_tokens_the_last_block_holds(self):
        # The same two batches as above, with a token count on the request.
        # Half the last block went out with the first batch and the rest of it
        # holds no tokens, so the batch that closes the request owes nothing --
        # sending those chunks would move KV the prefill never wrote.
        worker = self._worker(total=4)
        worker.block_size = 16
        set_shape(worker, chunk_mode=True)
        worker._shard_chunk_grids = {("eng", 0): (2, 2)}
        worker._valid_tokens = {"r0": 3 * 16 + 8}

        first = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        first.local_block_ids = pw.OfferedBlocks(
            first.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", first)

        second = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        second.local_block_ids = pw.OfferedBlocks(
            second.local_block_ids, 4 * 16, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", second)

        # One transfer, the first batch's. The second names no descriptor.
        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 1

    def test_a_closing_batch_that_owes_only_a_chunk_names_no_whole_block(self):
        # Four chunks a block. The first batch took the two the offer had
        # filled, so the handover owes the third and its block never closes:
        # nothing goes whole and the block list is empty. The cut has to take
        # the block from the piece, not from the list's last element.
        worker = self._worker(total=4)
        worker.block_size = 16
        set_shape(worker, chunk_mode=True)
        worker._shard_chunk_grids = {("eng", 0): (2, 4)}
        worker._valid_tokens = {"r0": 3 * 16 + 12}

        first = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        first.local_block_ids = pw.OfferedBlocks(
            first.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", first)

        second = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        second.local_block_ids = pw.OfferedBlocks(
            second.local_block_ids, 0, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", second)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 2
        descs = worker.nixl_wrapper.make_prepped_xfer.call_args.args[2]
        whole = 2 * worker.num_blocks
        assert all(d >= whole for d in descs)
        assert len(descs) == 2 * 2  # 2 regions x 2 heads x 1 chunk

    def test_streaming_alone_stops_the_closing_batch_at_the_same_place(self):
        # The same two batches with the knob off: streaming reaches the chunk
        # range on its own, so the bound the last block is cut to cannot be
        # chunk mode's. Without it the closing batch writes the rest of a block
        # the forward pass had not filled.
        worker = self._worker(total=4)
        worker.block_size = 16
        set_shape(worker, streams_prefix=True)
        worker._shard_chunk_grids = {("eng", 0): (2, 2)}
        worker._valid_tokens = {"r0": 3 * 16 + 8}

        first = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        first.local_block_ids = pw.OfferedBlocks(
            first.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", first)

        second = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        second.local_block_ids = pw.OfferedBlocks(
            second.local_block_ids, 4 * 16, worker._streamed["r0"].serial
        )
        worker._xfer_blocks_for_req("r0", second)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 1

    def test_a_peer_with_a_grid_is_counted_in_chunks_even_unstreamed(self):
        # No total, so there is no window and the whole list goes at once -- but
        # the peer still counts in chunks, and the `per_block` the notification
        # carries says so. A range named in blocks never reaches the total the
        # consumer computes from that unit, and the request never settles.
        gpb, heads = 2, 2
        worker = self._worker()
        worker._shard_chunk_grids = {("eng", 0): (heads, gpb)}

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5, 6],))
        )

        spans = {
            tuple(c.kwargs["notif_msg"].decode().split(":")[2:5])
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        # Three blocks at two chunks each, and the unit beside the range.
        assert spans == {("0", "6", "2")}

    def test_a_part_filled_chunk_is_not_offered(self):
        # The count rounds DOWN here, unlike at handover: a chunk the offer
        # reaches into but does not fill holds tokens the forward has not
        # computed, and the writer reads device memory without checking.
        # 49 tokens over blocks of 16 leaves one token in a chunk of eight.
        gpb, heads = 2, 2
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._shard_chunk_grids = {("eng", 0): (heads, gpb)}
        meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5, 6, 7],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 49, worker._streamed["r0"].serial
        )

        worker._xfer_blocks_for_req("r0", meta)

        spans = {
            tuple(c.kwargs["notif_msg"].decode().split(":")[2:4])
            for c in worker.nixl_wrapper.make_prepped_xfer.call_args_list
        }
        # Three whole blocks at two chunks each, and nothing of the fourth.
        assert spans == {("0", "6")}

    def test_a_prefill_writes_every_chunk_exactly_once(self):
        """Walk a whole prefill and account for every byte of it.

        Two properties at once: nothing written twice, which is what makes
        streaming in chunks cheaper than not streaming at all, and nothing left
        out, because the ranges the consumer settles on say it was written -- a
        hole there is a torn block hashed into its prefix cache. Where the
        offer STOPS is the sibling's (`test_a_part_filled_chunk_is_not_offered`):
        a range that rounds the wrong way still covers each piece once.
        """
        gpb, heads, regions = 2, 2, 2
        worker = self._worker(total=4)
        worker.block_size = 16
        worker._shard_chunk_grids = {("eng", 0): (heads, gpb)}
        whole_descs = regions * worker.num_blocks

        def decode(desc: int) -> set[tuple[int, int, int, int]]:
            """(region, block, head, chunk) a descriptor id covers."""
            if desc < whole_descs:
                region, block = divmod(desc, worker.num_blocks)
                return {(region, block, h, c) for h in range(heads) for c in range(gpb)}
            within = desc - whole_descs
            block_ix, rest = divmod(within, heads * gpb)
            region, block = divmod(block_ix, worker.num_blocks)
            head, chunk = divmod(rest, gpb)
            return {(region, block, head, chunk)}

        written: list[tuple[int, int, int, int]] = []
        spans: list[tuple[int, int]] = []
        # A chunk is 8 tokens, so the offer grows by one chunk a step until the
        # four blocks are full; the handover then carries no token count.
        # A step of six against a chunk of eight, so an offer lands inside a
        # chunk as often as on one -- a step of exactly one chunk would make
        # a range that rounds the wrong way land on the same boundary.
        for tokens in [*range(6, 4 * 16, 6), 4 * 16, 0]:
            meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5, 6, 7],))
            meta.local_block_ids = pw.OfferedBlocks(
                meta.local_block_ids, tokens, worker._streamed["r0"].serial
            )
            worker.nixl_wrapper.make_prepped_xfer.reset_mock()
            worker._xfer_blocks_for_req("r0", meta)
            for call in worker.nixl_wrapper.make_prepped_xfer.call_args_list:
                for desc in call.args[2]:
                    written.extend(decode(int(desc)))
                notif = call.kwargs["notif_msg"].decode()
                lo, hi = notif.split(":")[2:4]
                spans.append((int(lo), int(hi)))

        expected = {
            (r, b, h, c)
            for r in range(regions)
            for b in range(4)  # this rank's four blocks, all of them offered
            for h in range(heads)
            for c in range(gpb)
        }
        assert sorted(written) == sorted(expected), (
            "a chunk was written twice or not at all"
        )
        # And the ranges the consumer settles on tile the same span end to end.
        assert [s for s in spans if s[0] != s[1]] == sorted(
            {s for s in spans if s[0] != s[1]}
        )
        merged = [s for s in spans if s[0] != s[1]]
        assert merged[0][0] == 0 and merged[-1][1] == 4 * gpb
        assert all(a[1] == b[0] for a, b in zip(merged, merged[1:]))

    @pytest.mark.parametrize(
        "second, expected",
        [
            # Both peers count in halves of a block, so the request does.
            ((2, 2), b"RBLNS:0:0:3:2:"),
            # One of them does not, so neither gets chunks: one window serves
            # every peer and `issued_hwm` is the request's, so a per-peer unit
            # would count one request's coverage two ways. The block being
            # filled is then left alone -- one block, not two.
            (None, b"RBLNS:0:0:1:1:"),
            # Two peers that both count in chunks, but not the same ones. A set
            # holding one grid and a `None` cannot tell "they disagree" from
            # "take whichever is there": `set.pop()` answers `None` on it either
            # way. Two real grids separate the two.
            ((2, 4), b"RBLNS:0:0:1:1:"),
        ],
    )
    def test_every_peer_of_a_request_counts_in_one_unit(self, second, expected):
        worker = TestPerShardWrite._writing_worker(ranks=2)
        worker._physical_blocks_per_logical_kv_block = 1
        worker._recving_metadata = {}
        _send(worker, total=4)
        worker.block_size = 16
        worker._shard_chunk_grids = {("eng", 0): (2, 2), ("eng", 1): second}
        meta = TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 3 * 16 + 8, worker._streamed["r0"].serial
        )

        worker._xfer_blocks_for_req("r0", meta)

        assert worker.nixl_wrapper.make_prepped_xfer.call_count == 2
        for call in worker.nixl_wrapper.make_prepped_xfer.call_args_list:
            assert call.kwargs["notif_msg"].startswith(expected)

    def test_the_first_batch_starts_where_the_consumer_window_does(self):
        # Four blocks in the prompt, the consumer registered the last two: its
        # window begins at our block 2. Two closed so far, so exactly one of
        # them is inside it.
        worker = self._worker(total=4)
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )

        assert self._sent(worker) == [[2]]
        assert worker._streamed["r0"].issued_hwm == 1

    def test_the_next_batch_starts_where_the_last_one_stopped(self):
        worker = self._worker(total=4)
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )
        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2, 3],), ([4, 5],))
        )

        assert self._sent(worker) == [[2], [3]]
        assert worker._streamed["r0"].issued_hwm == 2

    def test_a_prefix_short_of_the_window_writes_nothing(self):
        # The consumer's cache covered everything this prefix has reached.
        worker = self._worker(total=4)

        worker._xfer_blocks_for_req("r0", TestPerShardWrite._meta(([0, 1],), ([4, 5],)))

        assert self._sent(worker) == []
        assert worker._streamed["r0"].issued_hwm == 0

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
        assert notifs == [b"RBLNS:0:0:1:1:r0:1", b"RBLNS:0:1:2:1:r0:1"]

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
        # different things. The peer's factor comes off the handshake, which
        # is the only place a producer has it -- the D-side metadata map holds
        # nothing for a request this side is sending.
        worker = self._worker(total=4)
        worker.transfer_topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1,
            remote_block_size=16,
            remote_physical_blocks_per_logical=2,
        )

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )

        assert self._sent(worker) == [[1, 2]]

    def test_a_batch_raises_the_writer_s_count_and_carries_the_mark(self):
        # The count and the mark are the writer's, and this is where a batch
        # begins. Seeding them in a test would leave the wiring from a real
        # batch to either one pinned by nothing.
        worker = self._worker(total=4)
        _send(worker, released=True)
        meta = TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))

        worker._xfer_blocks_for_req("r0", meta)
        send = worker._streamed["r0"]
        assert (send.issued, send.handover_written) == (1, False)

        meta = TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        meta.local_block_ids = pw.OfferedBlocks(
            meta.local_block_ids, 0, worker._streamed["r0"].serial, handover=True
        )
        worker._xfer_blocks_for_req("r0", meta)

        assert (send.issued, send.handover_written) == (2, True)

    def test_a_batch_that_issued_nothing_files_no_entry_upstream_can_read(self):
        # A step that computes tokens without closing a chunk submits no
        # write, and upstream owns the request on this route. Its transfer map
        # is a defaultdict and it counts a request whose handle list is empty
        # as done, so filing one here reports a send that never left.
        worker = self._worker(total=4)

        worker._submit_writes("r0", "eng", b"r0:1", [], None)

        assert "r0" not in worker._sending_transfers

    def test_a_batch_for_an_unreleased_record_is_not_the_writer_s(self):
        # A record exists from `start_early_push` until the next step's
        # release, and a list reaching the writer inside that window would
        # leave a count nothing can meet: `_submit_writes` files only a
        # released request. Upstream's ordering keeps the window shut today.
        worker = self._worker(total=4)  # the record it makes is not released

        worker._xfer_blocks_for_req(
            "r0", TestPerShardWrite._meta(([0, 1, 2],), ([4, 5],))
        )

        assert worker._streamed["r0"].issued == 0


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

    def test_a_fresh_registration_replaces_the_one_held(self, monkeypatch):
        # A second PUSH_REG for the same request describes the blocks the
        # consumer holds NOW. Keeping the first would write the batches that
        # follow into blocks the consumer has moved on from.
        worker = self._worker(monkeypatch)
        worker._pending_d_registrations["r0"] = {"request_id": "r0", "n": 1}
        fresh = {"request_id": "r0", "n": 2}

        worker._do_start_push_kv("r0", ([1],), fresh)

        assert worker._pending_d_registrations["r0"] is fresh

    def test_the_entry_is_filed_under_the_id_the_eviction_drops_by(self, monkeypatch):
        # Upstream files it under the CONSUMER's id, and the eviction drops by
        # this side's -- which a retry's suffix makes a different string. Left
        # as upstream filed it, the entry is one nothing ever reaches.
        worker = self._worker(monkeypatch)
        reg = {"request_id": "r0-1a2b3c4d", "decode_engine_id": "eng"}
        worker._pending_d_registrations["r0-1a2b3c4d"] = reg

        worker._do_start_push_kv("r0-9f8e7d6c", ([1],), reg)

        assert worker._pending_d_registrations == {"r0-9f8e7d6c": reg}
