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

# The scheduler-side shared layer: chunked-prefill save tracking, the chunk
# alignment a fetch is trimmed to, and the finished-request branches, exercised
# through the inherited entry points of whichever direction sits underneath.
# Built bare with only the state those paths read.

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorScheduler,
    NixlPushConnectorScheduler,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import RequestStatus

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_scheduler as sm
from tests.vllm.distributed.kv_connector.utils import (
    mock_vllm_config,
    shape,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
    transfer_shape,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_scheduler import (
    RblnNixlPullConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_scheduler import (
    RblnNixlPushConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker import (
    RblnNixlPushConnectorWorker,
)


@dataclass
class _NewReq:
    req_id: str
    block_ids: tuple


@dataclass
class _CachedReqs:
    req_ids: list = field(default_factory=list)
    new_block_ids: list = field(default_factory=list)
    resumed_req_ids: set = field(default_factory=set)


@dataclass
class _SchedOutput:
    scheduled_new_reqs: list
    scheduled_cached_reqs: _CachedReqs
    num_scheduled_tokens: dict
    preempted_req_ids: set = field(default_factory=set)


@dataclass
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.RUNNING
    kv_transfer_params: dict | None = field(
        default_factory=lambda: {"do_remote_decode": True}
    )
    prompt_token_ids: list = field(default_factory=list)


def _sched_output(
    req_id, block_ids, num_scheduled_tokens, *, is_new=True, resumed=False
):
    """A minimal SchedulerOutput for yield_req_data: a fresh req carries its
    block_ids on scheduled_new_reqs; a resumed chunk carries them on
    scheduled_cached_reqs (None once no new blocks are added). `resumed` marks
    a request coming back from preemption, whose block ids are its whole list
    again rather than the step's delta."""
    if is_new:
        return _SchedOutput(
            scheduled_new_reqs=[_NewReq(req_id, block_ids)],
            scheduled_cached_reqs=_CachedReqs(),
            num_scheduled_tokens={req_id: num_scheduled_tokens},
        )
    return _SchedOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=_CachedReqs(
            req_ids=[req_id],
            new_block_ids=[block_ids],
            resumed_req_ids={req_id} if resumed else set(),
        ),
        num_scheduled_tokens={req_id: num_scheduled_tokens},
    )


def _state_knobs(sched, *, specs=None, **knobs) -> None:
    """Set the config and the shape together, the way `__init__` does.

    Setting one without the other is how a test comes to assert a knob the
    code under test never saw -- the drift this shape exists to remove.
    """
    sched.vllm_config = mock_vllm_config(**knobs)
    sched.vllm_config.parallel_config.tensor_parallel_size = 1
    sched._shape = transfer_shape(
        sched.vllm_config,
        _kv_config(specs if specs is not None else [MagicMock()]).kv_cache_groups,
        writes_into_peer=type(sched)._writes_into_peer,
    )


def _scheduler(*, use_host_buffer=False, cls=RblnNixlPullConnectorScheduler):
    sched = object.__new__(cls)
    sched.vllm_config = mock_vllm_config()
    sched.vllm_config.parallel_config.tensor_parallel_size = 1
    sched.block_size = 16
    sched.engine_id = "test-engine"
    sched.kv_cache_config = MagicMock()
    sched.side_channel_host = "localhost"
    sched.side_channel_port = 5000
    # The save path is gated on this, so save tests must turn it on.
    sched.use_host_buffer = use_host_buffer
    _state_knobs(sched)
    sched._is_hma_required = False  # get_sw_clipped_blocks (inherited) reads this
    sched.blocks_per_sw = [0]
    sched._kv_lease_duration = 30
    sched._reqs_need_recv = {}
    sched._reqs_need_save = {}
    sched._valid_tokens = {}
    sched._reqs_need_send = {}
    sched._reqs_in_batch = set()
    sched._reqs_not_processed = set()
    sched._block_ids_need_save = {}
    # Upstream state the inherited entry points read.
    sched._heartbeat_by_engine = {}
    sched._heartbeat_req_engine = {}
    sched._last_heartbeat_time = 0.0
    sched._heartbeat_interval = 5
    sched.is_bidirectional_kv_xfer_enabled = False
    sched.decoder_kv_blocks_ttl = 480
    sched.kv_recompute_threshold = 64
    sched._has_mamba = False
    sched.vllm_config.scheduler_config.max_num_batched_tokens = 512
    if cls is RblnNixlPushConnectorScheduler:
        sched._streamed_chunks = {}
        sched._push_pending_registrations = {}
        sched._push_registration_deadlines = {}
        sched._push_registration_timeout = 480
        sched._finished_request_blocks = {}
        sched._newly_finished_push_blocks = {}
        # Off by default, as the environment variable is.
        sched._shape = shape(streams_prefix=False)
        sched._early_sent = set()
    return sched


class TestInit:
    @pytest.mark.parametrize(
        ("kv_buffer_device", "expected"),
        [("cpu", True), ("rbln", False)],
    )
    def test_use_host_buffer_follows_kv_buffer_device(
        self, monkeypatch, kv_buffer_device, expected
    ):
        # host-bounce ("cpu") stages through host DRAM; D2D ("rbln") does not.
        monkeypatch.setattr(
            sm.NixlPullConnectorScheduler, "__init__", lambda self, *a, **k: None
        )
        vllm_config = mock_vllm_config()
        vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device
        sched = object.__new__(RblnNixlPullConnectorScheduler)
        RblnNixlPullConnectorScheduler.__init__(
            sched, vllm_config, "eng", _kv_config([MagicMock()])
        )
        assert sched.use_host_buffer is expected
        assert sched._block_ids_need_save == {}


class TestBuildConnectorMeta:
    def test_single_step_prefill_saves_immediately(self):
        # A prefill that finishes in one step is added to the save metadata and
        # dropped from both tracking dicts.
        sched = _scheduler(use_host_buffer=True)
        req = _Request("prefill", num_prompt_tokens=256)
        sched._reqs_need_save["prefill"] = req

        meta = sched.build_connector_meta(
            _sched_output("prefill", ([1, 2, 3, 4],), 256)
        )
        assert "prefill" in meta.reqs_to_save
        assert "prefill" not in sched._reqs_need_save
        assert "prefill" not in sched._block_ids_need_save

    def test_chunked_prefill_defers_save_to_final_chunk(self):
        # Partial chunks accumulate blocks in _block_ids_need_save and are NOT
        # saved; only the final chunk (prompt fully computed) saves and clears.
        sched = _scheduler(use_host_buffer=True)
        req = _Request("chunked", num_prompt_tokens=512)
        sched._reqs_need_save["chunked"] = req

        meta = sched.build_connector_meta(
            _sched_output("chunked", ([1, 2, 3, 4],), 256)
        )
        assert "chunked" not in meta.reqs_to_save
        assert "chunked" in sched._block_ids_need_save
        assert "chunked" in sched._reqs_need_save

        req.num_computed_tokens = 256
        meta = sched.build_connector_meta(
            _sched_output("chunked", None, 256, is_new=False)
        )
        assert "chunked" in meta.reqs_to_save
        assert "chunked" not in sched._block_ids_need_save
        assert "chunked" not in sched._reqs_need_save

    def test_blocks_from_every_chunk_are_saved_together(self):
        # The host copy moves whole blocks once, so a chunk that brings new
        # blocks after the first appends to what is already held. Replacing
        # would stage the last chunk's blocks alone and lose the prefix.
        sched = _scheduler(use_host_buffer=True)
        req = _Request("chunked", num_prompt_tokens=768)
        sched._reqs_need_save["chunked"] = req

        sched.build_connector_meta(_sched_output("chunked", ([1, 2],), 256))
        req.num_computed_tokens = 256
        sched.build_connector_meta(_sched_output("chunked", ([3],), 256, is_new=False))
        req.num_computed_tokens = 512
        meta = sched.build_connector_meta(
            _sched_output("chunked", ([4],), 256, is_new=False)
        )

        assert meta.reqs_to_save["chunked"].local_block_ids == ([1, 2, 3, 4],)

    def test_recv_requests_added_and_tracking_cleared(self):
        # Requests awaiting a remote-KV load are emitted as recv entries, and the
        # per-step tracking sets are reset afterwards.
        sched = _scheduler()
        req = _Request("recv", num_prompt_tokens=64)
        req.kv_transfer_params = {
            "remote_block_ids": [7, 8],
            "remote_engine_id": "peer",
            "remote_request_id": "recv-remote",
            "remote_host": "1.2.3.4",
            "remote_port": 6000,
        }
        sched._reqs_need_recv["recv"] = (req, [1, 2])
        sched._reqs_in_batch = {"x"}
        sched._reqs_not_processed = {"y"}

        meta = sched.build_connector_meta(_sched_output("other", ([9],), 0))
        assert "recv" in meta.reqs_to_recv
        assert sched._reqs_need_recv == {}
        assert sched._reqs_in_batch == set()
        assert sched._reqs_not_processed == set()


class TestRequestFinished:
    def test_no_transfer_params_frees_immediately(self):
        sched = _scheduler()
        req = _Request("no-params", num_prompt_tokens=10)
        req.kv_transfer_params = None
        assert sched.request_finished(req, ([1],)) == (False, None)

    def test_aborted_before_schedule_queues_empty_recv(self):
        # do_remote_prefill still set at finish -> the request was aborted before
        # being scheduled; queue an empty recv so the worker frees remote blocks.
        sched = _scheduler()
        req = _Request("remote-prefill", num_prompt_tokens=10)
        req.kv_transfer_params = {"do_remote_prefill": True}

        delay, params = sched.request_finished(req, ([1],))
        assert (delay, params) == (False, None)
        assert sched._reqs_need_recv["remote-prefill"] == (req, [])
        assert req.kv_transfer_params["do_remote_prefill"] is False

    def test_not_remote_decode_frees_immediately(self):
        sched = _scheduler()
        req = _Request("not-decode", num_prompt_tokens=10)
        req.kv_transfer_params = {"foo": 1}
        assert sched.request_finished(req, ([1],)) == (False, None)

    def test_aborted_producer_cleans_up_tracking(self):
        # A remote-decode producer that ended without completing its prefill
        # stops being tracked and frees now -- there is no KV worth sending.
        sched = _scheduler()
        req = _Request("aborted", num_prompt_tokens=512)
        req.status = RequestStatus.FINISHED_ABORTED
        sched._reqs_need_save["aborted"] = req
        sched._block_ids_need_save["aborted"] = ([1, 2],)

        delay, params = sched.request_finished(req, ([],))
        assert (delay, params) == (False, None)
        assert "aborted" not in sched._reqs_need_save
        assert "aborted" not in sched._block_ids_need_save
        assert "aborted" in sched._reqs_not_processed

    def test_stopped_prefill_is_transferred_like_length_capped(self):
        # A producer whose single generated token happens to be a stop token
        # finishes STOPPED rather than LENGTH_CAPPED. Its KV is just as valid,
        # so it must still be handed to the decode side.
        sched = _scheduler()
        req = _Request("stopped", num_prompt_tokens=256)
        req.status = RequestStatus.FINISHED_STOPPED

        delay, params = sched.request_finished(req, ([1, 2, 3, 4],))
        assert delay is True
        assert params is not None
        assert params["do_remote_prefill"] is True
        assert "stopped" in sched._reqs_need_send
        assert "stopped" not in sched._reqs_not_processed

    def test_partial_save_state_is_dropped_when_the_request_ends(self):
        # _block_ids_need_save only holds blocks for a prefill still being
        # chunked; whatever survives to request_finished is stale.
        sched = _scheduler()
        req = _Request("leftover", num_prompt_tokens=256)
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        sched._block_ids_need_save["leftover"] = ([9],)

        sched.request_finished(req, ([1],))
        assert "leftover" not in sched._block_ids_need_save

    def test_completed_prefill_delays_free_and_returns_remote_params(self):
        # A LENGTH_CAPPED remote-decode producer with real blocks delays the free
        # (leased for the decode side to fetch) and returns the remote handshake.
        sched = _scheduler()
        req = _Request("done", num_prompt_tokens=256)
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED

        delay, params = sched.request_finished(req, ([1, 2, 3, 4],))
        assert delay is True
        assert params["do_remote_prefill"] is True
        assert params["do_remote_decode"] is False
        assert params["remote_engine_id"] == "test-engine"
        assert params["remote_request_id"] == "done"
        assert "done" in sched._reqs_need_send

    def test_completed_prefill_without_blocks_frees_now_but_returns_params(self):
        # LENGTH_CAPPED with no blocks to send: nothing is leased or tracked, but
        # the remote handshake still goes back to the decode side.
        sched = _scheduler()
        req = _Request("empty", num_prompt_tokens=256)
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED

        delay, params = sched.request_finished(req, ([],))
        assert delay is False
        assert params is not None
        assert params["do_remote_prefill"] is True
        assert "empty" not in sched._reqs_need_send


class TestChunkAlignedFetch:
    # A prefill has to resume on a chunk boundary, so the fetched amount is
    # trimmed to it. The base reports the peer's computed tokens, which is an
    # arbitrary count.

    @staticmethod
    def _reverse_req(*, prompt, remote):
        # What the decode side reports back for a prompt it partly computed.
        return _Request(
            request_id="r0",
            num_prompt_tokens=prompt,
            kv_transfer_params={
                "do_remote_decode": True,
                "remote_block_ids": [[1, 2]],
                "remote_engine_id": "eng",
                "remote_request_id": "r0",
                "remote_host": "h",
                "remote_port": 1,
                "remote_num_tokens": remote,
            },
        )

    def test_fetch_is_trimmed_to_the_chunk_below(self):
        # 900 tokens held remotely, 512-token chunks: fetch 512 and recompute
        # the 388 that would have put the next chunk mid-grid.
        sched = _scheduler()
        assert sched.get_num_new_matched_tokens(
            self._reverse_req(prompt=4096, remote=900), 0
        ) == (512, True)

    def test_an_already_aligned_fetch_passes_through(self):
        # The trim is a remainder, so a peer holding a whole number of chunks
        # needs none of it -- recomputing a chunk the transfer already carried.
        sched = _scheduler()
        assert sched.get_num_new_matched_tokens(
            self._reverse_req(prompt=4096, remote=1024), 0
        ) == (1024, True)

    def test_fetch_shorter_than_one_chunk_reports_no_match(self):
        # Trimming leaves nothing, and an async load of zero tokens trips the
        # scheduler's own assertion, so the answer has to be a plain no-match.
        sched = _scheduler()
        assert sched.get_num_new_matched_tokens(
            self._reverse_req(prompt=4096, remote=500), 0
        ) == (0, False)

    def test_whole_prompt_fetch_is_untouched(self):
        # Guard: the ordinary prefill-to-decode direction fetches through the
        # prompt's last token, so no chunk follows and nothing is trimmed --
        # trimming here would recompute what the transfer already carried.
        sched = _scheduler()
        req = _Request(
            request_id="r0",
            num_prompt_tokens=900,
            prompt_token_ids=list(range(900)),
            kv_transfer_params={"do_remote_prefill": True},
        )
        assert sched.get_num_new_matched_tokens(req, 0) == (900, True)


class TestSchedulerCleanupReachesBothDirections:
    @pytest.mark.parametrize(
        "scheduler_cls, direction_cls",
        [
            (RblnNixlPullConnectorScheduler, NixlPullConnectorScheduler),
            (RblnNixlPushConnectorScheduler, NixlPushConnectorScheduler),
        ],
    )
    def test_stale_chunk_accumulation_is_dropped(
        self, monkeypatch, scheduler_cls, direction_cls
    ):
        # The accumulation belongs to the shared layer, so its cleanup must run
        # and then hand over to whichever direction scheduler is underneath.
        seen = []

        def record(self, request, block_ids):
            seen.append(request.request_id)
            return False, None

        monkeypatch.setattr(direction_cls, "request_finished", record)
        scheduler = object.__new__(scheduler_cls)
        scheduler._block_ids_need_save = {"r0": ([1, 2],)}
        scheduler._streamed_chunks = {"r0": 2}
        scheduler._valid_tokens = {}

        # A real Request always carries the field, even when it is None.
        scheduler.request_finished(
            SimpleNamespace(request_id="r0", kv_transfer_params=None), ([1, 2],)
        )

        assert scheduler._block_ids_need_save == {}
        assert seen == ["r0"]
        if scheduler_cls is RblnNixlPushConnectorScheduler:
            # The offered prefix goes with it: a retry reusing the id would
            # otherwise be thought to have already streamed what it has not.
            assert scheduler._streamed_chunks == {}


class TestRejectedBeforeScheduling:
    """The serving layer can turn a request away before it is ever scheduled --
    a prompt past the context length, a client that left. The base registers an
    empty receive for it so the producer stops holding the blocks it pinned,
    and building that receive reads a field only `update_state_after_alloc`
    fills, which such a request never reaches."""

    @staticmethod
    def _rejected():
        return _Request(
            "rejected",
            num_prompt_tokens=1,
            status=RequestStatus.FINISHED_ABORTED,
            # What the serving layer hands back: still flagged for a remote
            # prefill, and without the field D fills for itself.
            kv_transfer_params={
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "remote_engine_id": "prefill0",
                "remote_request_id": "abc",
                "remote_host": "localhost",
                "remote_port": 5559,
                "tp_size": 1,
            },
        )

    def test_the_metadata_for_a_rejected_request_can_be_built(self):
        # Calls upstream's own builder rather than checking the key by hand:
        # what has to hold is that upstream's read of it succeeds.
        sched = _scheduler(cls=RblnNixlPushConnectorScheduler)
        req = self._rejected()

        sched.request_finished(req, ([],))
        meta = sched.build_connector_meta(_sched_output("other", ([9],), 16))

        assert "rejected" in meta.reqs_to_recv
        assert meta.reqs_to_recv["rejected"].remote.block_ids == ()


class TestEarlyOfferOnTheWritePath:
    """A prefill is offered to the writer as its chunks close blocks, so what
    it has finished can leave while the rest is still being computed. The
    direct path has no save of its own, so the offer is the only reader of the
    accumulation there."""

    @staticmethod
    def _push_scheduler(*, enabled=True, use_host_buffer=False):
        sched = _scheduler(
            use_host_buffer=use_host_buffer, cls=RblnNixlPushConnectorScheduler
        )
        sched._shape = shape(streams_prefix=enabled)
        return sched

    @staticmethod
    def _chunking_scheduler(chunk):
        """A push scheduler whose config asks for writes smaller than a block.

        The shared fixture's config is a mock, so the chunk derivation falls
        back to the block and no offer holds a partial one. Naming the prefill
        chunk is what turns that on.
        """
        sched = TestEarlyOfferOnTheWritePath._push_scheduler()
        sched.vllm_config.scheduler_config.max_num_batched_tokens = chunk
        return sched

    def test_an_offer_carries_the_block_being_filled(self):
        # Half a block computed: the block holding it comes with the offer, and
        # the token count says how much of it is there. Offering only closed
        # blocks leaves that half until the request ends.
        sched = self._chunking_scheduler(8)
        req = _Request("prefill", num_prompt_tokens=512)
        sched._reqs_need_save["prefill"] = req
        req.num_computed_tokens = 24

        meta = sched.build_connector_meta(_sched_output("prefill", ([1, 2],), 24))

        assert meta.reqs_to_save["prefill"].local_block_ids == ([1, 2],)
        assert meta.push_stream_tokens["prefill"] == 24

    def test_an_offer_grows_by_a_chunk_rather_than_by_a_block(self):
        # A step that closes no block still closes a chunk, and today that
        # step offers nothing at all.
        sched = self._chunking_scheduler(8)
        req = _Request("prefill", num_prompt_tokens=512)
        sched._reqs_need_save["prefill"] = req
        req.num_computed_tokens = 8
        sched.build_connector_meta(_sched_output("prefill", ([1, 2],), 8))
        assert sched._streamed_chunks == {"prefill": 1}

        req.num_computed_tokens = 16
        meta = sched.build_connector_meta(
            _sched_output("prefill", ([1, 2],), 16, is_new=False)
        )

        assert "prefill" in meta.reqs_to_save
        assert meta.push_stream_tokens["prefill"] == 16

    def test_a_hybrid_offers_its_full_attention_blocks_and_no_window(self):
        # The window's group holds one block whatever the prompt length, so a
        # prefix that never reaches its length would be capped at one block
        # forever; and that block is the live window the kernel keeps
        # overwriting, so nothing before the handover may send it.
        sched = self._chunking_scheduler(8)
        sched.blocks_per_sw = [0, 2]  # group 1 is the sliding window
        req = _Request("prefill", num_prompt_tokens=512)
        sched._reqs_need_save["prefill"] = req
        req.num_computed_tokens = 40

        meta = sched.build_connector_meta(
            _sched_output("prefill", ([1, 2, 3], [9]), 40)
        )

        assert meta.reqs_to_save["prefill"].local_block_ids == ([1, 2, 3], [])
        assert meta.push_stream_tokens["prefill"] == 40

    def test_a_part_chunk_does_not_count_as_a_whole_one(self):
        # The cursor rounds DOWN: counting the chunk a step reached into as
        # done would make the next step, which actually closes it, look like no
        # progress -- and that step's offer would never go out.
        sched = self._chunking_scheduler(8)
        req = _Request("prefill", num_prompt_tokens=512)
        sched._reqs_need_save["prefill"] = req

        req.num_computed_tokens = 44  # five whole chunks of eight, and half a sixth
        first = sched.build_connector_meta(_sched_output("prefill", ([1, 2, 3],), 44))
        assert "prefill" in first.reqs_to_save

        sched._reqs_need_save["prefill"] = req
        req.num_computed_tokens = 48  # the sixth chunk closes here
        second = sched.build_connector_meta(_sched_output("prefill", ([1, 2, 3],), 48))

        assert "prefill" in second.reqs_to_save
        assert second.push_stream_tokens["prefill"] == 48

    def test_the_window_s_group_is_not_the_one_counted(self):
        # The counted group is found by which one the window does not hold, not
        # by position. Sized from the window's group instead, an offer would be
        # one block whatever the prompt length -- and that block is the live
        # window.
        sched = self._chunking_scheduler(8)
        sched.blocks_per_sw = [2, 0]  # group 0 is the sliding window this time
        req = _Request("prefill", num_prompt_tokens=512)
        sched._reqs_need_save["prefill"] = req
        req.num_computed_tokens = 40

        meta = sched.build_connector_meta(
            _sched_output("prefill", ([9], [1, 2, 3]), 40)
        )

        assert meta.reqs_to_save["prefill"].local_block_ids == ([], [1, 2, 3])
        assert meta.push_stream_tokens["prefill"] == 40

    def test_an_offer_never_names_tokens_no_block_of_ours_holds(self):
        # A step can compute past the blocks we have -- the accumulation lags
        # by a step on a resume -- and a count past them names KV that is not
        # there for the writer to read.
        sched = self._chunking_scheduler(8)
        req = _Request("prefill", num_prompt_tokens=512)
        sched._reqs_need_save["prefill"] = req
        req.num_computed_tokens = 100

        meta = sched.build_connector_meta(_sched_output("prefill", ([1, 2],), 100))

        assert meta.push_stream_tokens["prefill"] == 2 * 16

    def test_a_producer_request_is_tracked_for_the_offer(self):
        # The offer reads the accumulation upstream builds only under host
        # staging, so the direct path has to enter the request itself. The
        # in-batch set is upstream's own side effect of the same call: if it
        # is missing, the chain to upstream did not run.
        sched = self._push_scheduler()
        req = _Request("prefill", num_prompt_tokens=256)

        sched.update_state_after_alloc(req, MagicMock(), 0)

        assert sched._reqs_need_save == {"prefill": req}
        assert sched._reqs_in_batch == {"prefill"}

    def test_a_consumer_request_is_not_tracked(self):
        # Only the side that produces KV has anything to offer.
        sched = self._push_scheduler()
        req = _Request("decode", num_prompt_tokens=256)
        req.kv_transfer_params = {"do_remote_prefill": False}

        sched.update_state_after_alloc(req, MagicMock(), 0)

        assert sched._reqs_need_save == {}

    def test_the_gate_off_tracks_nothing(self):
        sched = self._push_scheduler(enabled=False)
        req = _Request("prefill", num_prompt_tokens=256)

        sched.update_state_after_alloc(req, MagicMock(), 0)

        assert sched._reqs_need_save == {}

    def test_the_closing_chunk_offers_its_blocks(self):
        sched = self._push_scheduler()
        req = _Request("chunked", num_prompt_tokens=512)
        sched._reqs_need_save["chunked"] = req

        meta = sched.build_connector_meta(_sched_output("chunked", ([1, 2],), 256))
        assert "chunked" not in meta.reqs_to_save
        assert sched._early_sent == set()

        req.num_computed_tokens = 256
        meta = sched.build_connector_meta(
            _sched_output("chunked", ([3],), 256, is_new=False)
        )

        assert meta.reqs_to_save["chunked"].local_block_ids == ([1, 2, 3],)
        assert sched._early_sent == {"chunked"}

    def test_a_resumed_request_starts_its_offers_over(self):
        # Both halves of the accumulation have to reset here: the old ids
        # would offer blocks the request no longer owns, and the old count
        # would hold each re-prefilled block under the high-water mark.
        sched = self._push_scheduler()
        req = _Request("preempted", num_prompt_tokens=512)
        sched._reqs_need_save["preempted"] = req
        req.num_computed_tokens = 64
        sched.build_connector_meta(_sched_output("preempted", ([1, 2, 3],), 64))
        assert sched._streamed_chunks == {"preempted": 3}

        req.num_computed_tokens = 32
        meta = sched.build_connector_meta(
            _sched_output("preempted", ([7, 8, 9],), 32, is_new=False, resumed=True)
        )

        assert meta.reqs_to_save["preempted"].local_block_ids == ([7, 8],)

    def test_the_offer_carries_the_block_count_of_the_whole_prompt(self):
        # The consumer registered the tail of the final list, so its window is
        # placed from the total. Mid-stream the offer is shorter than that, and
        # a total taken from what has closed so far would put the window at the
        # wrong end of a prompt still being computed.
        sched = self._push_scheduler()
        # A length that is not a whole number of blocks: the last block is part
        # full, and rounding it away would place the window a block early.
        req = _Request("chunked", num_prompt_tokens=50)
        sched._reqs_need_save["chunked"] = req
        req.num_computed_tokens = 16

        meta = sched.build_connector_meta(_sched_output("chunked", ([1, 2, 3, 4],), 16))

        # 50 tokens over a block size of 16 is four blocks, the last holding two
        # tokens; one has closed.
        assert meta.reqs_to_save["chunked"].local_block_ids == ([1],)
        assert meta.push_stream_total == {"chunked": 4}

    def test_the_gate_off_offers_nothing(self):
        # Guard: the direct path had no offer before this change either, so
        # what this pins is that the flag is what turns it on -- dropping the
        # gate would make every direct-path run stream.
        sched = self._push_scheduler(enabled=False)
        sched._reqs_need_save["prefill"] = _Request("prefill", num_prompt_tokens=256)

        meta = sched.build_connector_meta(_sched_output("prefill", ([1, 2],), 256))

        assert meta.reqs_to_save == {}
        assert sched._early_sent == set()

    def test_the_offer_is_the_list_the_handover_would_have_carried(self):
        # The duplicate handover is dropped on the strength of the two lists
        # being the same one; if they diverged, the writer would send the early
        # list and the blocks the request actually ended with would never go.
        sched = self._push_scheduler()
        req = _Request("chunked", num_prompt_tokens=512)
        sched._reqs_need_save["chunked"] = req
        sched.build_connector_meta(_sched_output("chunked", ([1, 2],), 256))
        req.num_computed_tokens = 256
        meta = sched.build_connector_meta(
            _sched_output("chunked", ([3],), 256, is_new=False)
        )

        req.status = RequestStatus.FINISHED_STOPPED
        sched.request_finished(req, ([1, 2, 3],))

        assert (
            meta.reqs_to_save["chunked"].local_block_ids
            == sched._newly_finished_push_blocks["chunked"]
        )

    @pytest.mark.parametrize(
        "kind",
        ["preempted", "not-processed"],
    )
    def test_blocks_going_back_without_a_lease_are_flushed(self, kind):
        # Both hand the blocks to the allocator with a write still reading
        # them: a preempted request re-prefills into them, and one that ended
        # on a non-terminal status frees them outright.
        sched = self._push_scheduler()
        sched._early_sent = {"r0"}
        if kind == "preempted":
            output = _sched_output("other", ([9],), 16)
            output.preempted_req_ids = {"r0"}
        else:
            output = _sched_output("other", ([9],), 16)
            sched._reqs_not_processed = {"r0"}

        meta = sched.build_connector_meta(output)

        assert meta.push_early_flush == {"r0"}
        # Cleared, so a later step does not flush the same write twice.
        assert sched._early_sent == set()

    def test_a_terminal_finish_is_not_flushed(self):
        # The lease holds those blocks, and a flush would stall the engine on
        # every completed prefill.
        sched = self._push_scheduler()
        sched._early_sent = {"r0"}
        req = _Request(
            "r0", num_prompt_tokens=256, status=RequestStatus.FINISHED_STOPPED
        )

        sched.request_finished(req, ([1, 2],))
        meta = sched.build_connector_meta(_sched_output("other", ([9],), 16))

        assert sched._early_sent == set()
        assert meta.push_early_flush == set()


def _sw_spec(*, block_size, sliding_window):
    spec = MagicMock(spec=SlidingWindowSpec)
    spec.block_size = block_size
    spec.sliding_window = sliding_window
    return spec


def _kv_config(specs):
    return MagicMock(kv_cache_groups=[MagicMock(kv_cache_spec=spec) for spec in specs])


class TestEarlyPushGate:
    @pytest.mark.parametrize(
        ("flag", "pp_size", "groups", "host_buffer", "expected"),
        [
            (True, 4, 1, False, True),
            # One stage still closes blocks one chunk at a time, so what a
            # prefill has finished can leave before the request does.
            (True, 1, 1, False, True),
            (False, 4, 1, False, False),
            # A second group is carried at a different time, and telling the
            # two apart on the wire needs the descriptor list a sliding
            # window's view builds. Without a window there is no such list.
            (True, 4, 2, False, False),
            # Host staging holds no areas to write out of.
            (True, 4, 1, True, False),
        ],
    )
    def test_the_gate_needs_the_flag_and_one_block_scale(
        self, monkeypatch, flag, pp_size, groups, host_buffer, expected
    ):
        def stub_init(self, cfg, _engine_id, kv_cache_config):
            self._shape = transfer_shape(
                cfg, kv_cache_config.kv_cache_groups, writes_into_peer=True
            )

        monkeypatch.setattr(NixlPushConnectorScheduler, "__init__", stub_init)
        config = mock_vllm_config(push_stream=flag)
        config.parallel_config.pipeline_parallel_size = pp_size
        config.kv_transfer_config.kv_buffer_device = "cpu" if host_buffer else "rbln"

        sched = RblnNixlPushConnectorScheduler(
            config, "eng", _kv_config([MagicMock() for _ in range(groups)])
        )

        assert sched._shape.streams_prefix is expected

    def test_one_group_streams_although_upstream_calls_it_hybrid(self, monkeypatch):
        # A merged MLA-plus-indexer cache is reported as hybrid and is ONE
        # group. The reason a windowless hybrid is left out -- that the offer
        # and the handover carry different groups and the wire cannot tell
        # them apart -- has nothing to apply to here, and the two models this
        # connector cuts on the context axis are both this shape.
        def stub_init(self, cfg, _engine_id, kv_cache_config):
            self._shape = transfer_shape(
                cfg, kv_cache_config.kv_cache_groups, writes_into_peer=True
            )

        monkeypatch.setattr(NixlPushConnectorScheduler, "__init__", stub_init)
        config = mock_vllm_config(push_stream=True)

        sched = RblnNixlPushConnectorScheduler(config, "eng", _kv_config([MagicMock()]))

        assert sched._shape.streams_prefix is True

    def test_a_hybrid_streams_where_its_window_can_be_viewed(self, monkeypatch):
        # The offer carries the full-attention group and the handover carries
        # the window's block; telling them apart on the wire needs the one
        # descriptor list that names two groups, which the view builds.
        def stub_init(self, cfg, _engine_id, kv_cache_config):
            self._shape = transfer_shape(
                cfg, kv_cache_config.kv_cache_groups, writes_into_peer=True
            )

        monkeypatch.setattr(NixlPushConnectorScheduler, "__init__", stub_init)
        config = mock_vllm_config(push_stream=True)

        sched = RblnNixlPushConnectorScheduler(
            config, "eng", _kv_config([_sw_spec(block_size=1024, sliding_window=128)])
        )

        assert sched._shape.streams_prefix is True

    @pytest.mark.parametrize("hybrid", [False, True])
    @pytest.mark.parametrize("host_buffer", [False, True])
    def test_both_sides_answer_the_same_for_one_config(
        self, monkeypatch, hybrid, host_buffer
    ):
        """The scheduler stops building offers and the worker stops asking for
        per-shard descriptors. A side that takes one without the other sends
        its peers down a route nothing feeds -- and the route asserts on the
        group count a hybrid model has."""
        specs = [_sw_spec(block_size=1024, sliding_window=128)] if hybrid else []

        def stub_init(self, cfg, _engine_id, kv_cache_config):
            # Both bases run the same reduction over the same arguments; stub
            # them at the same depth so nothing but that is left to differ.
            self._shape = transfer_shape(
                cfg, kv_cache_config.kv_cache_groups, writes_into_peer=True
            )
            self._group_specs = specs

        monkeypatch.setattr(RblnNixlSchedulerBase, "__init__", stub_init)
        monkeypatch.setattr(RblnNixlWorkerBase, "__init__", stub_init)
        config = mock_vllm_config(push_stream=True)
        config.parallel_config.pipeline_parallel_size = 4

        sched = RblnNixlPushConnectorScheduler(config, "eng", _kv_config(specs))
        worker = RblnNixlPushConnectorWorker(config, "eng", _kv_config(specs))
        worker.shutdown = lambda: None  # the base __init__ was stubbed out

        assert worker._shape.streams_prefix is sched._shape.streams_prefix


class TestTailTokenCountOnTheReadPath:
    """The producer says how many tokens it holds in `kv_transfer_params`;
    upstream reads it once for the match length and drops it. The read path
    needs it to know how full the request's last block is."""

    @staticmethod
    def _params(remote_num_tokens):
        return {
            "do_remote_prefill": True,
            "remote_engine_id": "prefill0",
            "remote_request_id": "abc",
            "remote_host": "localhost",
            "remote_port": 5559,
            "remote_block_ids": ([4, 5],),
            "tp_size": 1,
            "remote_num_tokens": remote_num_tokens,
        }

    def _meta_for(self, monkeypatch, remote_num_tokens):
        sched = _scheduler()
        _state_knobs(sched, chunk_mode=True)
        req = _Request(
            "r0",
            # Apart from the producer's count: a consumer's own prompt length
            # is not what the producer holds, and only one of them sizes the
            # last block the producer hands over.
            num_prompt_tokens=41,
            kv_transfer_params=self._params(remote_num_tokens),
        )
        sched._reqs_need_recv["r0"] = (req, ([7],))
        return sched.build_connector_meta(_sched_output("other", ([9],), 16))

    def test_the_count_survives_into_the_metadata(self, monkeypatch):
        meta = self._meta_for(monkeypatch, 33)
        # Promoted, or the worker has no field to read it from.
        assert isinstance(meta, RblnNixlConnectorMetadata)
        assert meta.valid_tokens == {"r0": 33}

    def test_a_producer_holding_nothing_is_left_out(self, monkeypatch):
        # A request the serving layer turned away registers an empty receive
        # and reports zero, which is not a last block anyone can size.
        assert self._meta_for(monkeypatch, 0).valid_tokens == {}

    def test_both_flags_off_collect_nothing(self, monkeypatch):
        # The worker would not read it, and an entry nobody pops outlives its
        # request.
        assert self._with_knobs(chunk_mode=False).valid_tokens == {}

    def test_window_mode_alone_collects_the_count(self, monkeypatch):
        # The count is not the chunk range's alone: a window's group reads it
        # to say which granule of a block its window sits in.
        assert self._with_knobs(
            chunk_mode=False, swa_window_mode=True
        ).valid_tokens == {"r0": 33}

    def test_streaming_alone_collects_the_count(self, monkeypatch):
        # Streaming turns the window on without naming it, so it reaches the
        # same granule question by a third knob.
        assert self._with_knobs(chunk_mode=False, push_stream=True).valid_tokens == {
            "r0": 33
        }

    def _with_knobs(self, **knobs):
        sched = _scheduler()
        _state_knobs(sched, **knobs)
        req = _Request("r0", num_prompt_tokens=33, kv_transfer_params=self._params(33))
        sched._reqs_need_recv["r0"] = (req, ([7],))
        return sched.build_connector_meta(_sched_output("other", ([9],), 16))


class TestTailTokenCountOnTheWritePath:
    """The producer takes its own count where it hands the blocks over."""

    @staticmethod
    def _finish(monkeypatch, delay_free_blocks, trim=True, window=False, stream=False):
        monkeypatch.setattr(
            NixlPushConnectorScheduler,
            "request_finished",
            lambda self, request, block_ids: (delay_free_blocks, None),
        )
        sched = _scheduler(cls=RblnNixlPushConnectorScheduler)
        _state_knobs(sched, chunk_mode=trim, swa_window_mode=window, push_stream=stream)
        sched.request_finished(
            # Apart, so taking the prompt length instead of what was computed
            # is a different answer.
            _Request("r0", num_prompt_tokens=41, num_computed_tokens=33),
            ([1, 2],),
        )
        return sched

    def test_the_scheduler_builds_the_map_the_handover_writes_into(self, monkeypatch):
        # Every case here hands the map in already made, so none of them would
        # notice it going missing -- and production reaches this line before
        # any request finishes.
        monkeypatch.setattr(
            NixlPushConnectorScheduler,
            "__init__",
            # What our `__init__` reads of upstream's, and nothing else: the
            # point of the stub is that only our own lines run.
            lambda self, *a, **k: self.__dict__.update(
                _is_hma_required=False, use_host_buffer=False
            ),
        )
        monkeypatch.setattr(
            NixlPushConnectorScheduler,
            "request_finished",
            lambda self, request, block_ids: (True, None),
        )
        sched = RblnNixlPushConnectorScheduler(
            mock_vllm_config(chunk_mode=True), "eng", MagicMock()
        )
        _state_knobs(sched, chunk_mode=True)

        sched.request_finished(
            # Apart, so taking the prompt length instead of what was computed
            # is a different answer.
            _Request("r0", num_prompt_tokens=41, num_computed_tokens=33),
            ([1, 2],),
        )

        assert sched._valid_tokens == {"r0": 33}

    def test_the_count_is_taken_where_the_lease_takes_the_blocks(self, monkeypatch):
        assert self._finish(monkeypatch, True)._valid_tokens == {"r0": 33}

    def test_blocks_going_straight_back_leave_no_count(self, monkeypatch):
        # Nothing is handed over, so there is no write to size.
        assert self._finish(monkeypatch, False)._valid_tokens == {}

    def test_both_flags_off_collect_nothing(self, monkeypatch):
        # Same handover, and nothing kept: the worker would not read it.
        assert self._finish(monkeypatch, True, trim=False)._valid_tokens == {}

    def test_window_mode_alone_collects_the_count(self, monkeypatch):
        # As on the read path: the granule a window sits in is read off this
        # count, so a chunk range is not the only thing that asks for it.
        sched = self._finish(monkeypatch, True, trim=False, window=True)

        assert sched._valid_tokens == {"r0": 33}

    def test_streaming_alone_collects_the_count(self, monkeypatch):
        # And the third knob that turns the window on asks for it too.
        sched = self._finish(monkeypatch, True, trim=False, stream=True)

        assert sched._valid_tokens == {"r0": 33}

    def test_the_handover_carries_the_count_to_the_worker(self):
        # The positive direction of the case below: the count the scheduler
        # took at handover has to reach the worker in the same step's metadata,
        # and leave the scheduler's map so a later step does not resend it.
        sched = _scheduler(cls=RblnNixlPushConnectorScheduler)
        sched._valid_tokens = {"r0": 33, "r1": 64}
        sched._newly_finished_push_blocks = {"r0": ([1, 2],)}

        meta = sched.build_connector_meta(_sched_output("other", ([9],), 16))

        assert meta.valid_tokens == {"r0": 33}
        # r1 has not been handed over, so its count waits for the step that has.
        assert sched._valid_tokens == {"r1": 64}

    def test_a_streamed_offer_does_not_carry_the_count(self):
        # Only the handover reaches the request's last block; a mid-stream
        # offer is a closed prefix, and the count must wait for the batch that
        # can use it.
        sched = _scheduler(cls=RblnNixlPushConnectorScheduler)
        sched._valid_tokens = {"r0": 33}

        meta = sched.build_connector_meta(_sched_output("other", ([9],), 16))

        assert meta.valid_tokens == {}
        assert sched._valid_tokens == {"r0": 33}
