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
from vllm.v1.request import RequestStatus

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_scheduler as sm
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_scheduler import (
    RblnNixlPullConnectorScheduler,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_scheduler import (
    RblnNixlPushConnectorScheduler,
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


def _sched_output(req_id, block_ids, num_scheduled_tokens, *, is_new=True):
    """A minimal SchedulerOutput for yield_req_data: a fresh req carries its
    block_ids on scheduled_new_reqs; a resumed chunk carries them on
    scheduled_cached_reqs (None once no new blocks are added)."""
    if is_new:
        return _SchedOutput(
            scheduled_new_reqs=[_NewReq(req_id, block_ids)],
            scheduled_cached_reqs=_CachedReqs(),
            num_scheduled_tokens={req_id: num_scheduled_tokens},
        )
    return _SchedOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=_CachedReqs(req_ids=[req_id], new_block_ids=[block_ids]),
        num_scheduled_tokens={req_id: num_scheduled_tokens},
    )


def _scheduler(*, use_host_buffer=False, cls=RblnNixlPullConnectorScheduler):
    sched = object.__new__(cls)
    sched.vllm_config = MagicMock()
    sched.vllm_config.parallel_config.tensor_parallel_size = 1
    sched.block_size = 16
    sched.engine_id = "test-engine"
    sched.kv_cache_config = MagicMock()
    sched.side_channel_host = "localhost"
    sched.side_channel_port = 5000
    # The save path is gated on this, so save tests must turn it on.
    sched.use_host_buffer = use_host_buffer
    sched._is_hma_required = False  # get_sw_clipped_blocks (inherited) reads this
    sched.blocks_per_sw = [0]
    sched._kv_lease_duration = 30
    sched._reqs_need_recv = {}
    sched._reqs_need_save = {}
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
        sched._push_pending_registrations = {}
        sched._push_registration_deadlines = {}
        sched._push_registration_timeout = 480
        sched._finished_request_blocks = {}
        sched._newly_finished_push_blocks = {}
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
        vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_buffer_device=kv_buffer_device)
        )
        sched = object.__new__(RblnNixlPullConnectorScheduler)
        RblnNixlPullConnectorScheduler.__init__(sched, vllm_config, "eng", {"kv": 1})
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

        # A real Request always carries the field, even when it is None.
        scheduler.request_finished(
            SimpleNamespace(request_id="r0", kv_transfer_params=None), ([1, 2],)
        )

        assert scheduler._block_ids_need_save == {}
        assert seen == ["r0"]


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
        # Calls the base's own builder rather than checking the key by hand:
        # what has to hold is that upstream's read of it succeeds.
        sched = _scheduler(cls=RblnNixlPushConnectorScheduler)
        req = self._rejected()

        sched.request_finished(req, ([],))
        meta = sched.build_connector_meta(_sched_output("other", ([9],), 16))

        assert "rejected" in meta.reqs_to_recv
        assert meta.reqs_to_recv["rejected"].remote.block_ids == ()

    def test_the_producers_own_block_ids_are_not_clobbered(self):
        # The producer's reply carries both the remote-prefill flag and its
        # block ids, so a proxy that forwards it leaves the field already
        # filled; only one dispatching to both sides at once leaves it absent.
        sched = _scheduler(cls=RblnNixlPushConnectorScheduler)
        req = self._rejected()
        req.kv_transfer_params["remote_block_ids"] = ([4, 5],)

        sched.request_finished(req, ([],))

        assert req.kv_transfer_params["remote_block_ids"] == ([4, 5],)
