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

# RblnNixlConnectorScheduler: chunked-prefill save tracking in
# build_connector_meta and the finished-request branches in request_finished.
# Built bare with only the state its methods read.

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.request import RequestStatus

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.scheduler as sm
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.scheduler import (
    RblnNixlConnectorScheduler,
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


def _scheduler():
    sched = object.__new__(RblnNixlConnectorScheduler)
    sched.vllm_config = MagicMock()
    sched.vllm_config.parallel_config.tensor_parallel_size = 1
    sched.block_size = 16
    sched.engine_id = "test-engine"
    sched.kv_cache_config = MagicMock()
    sched.side_channel_host = "localhost"
    sched.side_channel_port = 5000
    sched.use_host_buffer = False
    sched._is_hma_required = False  # get_sw_clipped_blocks (inherited) reads this
    sched.blocks_per_sw = [0]
    sched._kv_lease_duration = 30
    sched._reqs_need_recv = {}
    sched._reqs_need_save = {}
    sched._reqs_need_send = {}
    sched._reqs_in_batch = set()
    sched._reqs_not_processed = set()
    sched._block_ids_need_save = {}
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
        sched = object.__new__(RblnNixlConnectorScheduler)
        RblnNixlConnectorScheduler.__init__(sched, vllm_config, "eng", {"kv": 1})
        assert sched.use_host_buffer is expected
        assert sched._block_ids_need_save == {}


class TestBuildConnectorMeta:
    def test_single_step_prefill_saves_immediately(self):
        # A prefill that finishes in one step is added to the save metadata and
        # dropped from both tracking dicts.
        sched = _scheduler()
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
        sched = _scheduler()
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

    def test_non_length_capped_cleans_up_tracking(self):
        # A remote-decode producer that did not finish LENGTH_CAPPED (e.g. an
        # aborted partial prefill) stops being tracked and frees now.
        sched = _scheduler()
        req = _Request("aborted", num_prompt_tokens=512)
        req.status = RequestStatus.FINISHED_STOPPED
        sched._reqs_need_save["aborted"] = req
        sched._block_ids_need_save["aborted"] = ([1, 2],)

        delay, params = sched.request_finished(req, ([],))
        assert (delay, params) == (False, None)
        assert "aborted" not in sched._reqs_need_save
        assert "aborted" not in sched._block_ids_need_save
        assert "aborted" in sched._reqs_not_processed

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
