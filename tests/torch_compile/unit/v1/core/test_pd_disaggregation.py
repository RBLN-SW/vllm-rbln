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

"""Unit tests for P/D (Prefill/Decode) disaggregation.

Tests cover:
- Scheduler: async KV transfer lifecycle and request scheduling
- NIXL connector: chunked prefill block tracking and request finish handling
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    MockKVConfig,
    advance_to_decode,
    create_requests,
    create_runner_output,
    create_scheduler,
)

_BLOCK_SIZE = 16
_NUM_BLOCKS = 512
_MAX_NUM_SEQS = 16


def _create_pd_scheduler(
    matched_tokens,
    block_size=_BLOCK_SIZE,
    num_blocks=_NUM_BLOCKS,
    max_num_seqs=_MAX_NUM_SEQS,
    max_num_batched_tokens=8192,
):
    """Create a scheduler with a mock async KV connector."""
    return create_scheduler(
        block_size=block_size,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        use_kv_connector=MockKVConfig(matched_tokens=matched_tokens, is_async=True),
    )


def _create_pd_request(num_tokens, req_id, do_remote_prefill=True):
    """Create a request for P/D disaggregation tests."""
    req = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        block_size=_BLOCK_SIZE,
        req_ids=[req_id],
    )[0]
    if do_remote_prefill:
        req.kv_transfer_params = {"do_remote_prefill": True}
    return req


def _simulate_kv_transfer_completion(
    scheduler, output, remote_req_id, sampled_token_id=1
):
    """Call update_from_output with a KVConnectorOutput that marks the
    remote request's KV transfer as finished."""
    model_runner_output = create_runner_output(output, sampled_token_id)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={remote_req_id}
    )
    scheduler.update_from_output(output, model_runner_output)


class TestPDDisaggregationScheduler:
    """Scheduler-level tests for P/D disaggregation.

    Each test exercises a distinct aspect of the async KV transfer flow
    that the RBLNScheduler implements on top of the upstream Scheduler.
    """

    def test_async_kv_transitions_to_waiting_for_remote_kvs(self):
        """Request with async KV connector goes to WAITING_FOR_REMOTE_KVS
        state and no tokens are scheduled for it in the current step."""
        num_tokens = 256
        scheduler = _create_pd_scheduler(matched_tokens=num_tokens)

        remote = _create_pd_request(num_tokens, "remote")
        scheduler.add_request(remote)

        output = scheduler.schedule()

        # No tokens scheduled in this step.
        assert len(output.num_scheduled_tokens) == 0
        assert len(output.scheduled_new_reqs) == 0
        # Request transitions to WAITING_FOR_REMOTE_KVS.
        assert remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert len(scheduler.running) == 0
        assert len(scheduler.skipped_waiting) == 1
        # Computed tokens reflect the connector match.
        assert remote.num_computed_tokens == num_tokens

    def test_promoted_remote_request_scheduled_as_decode(self):
        """After KV transfer completes for a full-match request, the
        scheduler re-schedules it as decode request."""
        num_tokens = 256
        scheduler = _create_pd_scheduler(matched_tokens=num_tokens)

        remote = _create_pd_request(num_tokens, "remote")
        scheduler.add_request(remote)

        # Step 1: async schedule → WAITING_FOR_REMOTE_KVS
        output = scheduler.schedule()
        assert remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS

        # Step 2: simulate KV transfer completion via KVConnectorOutput
        _simulate_kv_transfer_completion(scheduler, output, remote.request_id)

        # Step 3: schedule → promoted as decode
        output = scheduler.schedule()
        assert remote.request_id in output.num_scheduled_tokens
        assert output.num_scheduled_tokens[remote.request_id] == 1
        assert remote.status == RequestStatus.RUNNING

    def test_local_prefill_deferred_when_remote_already_scheduled(self):
        """When a remote-prefilled request is scheduled (as decode-like),
        a local prefill waiting request is deferred to the next step."""
        num_tokens = 256
        scheduler = _create_pd_scheduler(matched_tokens=num_tokens)

        remote = _create_pd_request(num_tokens, "remote")
        scheduler.add_request(remote)

        # Step 1: remote → WAITING_FOR_REMOTE_KVS
        output = scheduler.schedule()

        # Step 2: simulate KV transfer completion + add local prefill request
        _simulate_kv_transfer_completion(scheduler, output, remote.request_id)
        local = _create_pd_request(num_tokens, "local", do_remote_prefill=False)
        scheduler.add_request(local)

        # Step 3: remote promoted (decode-like) + local deferred
        output = scheduler.schedule()
        assert remote.request_id in output.num_scheduled_tokens
        assert output.num_scheduled_tokens[remote.request_id] == 1
        assert local.request_id not in output.num_scheduled_tokens

    def test_promoted_remote_coexists_with_running_decode(self):
        """A promoted remote request joins the decode batch alongside
        running decode requests, unlike a normal prefill which would
        kick them out."""
        num_tokens = 64
        scheduler = _create_pd_scheduler(matched_tokens=num_tokens)

        # Step 1: decode request does local prefill + enters decode
        decode = _create_pd_request(num_tokens, "decode", do_remote_prefill=False)
        advance_to_decode(scheduler, decode)

        # Step 2: remote request added → goes WAITING_FOR_REMOTE_KVS
        # (decode is scheduled for decode in this step)
        remote = _create_pd_request(num_tokens, "remote")
        scheduler.add_request(remote)
        output = scheduler.schedule()
        assert decode.request_id in output.num_scheduled_tokens
        assert remote.request_id not in output.num_scheduled_tokens

        # Step 3: simulate remote's KV completion via KVConnectorOutput
        _simulate_kv_transfer_completion(
            scheduler, output, remote.request_id, sampled_token_id=2
        )

        # Step 4: both decode and remote (promoted single-token) scheduled
        output = scheduler.schedule()
        assert decode.request_id in output.num_scheduled_tokens
        assert remote.request_id in output.num_scheduled_tokens
        assert output.num_scheduled_tokens[decode.request_id] == 1
        assert output.num_scheduled_tokens[remote.request_id] == 1

    def test_promotion_keeps_decode_batch_and_defers_local_prefill(self):
        """A ready remote-KV request should join the decode batch, while
        a later local prefill stays deferred to the next step.
        Also verifies the running and waiting queue contents."""
        num_tokens = 10
        scheduler = _create_pd_scheduler(
            matched_tokens=num_tokens, max_num_seqs=4, max_num_batched_tokens=16
        )

        # Running decode request
        decode = _create_pd_request(num_tokens, "decode", do_remote_prefill=False)
        advance_to_decode(scheduler, decode)

        # Remote request → WAITING_FOR_REMOTE_KVS
        remote = _create_pd_request(num_tokens, "remote")
        scheduler.add_request(remote)
        output = scheduler.schedule()
        assert remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS

        # Simulate KV transfer completion via KVConnectorOutput
        _simulate_kv_transfer_completion(
            scheduler, output, remote.request_id, sampled_token_id=1
        )

        # Add local prefill request
        local = _create_pd_request(num_tokens, "local", do_remote_prefill=False)
        scheduler.add_request(local)

        # Schedule: decode + remote promoted, local deferred
        output = scheduler.schedule()
        assert output.scheduled_cached_reqs.req_ids == [decode.request_id]
        assert [req.req_id for req in output.scheduled_new_reqs] == [remote.request_id]
        assert output.num_scheduled_tokens[remote.request_id] == 1
        assert local.request_id not in output.num_scheduled_tokens
        assert [req.request_id for req in scheduler.running] == [
            decode.request_id,
            remote.request_id,
        ]
        assert [req.request_id for req in scheduler.waiting] == [local.request_id]


# ===========================================================================
# NIXL connector tests
# ===========================================================================


@dataclass
class MockNewReqData:
    req_id: str
    block_ids: tuple


@dataclass
class MockCachedReqData:
    req_ids: list = field(default_factory=list)
    new_block_ids: list = field(default_factory=list)
    resumed_req_ids: set = field(default_factory=set)


@dataclass
class MockSchedulerOutput:
    scheduled_new_reqs: list
    scheduled_cached_reqs: MockCachedReqData
    num_scheduled_tokens: dict


@dataclass
class MockRequest:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.RUNNING
    kv_transfer_params: dict = field(default_factory=lambda: {"do_remote_decode": True})


def _make_scheduler_output(req_id, block_ids, num_scheduled_tokens, is_new=True):
    """Build a minimal SchedulerOutput-like object for yield_req_data."""
    if is_new:
        return MockSchedulerOutput(
            scheduled_new_reqs=[MockNewReqData(req_id=req_id, block_ids=block_ids)],
            scheduled_cached_reqs=MockCachedReqData(),
            num_scheduled_tokens={req_id: num_scheduled_tokens},
        )
    else:
        return MockSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=MockCachedReqData(
                req_ids=[req_id],
                new_block_ids=[block_ids],
            ),
            num_scheduled_tokens={req_id: num_scheduled_tokens},
        )


def _create_connector_scheduler():
    """Create an RblnNixlConnectorScheduler with mocked-out dependencies."""
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
        RblnNixlConnectorScheduler,
    )

    sched = object.__new__(RblnNixlConnectorScheduler)

    sched.vllm_config = MagicMock()
    sched.block_size = _BLOCK_SIZE
    sched.engine_id = "test-engine"
    sched.kv_cache_config = MagicMock()
    sched.side_channel_host = "localhost"
    sched.side_channel_port = 5000
    sched.use_host_buffer = False
    sched._is_hma_required = False
    sched.blocks_per_sw = [0]

    sched._reqs_need_recv = {}
    sched._reqs_need_save = {}
    sched._reqs_need_send = {}
    sched._reqs_in_batch = set()
    sched._reqs_not_processed = set()
    sched._block_ids_need_save = {}

    return sched


class TestPDDisaggregationNixlConnector:
    """Tests for RBLN-specific NIXL connector logic.

    Covers chunked prefill block tracking in build_connector_meta
    and cleanup in request_finished.
    """

    def test_single_step_prefill_saves_blocks_immediately(self):
        """When prefill completes in a single step, blocks are saved to
        connector metadata right away."""
        sched = _create_connector_scheduler()
        req = MockRequest("prefill", num_prompt_tokens=256, num_computed_tokens=0)
        sched._reqs_need_save["prefill"] = req

        block_ids = ([1, 2, 3, 4],)
        output = _make_scheduler_output("prefill", block_ids, num_scheduled_tokens=256)

        meta = sched.build_connector_meta(output)

        assert "prefill" in meta.reqs_to_save
        assert "prefill" not in sched._reqs_need_save
        assert "prefill" not in sched._block_ids_need_save

    def test_chunked_prefill_defers_save_until_final_chunk(self):
        """During chunked prefill, blocks are accumulated in
        _block_ids_need_save and only saved to metadata on the final chunk."""
        sched = _create_connector_scheduler()
        req = MockRequest("chunked", num_prompt_tokens=512, num_computed_tokens=0)
        sched._reqs_need_save["chunked"] = req

        # First chunk: 256 of 512 tokens — partial
        block_ids = ([1, 2, 3, 4],)
        output = _make_scheduler_output("chunked", block_ids, num_scheduled_tokens=256)
        meta = sched.build_connector_meta(output)

        assert "chunked" not in meta.reqs_to_save
        assert "chunked" in sched._block_ids_need_save
        assert "chunked" in sched._reqs_need_save

        # Final chunk: remaining 256 tokens — complete
        req.num_computed_tokens = 256
        output = _make_scheduler_output(
            "chunked", None, num_scheduled_tokens=256, is_new=False
        )
        meta = sched.build_connector_meta(output)

        assert "chunked" in meta.reqs_to_save
        assert "chunked" not in sched._block_ids_need_save
        assert "chunked" not in sched._reqs_need_save

    def test_aborted_partial_prefill_cleans_up_tracking(self):
        """When a request is aborted during partial prefill,
        request_finished cleans up both _reqs_need_save and
        _block_ids_need_save."""
        sched = _create_connector_scheduler()
        req = MockRequest("aborted", num_prompt_tokens=512, num_computed_tokens=0)
        req.status = RequestStatus.FINISHED_STOPPED
        sched._reqs_need_save["aborted"] = req
        sched._block_ids_need_save["aborted"] = ([1, 2],)

        delay, _ = sched.request_finished(req, block_ids=([],))

        assert not delay
        assert "aborted" not in sched._reqs_need_save
        assert "aborted" not in sched._block_ids_need_save
        assert "aborted" in sched._reqs_not_processed

    def test_completed_prefill_delays_block_free(self):
        """When a prefill request finishes with FINISHED_LENGTH_CAPPED,
        block free is delayed for remote decode to fetch."""
        sched = _create_connector_scheduler()
        req = MockRequest("done", num_prompt_tokens=256)
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED

        delay, params = sched.request_finished(req, block_ids=([1, 2, 3, 4],))

        assert delay is True
        assert params is not None
        assert params["do_remote_prefill"] is True
        assert params["remote_engine_id"] == "test-engine"
        assert "done" in sched._reqs_need_send
