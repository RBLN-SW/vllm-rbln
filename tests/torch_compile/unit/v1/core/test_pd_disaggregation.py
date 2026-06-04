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


# ===========================================================================
# RBLNSlidingWindowManager.allocate_new_computed_blocks
# ===========================================================================
#
# The D-side P/D receive path routes through `allocate_new_computed_blocks`
# rather than `allocate_new_blocks`. The RBLN SWA kernel uses a single
# in-place ring-buffered block, so this override forces "one block per
# request" regardless of how many computed tokens the scheduler hands us.


def _make_swa_manager():
    """Build an RBLNSlidingWindowManager with the minimum state its
    `allocate_new_computed_blocks` reaches into."""
    from collections import defaultdict

    from vllm_rbln.v1.kv_cache import RBLNSlidingWindowManager

    mgr = object.__new__(RBLNSlidingWindowManager)
    mgr.num_cached_block = {}
    mgr.req_to_blocks = defaultdict(list)
    mgr.block_pool = MagicMock()
    # `get_new_blocks(n)` returns a list of n fresh KVCacheBlock objects.
    mgr.block_pool.get_new_blocks.side_effect = lambda n: [
        MagicMock(name=f"block_{i}") for i in range(n)
    ]
    return mgr


class TestRBLNSlidingWindowManager:
    """`allocate_new_computed_blocks` enforces the SWA kernel's
    one-block-per-request invariant on the receive path."""

    def test_allocates_single_block_for_remote_prefill(self):
        """One block regardless of `num_external_computed_tokens` size."""
        mgr = _make_swa_manager()

        mgr.allocate_new_computed_blocks(
            request_id="req-0",
            new_computed_blocks=[],
            num_local_computed_tokens=0,
            num_external_computed_tokens=2674,
        )

        assert len(mgr.req_to_blocks["req-0"]) == 1
        # Sentinel set so a follow-up call hits the fast path.
        assert mgr.num_cached_block["req-0"] == 0
        mgr.block_pool.get_new_blocks.assert_called_once_with(1)

    def test_no_allocation_when_no_external_tokens(self):
        """Setting num_external_computed_tokens=0 still records the
        sentinel but does not allocate."""
        mgr = _make_swa_manager()

        mgr.allocate_new_computed_blocks(
            request_id="req-0",
            new_computed_blocks=[],
            num_local_computed_tokens=0,
            num_external_computed_tokens=0,
        )

        assert mgr.req_to_blocks["req-0"] == []
        assert mgr.num_cached_block["req-0"] == 0
        mgr.block_pool.get_new_blocks.assert_not_called()

    def test_running_request_fast_path_is_noop(self):
        """Second call for the same request (already in num_cached_block)
        is a no-op."""
        mgr = _make_swa_manager()
        mgr.num_cached_block["req-0"] = 0
        mgr.req_to_blocks["req-0"] = [MagicMock(name="existing")]

        mgr.allocate_new_computed_blocks(
            request_id="req-0",
            new_computed_blocks=[],
            num_local_computed_tokens=128,
            num_external_computed_tokens=512,
        )

        assert len(mgr.req_to_blocks["req-0"]) == 1
        mgr.block_pool.get_new_blocks.assert_not_called()

    def test_rejects_prefix_cache_hits(self):
        """`find_longest_cache_hit` is overridden to return empty, so a
        non-empty new_computed_blocks is a contract violation."""
        import pytest

        mgr = _make_swa_manager()

        with pytest.raises(AssertionError):
            mgr.allocate_new_computed_blocks(
                request_id="req-0",
                new_computed_blocks=[MagicMock(name="leaked_hit")],
                num_local_computed_tokens=0,
                num_external_computed_tokens=128,
            )


# ===========================================================================
# RblnNixlConnectorWorker
# ===========================================================================
#
# We exercise the worker __init__ and host-buffer helpers without touching
# the upstream NIXL agent — `super().__init__` is patched out, and we
# inject only the attributes our overrides read.


def _build_connector_worker(
    kv_buffer_device="cpu",
    num_blocks=128,
    block_size=64,
):
    """Create a RblnNixlConnectorWorker through its __init__ with the
    upstream NixlConnectorWorker side effects stubbed out.

    Returns the constructed worker so tests can inspect post-__init__ state.
    """
    from unittest.mock import patch

    from vllm.config import CacheConfig

    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
        RblnNixlConnectorWorker,
    )

    vllm_config = MagicMock()
    vllm_config.cache_config = CacheConfig(block_size=block_size)
    kv_cache_config = MagicMock()
    kv_cache_config.num_blocks = num_blocks

    def fake_super_init(self, vllm_config_, engine_id_, kv_cache_config_):
        # Set just the attributes our overrides touch / depend on. The real
        # NixlConnectorWorker.__init__ does a lot more, including NIXL agent
        # creation — we don't want any of that in a unit test.
        self.vllm_config = vllm_config_
        self.engine_id = engine_id_
        self.kv_cache_config = kv_cache_config_
        self.kv_buffer_device = kv_buffer_device
        self._block_size = {}

    with patch.object(
        RblnNixlConnectorWorker.__mro__[1], "__init__", fake_super_init
    ):
        return RblnNixlConnectorWorker(
            vllm_config=vllm_config,
            engine_id="test-engine",
            kv_cache_config=kv_cache_config,
        )


class TestRblnNixlConnectorWorkerInit:
    """`__init__` recovers the host-buffer flag (upstream sets it False
    because `RblnPlatform.device_type == 'cpu'`) and pins block sizes to
    logical values."""

    def test_recovers_host_buffer_for_cpu_kv_device(self):
        worker = _build_connector_worker(kv_buffer_device="cpu")
        assert worker.use_host_buffer is True

    def test_no_host_buffer_when_kv_device_is_non_cpu(self):
        worker = _build_connector_worker(kv_buffer_device="cuda")
        assert worker.use_host_buffer is False

    def test_pins_logical_block_sizes(self):
        worker = _build_connector_worker(num_blocks=128, block_size=64)
        assert worker.num_blocks == 128
        assert worker.block_size == 64
        assert worker._physical_blocks_per_logical_kv_block == 1
        assert worker._logical_num_blocks == 128
        assert worker._block_size["test-engine"] == 64


class TestRblnNixlConnectorWorkerHostBuffer:
    """`initialize_host_xfer_buffer` / `set_host_xfer_buffer_ops` honor
    HND layout, allocate one rebel-aligned buffer per filtered layer, and
    preserve insertion order (matters for NIXL region indexing in P/D)."""

    def _patch_worker(self, kv_cache_layout="HND"):
        worker = _build_connector_worker()
        worker.kv_cache_layout = kv_cache_layout
        return worker

    def test_one_buffer_per_layer_preserves_order(self):
        """Iterates `kv_caches.items()` in input order; result dict keeps
        that order — load-bearing for P/D region <-> layer mapping."""
        import torch

        worker = self._patch_worker()
        kv_caches = {
            f"model.layers.{i}.attn": torch.zeros(4, 2, dtype=torch.float32)
            for i in (3, 1, 7, 0)
        }

        worker.initialize_host_xfer_buffer(kv_caches)

        assert list(worker.host_xfer_buffers.keys()) == list(kv_caches.keys())
        for name, original in kv_caches.items():
            assert worker.host_xfer_buffers[name].shape == original.shape

    def test_asserts_hnd_layout(self):
        import pytest
        import torch

        worker = self._patch_worker(kv_cache_layout="NHD")
        with pytest.raises(AssertionError, match="HND"):
            worker.initialize_host_xfer_buffer(
                {"layer0": torch.zeros(4, 2, dtype=torch.float32)}
            )

    def test_set_ops_noop_when_kv_buffer_not_cpu(self):
        """When kv_buffer_device is not 'cpu' the operation is a no-op —
        host-buffer copies aren't needed."""
        worker = _build_connector_worker(kv_buffer_device="cuda")

        sentinel = MagicMock(name="copy_op")
        worker.set_host_xfer_buffer_ops(sentinel)

        assert not hasattr(worker, "copy_blocks") or worker.copy_blocks is not sentinel

    def test_set_ops_assigns_copy_when_kv_buffer_is_cpu(self):
        worker = _build_connector_worker(kv_buffer_device="cpu")

        sentinel = MagicMock(name="copy_op")
        worker.set_host_xfer_buffer_ops(sentinel)

        assert worker.copy_blocks is sentinel


# ===========================================================================
# RblnNixlConnectorWorker — emulation toggles
# ===========================================================================
#
# Cost-isolation toggles that short-circuit either the host xfer
# (`VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP`) or the NIXL RDMA `READ`
# (`VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP`); both are read lazily through
# `vllm_rbln.rbln_envs`, so tests patch `os.environ` directly.


def _emulation_env(host_xfer_noop=False, remote_xfer_noop=False):
    """`patch.dict(os.environ, ...)` payload that mirrors the lazy lookups
    in `rbln_envs.VLLM_RBLN_EMULATE_*`."""
    return {
        "VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP": "1" if host_xfer_noop else "0",
        "VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP": "1" if remote_xfer_noop else "0",
    }



class TestRblnNixlConnectorWorkerNoopCopyBlocks:
    """`_noop_copy_blocks` is the static stand-in installed under
    `VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP=1`."""

    def test_accepts_arbitrary_signature_returns_none(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
            RblnNixlConnectorWorker,
        )

        assert RblnNixlConnectorWorker._noop_copy_blocks() is None
        assert RblnNixlConnectorWorker._noop_copy_blocks(1, 2, 3) is None
        assert (
            RblnNixlConnectorWorker._noop_copy_blocks(
                src={}, dst={}, direction="h2d", view=128
            )
            is None
        )


class TestRblnNixlConnectorWorkerEmulationHostXfer:
    """`set_host_xfer_buffer_ops` and `initialize_host_xfer_buffer` swap
    to emulation-mode behavior when the env-var-backed `envs.VLLM_RBLN_*`
    toggles are set. The reads are lazy through `rbln_envs.__getattr__`,
    so `patch.dict(os.environ, ...)` is sufficient."""

    def test_set_ops_swaps_to_noop_under_emulation(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, _emulation_env(host_xfer_noop=True)):
            worker = _build_connector_worker(kv_buffer_device="cpu")
            caller_op = MagicMock(name="real_copy_op")
            worker.set_host_xfer_buffer_ops(caller_op)
            assert worker.copy_blocks is worker._noop_copy_blocks
            assert worker.copy_blocks is not caller_op

    def test_set_ops_uses_caller_when_not_emulating(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, _emulation_env(host_xfer_noop=False)):
            worker = _build_connector_worker(kv_buffer_device="cpu")
            caller_op = MagicMock(name="real_copy_op")
            worker.set_host_xfer_buffer_ops(caller_op)
            assert worker.copy_blocks is caller_op

    def test_initialize_buffer_shares_one_allocation_under_emulation(self):
        import os
        import torch
        from unittest.mock import patch

        with patch.dict(
            os.environ, _emulation_env(host_xfer_noop=True, remote_xfer_noop=False)
        ):
            worker = _build_connector_worker()
            worker.kv_cache_layout = "HND"
            kv_caches = {
                f"layer{i}": torch.zeros(4, 2, dtype=torch.float32)
                for i in range(3)
            }
            worker.initialize_host_xfer_buffer(kv_caches)

            buffers = list(worker.host_xfer_buffers.values())
            # All layer entries map to the same underlying allocation.
            for b in buffers[1:]:
                assert b is buffers[0]
            # Order preserved (P/D NIXL region indexing depends on it).
            assert list(worker.host_xfer_buffers.keys()) == list(
                kv_caches.keys()
            )

    def test_emulation_also_active_when_remote_xfer_noop_set(self):
        """Either emulation flag triggers shared-allocation: NIXL-noop on
        its own still expects host buffers to be a fixed allocation since
        no actual transport ever touches them."""
        import os
        import torch
        from unittest.mock import patch

        with patch.dict(
            os.environ, _emulation_env(host_xfer_noop=False, remote_xfer_noop=True)
        ):
            worker = _build_connector_worker()
            worker.kv_cache_layout = "HND"
            kv_caches = {
                f"layer{i}": torch.zeros(4, 2, dtype=torch.float32)
                for i in range(2)
            }
            worker.initialize_host_xfer_buffer(kv_caches)

            buffers = list(worker.host_xfer_buffers.values())
            assert buffers[0] is buffers[1]

    def test_emulation_asserts_uniform_shape(self):
        import os
        import pytest
        import torch
        from unittest.mock import patch

        with patch.dict(os.environ, _emulation_env(host_xfer_noop=True)):
            worker = _build_connector_worker()
            worker.kv_cache_layout = "HND"
            kv_caches = {
                "a": torch.zeros(4, 2, dtype=torch.float32),
                "b": torch.zeros(8, 2, dtype=torch.float32),  # different shape
            }
            with pytest.raises(AssertionError, match="uniform"):
                worker.initialize_host_xfer_buffer(kv_caches)

    def test_no_emulation_keeps_per_layer_allocation(self):
        import os
        import torch
        from unittest.mock import patch

        with patch.dict(
            os.environ, _emulation_env(host_xfer_noop=False, remote_xfer_noop=False)
        ):
            worker = _build_connector_worker()
            worker.kv_cache_layout = "HND"
            kv_caches = {
                f"layer{i}": torch.zeros(4, 2, dtype=torch.float32)
                for i in range(3)
            }
            worker.initialize_host_xfer_buffer(kv_caches)

            buffers = list(worker.host_xfer_buffers.values())
            # Distinct storages — non-emulation path allocates per layer.
            ptrs = {b.data_ptr() for b in buffers}
            assert len(ptrs) == 3


class TestRblnNixlConnectorWorkerEmulationReadBlocks:
    """`_read_blocks` short-circuits the RDMA `READ` under
    `VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP=1`: send P-side notif + touch the
    `_recving_transfers` key so `_pop_done_transfers` marks completion."""

    def _setup_worker(self):
        from collections import defaultdict

        worker = _build_connector_worker()
        worker.world_size = 1
        worker._remote_agents = {"dst-eng": {0: "agent-A"}}
        worker.nixl_wrapper = MagicMock()
        worker.xfer_stats = MagicMock()
        worker._log_failure = MagicMock()
        worker._recving_transfers = defaultdict(list)
        return worker

    def test_emulation_skips_super_sends_notif_marks_done(self):
        import os
        from unittest.mock import patch

        worker = self._setup_worker()
        super_cls = type(worker).__mro__[1]
        super_read = MagicMock(name="super_read_blocks")
        with patch.dict(
            os.environ, _emulation_env(remote_xfer_noop=True)
        ), patch.object(super_cls, "_read_blocks", super_read):
            worker._read_blocks(
                local_block_ids=[[1]],
                remote_block_ids=[[1]],
                dst_engine_id="dst-eng",
                request_id="req-1",
                remote_request_id="rreq-1",
                remote_rank=0,
                local_xfer_side_handle=10,
                remote_xfer_side_handle=20,
            )

        super_read.assert_not_called()
        # P-side notified — notif_id format is `{remote_request_id}:{world_size}`
        worker.nixl_wrapper.send_notif.assert_called_once()
        call = worker.nixl_wrapper.send_notif.call_args
        assert call.args[0] == "agent-A"
        assert call.kwargs.get("notif_msg") == b"rreq-1:1"
        # D-side: key created in defaultdict, value is the empty handle list
        # that `_pop_done_transfers` treats as completed.
        assert "req-1" in worker._recving_transfers
        assert worker._recving_transfers["req-1"] == []

    def test_emulation_swallows_send_notif_exception(self):
        import os
        from unittest.mock import patch

        worker = self._setup_worker()
        worker.nixl_wrapper.send_notif.side_effect = RuntimeError("boom")
        super_cls = type(worker).__mro__[1]

        with patch.dict(
            os.environ, _emulation_env(remote_xfer_noop=True)
        ), patch.object(super_cls, "_read_blocks", MagicMock()):
            # Should not raise; failure is logged and stats recorded.
            worker._read_blocks(
                local_block_ids=[[1]],
                remote_block_ids=[[1]],
                dst_engine_id="dst-eng",
                request_id="req-1",
                remote_request_id="rreq-1",
                remote_rank=0,
                local_xfer_side_handle=10,
                remote_xfer_side_handle=20,
            )

        worker._log_failure.assert_called_once()
        worker.xfer_stats.record_failed_notification.assert_called_once()
        # The done marker is still set so the request doesn't stick around.
        assert "req-1" in worker._recving_transfers

    def test_no_emulation_delegates_to_super(self):
        import os
        from unittest.mock import patch

        worker = self._setup_worker()
        super_cls = type(worker).__mro__[1]
        super_read = MagicMock(name="super_read_blocks")

        with patch.dict(
            os.environ, _emulation_env(remote_xfer_noop=False)
        ), patch.object(super_cls, "_read_blocks", super_read):
            worker._read_blocks(
                local_block_ids=[[1]],
                remote_block_ids=[[1]],
                dst_engine_id="dst-eng",
                request_id="req-1",
                remote_request_id="rreq-1",
                remote_rank=0,
                local_xfer_side_handle=10,
                remote_xfer_side_handle=20,
            )

        super_read.assert_called_once()
        # The emulation-only side effects must NOT fire.
        worker.nixl_wrapper.send_notif.assert_not_called()
        assert "req-1" not in worker._recving_transfers
