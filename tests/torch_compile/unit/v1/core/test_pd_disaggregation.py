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


def _spec_mock(cls, **attrs):
    """Build a MagicMock that `isinstance(..., cls)` matches.

    Tests build per-group `kv_cache_spec` stand-ins this way so
    `__init__` can walk them without pulling in the real
    `KVCacheConfig` / `KVCacheGroup` plumbing.
    """
    m = MagicMock(spec=cls)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _build_connector_worker(
    kv_buffer_device="cpu",
    num_blocks=128,
    block_size=64,
    kv_cache_specs=None,
):
    """Create a RblnNixlConnectorWorker through its __init__ with the
    upstream NixlConnectorWorker side effects stubbed out.

    `kv_cache_specs` (optional): list of `kv_cache_spec` objects (one per
    group). When provided, `kv_cache_config.kv_cache_groups` is wired so
    `__init__` populates `_group_specs` / `_sw_ratio` from real specs.
    Omit it to leave `MagicMock`'s default empty iteration in place —
    keeps `_group_specs == []` and `_sw_ratio is None`.

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
    if kv_cache_specs is not None:
        kv_cache_config.kv_cache_groups = [
            MagicMock(kv_cache_spec=spec) for spec in kv_cache_specs
        ]

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


# ---------------------------------------------------------------------------
# VLLM_RBLN_NIXL_SWA_VIEW_OPT — NIXL desc layout scope
# ---------------------------------------------------------------------------
#
# `_group_specs` / `_sw_ratio` derivation in `__init__`, and the
# per-group dispatch in `_get_block_descs_ids`. Host h2d/d2h copy is
# out of scope for this toggle.


class TestRblnNixlConnectorWorkerSWARatio:
    """`__init__` walks `kv_cache_groups` to populate `_group_specs` and
    derive `_sw_ratio` (drives the dual desc-range layout downstream)."""

    def test_pure_full_keeps_sw_ratio_none(self):
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            worker = _build_connector_worker(
                kv_cache_specs=[_spec_mock(FullAttentionSpec, block_size=1024)]
            )
        assert len(worker._group_specs) == 1
        assert worker._sw_ratio is None

    def test_hybrid_full_swa_derives_ratio(self):
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            worker = _build_connector_worker(
                kv_cache_specs=[
                    _spec_mock(FullAttentionSpec, block_size=1024),
                    _spec_mock(
                        SlidingWindowSpec, block_size=1024, sliding_window=128
                    ),
                ]
            )
        assert worker._sw_ratio == 8
        assert len(worker._group_specs) == 2

    def test_ratio_one_collapses_to_none(self):
        """`sliding_window == block_size` is degenerate: SWA desc length
        equals Full desc length, so the dual-range layout would be a
        no-op. `_sw_ratio` stays `None` to keep the pure-Full path."""
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            worker = _build_connector_worker(
                kv_cache_specs=[
                    _spec_mock(
                        SlidingWindowSpec, block_size=128, sliding_window=128
                    )
                ]
            )
        assert worker._sw_ratio is None

    def test_multiple_swa_groups_with_consistent_ratio(self):
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            worker = _build_connector_worker(
                kv_cache_specs=[
                    _spec_mock(
                        SlidingWindowSpec, block_size=1024, sliding_window=128
                    ),
                    _spec_mock(
                        SlidingWindowSpec, block_size=1024, sliding_window=128
                    ),
                ]
            )
        assert worker._sw_ratio == 8

    def test_mismatched_swa_ratios_assert(self):
        """The dual-range layout assumes a single SWA desc length, so
        groups with different ratios are rejected at __init__."""
        import os
        import pytest
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            with pytest.raises(AssertionError, match="single SWA ratio"):
                _build_connector_worker(
                    kv_cache_specs=[
                        _spec_mock(
                            SlidingWindowSpec,
                            block_size=1024,
                            sliding_window=128,
                        ),
                        _spec_mock(
                            SlidingWindowSpec,
                            block_size=1024,
                            sliding_window=256,
                        ),
                    ]
                )

    def test_non_multiple_sliding_window_asserts(self):
        import os
        import pytest
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            with pytest.raises(AssertionError):
                _build_connector_worker(
                    kv_cache_specs=[
                        _spec_mock(
                            SlidingWindowSpec,
                            block_size=1024,
                            sliding_window=300,
                        )
                    ]
                )


class TestRblnNixlConnectorWorkerGetBlockDescsIds:
    """`_get_block_descs_ids` lays out dual desc ranges sharing the same
    Full-region base addresses: Full descs in [0, num_regions*num_blocks),
    SWA descs in [num_regions*num_blocks, 2*num_regions*num_blocks).

    With `_sw_ratio is None` (opt off, pure Full, or degenerate ratio==1)
    it collapses to upstream's single concat path.
    """

    def _worker(self, specs, num_blocks=4, num_regions=2, opt_on=True):
        import os
        from unittest.mock import patch

        env_val = "1" if opt_on else "0"
        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": env_val}):
            worker = _build_connector_worker(
                kv_cache_specs=specs, num_blocks=num_blocks
            )
        worker.dst_num_blocks = {"test-engine": num_blocks}
        worker.num_regions = num_regions
        return worker

    def test_pure_full_uses_lower_range_no_offset(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        worker = self._worker(
            [_spec_mock(FullAttentionSpec, block_size=1024)],
            num_blocks=4,
            num_regions=2,
        )
        # `_sw_ratio` is None -> single-range concat path (matches upstream).
        # region 0: 0*4 + [1,2] = [1, 2]
        # region 1: 1*4 + [1,2] = [5, 6]
        out = worker._get_block_descs_ids("test-engine", [[1, 2]])
        assert list(out) == [1, 2, 5, 6]

    def test_hybrid_routes_full_low_swa_high(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        worker = self._worker(
            [
                _spec_mock(FullAttentionSpec, block_size=1024),
                _spec_mock(
                    SlidingWindowSpec, block_size=1024, sliding_window=128
                ),
            ],
            num_blocks=4,
            num_regions=2,
        )
        # num_full_descs = num_regions * num_blocks = 8
        # Full  [g=0, block 0]: regions -> [0, 4]
        # SWA   [g=1, block 1]: regions -> [8+1, 8+5] = [9, 13]
        out = worker._get_block_descs_ids("test-engine", [[0], [1]])
        assert list(out) == [0, 4, 9, 13]

    def test_empty_group_skipped(self):
        """Some requests touch only one group's blocks — the empty groups
        must drop out without contributing stray indices."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        worker = self._worker(
            [
                _spec_mock(FullAttentionSpec, block_size=1024),
                _spec_mock(
                    SlidingWindowSpec, block_size=1024, sliding_window=128
                ),
            ],
            num_blocks=4,
            num_regions=2,
        )
        out = worker._get_block_descs_ids("test-engine", [[], [2]])
        # Only SWA contributes; num_full_descs=8; regions -> [8+2, 8+6]
        assert list(out) == [10, 14]

    def test_all_empty_returns_empty_array(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        worker = self._worker(
            [
                _spec_mock(FullAttentionSpec, block_size=1024),
                _spec_mock(
                    SlidingWindowSpec, block_size=1024, sliding_window=128
                ),
            ],
            num_blocks=4,
            num_regions=2,
        )
        out = worker._get_block_descs_ids("test-engine", [[], []])
        assert list(out) == []


class TestRblnNixlConnectorWorkerSWAViewOptToggle:
    """`VLLM_RBLN_NIXL_SWA_VIEW_OPT=0` collapses the hybrid SWA path back to
    the single Full-sized desc layout even when SlidingWindowSpec groups
    exist. The opt gates `_group_specs` / `_sw_ratio` derivation *and*
    the per-group desc dispatch.
    """

    def test_opt_off_skips_group_specs_and_ratio(self):
        """Env off: derivation block doesn't run, so both `_group_specs`
        and `_sw_ratio` stay at their no-op defaults — the desc dispatch
        consults neither (early returns on `_sw_ratio is None`)."""
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "0"}):
            worker = _build_connector_worker(
                kv_cache_specs=[
                    _spec_mock(FullAttentionSpec, block_size=1024),
                    _spec_mock(
                        SlidingWindowSpec, block_size=1024, sliding_window=128
                    ),
                ]
            )
        assert worker._sw_ratio is None
        assert worker._group_specs == []

    def test_opt_off_get_block_descs_uses_single_range(self):
        """No SWA-side desc offset: SWA group's block_ids resolve into the
        same Full-sized range as Full groups."""
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "0"}):
            worker = _build_connector_worker(
                kv_cache_specs=[
                    _spec_mock(FullAttentionSpec, block_size=1024),
                    _spec_mock(
                        SlidingWindowSpec, block_size=1024, sliding_window=128
                    ),
                ],
                num_blocks=4,
            )
        worker.dst_num_blocks = {"test-engine": 4}
        worker.num_regions = 2
        # Both groups' block_ids concat into one Full-sized range
        # without the `num_full_descs` offset.
        # Concatenated block_ids: [0, 1] → region 0: [0, 1], region 1: [4, 5]
        out = worker._get_block_descs_ids("test-engine", [[0], [1]])
        assert list(out) == [0, 1, 4, 5]

    def test_opt_on_default_keeps_derivation(self):
        """Sanity: with the toggle explicitly set ON, derivation works
        exactly like `TestRblnNixlConnectorWorkerSWARatio
        ::test_hybrid_full_swa_derives_ratio`."""
        import os
        from unittest.mock import patch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        with patch.dict(os.environ, {"VLLM_RBLN_NIXL_SWA_VIEW_OPT": "1"}):
            worker = _build_connector_worker(
                kv_cache_specs=[
                    _spec_mock(FullAttentionSpec, block_size=1024),
                    _spec_mock(
                        SlidingWindowSpec, block_size=1024, sliding_window=128
                    ),
                ]
            )
        assert worker._sw_ratio == 8
        assert len(worker._group_specs) == 2


# ---------------------------------------------------------------------------
# kv_buffer_device="rbln" (D2D) coverage on the unified connector
# ---------------------------------------------------------------------------
#
# Cover the D2D branches on the unified worker: init-time invariants,
# deferred-registration flow, host-xfer-ops gating, runtime_holder
# forwarding, and the `_register_kv_caches_impl` body. NIXL/RBLN
# runtime is fully mocked.


def _build_d2d_worker(**kw):
    """Convenience around `_build_connector_worker(kv_buffer_device='rbln')`
    that turns env-toggle-gated logger.warning calls into no-ops for
    cleaner test failure messages."""
    kw.setdefault("kv_buffer_device", "rbln")
    return _build_connector_worker(**kw)


class TestRblnNixlConnectorInitAsserts:
    """Outer `RblnNixlConnector.__init__` validates `kv_buffer_device` and
    requires `VLLM_RBLN_USE_DEVICE_TENSOR=1` before constructing either
    the scheduler or worker side."""

    def _vllm_config(self, kv_buffer_device="cpu"):
        cfg = MagicMock()
        cfg.kv_transfer_config = MagicMock()
        cfg.kv_transfer_config.engine_id = "test-engine"
        cfg.kv_transfer_config.kv_buffer_device = kv_buffer_device
        return cfg

    def test_rejects_unknown_kv_buffer_device(self):
        import os
        import pytest
        from unittest.mock import patch
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (  # noqa: E501
            RblnNixlConnector,
        )

        with patch.dict(os.environ, {"VLLM_RBLN_USE_DEVICE_TENSOR": "1"}):
            with pytest.raises(AssertionError, match="kv_buffer_device"):
                RblnNixlConnector(
                    vllm_config=self._vllm_config(kv_buffer_device="cuda"),
                    role=KVConnectorRole.SCHEDULER,
                    kv_cache_config=MagicMock(),
                )

    def test_requires_use_device_tensor(self):
        import os
        import pytest
        from unittest.mock import patch
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (  # noqa: E501
            RblnNixlConnector,
        )

        with patch.dict(os.environ, {"VLLM_RBLN_USE_DEVICE_TENSOR": "0"}):
            with pytest.raises(
                AssertionError, match="VLLM_RBLN_USE_DEVICE_TENSOR"
            ):
                RblnNixlConnector(
                    vllm_config=self._vllm_config(kv_buffer_device="rbln"),
                    role=KVConnectorRole.SCHEDULER,
                    kv_cache_config=MagicMock(),
                )


class TestRblnNixlConnectorSchedulerHostBufferFlag:
    """`RblnNixlConnectorScheduler.use_host_buffer` mirrors
    `kv_buffer_device == 'cpu'` so scheduler-side helpers can branch
    without re-reading the env at every call."""

    def _vllm_config(self, kv_buffer_device):
        cfg = MagicMock()
        cfg.kv_transfer_config = MagicMock()
        cfg.kv_transfer_config.engine_id = "test-engine"
        cfg.kv_transfer_config.kv_buffer_device = kv_buffer_device
        cfg.parallel_config = MagicMock()
        cfg.parallel_config.tensor_parallel_size = 1
        return cfg

    def _build(self, kv_buffer_device):
        from unittest.mock import patch
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (  # noqa: E501
            RblnNixlConnectorScheduler,
        )

        def fake_super_init(self, vllm_config_, engine_id_, kv_cache_config_):
            self.vllm_config = vllm_config_
            self.engine_id = engine_id_
            self.kv_cache_config = kv_cache_config_

        with patch.object(
            RblnNixlConnectorScheduler.__mro__[1], "__init__", fake_super_init
        ):
            return RblnNixlConnectorScheduler(
                vllm_config=self._vllm_config(kv_buffer_device),
                engine_id="test-engine",
                kv_cache_config=MagicMock(),
            )

    def test_cpu_sets_host_buffer_true(self):
        scheduler = self._build("cpu")
        assert scheduler.use_host_buffer is True

    def test_rbln_clears_host_buffer(self):
        scheduler = self._build("rbln")
        assert scheduler.use_host_buffer is False


class TestRblnNixlConnectorWorkerD2DInit:
    """Worker `__init__` under `kv_buffer_device='rbln'` flips
    `use_host_buffer`, raises `nixl_memory_type` to VRAM, and primes the
    deferred-registration slots that `set_runtime_holder` /
    `finalize_kv_cache_registration` later consume."""

    def test_d2d_init_state(self):
        worker = _build_d2d_worker()
        # Host buffer disabled — D2D path runs directly off the device tensor.
        assert worker.use_host_buffer is False
        # VRAM segment: nixl-rbln uses ibv_reg_dmabuf_mr for device dmabufs.
        assert worker.nixl_memory_type == "VRAM"
        # Both deferred-registration slots are declared up-front so other
        # methods can check them without a `getattr` probe.
        assert worker._pending_kv_caches is None
        assert worker._runtime_holder is None
        # Block-size pins still apply on the D2D path.
        assert worker.num_blocks == 128
        assert worker.block_size == 64

    def test_cpu_init_leaves_nixl_memory_type_alone(self):
        """The `kv_buffer_device == 'rbln'` block is the only place
        `nixl_memory_type` is raised to VRAM. On the cpu path our
        override does nothing — upstream's value (whatever the platform
        returns from `get_nixl_memory_type`, default DRAM) survives.
        The fake super-init in the helper doesn't replay upstream's
        full set-up, so the attribute may simply not exist; assert
        only that we didn't accidentally raise it to VRAM."""
        worker = _build_connector_worker(kv_buffer_device="cpu")
        assert getattr(worker, "nixl_memory_type", "DRAM") != "VRAM"


class TestRblnNixlConnectorWorkerEmulationWarningsAtInit:
    """`VLLM_RBLN_NIXL_EMULATE_*_NOOP=1` are dev-only toggles that produce
    garbage outputs in a real run; the worker logs a loud warning at
    startup so a stale `1` left in the env doesn't go unnoticed."""

    def test_warns_on_host_xfer_noop(self):
        import os
        from unittest.mock import patch

        with patch.dict(
            os.environ, {"VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP": "1"}
        ):
            with patch(
                "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector.logger"
            ) as mock_logger:
                _build_connector_worker(kv_buffer_device="cpu")
        # At least one warning mentioning the env var fired.
        assert any(
            "VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP" in str(call_args)
            for call_args in mock_logger.warning.call_args_list
        )

    def test_warns_on_remote_xfer_noop(self):
        import os
        from unittest.mock import patch

        with patch.dict(
            os.environ, {"VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP": "1"}
        ):
            with patch(
                "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector.logger"
            ) as mock_logger:
                _build_connector_worker(kv_buffer_device="cpu")
        assert any(
            "VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP" in str(call_args)
            for call_args in mock_logger.warning.call_args_list
        )

    def test_quiet_when_toggles_off(self):
        import os
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP": "0",
                "VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP": "0",
            },
        ):
            with patch(
                "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector.logger"
            ) as mock_logger:
                _build_connector_worker(kv_buffer_device="cpu")
        # No emulation warnings at all — separate from upstream's own logs.
        emul_calls = [
            c
            for c in mock_logger.warning.call_args_list
            if "EMULATE" in str(c)
        ]
        assert emul_calls == []


class TestRblnNixlConnectorWorkerRegisterKvCachesDispatch:
    """`register_kv_caches` is the path fork. On D2D the backing memory
    isn't allocated yet (warm-up does that), so the call stashes the
    kv_caches and returns; the real NIXL registration runs later in
    `finalize_kv_cache_registration`. On host-bounce it goes through
    upstream's flow after creating the RBLN backend on the agent."""

    def test_rbln_stashes_kv_caches_and_returns(self):
        from unittest.mock import patch

        worker = _build_d2d_worker()
        kv_caches = {"l0": MagicMock(), "l1": MagicMock()}

        # Make sure super().register_kv_caches is NOT called on the D2D
        # path and that ensure_rbln_backend stays untouched.
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1 import (
            rbln_nixl_connector as conn_mod,
        )

        with patch.object(
            conn_mod.NixlConnectorWorker, "register_kv_caches"
        ) as super_register:
            with patch("nixl_rbln.ensure_rbln_backend") as ensure_backend:
                worker.register_kv_caches(kv_caches)

        super_register.assert_not_called()
        ensure_backend.assert_not_called()
        assert worker._pending_kv_caches is kv_caches

    def test_cpu_creates_backend_and_delegates_to_super(self):
        from unittest.mock import patch
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1 import (
            rbln_nixl_connector as conn_mod,
        )

        worker = _build_connector_worker(kv_buffer_device="cpu")
        # nixl_wrapper is provided by upstream __init__; helper's fake
        # super_init doesn't, so install a stand-in.
        worker.nixl_wrapper = MagicMock()
        kv_caches = {"l0": MagicMock()}

        with patch.object(
            conn_mod.NixlConnectorWorker, "register_kv_caches"
        ) as super_register:
            with patch("nixl_rbln.ensure_rbln_backend") as ensure_backend:
                worker.register_kv_caches(kv_caches)

        ensure_backend.assert_called_once_with(worker.nixl_wrapper, device_id=0)
        super_register.assert_called_once_with(kv_caches)
        # The D2D-only slot must stay clear on the host-bounce path.
        assert worker._pending_kv_caches is None


class TestRblnNixlConnectorWorkerFinalizeKvCacheRegistration:
    """`finalize_kv_cache_registration` is the deferred-dispatch hook.
    No pending kv_caches → no-op (idempotent, also fires on the
    host-bounce path where there's nothing to do). Pending present →
    swap to `None` and hand to `_register_kv_caches_impl`."""

    def test_no_pending_is_noop(self):
        from unittest.mock import patch

        worker = _build_d2d_worker()
        assert worker._pending_kv_caches is None
        with patch.object(worker, "_register_kv_caches_impl") as impl:
            worker.finalize_kv_cache_registration()
        impl.assert_not_called()

    def test_pending_dispatches_to_impl_and_clears(self):
        from unittest.mock import patch

        worker = _build_d2d_worker()
        kv_caches = {"l0": MagicMock()}
        worker._pending_kv_caches = kv_caches

        with patch.object(worker, "_register_kv_caches_impl") as impl:
            worker.finalize_kv_cache_registration()

        impl.assert_called_once_with(kv_caches)
        # Slot must be cleared before impl is called so a second invocation
        # is a guaranteed no-op (idempotent on re-entry).
        assert worker._pending_kv_caches is None

    def test_double_call_is_safe(self):
        """Second invocation finds `_pending_kv_caches is None` and exits
        before `_register_kv_caches_impl` is touched again."""
        from unittest.mock import patch

        worker = _build_d2d_worker()
        worker._pending_kv_caches = {"l0": MagicMock()}
        with patch.object(worker, "_register_kv_caches_impl") as impl:
            worker.finalize_kv_cache_registration()
            worker.finalize_kv_cache_registration()
        assert impl.call_count == 1


class TestRblnNixlConnectorWorkerSetHostXferBufferOps:
    """`set_host_xfer_buffer_ops` is called by the runner regardless of
    `kv_buffer_device`. On D2D there's no host buffer to copy through,
    so the call must short-circuit and leave `copy_blocks` unset."""

    def test_rbln_returns_without_setting_copy_blocks(self):
        worker = _build_d2d_worker()
        sentinel = MagicMock()
        worker.set_host_xfer_buffer_ops(sentinel)
        # copy_blocks must not have been assigned to the runner-supplied op.
        assert getattr(worker, "copy_blocks", None) is not sentinel

    def test_cpu_assigns_copy_blocks(self):
        """Sanity: on the host-bounce path the runner-supplied callback
        is installed verbatim (the emulation toggle, if set, would
        replace it — that's a separate test class above)."""
        import os
        from unittest.mock import patch

        worker = _build_connector_worker(kv_buffer_device="cpu")
        worker.use_host_buffer = True
        sentinel = MagicMock()
        with patch.dict(
            os.environ, {"VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP": "0"}
        ):
            worker.set_host_xfer_buffer_ops(sentinel)
        assert worker.copy_blocks is sentinel


class TestRblnNixlConnectorSetRuntimeHolder:
    """`RblnNixlConnector.set_runtime_holder` forwards the runner's holder
    onto the worker so `_register_kv_caches_impl` can resolve the
    RblnContext pointer at registration time. Scheduler-role connector
    has no worker — the forward is a guarded no-op."""

    def _make_connector(self, role):
        import os
        from unittest.mock import patch
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (  # noqa: E501
            RblnNixlConnector,
        )

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.engine_id = "test-engine"
        vllm_config.kv_transfer_config.kv_buffer_device = "rbln"
        # Side-step the scheduler/worker construction — we only test the
        # outer set_runtime_holder method.
        with patch.dict(os.environ, {"VLLM_RBLN_USE_DEVICE_TENSOR": "1"}):
            with patch(
                "vllm_rbln.distributed.kv_transfer.kv_connector.v1."
                "rbln_nixl_connector.RblnNixlConnectorScheduler"
            ), patch(
                "vllm_rbln.distributed.kv_transfer.kv_connector.v1."
                "rbln_nixl_connector.RblnNixlConnectorWorker"
            ), patch.object(
                RblnNixlConnector.__mro__[1], "__init__", lambda *a, **k: None
            ):
                return RblnNixlConnector(
                    vllm_config=vllm_config,
                    role=role,
                    kv_cache_config=MagicMock(),
                )

    def test_worker_role_forwards_holder(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        connector = self._make_connector(KVConnectorRole.WORKER)
        # Worker is a MagicMock from our patched class above; set
        # _runtime_holder explicitly so the assignment is observable.
        connector.connector_worker._runtime_holder = None
        holder = ["sentinel"]
        connector.set_runtime_holder(holder)
        assert connector.connector_worker._runtime_holder is holder

    def test_scheduler_role_is_safe_noop(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        connector = self._make_connector(KVConnectorRole.SCHEDULER)
        # No worker on the scheduler side — must not raise.
        connector.connector_worker = None
        connector.set_runtime_holder(["sentinel"])


class TestRblnNixlConnectorWorkerRegisterKvCachesImpl:
    """`_register_kv_caches_impl` (the deferred D2D body) is what
    `finalize_kv_cache_registration` calls after warm-up materializes
    the backing memory. End-to-end mocking: stub `nixl_rbln`,
    `nixl_connector.TpKVTopology`, the metadata encoder, and the worker
    attributes the upstream `__init__` would normally populate.

    Verifies the post-call worker state captures what nixl-rbln returned
    (base addrs, block lens, region counts) and that `nixl-rbln` is
    invoked with the right memory segment for D2D (`mem='VRAM'`).
    """

    def _prep_worker(self, runtime_holder=("ctx_sentinel",)):
        """Build a D2D worker and back-fill every attribute that the real
        upstream `NixlConnectorWorker.__init__` would have set, so
        `_register_kv_caches_impl` can run without `AttributeError`."""
        worker = _build_d2d_worker()
        worker.tp_rank = 0
        worker._tp_size = {worker.engine_id: 1}
        worker.use_mla = False
        worker._has_mamba = False
        worker.attn_backends = []
        worker.backend_name = "rbln"
        worker.kv_cache_layout = "HND"
        worker._mamba_ssm_size = 0
        worker.model_config = MagicMock()
        worker.model_config.get_total_num_kv_heads.return_value = 8
        worker.host_xfer_buffers = {}
        worker.nixl_wrapper = MagicMock()
        worker.nixl_wrapper.get_agent_metadata.return_value = b"agent-meta"
        worker.kv_caches_base_addr = {worker.engine_id: {0: None}}
        worker._registered_descs = []
        worker.dst_num_blocks = {}
        worker.src_xfer_handles_by_block_size = {}

        if runtime_holder is not None:
            ctx = MagicMock()
            ctx.rbln_ctx_ptr = 0x1000
            holder_entry = MagicMock()
            holder_entry._runtime_handle.get_context.return_value = ctx
            worker._runtime_holder = [holder_entry]
        else:
            worker._runtime_holder = None
        return worker

    def _layer_spec(self, page_size_bytes=4096):
        """Stand-in for FullAttentionSpec — only attrs the impl reads."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = MagicMock(spec=FullAttentionSpec)
        spec.page_size_bytes = page_size_bytes
        return spec

    def _kv_caches(self, base_ptr=0x10000, num_blocks=128):
        """Two-layer kv_caches; each layer's tensor must support
        `.shape`, `.data_ptr()`, `.zero_()`, `.get_device()`. The impl
        also walks the values via `next(iter(...))` for tensor_shape /
        device_id."""
        kv = {}
        for i, name in enumerate(("l0", "l1")):
            t = MagicMock()
            # Full layer tensor: (K/V=2, num_blocks, heads, ..., head_dim).
            t.shape = (2, num_blocks, 8, 1, 64, 64)
            t.data_ptr.return_value = base_ptr + i * 0x10000
            t.get_device.return_value = 0
            t.zero_ = MagicMock()
            kv[name] = t
        return kv

    def _split_kv(self, num_blocks):
        """Mimic `kv_topo.get_transfer_cache_regions` returning K and V
        as separate region tensors, each with `shape[0] == num_blocks`
        (the per-K / per-V view the impl asserts against)."""

        def _split(cache, _spec):
            k = MagicMock()
            v = MagicMock()
            for region, off in ((k, 0), (v, 0x100)):
                region.shape = (num_blocks, 8, 1, 64, 64)
                region.data_ptr.return_value = cache.data_ptr() + off
            return [k, v]

        return _split

    def _patch_nixl_connector(self, topo):
        """Patch the upstream nixl_connector module used by the impl.

        `isinstance(layer_spec, nixl_connector.{UniformTypeKVCacheSpecs,
        MambaSpec})` requires real types (MagicMock breaks isinstance),
        so install dummy classes the spec MagicMocks won't match.
        """
        from unittest.mock import patch

        nc_patch = patch(
            "vllm_rbln.distributed.kv_transfer.kv_connector.v1."
            "rbln_nixl_connector.nixl_connector"
        )
        nc = nc_patch.start()
        nc.TpKVTopology.return_value = topo
        nc.compute_nixl_compatibility_hash.return_value = "hash"
        nc.MambaSpec = type("MambaSpec", (), {})
        nc.UniformTypeKVCacheSpecs = type("UniformTypeKVCacheSpecs", (), {})
        nc.NixlAgentMetadata = MagicMock()
        nc.NixlHandshakePayload = MagicMock()
        nc.msgspec.msgpack.Encoder.return_value.encode.return_value = b"meta"
        return nc, nc_patch

    def test_asserts_runtime_holder_present(self):
        """The runner sets `_runtime_holder` via
        `connector.set_runtime_holder` before warm-up. If
        `_register_kv_caches_impl` runs first, the impl must fail
        loudly instead of dereferencing `None`."""
        import pytest

        worker = self._prep_worker(runtime_holder=None)
        worker._layer_specs = {
            "l0": self._layer_spec(),
            "l1": self._layer_spec(),
        }
        kv_caches = self._kv_caches(num_blocks=worker.num_blocks)
        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            _cross_layers_blocks=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = self._split_kv(
            worker.num_blocks
        )
        _, nc_patch = self._patch_nixl_connector(topo)
        try:
            with pytest.raises(AssertionError, match="runtime_holder"):
                worker._register_kv_caches_impl(kv_caches)
        finally:
            nc_patch.stop()

    def test_registers_with_vram_segment_and_captures_xfer_tables(self):
        """Happy path: nixl-rbln is invoked with `mem='VRAM'` (D2D
        signature), and the returned `base_addrs` / `block_lens` /
        `reg_handle` are absorbed into the worker's transfer state."""
        from unittest.mock import patch
        import sys
        import types

        worker = self._prep_worker()
        spec = self._layer_spec(page_size_bytes=4096)
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = self._kv_caches(num_blocks=worker.num_blocks)

        # nixl-rbln returns base addrs / block lens / a reg handle.
        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000, 0x20100]
        xfer_result.block_lens = [256, 256]
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = 1

        fake_nixl_rbln = types.ModuleType("nixl_rbln")
        fake_nixl_rbln.register_kv_regions = MagicMock(return_value=xfer_result)
        fake_nixl_rbln.ensure_rbln_backend = MagicMock()

        # Stub upstream nixl_connector helpers used inside the impl.
        with patch.dict(sys.modules, {"nixl_rbln": fake_nixl_rbln}):
            with patch(
                "vllm_rbln.distributed.kv_transfer.kv_connector.v1."
                "rbln_nixl_connector.nixl_connector"
            ) as nc:
                topo = MagicMock(
                    is_kv_layout_blocks_first=False,
                    _cross_layers_blocks=False,
                    cross_layers_blocks=False,
                )
                topo.get_transfer_cache_regions.side_effect = self._split_kv(
                    worker.num_blocks
                )
                nc.TpKVTopology.return_value = topo
                nc.compute_nixl_compatibility_hash.return_value = "hash"
                # Don't import MambaSpec / UniformTypeKVCacheSpecs paths.
                nc.MambaSpec = type("MambaSpec", (), {})
                nc.UniformTypeKVCacheSpecs = type(
                    "UniformTypeKVCacheSpecs", (), {}
                )
                # Encode handshake payload without touching real msgspec.
                nc.NixlAgentMetadata = MagicMock()
                nc.NixlHandshakePayload = MagicMock()
                nc.msgspec.msgpack.Encoder.return_value.encode.return_value = (
                    b"meta"
                )
                # Short-circuit register_local_xfer_handler so the test
                # focuses on _register_kv_caches_impl's own state writes.
                with patch.object(
                    worker,
                    "register_local_xfer_handler",
                    return_value=("local-handle", [(0x0, 0, 0)]),
                ):
                    worker._register_kv_caches_impl(kv_caches)

        # nixl-rbln was called once, with VRAM segment.
        fake_nixl_rbln.register_kv_regions.assert_called_once()
        called_kwargs = fake_nixl_rbln.register_kv_regions.call_args.kwargs
        assert called_kwargs["mem"] == "VRAM"
        assert called_kwargs["rbln_ctx_ptr"] == 0x1000

        # Returned tables threaded into the worker state.
        assert worker.device_id == 0
        assert worker.block_len_per_layer == [256, 256]
        assert (
            worker.kv_caches_base_addr[worker.engine_id][0]
            == [0x20000, 0x20100]
        )
        assert worker._registered_descs == ["reg-handle"]

        # Region counts: 2 regions, layout-blocks-first=False so no x2.
        assert worker.num_regions == 2
        assert worker.num_descs == 2 * worker.num_blocks

        # Final hand-offs.
        assert worker.device_kv_caches is kv_caches
        assert worker.dst_num_blocks[worker.engine_id] == worker.num_blocks
        assert (
            worker.src_xfer_handles_by_block_size[worker.block_size]
            == "local-handle"
        )
