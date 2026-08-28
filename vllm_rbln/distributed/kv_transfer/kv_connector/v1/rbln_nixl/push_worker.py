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

import threading
import time
from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
    connector_option,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        ReqMeta,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# How long a flush waits for an early write to leave the NIC before giving up
# on it. Bounded because it runs on the engine main thread: a wedged transfer
# must not take the engine with it.
_EARLY_FLUSH_DRAIN_TIMEOUT_S = 1.0
_EARLY_FLUSH_POLL_INTERVAL_S = 0.001


class RblnNixlPushConnectorWorker(RblnNixlWorkerBase, NixlPushConnectorWorker):
    """Writes a request's KV to the consumer that registered for it.

    The producer drives the transfer here, so the peer this rank hands its
    metadata to is the consumer rather than the other way round. Pairing does
    not care -- it describes what two peers hold -- so what belongs here is the
    write: which peers to issue it against and the writer thread that issues it.
    """

    _writes_into_peer = True

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # Tokens a handed-over request holds, for `_tail_chunks`.
        self._valid_tokens: dict[str, int] = {}

        # A stage's write overlaps the stages that come after it, so there is
        # nothing to overlap with when this producer is one stage.
        self._early_push_enabled = (
            connector_option(vllm_config, "push_stream", False)
            and vllm_config.parallel_config.pipeline_parallel_size > 1
        )
        # Requests written from their closing prefill chunk, before the engine
        # handed their blocks over. Membership survives several writes.
        self._early_sends: set[ReqId] = set()
        # Their handles, kept out of `_sending_transfers` so the base cannot
        # report the request as finished_sending: the scheduler frees a
        # request's blocks on that report unconditionally, and this one is
        # still prefilling.
        self._early_transfers: defaultdict[ReqId, list[int]] = defaultdict(list)
        # Offers waiting for this rank's next step to order them (see
        # `start_early_push`).
        self._pending_early_offers: dict[ReqId, BlockIds] = {}

    def start_load_kv(self, metadata: "NixlConnectorMetadata") -> None:
        """Hand this step's work to the writer, once the KV it names is settled.

        NOTE(RBLN): the writer is woken here, at the START of a step, while the
        host copy for the same step's saves runs at its END (`wait_for_save`).
        Today no request is in both: a producer hands its blocks over in
        `request_finished`, which runs after this step's metadata was already
        built, so the blocking copy always lands a step earlier. Nothing states
        that, and if it ever stopped holding, the writer would ship a staging
        buffer still being filled -- silently, and only under host staging.
        """
        assert isinstance(metadata, RblnNixlConnectorMetadata)
        self._valid_tokens.update(metadata.valid_tokens)
        if self.use_host_buffer and metadata.push_finished_blocks:
            both = metadata.push_finished_blocks.keys() & metadata.reqs_to_save.keys()
            assert not both, (
                "RBLN NIXL push: request(s) staged for the host copy and handed "
                f"to the writer in one step: {sorted(both)}. The copy runs after "
                "this call, so the write would read an unfilled buffer."
            )
        self._adopt_early_sends(metadata)
        super().start_load_kv(metadata)

    def _adopt_early_sends(self, metadata: "NixlConnectorMetadata") -> None:
        """Take a request the engine has now finished out of the early hold.

        Its arrival in `push_finished_blocks` IS the engine saying the request
        finished, which is what the base's completion report is allowed to
        follow. So publish the handles the early write left parked, and drop
        the handover itself -- the blocks it carries are the ones already
        written, or the ones the writer still holds unmatched.
        """
        with self._sending_transfers_lock:
            for req_id in list(metadata.push_finished_blocks):
                if req_id not in self._early_sends:
                    continue
                self._early_sends.discard(req_id)
                handles = self._early_transfers.pop(req_id, [])
                if handles:
                    self._sending_transfers[req_id].extend(handles)
                del metadata.push_finished_blocks[req_id]

    def start_early_push(self, metadata: "NixlConnectorMetadata") -> None:
        """Hold the prefill this stage has just closed, for the writer.

        NOTE(RBLN): `reqs_to_save` says a request's KV for this rank's layers
        is complete. Host staging reads that to fill its buffer, and this
        reads it to write straight out of device memory. The two never run on
        the same request because `push_stream_enabled` refuses host staging.

        Held rather than handed over, because the forward that produced the KV
        completes asynchronously and the runtime's wait covers transfers rather
        than compute: a write issued from here reads KV still being written and
        reports no error. `release_early_offers` holds it a step.

        Called from `wait_for_save` rather than `get_finished` so the host copy
        for the step is already done: speculative decoding on the last stage
        defers `wait_for_save` past `get_finished`, which would reverse them.
        """
        if not self._early_push_enabled or self.use_host_buffer:
            return
        for req_id, meta in metadata.reqs_to_save.items():
            self._pending_early_offers[req_id] = meta.local_block_ids

    def release_early_offers(self) -> None:
        """Hand the previous step's held offers to the writer.

        Called at the start of a step, before the handover is adopted, so a
        request whose handover lands on this same step is written by the offer
        rather than dropped with it. Runs on every step -- one that closes no
        chunk and one with no forward included -- so nothing is left held.

        What guarantees a next step at all: while the request runs, it is
        unfinished; once it ends, the scheduler keeps stepping on the
        connector's pending-push-work hook until the send is reported. The
        second half rests on a hook upstream documents as a placeholder, so a
        held offer outliving the engine is what to suspect if a request ever
        stalls with its KV never arriving.
        """
        if not self._pending_early_offers:
            return
        offers = self._pending_early_offers
        self._pending_early_offers = {}
        with self._sending_transfers_lock:
            self._early_sends.update(offers)
        for req_id, block_ids in offers.items():
            self._finished_blocks_inbox.put((req_id, block_ids))
        self._push_writer_wake.set()

    def flush_early_sends(self, req_ids: set[ReqId]) -> None:
        """Let an early write finish before its source blocks are reused.

        The prefill did finish, so the bytes already on their way are correct
        and complete; cancelling would leave the consumer a torn block. Wait
        for them instead -- bounded, because this runs on the engine main
        thread. Nothing here is reported as finished_sending: a preempted
        request re-prefills into these blocks, and an aborted one is gone from
        the scheduler, which asserts on a report for a request it does not
        hold.
        """
        drained = False
        for req_id in req_ids:
            self._pending_early_offers.pop(req_id, None)
            with self._sending_transfers_lock:
                self._early_sends.discard(req_id)
                handles = self._early_transfers.pop(req_id, [])
            for handle in handles:
                self._drain_early_handle(req_id, handle)
            self._evict_finished_inbox.put(req_id)
            drained = True
        if drained:
            self._push_writer_wake.set()

    def _drain_early_handle(self, req_id: ReqId, handle: int) -> None:
        deadline = time.perf_counter() + _EARLY_FLUSH_DRAIN_TIMEOUT_S
        while self.nixl_wrapper.check_xfer_state(handle) == "PROC":
            if time.perf_counter() >= deadline:
                logger.warning(
                    "RBLN NIXL push: early write for request %s still in "
                    "flight after %.1fs; releasing it and letting the step "
                    "go on. The blocks it reads are about to be reused.",
                    req_id,
                    _EARLY_FLUSH_DRAIN_TIMEOUT_S,
                )
                break
            time.sleep(_EARLY_FLUSH_POLL_INTERVAL_S)
        self.nixl_wrapper.release_xfer_handle(handle)

    def shutdown(self) -> None:
        with self._sending_transfers_lock:
            for handles in self._early_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._early_transfers.clear()
            self._early_sends.clear()
        self._pending_early_offers = {}
        super().shutdown()

    def finalize_kv_cache_registration(self) -> None:
        """Register the deferred D2D memory, then make sure the writer runs.

        NOTE(RBLN): D2D defers registration past `register_kv_caches`, and that
        early return skips the writer-thread start hung off it upstream -- the
        pushes would then queue with nothing draining them. Start it here; the
        start is guarded on the thread being unset, so host staging, which does
        reach the upstream method, is unaffected.
        """
        super().finalize_kv_cache_registration()
        self._ensure_push_writer()

    def _ensure_push_writer(self) -> None:
        # NOTE(RBLN): the writer start is inlined in the upstream
        # `register_kv_caches`, which the D2D deferral never reaches, so it is
        # mirrored here. Both are guarded on the thread being unset, so exactly
        # one of them starts it whichever path ran.
        if self._push_writer_thread is not None:
            return
        self._push_writer_thread: threading.Thread | None = threading.Thread(
            target=self._push_writer_loop,
            daemon=True,
            name="nixl-push-writer",
        )
        self._push_writer_thread.start()
        logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending, done_recving = super().get_finished()
        for req_id in done_sending:
            self._valid_tokens.pop(req_id, None)
        return done_sending, done_recving

    def _xfer_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        """Write this request's blocks, one transfer per paired peer rank.

        Runs on the writer thread, which is also where upstream writes
        `_engine_last_active` and runs the eviction sweep on this path, so the
        touch below needs no lock. Handles go out under the sending lock.
        """
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Keep the engine off the staleness sweep: a swept peer loses the state
        # this path reads, and upstream refreshes it on the route it replaces.
        self._engine_last_active[engine_id] = time.perf_counter()
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        # Per-shard lists exist exactly for peers serving part of what a
        # whole-engine handle covers; without any, upstream describes them all.
        # Read once: a handshake replacing the entry between uses would leave the
        # count and the loop below describing different peers.
        peer_ranks = self._overlapping_ranks.get(engine_id)
        if not peer_ranks:
            # An early send exists only at pipeline_parallel_size > 1, and a
            # consumer holds every layer, so this producer's band is always a
            # part of what the peer covers -- which is what puts it on the
            # per-shard route. Reaching upstream's route with one means the
            # pipeline gate leaked.
            assert req_id not in self._early_sends
            # Chunk mode asks for per-shard state, so a request written in
            # pieces cannot arrive on this route -- unless a sliding window
            # kept it here.
            assert not self._chunk_mode or self._own_engine_layout
            tail: AbstractContextManager = (
                self._tail_viewed_as(
                    self._valid_tokens.get(req_id),
                    # Counted before the trim below, which cuts the head off
                    # the local list: the token count describes the request's
                    # own blocks.
                    self._prompt_blocks(meta.local_physical_block_ids),
                )
                if self._chunk_mode or self._own_engine_layout
                else nullcontext()
            )
            # NOTE(RBLN): upstream aligns by truncating the longer list and
            # keeping its HEAD -- the wrong end (see _trim_to_consumer_blocks)
            # -- and the lengths match either way so nothing catches it. Trim
            # first, only where upstream's expansion of the remote list is the
            # identity: past that the two lengths are not the same unit.
            if remote_info.remote_physical_blocks_per_logical == 1:
                meta.local_physical_block_ids = self._trim_to_consumer_blocks(
                    meta.local_physical_block_ids,
                    meta.remote.block_ids,
                    engine_id,
                    meta.remote.request_id,
                )
            with tail:
                return super()._xfer_blocks_for_req(req_id, meta)

        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        assert block_size_ratio == 1, (
            "RBLN NIXL per-shard write path requires equal P/D block sizes "
            f"(got block_size_ratio={block_size_ratio})"
        )
        remote_block_size = remote_info.remote_block_size

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids, remote_info.remote_physical_blocks_per_logical
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        notif_id = self._xfer_notif_id(
            engine_id,
            meta.remote.request_id,
            remote_info.remote_tp_size,
            count_stages=False,
        )

        # Counted before the consumer trim below, which cuts the front: the
        # token count describes this producer's whole list, and both lists keep
        # their tail, so the last element is still the request's last block.
        n_prompt_blocks = sum(len(g) for g in local_block_ids)
        tail_tokens = self._valid_tokens.get(req_id)
        local_block_ids = self._trim_to_consumer_blocks(
            local_block_ids, remote_block_ids, engine_id, meta.remote.request_id
        )
        n_write_blocks = sum(len(g) for g in local_block_ids)
        if not n_write_blocks:
            logger.warning("per-shard write req %s: no blocks to push", req_id)
            return

        logger.debug(
            "per-shard write req %s: ranks=%d write_blocks=%d",
            req_id,
            len(peer_ranks),
            n_write_blocks,
        )

        # Publish once, for the reason the read path states (see
        # `_read_blocks_for_req`); failure is per peer here, not per request.
        handles: list[int] = []
        for global_rank in peer_ranks:
            remote_descs = self._shard_descs_for_tokens(
                engine_id,
                global_rank,
                self.dst_num_blocks[engine_id],
                remote_block_ids,
                num_valid_tokens=tail_tokens,
                num_prompt_blocks=n_prompt_blocks,
            )
            local_descs = self._shard_descs_for_tokens(
                engine_id,
                global_rank,
                self.num_blocks,
                local_block_ids,
                num_valid_tokens=tail_tokens,
                num_prompt_blocks=n_prompt_blocks,
            )
            assert len(local_descs) == len(remote_descs)
            local_handle = self.src_xfer_handles_by_remote[
                (engine_id, global_rank, remote_block_size)
            ]
            remote_handle = self.dst_xfer_side_handles[engine_id][global_rank]

            handle = None
            try:
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    local_handle,
                    local_descs,
                    remote_handle,
                    remote_descs,
                    notif_msg=notif_id,
                )
                self.nixl_wrapper.transfer(handle)
                handles.append(handle)
            except Exception as e:
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=req_id,
                    msg="Push WRITE submission failed; releasing handle",
                    error=e,
                    dst_engine_id=engine_id,
                    remote_pp_rank=global_rank,
                )
                # Outbound only: there is no local metadata to invalidate, so
                # release this peer's handle and let the remaining peers go.
                if handle is not None:
                    self.nixl_wrapper.release_xfer_handle(handle)
                self.xfer_stats.record_failed_transfer()

        if handles:
            with self._sending_transfers_lock:
                target = (
                    self._early_transfers
                    if req_id in self._early_sends
                    else self._sending_transfers
                )
                target[req_id].extend(handles)

    @staticmethod
    def _trim_to_consumer_blocks(
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        engine_id: str,
        req_id: str,
    ) -> BlockIds:
        """Drop from our side the blocks the consumer already had.

        NOTE(RBLN): the same trim the read path gets from upstream's
        `_apply_prefix_caching`, with the roles swapped -- there the consumer is
        local and the producer's longer list is trimmed to it, here the consumer
        is the peer and ours is the longer one. It has to come off the END: the
        consumer registers the uncached SUFFIX of a prompt, so dropping our tail
        would hand it the wrong blocks under a partial prefix hit.

        TODO(vllm>=0.27.2): delete -- upstream trims the write path itself there.
        """
        local = list(local_block_ids)
        for i, remote_group in enumerate(remote_block_ids):
            num_remote = len(remote_group)
            if num_remote > len(local[i]):
                raise RuntimeError(
                    f"RBLN NIXL: consumer {engine_id} registered {num_remote} "
                    f"block(s) for request {req_id} in group {i}, more than the "
                    f"{len(local[i])} this producer holds; the peer's advertised "
                    "length crosses the handshake, so the pair is refused rather "
                    "than trimmed to it."
                )
            if num_remote < len(local[i]):
                local[i] = local[i][-num_remote:]
        return tuple(local)
