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

import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
)
from vllm.distributed.parallel_state import get_pp_group

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RBLN_COVERAGE_NOTIF_PREFIX,
    RblnNixlConnectorMetadata,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        ReqMeta,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

#: Half-open range of a request's blocks, as positions in the list this rank
#: registered.
Span = tuple[int, int]

# How long a flush waits for an early write to leave the NIC before giving up
# on it. Bounded because it runs on the engine main thread: a wedged transfer
# must not take the engine with it.
#
# The bound is a fault detector, not a deadline -- nothing here knows what a
# healthy write of this size costs, so it is set far above any of them. Giving
# it a real deadline needs that cost, and until then a smaller value would
# start abandoning writes that were about to land.
_EARLY_FLUSH_DRAIN_TIMEOUT_S = 1.0
_EARLY_FLUSH_POLL_INTERVAL_S = 0.001


class RblnNixlPushConnectorWorker(RblnNixlWorkerBase, NixlPushConnectorWorker):
    """Writes a request's KV to the consumer that registered for it.

    The producer drives the transfer here, so the peer this rank hands its
    metadata to is the consumer rather than the other way round. Pairing does
    not care -- it describes what two peers hold -- so what belongs here is the
    write: which peers to issue it against, the writer thread that issues it,
    and the count a consumer settles a request by.
    """

    _writes_into_peer = True

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Completion notifications seen per request being received, counted
        # against the number of writers the peer put in them. Used for a peer
        # that names no range (see `_with_coverage`).
        self._writer_counts_by_req: defaultdict[str, int] = defaultdict(int)

        # Ranges of this request's blocks each writer has reported filling,
        # for a peer that does name them. Kept per writer because a request
        # settles only once every one of them has covered the whole list.
        self._coverage_by_req: defaultdict[str, defaultdict[int, list[Span]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        self._early_push_enabled = envs.VLLM_RBLN_NIXL_PUSH_STREAM
        # Requests written from their closing prefill chunk, before the engine
        # handed their blocks over. Membership survives several writes.
        self._early_sends: set[ReqId] = set()
        self._empty_receives: set[ReqId] = set()
        # Their handles, kept out of `_sending_transfers` so upstream cannot
        # report the request as finished_sending: the scheduler frees a
        # request's blocks on that report unconditionally, and this one is
        # still prefilling. Grouped by the batch that issued them, because a
        # request is finished when every batch is, not when any list empties.
        self._early_transfers: defaultdict[ReqId, list[list[int]]] = defaultdict(list)
        # Batches handed to the writer, and batches whose writes have landed.
        # Counted where they are handed over rather than where they are
        # issued: both happen on the engine thread, so the seal below sees the
        # final number without having to ask the writer what it has reached.
        self._batches_queued: defaultdict[ReqId, int] = defaultdict(int)
        self._batches_done: defaultdict[ReqId, int] = defaultdict(int)
        # How many batches a request will have, once the engine says it is
        # over. Absent until then: an unsealed request is never finished, however
        # many of its batches have landed.
        self._batches_expected: dict[ReqId, int] = {}
        # Blocks a streamed request will hold once its whole prompt is
        # computed, and how many of the consumer's the writer has already
        # filled. The first places the consumer's window inside our list; the
        # second says where in that window this batch starts.
        self._stream_total: dict[ReqId, int] = {}
        self._issued_hwm: defaultdict[ReqId, int] = defaultdict(int)
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
        if self.use_host_buffer and metadata.push_finished_blocks:
            both = metadata.push_finished_blocks.keys() & metadata.reqs_to_save.keys()
            assert not both, (
                "RBLN NIXL push: request(s) staged for the host copy and handed "
                f"to the writer in one step: {sorted(both)}. The copy runs after "
                "this call, so the write would read an unfilled buffer."
            )
        self._seal_at_handover(metadata)
        super().start_load_kv(metadata)
        self._settle_empty_receives(metadata)

    def _settle_empty_receives(self, metadata: "NixlConnectorMetadata") -> None:
        """Finish a receive that has nothing to receive, on the step it arrives.

        NOTE(RBLN): the serving layer can turn a request away before it was
        ever scheduled, and upstream registers a receive of no blocks for it so
        the producer stops holding what it pinned. Nothing is ever written into
        no blocks, so the completion notification that would settle the request
        never comes: it sits in the receive metadata for the life of the
        engine, and the one place that drops an entry is the report this
        request never reaches.

        Settled here rather than through upstream's transfer table, even
        though an empty entry there would pop as done: everything upstream
        does with a completed receive reads the blocks it landed in, starting
        with the engine those blocks came from. A request turned away has
        neither -- it never gave this rank a reason to handshake with the
        producer, so that lookup finds nothing and takes the engine down.
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            if not sum(len(group) for group in meta.local_block_ids):
                self._empty_receives.add(req_id)

    def _seal_at_handover(self, metadata: "NixlConnectorMetadata") -> None:
        """Fix how many batches a request written early will have.

        Its arrival in `push_finished_blocks` IS the engine saying the request
        is over, so nothing further will be handed to the writer for it and
        what has been handed over is all there will be.

        The handover is one of those batches, not a duplicate of them. What was
        streamed is the prefix of blocks a prefill CLOSED, and the last block of
        a prompt is closed by nothing -- its tokens end mid-block. The handover
        carries the whole list, so it is what finally covers that tail, and the
        writer sends only the part past what it has already written.

        Runs before the call that hands the same metadata to the writer:
        that call wakes it, and a batch it finishes raises the landed count.
        A count that moves before the seal is set leaves the request either
        unreportable or short of a total that already passed it.

        Sealing rather than publishing the parked handles: upstream reports a
        request as finished the moment the handles it can see have landed, and
        a batch handed over on this same step has not been issued yet. What the
        base cannot see cannot be reported early.
        """
        with self._sending_transfers_lock:
            for req_id in metadata.push_finished_blocks:
                if req_id not in self._early_sends:
                    continue
                self._batches_queued[req_id] += 1
                self._batches_expected[req_id] = self._batches_queued[req_id]

    def _writes_less_than_a_request(self) -> bool:
        return self._early_push_enabled and not self.use_host_buffer

    def start_early_push(self, metadata: "RblnNixlConnectorMetadata") -> None:
        """Hold the prefill this stage has just closed, for the writer.

        NOTE(RBLN): `reqs_to_save` says a request's KV for this rank's layers
        is complete. Host staging reads that to fill its buffer, and this
        reads it to write straight out of device memory, so the two never run
        on the same request -- see the assertion in `start_load_kv`.

        Held rather than handed over, because the forward that produced the KV
        completes asynchronously: this rank's Python returning does not mean
        its writes are visible to the NIC, and the wait the runtime exposes
        covers pending transfers rather than compute. Writing from here reads
        KV that is still being written, and does so silently -- the transfer
        reports no error. `release_early_offers` lets the offer go one step of
        this rank later, which is late enough; what makes it late enough is not
        something this side can name, so treat the delay as load-bearing.

        Called from `wait_for_save` rather than `get_finished` so the host copy
        for the step is already done: speculative decoding on the last stage
        defers `wait_for_save` past `get_finished`, which would reverse them.
        """
        if not self._early_push_enabled or self.use_host_buffer:
            return
        for req_id, meta in metadata.reqs_to_save.items():
            self._pending_early_offers[req_id] = meta.local_block_ids
            total = metadata.push_stream_total.get(req_id)
            if total is not None:
                self._stream_total[req_id] = total

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
            self._batches_queued[req_id] += 1
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
                batches = self._early_transfers.get(req_id, [])
                handles = [h for batch in batches for h in batch]
                self._forget_send(req_id)
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
            for batches in self._early_transfers.values():
                for handles in batches:
                    for handle in handles:
                        self.nixl_wrapper.release_xfer_handle(handle)
            self._early_transfers.clear()
            self._early_sends.clear()
            self._batches_queued.clear()
            self._batches_done.clear()
            self._batches_expected.clear()
            self._stream_total.clear()
            self._issued_hwm.clear()
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

    def _get_new_notifs(self) -> set[str]:
        """Hold a request back until every writer of it has reported.

        NOTE(RBLN): upstream settles a pushed request on the FIRST completion
        notification, which is right only while one peer rank writes the whole
        thing. Several do as soon as either axis is cut finer on their side, and
        the rest of them are still writing -- host staging would copy a
        half-filled buffer to the device. Each writer's notification carries the
        count (see `_xfer_notif_id`), so drop all but the last one and let
        upstream settle the request on that.
        """
        for notif in self._drain_completion_notifs():
            writer, span, notif = self._split_coverage(notif)
            if self._writer_still_pending(notif, writer, span):
                continue
            self._pending_completion_notifs.put(notif)
        return super()._get_new_notifs()

    def _drain_completion_notifs(self) -> list[bytes]:
        notifs = []
        while True:
            try:
                notifs.append(self._pending_completion_notifs.get_nowait())
            except queue.Empty:
                return notifs

    def _writer_still_pending(
        self, notif: bytes, writer: int | None, span: "Span | None"
    ) -> bool:
        """Whether this notification leaves a request short of its writers.

        False for anything upstream has to see itself: heartbeats, our own
        outbound accounting, and a request we are not receiving.

        A writer that names the range it filled is settled on the ranges: it is
        done when what it has reported spans every block this rank registered,
        and the request is done when every writer is. A writer that names none
        is settled on how many notifications it sent, which is the same thing
        while one write carries a whole request.

        The block count comes from what this rank registered, not from anything
        the peer said -- a peer that sends the wrong ranges must stall the
        request, not settle it early.
        """
        msg = notif.decode("utf-8")
        if msg.startswith("HB:"):
            return False
        req_id, count = msg.rsplit(":", 1)
        if req_id in self._reqs_to_send or req_id in self._reqs_to_process:
            return False
        meta = self._recving_metadata.get(req_id)
        if meta is None:
            return False
        # The peer scales the count by our tensor-parallel size, the unit
        # upstream divides by on the other direction (see `_xfer_notif_id`).
        writers = max(1, int(count) // self.world_size)
        if span is None or writer is None:
            self._writer_counts_by_req[req_id] += 1
            return self._writer_counts_by_req[req_id] < writers

        spans_by_writer = self._coverage_by_req[req_id]
        spans_by_writer[writer].append(span)
        num_blocks = len(meta.local_physical_block_ids[0])
        covered = sum(
            self._covers(spans, num_blocks) for spans in spans_by_writer.values()
        )
        return covered < writers

    @staticmethod
    def _covers(spans: list["Span"], num_blocks: int) -> bool:
        """Whether the half-open ranges together leave no gap below num_blocks.

        Ranges rather than a running total because a preempted request is
        rescheduled from the start of its block list, so a writer re-sends what
        it already sent. Adding those up reaches the count with a hole still in
        the middle and settles a request whose KV is incomplete -- silently.
        """
        reach = 0
        for lo, hi in sorted(spans):
            if lo > reach:
                return False
            reach = max(reach, hi)
            if reach >= num_blocks:
                return True
        return num_blocks == 0

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Keep the registration this write matched, however it was matched.

        NOTE(RBLN): a registration reaches the writer two ways -- it arrives
        and finds the blocks already parked, or it is already held when the
        blocks arrive. Upstream stores it only on the second, because on the
        first it has just been used and, for a request written once, will not
        be wanted again. A request written in batches wants it for every one
        of them, and the batches that follow a registration which arrived late
        find nothing to match: they park, and park forever, because the
        registration that would release them came and went.

        Stored here rather than where it arrives because both ways run through
        this call, and because reading it back out of the notification would
        mean repeating the decode and the validation upstream has already done.
        """
        self._pending_d_registrations.setdefault(
            registration_data["request_id"], registration_data
        )
        return super()._do_start_push_kv(request_id, local_block_ids, registration_data)

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Find the consumer's registration without consuming it.

        NOTE(RBLN): upstream takes the registration out on the first batch it
        matches, which is right while a request is written once. A request
        written in several batches needs it for every one of them: the second
        would find nothing, park, and wait for a registration that already
        arrived and will not arrive again -- its blocks reaching the consumer
        only when the lease gives up on them.

        Kept until the request is done being written, which the eviction the
        writer already drains does: it drops the registration for the same
        request whose completion it drops the parked blocks for.
        """
        data = self._pending_d_registrations.get(request_id)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id, reg_data in self._pending_d_registrations.items():
            if get_base_request_id(reg_id) == base_id:
                return reg_data
        return None

    def _handle_failed_transfer(self, req_id: str, handle: int | None) -> None:
        """Record a failed WRITE as a failed send, not as a failed receive.

        NOTE(RBLN): upstream's handler is written for the read direction --
        it invalidates the blocks the transfer was filling and queues the
        request as a failed receive. The write path runs the same completion
        check over its outbound handles, where neither holds: the blocks are
        this producer's own, and upstream's `get_finished` asserts that every
        request it reports as received carries receive metadata, which one we
        were sending never does. So the queued failure kills the engine a step
        later, on the assertion rather than on the failure.

        The blocks stay held until the lease expires, which is already how a
        push that never completes is unwound.
        """
        if req_id not in self._recving_metadata:
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            self.xfer_stats.record_failed_transfer()
            return
        super()._handle_failed_transfer(req_id, handle)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending, done_recving = super().get_finished()
        while self._empty_receives:
            req_id = self._empty_receives.pop()
            self._recving_metadata.pop(req_id, None)
            done_recving.add(req_id)
        # Both completion and failure land here, and a retried request must not
        # inherit a partial count.
        for req_id in done_recving:
            self._writer_counts_by_req.pop(req_id, None)
            self._coverage_by_req.pop(req_id, None)
        sealed_done = self._finish_sealed_requests()
        if sealed_done:
            # Upstream drops the writer's state for what it reports itself,
            # and it has already been past that for this step.
            for req_id in sealed_done:
                self._evict_finished_inbox.put(req_id)
            self._push_writer_wake.set()
            done_sending |= sealed_done
        return done_sending, done_recving

    def _finish_sealed_requests(self) -> set[ReqId]:
        """Report a request written early once every batch of it has landed.

        Upstream reports what it can see, and it cannot see a batch parked
        here, so this side owns the report for these requests -- including the
        state upstream drops on its own reports, which the writer and the
        lease both read.

        Checked every step rather than only when a batch lands: a request
        whose batches all landed before the engine finished it is completed by
        the seal, not by a completion.
        """
        finished: set[ReqId] = set()
        with self._sending_transfers_lock:
            for req_id, batches in list(self._early_transfers.items()):
                still_going = []
                for handles in batches:
                    probe = {req_id: handles}
                    if self._pop_done_transfers(probe):
                        self._batches_done[req_id] += 1
                    else:
                        still_going.append(probe[req_id])
                if still_going:
                    self._early_transfers[req_id] = still_going
                else:
                    del self._early_transfers[req_id]

            for req_id, expected in list(self._batches_expected.items()):
                if self._batches_done[req_id] < expected:
                    continue
                finished.add(req_id)
                self._forget_send(req_id)

        for req_id in finished:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
        return finished

    def _forget_send(self, req_id: ReqId) -> None:
        """Drop what this side tracked for a request it is done pushing."""
        self._early_sends.discard(req_id)
        self._early_transfers.pop(req_id, None)
        self._batches_queued.pop(req_id, None)
        self._batches_done.pop(req_id, None)
        self._batches_expected.pop(req_id, None)
        self._stream_total.pop(req_id, None)
        self._issued_hwm.pop(req_id, None)

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
            # Streaming asks for per-shard state from every peer it writes to,
            # so a request written in pieces cannot arrive on this route.
            assert req_id not in self._early_sends, (
                f"RBLN NIXL push: request {req_id} was written early but is "
                f"served by a whole-engine handle (peer {engine_id}), which "
                "the streaming handshake asks not to be given."
            )
            # NOTE(RBLN): upstream aligns by truncating the longer list and
            # keeping its HEAD -- the wrong end (see _trim_to_consumer_blocks)
            # -- and the lengths match either way so nothing catches it. Trim
            # first, only where upstream's expansion of the remote list is the
            # identity: past that the two lengths are not the same unit.
            if remote_info.remote_physical_blocks_per_logical == 1:
                meta.local_physical_block_ids = self._trim_to_consumer_blocks(
                    meta.local_physical_block_ids, meta.remote.block_ids
                )
            return super()._xfer_blocks_for_req(req_id, meta)

        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        assert block_size_ratio == 1, (
            "RBLN NIXL per-shard write path requires equal P/D block sizes "
            f"(got block_size_ratio={block_size_ratio})"
        )
        remote_block_size = remote_info.remote_block_size

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids, remote_info.remote_physical_blocks_per_logical
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        notif_id = self._xfer_notif_id(
            engine_id, meta.remote.request_id, remote_info.remote_tp_size
        )

        window = self._stream_window(req_id, local_block_ids, remote_block_ids)
        if window is None:
            local_block_ids = self._trim_to_consumer_blocks(
                local_block_ids, remote_block_ids
            )
            span = (0, len(remote_block_ids[0])) if len(remote_block_ids) == 1 else None
        else:
            local_block_ids, remote_block_ids, span = window
        notif_id = self._with_coverage(notif_id, span)
        n_write_blocks = sum(len(g) for g in local_block_ids)
        if not n_write_blocks:
            logger.warning("per-shard write req %s: no blocks to push", req_id)
            with self._sending_transfers_lock:
                if req_id in self._early_sends:
                    self._batches_done[req_id] += 1
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
            remote_descs = self._get_block_descs_ids_for_shard(
                engine_id,
                global_rank,
                self.dst_num_blocks[engine_id],
                remote_block_ids,
            )
            local_descs = self._get_block_descs_ids_for_shard(
                engine_id, global_rank, self.num_blocks, local_block_ids
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

        with self._sending_transfers_lock:
            if req_id not in self._early_sends:
                self._sending_transfers[req_id].extend(handles)
            elif handles:
                self._early_transfers[req_id].append(handles)
            else:
                # Every peer's submission failed. The batch is over either
                # way, and a request whose count never reaches its seal is a
                # request that never finishes.
                self._batches_done[req_id] += 1

    def _stream_window(
        self, req_id: str, local_block_ids: BlockIds, remote_block_ids: BlockIds
    ) -> "tuple[BlockIds, BlockIds, Span] | None":
        """The part of the consumer's list this batch is the first to fill.

        None for a request that is not streamed, which is every request while
        the flag is off and every one whose whole prompt is offered at once.
        Those take the trim, and this leaves them exactly as they were.

        A streamed offer is a growing prefix of the producer's blocks, and the
        consumer registered the TAIL of the prompt -- what its own cache did
        not cover. So the window it wants starts at `total - registered` of
        ours, a place a prefix shorter than the whole prompt cannot be asked
        for. The total comes over with the offer for that reason.

        Only where both sides expand a logical block by the same factor: past
        that, the two lengths this arithmetic subtracts are counted in
        different units.
        """
        total = self._stream_total.get(req_id)
        if total is None or len(remote_block_ids) != 1:
            return None
        expand = self._physical_blocks_per_logical_kv_block
        if expand != self._remote_expand_for(req_id):
            return None

        registered = len(remote_block_ids[0])
        offset = total * expand - registered
        have = len(local_block_ids[0])
        lo = self._issued_hwm[req_id]
        hi = max(0, min(registered, have - offset))
        if hi <= lo:
            return (), (), (lo, lo)
        self._issued_hwm[req_id] = hi
        return (
            (local_block_ids[0][offset + lo : offset + hi],),
            (remote_block_ids[0][lo:hi],),
            (lo, hi),
        )

    def _with_coverage(self, notif_id: bytes, span: "Span | None") -> bytes:
        """Name the half-open range of consumer blocks this write filled.

        The range is positions into the list the consumer registered. A request
        written once covers all of it; one written in batches covers the part
        this batch is the first to reach, which is what lets the consumer tell
        a complete request from a partly written one.

        Left off when no single range describes the write -- a request with
        more than one KV cache group, whose groups go out together but carry
        their own lengths. A consumer has to accept a message without the
        prefix for that reason alone, which is also what lets the per-shard
        route carry it while the route upstream drives does not.
        """
        if span is None:
            return notif_id
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        pp_rank = get_pp_group().rank_in_group if pp_size > 1 else 0
        writer = pp_rank * self.world_size + self.tp_rank
        head = f"{writer}:{span[0]}:{span[1]}:".encode()
        return RBLN_COVERAGE_NOTIF_PREFIX + head + notif_id

    def _remote_expand_for(self, req_id: str) -> int:
        """The factor the peer expands a logical block by, for this request."""
        assert self.transfer_topo is not None
        meta = self._recving_metadata.get(req_id)
        engine_id = meta.remote.engine_id if meta and meta.remote else None
        if engine_id is None:
            return self._physical_blocks_per_logical_kv_block
        return self.transfer_topo.get_engine_info(
            engine_id
        ).remote_physical_blocks_per_logical

    @staticmethod
    def _split_coverage(notif: bytes) -> "tuple[int | None, Span | None, bytes]":
        """Take the coverage prefix off, returning what it said and what is left.

        Upstream reads a completion notification as `req_id:count` with
        `rsplit`, so it would take the whole prefixed string as the request id
        and find no such request. What upstream is handed here is what it was
        handed before this rank started naming ranges.
        """
        if not notif.startswith(RBLN_COVERAGE_NOTIF_PREFIX):
            return None, None, notif
        writer, lo, hi, rest = notif[len(RBLN_COVERAGE_NOTIF_PREFIX) :].split(b":", 3)
        return int(writer), (int(lo), int(hi)), rest

    @staticmethod
    def _trim_to_consumer_blocks(
        local_block_ids: BlockIds, remote_block_ids: BlockIds
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
            assert num_remote <= len(local[i]), (
                f"group {i}: consumer registered {num_remote} blocks but this "
                f"producer holds {len(local[i])}; the pair cannot be aligned"
            )
            if num_remote < len(local[i]):
                local[i] = local[i][-num_remote:]
        return tuple(local)
