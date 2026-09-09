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
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
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

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RBLN_COVERAGE_NOTIF_PREFIX,
    RblnNixlConnectorMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_scheduler import (
    push_stream_enabled,
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


class OfferedBlocks(tuple):
    """A released offer's block list, carrying the tokens it was built for.

    Carried on the list rather than kept per request because the writer runs
    on its own thread: a step can hold and release the next offer before the
    writer picks the previous one up, and a count read off the request then
    describes the wrong one. What that costs is a write claiming tokens this
    rank has not computed, into the consumer's blocks, silently.

    A tuple subclass so every reader of a block list -- upstream's parking
    dict, its grouping helper, its `ReqMeta` -- keeps working untouched.
    """

    offered_tokens: int

    def __new__(cls, groups: "BlockIds", offered_tokens: int) -> "OfferedBlocks":
        self = super().__new__(cls, groups)
        self.offered_tokens = offered_tokens
        return self


# How long a flush waits for an early write to leave the NIC before giving up
# on it. Bounded because it runs on the engine main thread: a wedged transfer
# must not take the engine with it. A fault detector, not a deadline -- what a
# healthy write of this size costs is not known here, so a smaller value would
# start abandoning writes that were about to land.
_EARLY_FLUSH_DRAIN_TIMEOUT_S = 1.0
_EARLY_FLUSH_POLL_INTERVAL_S = 0.001


@dataclass
class _StreamedSend:
    """What one request being pushed in pieces needs tracked between batches.

    One record on the request id rather than a map per field. The fields are
    written at different points of a single lifetime -- offered, released to
    the writer, issued, sealed, landed -- and there is no state in which only
    some of them should exist, which is why ending it used to mean remembering
    to drop the request from all eight.

    Reached only under `_sending_transfers_lock`, except where the maps it
    replaces were not: `start_early_push` and `_stream_window` run on the
    engine thread and touch nothing the writer writes.
    """

    # Blocks this rank has closed, held for the next step to hand over. None
    # once released, or once an abort dropped the offer.
    pending_offer: "BlockIds | None" = None
    # Tokens that offer holds. Travels with it to the writer -- see
    # `OfferedBlocks`.
    pending_offer_tokens: int = 0
    # Whether the writer has been told about this request at all. Not derivable
    # from `pending_offer`: an aborted offer clears that without ever releasing.
    released: bool = False
    # Blocks the request will hold once its whole prompt is computed, which is
    # what places the consumer's window inside our list.
    total: int | None = None
    # Handles grouped by the batch that issued them, kept out of
    # `_sending_transfers` so upstream cannot report the request as finished:
    # the scheduler frees a request's blocks on that report unconditionally,
    # and this one is still prefilling.
    transfers: list[list[int]] = field(default_factory=list)
    # Batches handed to the writer, and batches whose writes have landed.
    queued: int = 0
    done: int = 0
    # How many batches the request will have, once the engine says it is over.
    # None until then: an unsealed request is never finished, however many of
    # its batches have landed.
    expected: int | None = None
    # How many of the consumer's blocks the writer has already filled.
    issued_hwm: int = 0
    # And how many chunks of the one after those, for a block written in
    # pieces. Reset when that block is written whole, which subsumes them.
    issued_chunks: int = 0


class _CoverageNotifQueue(queue.Queue):
    """Upstream's completion queue, with this connector's coverage prefix taken
    off on the way out and a writer's reports held until its ranges cover the
    request -- see `_writer_still_pending` for what that leaves upstream.

    On the way out rather than in a pass of its own. The writer thread puts
    into this queue throughout the step, so a pass that drained it, stripped
    what it found and queued the results back left a window: a notification
    arriving after that pass and before upstream's drain reached upstream still
    prefixed, and upstream read the prefix as part of the request id. No
    request has that id, so the range that notification carried was dropped --
    and a request missing a range never settles (`_covers`), which makes the
    loss silent until it hangs.
    """

    def __init__(self, worker: "RblnNixlPushConnectorWorker") -> None:
        super().__init__()
        self._worker = worker

    def get_nowait(self) -> bytes:
        """The next notification upstream should act on, or raise `Empty`.

        Held-back notifications are consumed rather than returned, so the loop
        upstream drains this with is unchanged: it ends on `Empty` either way.
        """
        while True:
            notif = super().get_nowait()
            worker = self._worker
            writer, span, per_block, notif = worker._split_coverage(notif)
            if not worker._writer_still_pending(notif, writer, span, per_block):
                return notif


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
        # Tokens a handed-over request holds, for `_tail_chunks`. Every
        # request has one, streamed or not, so it is not part of
        # `_StreamedSend`.
        self._valid_tokens: dict[str, int] = {}

        # Ranges of this request's blocks each writer has reported filling,
        # for a peer that does name them. Kept per writer because a request
        # settles only once every one of them has covered the whole list.
        self._coverage_by_req: defaultdict[str, defaultdict[int, list[Span]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        # The unit those ranges are counted in, per writer. A writer states it
        # once and is held to it: a range read in the wrong unit reaches the
        # terminal count early and settles a request whose KV is incomplete.
        self._coverage_units_by_req: defaultdict[str, dict[int, int]] = defaultdict(
            dict
        )

        # Replaces the plain queue upstream made, so its own drain is the only
        # one and nothing can reach it unstripped.
        self._pending_completion_notifs = _CoverageNotifQueue(self)

        self._early_push_enabled = push_stream_enabled(
            vllm_config,
            is_hma_required=self._is_hma_required,
            use_host_buffer=self.use_host_buffer,
        )
        # Per request, for as long as it is being pushed in pieces. Created
        # when this rank first closes a chunk of it, dropped when the send is
        # over -- see _StreamedSend.
        self._streamed: dict[ReqId, _StreamedSend] = {}
        self._empty_receives: set[ReqId] = set()

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
        assert isinstance(metadata, RblnNixlConnectorMetadata)
        self._valid_tokens.update(metadata.valid_tokens)
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

        Settled here rather than through upstream's transfer table, though an
        empty entry there would pop as done: everything upstream does with a
        completed receive reads the blocks it landed in, starting with the
        engine they came from. A request turned away never handshook with a
        producer, so that lookup finds nothing and takes the engine down.
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            if not sum(len(group) for group in meta.local_block_ids):
                self._empty_receives.add(req_id)

    def _seal_at_handover(self, metadata: "NixlConnectorMetadata") -> None:
        """Fix how many batches a request written early will have.

        Its arrival in `push_finished_blocks` IS the engine saying the request
        is over, so nothing further will be handed to the writer for it.

        The handover is one of those batches, not a duplicate of them. What was
        streamed is the prefix of blocks a prefill CLOSED, and a prompt's last
        block is closed by nothing -- its tokens end mid-block. The handover
        carries the whole list, so it covers that tail, and the writer sends
        only the part past what it already wrote.

        Runs before the call that hands the same metadata to the writer, whose
        landed count would otherwise pass a total not yet set. Sealed rather
        than published: upstream reports a request finished once the handles it
        can see have landed, and this step's batch is not one of them.
        """
        with self._sending_transfers_lock:
            for req_id in metadata.push_finished_blocks:
                send = self._streamed.get(req_id)
                if send is None or not send.released:
                    continue
                send.queued += 1
                send.expected = send.queued

    def _writes_less_than_a_request(self) -> bool:
        return self._early_push_enabled

    def start_early_push(self, metadata: "RblnNixlConnectorMetadata") -> None:
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
        if not self._early_push_enabled:
            return
        for req_id, meta in metadata.reqs_to_save.items():
            send = self._streamed.setdefault(req_id, _StreamedSend())
            send.pending_offer = meta.local_block_ids
            send.pending_offer_tokens = metadata.push_stream_tokens.get(req_id, 0)
            total = metadata.push_stream_total.get(req_id)
            if total is not None:
                send.total = total

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
        offers = [
            (req_id, send, OfferedBlocks(send.pending_offer, send.pending_offer_tokens))
            for req_id, send in self._streamed.items()
            if send.pending_offer is not None
        ]
        if not offers:
            return
        with self._sending_transfers_lock:
            for _, send, _ in offers:
                send.released = True
        for req_id, send, block_ids in offers:
            send.pending_offer = None
            send.pending_offer_tokens = 0
            send.queued += 1
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
            with self._sending_transfers_lock:
                send = self._streamed.get(req_id)
                handles = [h for batch in send.transfers for h in batch] if send else []
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
            for send in self._streamed.values():
                for handles in send.transfers:
                    for handle in handles:
                        self.nixl_wrapper.release_xfer_handle(handle)
            self._streamed.clear()
            self._valid_tokens.clear()
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

    def _writer_still_pending(
        self, notif: bytes, writer: int | None, span: "Span | None", per_block: int
    ) -> bool:
        """Whether this notification leaves its writer short of the request.

        False for anything upstream has to see itself: heartbeats, our own
        outbound accounting, and a request we are not receiving.

        A writer that names the range it filled is held until what it has
        reported spans every block this rank registered, which is what leaves
        upstream one notification per writing rank. A writer that names none
        sends one already.

        The block count comes from what this rank registered, not from anything
        the peer said -- a peer that sends the wrong ranges must stall the
        request, not settle it early. The peer does say what unit it counts in,
        which it has to (the alternative is assuming its chiplet geometry
        equals ours), and is held to one per request for the same reason.
        """
        msg = notif.decode("utf-8")
        if msg.startswith("HB:"):
            return False
        req_id = msg.rsplit(":", 1)[0]
        if req_id in self._reqs_to_send or req_id in self._reqs_to_process:
            return False
        meta = self._recving_metadata.get(req_id)
        if meta is None:
            return False
        if span is None or writer is None:
            return False

        if per_block < 1:
            raise RuntimeError(
                f"RBLN NIXL push: writer {writer} named coverage of request "
                f"{req_id} in {per_block} units per block"
            )
        units = self._coverage_units_by_req[req_id]
        if units.setdefault(writer, per_block) != per_block:
            raise RuntimeError(
                f"RBLN NIXL push: writer {writer} changed the coverage unit of "
                f"request {req_id} from {units[writer]} to {per_block}"
            )
        spans = self._coverage_by_req[req_id][writer]
        spans.append(span)
        # Registration refuses a streamed engine with no full-attention group,
        # which is what would leave nothing to count in.
        prompt_blocks = self._prompt_blocks(meta.local_physical_block_ids)
        assert prompt_blocks is not None
        return not self._covers(spans, prompt_blocks * per_block)

    @staticmethod
    def _covers(spans: list["Span"], total: int) -> bool:
        """Whether the half-open ranges together leave no gap below `total`.

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
            if reach >= total:
                return True
        return total == 0

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
            self._coverage_by_req.pop(req_id, None)
            self._coverage_units_by_req.pop(req_id, None)
        sealed_done = self._finish_sealed_requests()
        if sealed_done:
            # Upstream drops the writer's state for what it reports itself,
            # and it has already been past that for this step.
            for req_id in sealed_done:
                self._evict_finished_inbox.put(req_id)
            self._push_writer_wake.set()
            done_sending |= sealed_done
        # `_forget_send` only reaches a request written in batches; one written
        # in a single transfer is reported here and nowhere else.
        for req_id in done_sending:
            self._valid_tokens.pop(req_id, None)
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
            for req_id, send in self._streamed.items():
                still_going = []
                for handles in send.transfers:
                    probe = {req_id: handles}
                    if self._pop_done_transfers(probe):
                        send.done += 1
                    else:
                        still_going.append(probe[req_id])
                send.transfers = still_going

            for req_id, send in list(self._streamed.items()):
                if send.expected is None or send.done < send.expected:
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
        self._streamed.pop(req_id, None)
        self._valid_tokens.pop(req_id, None)

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
            send = self._streamed.get(req_id)
            assert send is None or not send.released, (
                f"RBLN NIXL push: request {req_id} was written early but is "
                f"served by a whole-engine handle (peer {engine_id}), which "
                "the streaming handshake asks not to be given."
            )
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
        # Both read before the window and the consumer trim reshape them: the
        # token count describes this producer's whole list, and the request's
        # last block is only in the write that reaches the consumer's end.
        n_prompt_blocks = sum(len(g) for g in local_block_ids)
        # Registration refuses a streamed engine with no full-attention group,
        # which is what would leave nothing to count in.
        counted = self._counted_group(remote_block_ids)
        assert counted is not None
        registered = len(remote_block_ids[counted])

        # One window serves every peer of this request, and `issued_hwm` is
        # the request's, so the peers have to agree on the unit it counts in.
        # Where they do not, none of them gets chunks.
        grids = {self._shard_chunk_grids.get((engine_id, r)) for r in peer_ranks}
        chunk_grid = grids.pop() if len(grids) == 1 else None
        gpb = 1 if chunk_grid is None else chunk_grid[1]
        # Only the handover carries the request's final token count, and that
        # is what says how many chunks of its last block hold tokens. The
        # descriptor builders below derive the same number from the count they
        # are handed; the window needs it too, to stop a block it writes in
        # pieces at the same place.
        needed = self._tail_chunks(
            n_prompt_blocks, self._valid_tokens.get(req_id), chunks_per_span=gpb
        )
        window = self._stream_window(
            req_id,
            local_block_ids,
            remote_block_ids,
            chunk_grid=chunk_grid,
            offered_tokens=getattr(meta.local_block_ids, "offered_tokens", 0),
            tail=needed,
        )
        pieces = window[3] if window is not None else ()
        if window is None:
            local_block_ids = self._trim_to_consumer_blocks(
                local_block_ids, remote_block_ids, engine_id, meta.remote.request_id
            )
            span = (0, registered * gpb)
        else:
            local_block_ids, remote_block_ids, span, _ = window
        notif_id = self._with_coverage(notif_id, span, gpb)
        # Only the write that carries the request's last block may cut it. The
        # span counts in chunks now, so the consumer's end is that many past
        # its last block rather than the block count itself.
        tail_tokens = (
            self._valid_tokens.get(req_id) if span[1] == registered * gpb else None
        )
        n_write_blocks = sum(len(g) for g in local_block_ids)
        if not n_write_blocks and not pieces:
            # Ordinary once an offer grows every step: a step that computes
            # tokens without closing a chunk leaves the window where it was.
            logger.debug("per-shard write req %s: nothing new to push", req_id)
            with self._sending_transfers_lock:
                send = self._streamed.get(req_id)
                if send is not None and send.released:
                    send.done += 1
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
            # The chunks come from the second range of the same two lists, so
            # they join this batch rather than costing it a second transfer --
            # and one range keeps the notification single.
            for local_block, remote_block, chunk_span in pieces:
                remote_descs = np.concatenate(
                    (
                        remote_descs,
                        self._chunk_descs_ids_for_shard(
                            engine_id,
                            global_rank,
                            self.dst_num_blocks[engine_id],
                            remote_block,
                            chunk_span,
                        ),
                    )
                )
                local_descs = np.concatenate(
                    (
                        local_descs,
                        self._chunk_descs_ids_for_shard(
                            engine_id,
                            global_rank,
                            self.num_blocks,
                            local_block,
                            chunk_span,
                        ),
                    )
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
            send = self._streamed.get(req_id)
            if send is None or not send.released:
                self._sending_transfers[req_id].extend(handles)
            elif handles:
                send.transfers.append(handles)
            else:
                # Every peer's submission failed. The batch is over either
                # way, and a request whose count never reaches its seal is a
                # request that never finishes.
                send.done += 1

    def _stream_window(
        self,
        req_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        chunk_grid: tuple[int, int] | None = None,
        offered_tokens: int = 0,
        tail: int | None = None,
    ) -> "tuple[BlockIds, BlockIds, Span, tuple[tuple[int, int, Span], ...]] | None":
        """The part of the consumer's list this batch is the first to fill.

        Whole blocks, then the pieces of a block written in chunks: at most the
        rest of the one a previous batch left part-written and the start of the
        one being filled now. None where a request is not streamed -- the flag
        off, or a whole prompt offered at once -- which takes the trim instead.

        A streamed offer is a growing prefix of the producer's blocks, and the
        consumer registered the TAIL of the prompt, what its own cache did not
        cover. So the window starts at `total - registered` of ours, which a
        prefix shorter than the prompt cannot name -- hence the total travels
        with the offer.

        Only where both sides expand a logical block by the same factor: past
        that, the two lengths subtracted here count different units.
        """
        send = self._streamed.get(req_id)
        total = send.total if send else None
        if total is None:
            return None
        expand = self._physical_blocks_per_logical_kv_block
        if expand != self._remote_expand_for(req_id):
            return None

        f = self._counted_group(remote_block_ids)
        assert f is not None
        registered = len(remote_block_ids[f])
        offset = total * expand - registered
        have = len(local_block_ids[f])
        assert send is not None
        gpb = 1 if chunk_grid is None else chunk_grid[1]
        lo = send.issued_hwm
        # How many of our blocks the offer has CLOSED. Its length says so only
        # while it holds nothing else; once it carries the block being filled, a
        # peer that takes no chunks would write that one whole -- KV this rank has
        # not computed. So the token count decides, and the length is left to the
        # handover offer, which carries no count.
        closed = offered_tokens // self.block_size if offered_tokens else have
        hi = max(0, min(registered, closed - offset))

        # Chunks the offer holds of the block after those. Without a grid there
        # are none, which is a block written whole or not at all. An offer that
        # has not reached the consumer's window holds none either: `hi` clamps
        # to 0 there, and the block it would take them from is one the offer
        # does not have.
        tail_chunks = 0
        if (
            gpb > 1
            and offered_tokens
            and offset <= closed < have
            and 0 <= hi < registered
        ):
            rem = offered_tokens - closed * self.block_size
            if rem > 0:
                tail_chunks = rem * gpb // self.block_size
        # Chunks of the block at `lo` that earlier batches already wrote.
        head_chunks = send.issued_chunks

        if hi <= lo and tail_chunks <= head_chunks:
            reach = lo * gpb + head_chunks
            return (
                self._windowed(local_block_ids, f, []),
                self._windowed(remote_block_ids, f, []),
                (reach, reach),
                (),
            )

        def piece(index: int, chunk_span: "Span") -> tuple[int, int, "Span"]:
            return (
                local_block_ids[f][offset + index],
                remote_block_ids[f][index],
                chunk_span,
            )

        # The block at `lo` closes with part of it already gone, so the write takes
        # the rest in chunks and the whole-block range starts after it; writing it
        # whole would repeat what a previous batch sent. That rest stops at `tail`
        # where the block is the request's last, since the chunks past its tokens
        # are the ones this mode exists not to send.
        pieces: list[tuple[int, int, Span]] = []
        first = lo
        if hi > lo and head_chunks:
            end = tail if tail is not None and lo == registered - 1 else gpb
            if end > head_chunks:
                pieces.append(piece(lo, (head_chunks, end)))
            first = lo + 1
        tail_lo = head_chunks if hi == lo else 0
        if tail_chunks > tail_lo:
            pieces.append(piece(hi, (tail_lo, tail_chunks)))

        send.issued_hwm = hi
        send.issued_chunks = (
            tail_chunks if tail_chunks > tail_lo else (0 if hi > lo else head_chunks)
        )
        return (
            self._windowed(
                local_block_ids, f, local_block_ids[f][offset + first : offset + hi]
            ),
            self._windowed(remote_block_ids, f, remote_block_ids[f][first:hi]),
            (lo * gpb + head_chunks, hi * gpb + tail_chunks),
            tuple(pieces),
        )

    @staticmethod
    def _windowed(block_ids: BlockIds, group: int, window: list[int]) -> BlockIds:
        """This batch's slice of one group, with every other group untouched.

        A sliding window's group is not streamed -- its one block holds the
        live window and the kernel keeps overwriting it, so it is final only
        once the prefill is -- and the offer that carries it is the handover.
        Passing the other groups through is what lets that block ride the
        write this window ends on, and keeps them out of the ones before it.
        """
        return tuple(
            window if g == group else list(ids) for g, ids in enumerate(block_ids)
        )

    def _with_coverage(
        self, notif_id: bytes, span: "Span | None", per_block: int
    ) -> bytes:
        """Name the half-open range this write filled, and its unit.

        The range is positions into the list the consumer registered, counted
        in units of one consumer block divided by `per_block`. A request
        written once covers all of it; one written in batches covers the part
        this batch is the first to reach, which is what lets the consumer tell
        a complete request from a partly written one.

        The unit travels with the range because the consumer cannot derive it:
        it would have to assume the writer's chiplet geometry equals its own.

        Left off when no single range describes the write -- a request with more
        than one KV cache group, whose groups go out together carrying their own
        lengths. A consumer has to accept a message without the prefix for that
        reason alone, which is what lets the per-shard route carry it anyway.
        """
        if span is None:
            return notif_id
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        pp_rank = get_pp_group().rank_in_group if pp_size > 1 else 0
        writer = pp_rank * self.world_size + self.tp_rank
        head = f"{writer}:{span[0]}:{span[1]}:{per_block}:".encode()
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
    def _split_coverage(
        notif: bytes,
    ) -> "tuple[int | None, Span | None, int, bytes]":
        """Take the coverage prefix off, returning what it said and what is left.

        Upstream reads a completion notification as `req_id:count` with
        `rsplit`, so it would take the whole prefixed string as the request id
        and find no such request. What upstream is handed here is what it was
        handed before this rank started naming ranges.
        """
        if not notif.startswith(RBLN_COVERAGE_NOTIF_PREFIX):
            return None, None, 1, notif
        writer, lo, hi, per_block, rest = notif[
            len(RBLN_COVERAGE_NOTIF_PREFIX) :
        ].split(b":", 4)
        return int(writer), (int(lo), int(hi)), int(per_block), rest

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
