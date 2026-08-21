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
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPushConnectorWorker,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        ReqMeta,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


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
        # against the number of writers the peer put in them.
        self._writer_counts_by_req: defaultdict[str, int] = defaultdict(int)

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
        super().start_load_kv(metadata)

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

        NOTE(RBLN): the base settles a pushed request on the FIRST completion
        notification, which is right only while one peer rank writes the whole
        thing. Several do as soon as either axis is cut finer on their side, and
        the rest of them are still writing -- host staging would copy a
        half-filled buffer to the device. Each writer's notification carries the
        count (see `_xfer_notif_id`), so drop all but the last one and let the
        base settle the request on that.
        """
        for notif in self._drain_completion_notifs():
            if self._writer_still_pending(notif):
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

    def _writer_still_pending(self, notif: bytes) -> bool:
        """Whether this notification leaves a request short of its writers.

        False for anything the base has to see itself: heartbeats, our own
        outbound accounting, and a request we are not receiving.
        """
        msg = notif.decode("utf-8")
        if msg.startswith("HB:") or ":" not in msg:
            return False
        req_id, count = msg.rsplit(":", 1)
        if req_id in self._reqs_to_send or req_id in self._reqs_to_process:
            return False
        if req_id not in self._recving_metadata:
            return False
        # The peer scales the count by our tensor-parallel size, the unit the
        # base divides by on the other direction (see `_xfer_notif_id`).
        writers = max(1, int(count) // self.world_size)
        self._writer_counts_by_req[req_id] += 1
        return self._writer_counts_by_req[req_id] < writers

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending, done_recving = super().get_finished()
        # Both completion and failure land here, and a retried request must not
        # inherit a partial count.
        for req_id in done_recving:
            self._writer_counts_by_req.pop(req_id, None)
        return done_sending, done_recving

    def _xfer_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Keep the engine off the staleness sweep: a swept peer loses the state
        # this path reads, and the base refreshes it on the route it replaces.
        self._engine_last_active[engine_id] = time.perf_counter()
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        # Per-shard lists exist exactly for peers serving part of what a
        # whole-engine handle covers; without any, the base describes them all.
        if not self._overlapping_ranks.get(engine_id):
            return super()._xfer_blocks_for_req(req_id, meta)

        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        assert block_size_ratio == 1, (
            "RBLN NIXL per-shard transfers require equal P/D block sizes"
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

        local_block_ids = self._trim_to_consumer_blocks(
            local_block_ids, remote_block_ids
        )
        n_write_blocks = sum(len(g) for g in local_block_ids)
        if not n_write_blocks:
            logger.warning("per-shard write req %s: no blocks to push", req_id)
            return

        logger.info(
            "per-shard write req %s: ranks=%d write_blocks=%d",
            req_id,
            len(self._overlapping_ranks[engine_id]),
            n_write_blocks,
        )

        # Publish once, for the reason the read path states (see
        # `_read_blocks_for_req`); failure is per peer here, not per request.
        handles: list[int] = []
        for global_rank in self._overlapping_ranks[engine_id]:
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

        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)

    @staticmethod
    def _trim_to_consumer_blocks(
        local_block_ids: BlockIds, remote_block_ids: BlockIds
    ) -> BlockIds:
        """Drop from our side the blocks the consumer already had.

        NOTE(RBLN): the same trim the read path gets from the base's
        `_apply_prefix_caching`, with the roles swapped -- there the consumer is
        local and the producer's longer list is trimmed to it, here the consumer
        is the peer and ours is the longer one. It has to come off the END: the
        consumer registers the uncached SUFFIX of a prompt, so dropping our tail
        would hand it the wrong blocks under a partial prefix hit.
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
