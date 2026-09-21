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

import time
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorWorker,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_scheduler import (
    ABORT_NOTIFY_ATTR,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        ReqMeta,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlPullConnectorWorker(RblnNixlWorkerBase, NixlPullConnectorWorker):
    """Reads a request's KV from the producers whose regions this rank shares.

    The pairing itself lives in `RblnNixlWorkerBase`; what belongs here is the
    read -- which peers to issue it against.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        #: Requests whose read settled without moving a block, waiting for the
        #: `get_finished` that reports them.
        self._zero_read_reqs: set[str] = set()

        #: Of those, the ones already reported as failed this step, so that a
        #: second unreachable peer does not report the request twice.
        self._failed_zero_reads: set[str] = set()

        #: Reads queued only to release an aborted request's blocks on the
        #: producer. They look exactly like a full prefix hit here, and the
        #: scheduler tags them because only it can tell the two apart.
        self._abort_notify_reqs: set[str] = set()

    def start_load_kv(self, metadata: "NixlConnectorMetadata") -> None:
        """Take the abort tags off the metadata before the reads run."""
        self._abort_notify_reqs = set(getattr(metadata, ABORT_NOTIFY_ATTR, ()) or ())
        super().start_load_kv(metadata)

    def _settle_zero_read(self, req_id: str) -> None:
        """Record a request whose read completed without moving a block.

        `get_finished` reports exactly `_recving_transfers` and
        `_failed_recv_reqs`, so a read that posts no transfer and raises nothing
        is never named again, and a scheduler holding the request in
        WAITING_FOR_REMOTE_KVS has nothing to end that wait on (ICR-47).

        This is the route for a read that reached its peers: every block it
        needs is already local, so it is complete rather than failed. It stays
        out of `_recving_transfers` too, since `get_finished` follows that
        structure with the block-size and heterogeneous-attention
        post-processing, which would permute blocks nothing wrote.

        A read that only releases an aborted request's blocks reaches here the
        same way and is left out: the scheduler dropped that request before
        queueing the read, so naming it is what kills the engine (ICR-47).
        """
        if req_id in self._abort_notify_reqs:
            return
        self._zero_read_reqs.add(req_id)

    def _fail_zero_read(
        self,
        req_id: str,
        error: Exception,
        *,
        engine_id: str,
        remote_rank: int | None = None,
    ) -> None:
        """Report a notify-only read whose notification could not be sent.

        A notification that raises is how a dead connection shows up on this
        path, and upstream only logs it -- leaving the request unreported and
        the scheduler waiting without a bound (ICR-47). Routing it to the same
        place a failed transfer goes lets the scheduler recompute locally and
        end the request with a reason.
        """
        self._log_failure(
            failure_type="notification_failed",
            req_id=req_id,
            msg="Marking blocks as invalid",
            error=error,
            dst_engine_id=engine_id,
            remote_pp_rank=remote_rank,
        )
        self.xfer_stats.record_failed_notification()
        # A peer that answered earlier in this step does not make the request
        # complete; the failure is what the scheduler has to see.
        self._zero_read_reqs.discard(req_id)
        if req_id in self._abort_notify_reqs:
            # Nobody is waiting on it. The producer keeps the blocks until
            # their lease expires, which is the cost of the peer being gone.
            return
        if req_id in self._failed_zero_reads:
            # Every unreachable peer is worth logging, but the engine's
            # aggregator counts one report per worker, and a second would
            # settle some later request early.
            return
        self._failed_zero_reads.add(req_id)
        self._report_failed_recv(req_id)

    def _is_notify_only(self, read_spec: "ReadSpec", dst_engine_id: str) -> bool:
        """Whether upstream's read would notify this peer and move nothing.

        Upstream decides on the local list it reads with, which is the one on
        the spec except where a coarser local block maps onto several remote
        ones. That mapping expands every id, so it comes out empty exactly when
        the group it was given is empty.
        """
        local_block_ids = read_spec.local_block_ids
        if not local_block_ids:
            return True
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        if self.transfer_topo.block_size_ratio(remote_info.remote_block_size) > 1:
            return not local_block_ids[0]
        return False

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Upstream's report, plus the reads that completed without a transfer.

        Upstream goes first, so a request it already accounts for -- because a
        peer's transfer failed, or one completed -- is left to it. Anything
        still in flight keeps its `_recving_transfers` entry and is that
        handle's to report, so only what upstream will never name is settled
        here, and the metadata is dropped for exactly those.
        """
        pending = self._zero_read_reqs
        self._zero_read_reqs = set()
        self._failed_zero_reads = set()

        done_sending, done_recving = super().get_finished()

        settled = {
            r
            for r in pending
            if r not in done_recving and r not in self._recving_transfers
        }
        for req_id in settled:
            self._recving_metadata.pop(req_id, None)
        return done_sending, done_recving | settled

    def _read_blocks(
        self,
        read_spec: "ReadSpec",
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> None:
        """Report the reads upstream leaves unreported.

        Peers whose ranks do not overlap ours go through upstream's read (see
        `_read_blocks_for_req`), which touches neither `_recving_transfers` nor
        `_failed_recv_reqs` when it only notifies -- whether the notification
        landed or not. The notify-only case is taken over here so the two
        outcomes can be told apart; everything else stays upstream's, and is
        settled only when it posted no transfer and raised nothing.
        """
        if not self._is_notify_only(read_spec, dst_engine_id):
            super()._read_blocks(
                read_spec=read_spec,
                dst_engine_id=dst_engine_id,
                request_id=request_id,
                remote_request_id=remote_request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )
            if request_id not in self._recving_transfers:
                self._settle_zero_read(request_id)
            return

        # Upstream's own notification, sent here so its failure is reported.
        agent_name = self._remote_agents[dst_engine_id][(0, read_spec.remote_rank)]
        notif_id = f"{remote_request_id}:{self.world_size}".encode()
        try:
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
        except Exception as e:
            self._fail_zero_read(
                request_id,
                e,
                engine_id=dst_engine_id,
                remote_rank=read_spec.remote_rank,
            )
            return
        self._settle_zero_read(request_id)

    def _read_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Keep the engine off the staleness sweep: upstream does this on the
        # read path this one replaces, and a swept producer loses the state
        # mid-transfer.
        self._engine_last_active[engine_id] = time.perf_counter()
        pp_size = self._remote_pp_size.get(engine_id, 1)
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        # Per-shard lists exist exactly for peers serving part of what a
        # whole-engine handle covers. Re-deriving that from the parallel sizes
        # misses the reverse case: a producer without pipelining still serves
        # several of our ranks when ours is the finer one.
        if not self._overlapping_ranks.get(engine_id):
            return super()._read_blocks_for_req(req_id, meta)

        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        assert block_size_ratio == 1, (
            "RBLN NIXL per-shard read path requires equal P/D block sizes "
            f"(got block_size_ratio={block_size_ratio})"
        )
        remote_block_size = remote_info.remote_block_size

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids, remote_info.remote_physical_blocks_per_logical
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        notif_id = self._xfer_notif_id(
            engine_id, meta.remote.request_id, remote_info.remote_tp_size
        )
        prefix_hit = len(local_block_ids) == 0
        n_prompt_blocks = sum(len(g) for g in remote_block_ids)

        if not prefix_hit:
            # _apply_prefix_caching indexes per KV-cache group, so a group-count
            # mismatch must fail loudly here rather than as an opaque IndexError.
            assert (
                len(remote_block_ids)
                == len(local_block_ids)
                == len(self.kv_cache_config.kv_cache_groups)
            )
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                local_block_ids,
                remote_block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )

        n_read_blocks = sum(len(g) for g in local_block_ids)
        logger.debug(
            "per-shard read req %s: pp_size=%d prompt_blocks=%d read_blocks=%d "
            "prefix_skipped=%d%s",
            req_id,
            pp_size,
            n_prompt_blocks,
            n_read_blocks,
            n_prompt_blocks - n_read_blocks,
            " (full prefix hit, notif only)" if prefix_hit else "",
        )

        # Publish once and fail as one request: a handle visible while a later
        # stage is still being prepped settles the request early, and the stages
        # landing after that are a second completion with its metadata gone.
        # A failed stage means recompute, so in-flight handles are released.
        handles: list[int] = []
        notif_failed = False
        for global_rank in self._overlapping_ranks[engine_id]:
            if prefix_hit:
                # Stages are tracked by the flat rank, agents by the pair it
                # decomposes into (see add_remote_agent).
                agent_name = self._remote_agents[engine_id][
                    divmod(global_rank, remote_info.remote_tp_size)
                ]
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
                except Exception as e:
                    # The notification is all this read does, so a peer that
                    # cannot be reached leaves nothing to report the request
                    # with. Fail it rather than drop it (ICR-47).
                    self._fail_zero_read(
                        req_id, e, engine_id=engine_id, remote_rank=global_rank
                    )
                    notif_failed = True
                continue

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
                    "READ",
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
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=engine_id,
                    remote_pp_rank=global_rank,
                )
                for submitted in handles:
                    self.nixl_wrapper.release_xfer_handle(submitted)
                self._handle_failed_transfer(req_id, handle)
                return

        if handles:
            self._recving_transfers[req_id].extend(handles)
        elif not notif_failed:
            # Nothing was posted and every peer answered: a full prefix hit
            # notifies and reads nothing, and the blocks are already local.
            # Settle it, or it is never reported (ICR-47).
            self._settle_zero_read(req_id)
