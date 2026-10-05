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
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorWorker,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        ReqMeta,
    )

logger = init_logger(__name__)


class RblnNixlPullConnectorWorker(RblnNixlWorkerBase, NixlPullConnectorWorker):
    """Reads a request's KV from the producers whose regions this rank shares.

    The pairing itself lives in `RblnNixlWorkerBase`; what belongs here is the
    read -- which peers to issue it against.
    """

    def _read_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        if engine_id not in self._remote_agents:
            # A handshake publishes the agent before `_ready_requests` carries
            # its read here a step later, and a heartbeat can declare the engine
            # gone in between. Upstream's only teardown cannot open that window
            # -- an engine this fresh is never stale -- and the descriptors this
            # read needs went with it.
            self._log_failure(
                failure_type="peer_unreachable",
                req_id=req_id,
                error=None,
                dst_engine_id=engine_id,
            )
            self._handle_failed_transfer(req_id, None, self._recv_failures)
            return
        # Keep the engine off the staleness sweep: upstream does this on the
        # read path this one replaces, and a swept producer loses the state
        # mid-transfer.
        self._engine_last_active[engine_id] = time.perf_counter()
        pp_size = self._remote_pp_size.get(engine_id, 1)
        # `or None`: a producer that kept no blocks reports zero, and zero is
        # not a last block anyone can size -- the whole block goes.
        valid_tokens = meta.remote.num_tokens or None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        # Per-shard lists exist exactly for peers serving part of what a
        # whole-engine handle covers. Re-deriving that from the parallel sizes
        # misses the reverse case: a producer without pipelining still serves
        # several of our ranks when ours is the finer one.
        if not self._overlapping_ranks.get(engine_id):
            # Chunk mode registers per-shard state against every peer unless
            # this engine owns the whole-engine lists, so reaching upstream's
            # read without either means it did not.
            assert not self._shape.chunk_mode or self._own_engine_layout, (
                f"RBLN NIXL: chunk mode reached upstream's whole-engine read "
                f"for {engine_id}, whose notification cannot name the part of "
                "a request a chunked read fills"
            )
            # Counted before the call: upstream trims the front of both lists
            # against the local prefix cache, and the token count describes the
            # request's own blocks.
            tail: AbstractContextManager = (
                self._tail_viewed_as(
                    valid_tokens, self._prompt_blocks(meta.remote.block_ids)
                )
                if self._shape.chunk_mode or self._own_engine_layout
                else nullcontext()
            )
            with tail:
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
        # Counted before the prefix trim below, which cuts the front: the token
        # count describes the producer's whole list, and both lists keep their
        # tail, so the last element is still the request's last block.
        n_prompt_blocks = sum(len(g) for g in remote_block_ids)

        if not prefix_hit:
            # _apply_prefix_caching indexes per KV-cache group, so a group-count
            # mismatch must fail loudly here rather than as an opaque IndexError.
            assert (
                len(remote_block_ids)
                == len(local_block_ids)
                == len(self.kv_cache_config.transfer_groups)
            )
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                decode_block_ids=local_block_ids,
                prefill_block_ids=remote_block_ids,
                decode_physical_per_logical=(
                    self._physical_blocks_per_logical_kv_block
                ),
                prefill_physical_per_logical=(
                    remote_info.remote_physical_blocks_per_logical
                ),
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
                    # As upstream's own notification path does: a dropped
                    # notification leaves the remote blocks pinned until their
                    # lease expires, which is not worth failing the step over.
                    self._log_failure(
                        failure_type="notification_failed",
                        req_id=req_id,
                        msg="Remote blocks will be freed after timeout",
                        error=e,
                        dst_engine_id=engine_id,
                        remote_pp_rank=global_rank,
                    )
                    self.xfer_stats.record_failed_notification()
                continue

            # Per peer, because the chunk grid is: a read reaches several
            # producers, and there is no step where they agree on one.
            remote_descs = self._shard_descs_for_tokens(
                engine_id,
                global_rank,
                self.dst_num_blocks[engine_id],
                remote_block_ids,
                num_valid_tokens=valid_tokens,
                num_prompt_blocks=n_prompt_blocks,
            )
            local_descs = self._shard_descs_for_tokens(
                engine_id,
                global_rank,
                self.num_blocks,
                local_block_ids,
                num_valid_tokens=valid_tokens,
                num_prompt_blocks=n_prompt_blocks,
            )
            assert len(local_descs) == len(remote_descs), (
                f"RBLN NIXL: {len(local_descs)} local vs {len(remote_descs)} "
                f"remote descriptor(s) for {engine_id} rank {global_rank}; the "
                "two lists pair by position, so a transfer would misread"
            )
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
                # 0.30.0 no longer queues the request itself: the handler
                # records the failure into the set its caller hands it, and
                # get_finished drains that set.
                self._handle_failed_transfer(req_id, handle, self._recv_failures)
                return

        if handles:
            self._recving_transfers[req_id].extend(handles)
