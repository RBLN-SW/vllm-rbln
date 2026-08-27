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

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorWorker,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta

logger = init_logger(__name__)


class RblnNixlPullConnectorWorker(RblnNixlWorkerBase, NixlPullConnectorWorker):
    """Reads a request's KV from the producers whose regions this rank shares.

    The pairing itself lives in `RblnNixlWorkerBase`; what belongs here is the
    read -- which peers to issue it against.
    """

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

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
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
        for global_rank in self._overlapping_ranks[engine_id]:
            if prefix_hit:
                agent_name = self._remote_agents[engine_id][global_rank]
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
