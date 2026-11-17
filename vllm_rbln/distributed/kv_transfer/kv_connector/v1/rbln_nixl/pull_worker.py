# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
import time
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorWorker,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnReqMeta,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.transfer import (
    trim_producer_to_consumer,
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Remote agent -> the (mode, rail) its connection had before a request
        # changed it; present only while the connection is off that default.
        self._xfer_defaults: dict[bytes, tuple[int, int]] = {}

    def _read_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        assert meta.remote is not None and self.transfer_topo is not None
        # Upstream's drain reads every queued entry, and a handshake completing
        # after its request was reported failed still queues one. An awaited
        # request (upstream's test for holding the metadata) keeps it until it
        # is reported, so one without it is a read nothing waits on.
        if req_id not in self._recving_metadata and (
            meta.awaiting_kvs or any(meta.local_block_ids)
        ):
            return
        engine_id = meta.remote.engine_id
        # Upstream's `start_load_kv` hands a read here only for an engine that
        # still holds its agent, and re-handshakes one released since then.
        assert engine_id in self._remote_agents, (
            f"RBLN NIXL pull: request {req_id} reached the read path for engine "
            f"{engine_id}, which holds no agent"
        )
        # Keep the engine off the staleness sweep: upstream does this on the
        # read path this one replaces, and a swept producer loses the state
        # mid-transfer.
        self._engine_last_active[engine_id] = time.perf_counter()

        # nixl-rbln keeps mode and rail per connection, not per transfer, and
        # applies them from the next one posted. So a read that names a lane
        # sets it on every agent of the engine first, and one that names none
        # puts back the default a previous request replaced.
        asked = isinstance(meta, RblnReqMeta)
        # A notify-only read (full local hit, aborted cleanup) moves no KV, and
        # failing it here would drop the notice that frees the producer's blocks.
        moves_kv = meta.awaiting_kvs or any(meta.local_block_ids)
        agents = [
            agent
            for agent in self._remote_agents[engine_id].values()
            if moves_kv and (asked or agent in self._xfer_defaults)
        ]
        if agents:
            try:
                if not self._use_rbln_nixl_backend:
                    raise RuntimeError("xfer_mode / xfer_rail need nixl-rbln")
                if asked and meta.xfer_mode not in (None, "single_lane", "multi_lane"):
                    raise ValueError(
                        f"xfer_mode {meta.xfer_mode!r} is neither 'single_lane' "
                        "nor 'multi_lane'"
                    )
                if asked and isinstance(meta.xfer_rail, bool):
                    raise ValueError(f"xfer_rail {meta.xfer_rail!r} is not a rail")
                import nixl_rbln

                for agent in agents:
                    # NIXL's binding hands agent names back as bytes despite the
                    # str in `add_remote_agent`'s annotation, and nixl-rbln
                    # encodes what it is given.
                    peer = agent.decode()
                    if agent not in self._xfer_defaults:
                        self._xfer_defaults[agent] = (
                            nixl_rbln.get_xfer_mode(self.nixl_wrapper, peer),
                            nixl_rbln.get_xfer_rail(self.nixl_wrapper, peer),
                        )
                    mode, rail = self._xfer_defaults[agent]
                    if asked and meta.xfer_mode is not None:
                        mode = (
                            nixl_rbln.XFER_MULTI_LANE
                            if meta.xfer_mode == "multi_lane"
                            else nixl_rbln.XFER_SINGLE_LANE
                        )
                    if asked and meta.xfer_rail is not None:
                        rail = meta.xfer_rail
                    nixl_rbln.set_xfer_mode(self.nixl_wrapper, peer, mode)
                    nixl_rbln.set_xfer_rail(self.nixl_wrapper, peer, rail)
                    if not asked:
                        del self._xfer_defaults[agent]
            except (ValueError, RuntimeError, struct.error) as e:
                # What the request asked for decides it, so it fails the
                # request rather than the engine.
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=req_id,
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=engine_id,
                )
                self._handle_failed_transfer(req_id, None, self._recv_failures)
                return

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

        if not prefix_hit and len(remote_block_ids) != len(local_block_ids):
            assert len(local_block_ids) == len(self.kv_cache_config.transfer_groups)
            try:
                remote_block_ids = trim_producer_to_consumer(
                    remote_block_ids,
                    local_block_ids,
                    self._region_group_pairs(
                        engine_id,
                        self._overlapping_ranks[engine_id],
                        producer_is_peer=True,
                    ),
                    consumer_name="this decoder",
                    req_id=req_id,
                )
            except RuntimeError as e:
                # What this decoder had cached decides it, so it fails the
                # request rather than the engine.
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=req_id,
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=engine_id,
                )
                self._handle_failed_transfer(req_id, None, self._recv_failures)
                return
        elif not prefix_hit:
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
                remote=True,
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
