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


from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorMetadata,
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class RblnNixlPushConnectorScheduler(RblnNixlSchedulerBase, NixlPushConnectorScheduler):
    """Scheduler side of the write path.

    Beyond binding the two bases, this offers a prefill's blocks to the worker
    as the chunks close them rather than at the request's end, so what a
    prefill has finished can leave while the rest of it is still being
    computed -- on later pipeline stages, or in this rank's own later chunks.
    What it offers is a prefix of the list the request's end would have
    offered; the end still carries the block that prefix never closes.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # What a prefill closes can leave before the request ends, whether the
        # rest of it is still running on later pipeline stages or in this
        # rank's own later chunks. Which peers can be written a prefix is not
        # known until the handshake, so that part is settled per write.
        #
        # A hybrid model is left out because its groups do not close together:
        # they hold different numbers of blocks for the same tokens, and the
        # offer below advances every group by one count. Its handover clips
        # each group to its own window from the tail, which is the end a
        # prefix sent from the front never reaches.
        self._early_push_enabled = (
            envs.VLLM_RBLN_NIXL_PUSH_STREAM and not self._is_hma_required
        )
        # How much of each request's prefix has already been offered, so a
        # step that closes no new block offers nothing.
        self._streamed_blocks: dict[str, int] = {}
        # Requests offered early, kept until either the lease takes over
        # (terminal finish) or their blocks go back to the allocator without
        # one, which the worker has to be told about (`push_early_flush`).
        self._early_sent: set[ReqId] = set()

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        # Upstream tracks a producer's request for the save path only under
        # host staging, and the accumulation that path builds is what the
        # early offer reads.
        params = request.kv_transfer_params
        if self._early_push_enabled and params and params.get("do_remote_decode"):
            self._reqs_need_save[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        base_meta = super().build_connector_meta(scheduler_output)
        assert isinstance(base_meta, NixlConnectorMetadata)
        meta = RblnNixlConnectorMetadata.promote(base_meta)

        if self._early_push_enabled and not self.use_host_buffer:
            # Upstream fills `reqs_to_save` for host staging only (see
            # `_build_stream_meta`).
            self._build_stream_meta(meta, scheduler_output)
            self._early_sent.update(meta.reqs_to_save)

        # A preempted request re-prefills into these blocks and one that
        # finished on a non-terminal status hands them straight back, neither
        # of them behind the lease that protects a terminal finish.
        flush = self._early_sent & (
            set(scheduler_output.preempted_req_ids or ()) | meta.reqs_not_processed
        )
        meta.push_early_flush = flush
        self._early_sent -= flush
        return meta

    def _build_stream_meta(
        self, meta: RblnNixlConnectorMetadata, scheduler_output: SchedulerOutput
    ) -> None:
        """Offer the prefix a prefill has closed, on every step it grows.

        A prefill's blocks close one at a time and the request ends long after
        the first of them does. Offering the closed prefix each step lets a
        stage write a block while the chunks behind it are still running, so
        what is left to move when the prefill ends is the last block rather
        than the whole prompt.

        The offer is the accumulated prefix, not the step's new blocks. The
        writer parks an unmatched offer by overwriting what it held for the
        request, so a prefix survives that overwrite and a delta is silently
        dropped -- and an offer parked behind a late registration is exactly
        when several of them queue up.

        NOTE(RBLN): a parallel method rather than the generalisation of
        `_build_save_meta` the plan called for. That one emits once, at the
        closing chunk, and drops its accumulation there; this emits on every
        step and keeps it. Folding both into one would put a mode switch in the
        method host staging depends on, which would make that path share this
        one's risk for nothing.

        `closed` counts blocks whose tokens were computed before this step, so
        the writer -- which takes the offer at the start of a later step -- only
        ever reads KV a forward has already finished with.
        """
        for req_id, new_block_id_groups, resumed in yield_req_data(scheduler_output):
            req = self._reqs_need_save.get(req_id)
            if req is None:
                continue
            assert req.kv_transfer_params is not None

            if self._accumulate_blocks_to_save(req_id, new_block_id_groups, resumed):
                self._streamed_blocks.pop(req_id, None)

            groups = self._block_ids_need_save.get(req_id)
            # A request enters the table on the step it is admitted, which is
            # the step it first carries blocks, so nothing reaches here with
            # neither the accumulation nor a delta.
            assert groups is not None, (
                "RBLN push stream reached with no blocks: "
                f"req_id={req_id} resumed={resumed} "
                f"num_computed={req.num_computed_tokens} "
                f"num_prompt={req.num_prompt_tokens}"
            )
            closed = min(
                req.num_computed_tokens // self.block_size,
                min(len(group) for group in groups),
            )
            if closed <= self._streamed_blocks.get(req_id, 0):
                continue
            self._streamed_blocks[req_id] = closed
            meta.push_stream_total[req_id] = cdiv(
                req.num_prompt_tokens, self.block_size
            )
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=tuple(group[:closed] for group in groups),
                kv_transfer_params=req.kv_transfer_params,
            )

    def request_finished(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        """Seed the field a rejected request never filled, then let go of an
        early write once the lease holds its blocks.

        NOTE(RBLN): a request the serving layer rejects -- a prompt past the
        context length, a client that left -- reaches upstream still flagged for
        a remote prefill, and upstream registers an empty receive for it so the
        producer stops holding the blocks it pinned. Building that receive reads
        `remote_block_ids`, which on this direction is filled by
        `update_state_after_alloc`, the one call a rejected request never makes,
        and the engine dies on the missing key. The read path is unaffected:
        there the field arrives with the producer's own reply.
        """
        self._streamed_blocks.pop(request.request_id, None)
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            params.setdefault("remote_block_ids", ())

        delay_free_blocks, out_params = super().request_finished(request, block_ids)
        if delay_free_blocks:
            # The lease now holds the blocks, so the write no longer needs
            # watching for their reuse.
            self._early_sent.discard(request.request_id)
        return delay_free_blocks, out_params
