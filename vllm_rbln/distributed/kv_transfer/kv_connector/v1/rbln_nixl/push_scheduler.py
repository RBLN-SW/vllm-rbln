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
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorMetadata,
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
    connector_option,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def push_stream_enabled(
    vllm_config: VllmConfig, *, is_hma_required: bool, use_host_buffer: bool
) -> bool:
    """Whether a prefill's closed prefix leaves before the request ends.

    A hybrid model is left out because its groups do not close together: they
    hold different numbers of blocks for the same tokens, and an offer advances
    every group by one count. Its handover clips each group to its own window
    from the tail, which is the end a prefix sent from the front never reaches.
    Host staging is left out because it holds no areas to write out of.

    Derived in one place because the two sides decide different things from it
    and must not disagree: the scheduler stops building offers, and the worker
    stops asking a peer for per-shard descriptors. A side that takes one
    without the other routes its peers to a path nothing feeds.
    """
    return (
        connector_option(vllm_config, "push_stream", False)
        and not is_hma_required
        and not use_host_buffer
    )


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
        self._early_push_enabled = push_stream_enabled(
            vllm_config,
            is_hma_required=self._is_hma_required,
            use_host_buffer=self.use_host_buffer,
        )
        # How much of each request's prefix has already been offered, so a
        # step that closes no new block offers nothing.
        self._streamed_chunks: dict[str, int] = {}
        # Requests offered early, kept until either the lease takes over
        # (terminal finish) or their blocks go back to the allocator without
        # one, which the worker has to be told about (`push_early_flush`).
        self._early_sent: set[ReqId] = set()
        # Tokens each handed-over request holds, for the worker's `_tail_chunks`.
        self._valid_tokens: dict[ReqId, int] = {}

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

        if self._early_push_enabled:
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
        # Only the handover carries the request's last block; a streamed offer
        # is a closed prefix and never reaches it.
        meta.valid_tokens = {
            req_id: self._valid_tokens.pop(req_id)
            for req_id in meta.push_finished_blocks
            if req_id in self._valid_tokens
        }
        return meta

    def _build_stream_meta(
        self, meta: RblnNixlConnectorMetadata, scheduler_output: SchedulerOutput
    ) -> None:
        """Offer the prefix a prefill has closed, on every step it grows.

        The offer is the accumulated prefix, not the step's new blocks. The
        writer parks an unmatched offer by overwriting what it held for the
        request, so a prefix survives that overwrite and a delta is silently
        dropped -- and an offer parked behind a late registration is exactly
        when several of them queue up.

        Kept apart from `_build_save_meta`, which emits once at the closing
        chunk and drops its accumulation there: one method serving both would
        put a mode switch in the one host staging depends on.

        `closed` counts blocks whose tokens were computed before this step, so
        the writer -- which takes the offer at the start of a later step -- only
        ever reads KV a forward has already finished with.
        """
        # What the offer has to grow by to be worth releasing. One prefill
        # step is the finest unit any write can name -- the write path floors
        # its chunk at one step (`kv_chunk_tokens`) -- so a step's worth of
        # tokens is the right cursor here whether or not it divides a block:
        # all this counter asks is whether the offer grew.
        chunk = min(
            self.vllm_config.scheduler_config.max_num_batched_tokens, self.block_size
        )
        for req_id, new_block_id_groups, resumed in yield_req_data(scheduler_output):
            req = self._reqs_need_save.get(req_id)
            if req is None:
                continue
            assert req.kv_transfer_params is not None

            if self._accumulate_blocks_to_save(req_id, new_block_id_groups, resumed):
                self._streamed_chunks.pop(req_id, None)

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
            # Tokens the blocks we hold actually back. A step can compute
            # past them -- the accumulation lags by a step on a resume -- and
            # offering tokens no block of ours holds names KV that is not
            # there.
            held = min(
                req.num_computed_tokens,
                min(len(group) for group in groups) * self.block_size,
            )
            chunks = held // chunk
            if chunks <= self._streamed_chunks.get(req_id, 0):
                continue
            self._streamed_chunks[req_id] = chunks
            meta.push_stream_total[req_id] = cdiv(
                req.num_prompt_tokens, self.block_size
            )
            meta.push_stream_tokens[req_id] = held
            # The block being filled comes too. A write takes only the chunks
            # of it the token count backs, and a peer that takes no chunks
            # leaves it alone -- see `_stream_window`.
            offered = cdiv(held, self.block_size)
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=tuple(group[:offered] for group in groups),
                kv_transfer_params=req.kv_transfer_params,
            )

    def request_finished(
        self, request: "Request", block_ids: "BlockIds"
    ) -> tuple[bool, dict[str, Any] | None]:
        self._streamed_chunks.pop(request.request_id, None)
        delay_free_blocks, out_params = super().request_finished(request, block_ids)
        if delay_free_blocks:
            # The lease now holds the blocks, so the write no longer needs
            # watching for their reuse.
            self._early_sent.discard(request.request_id)
            # The same count upstream reports as `remote_num_tokens`, taken
            # here because it is the one that matches the handed-over blocks.
            if self._sends_token_count:
                self._valid_tokens[request.request_id] = request.num_computed_tokens
        return delay_free_blocks, out_params
