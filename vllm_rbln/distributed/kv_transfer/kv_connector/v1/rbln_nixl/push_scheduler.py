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
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorMetadata,
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId
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


class RblnNixlPushConnectorScheduler(RblnNixlSchedulerBase, NixlPushConnectorScheduler):
    """Scheduler side of the write path.

    Beyond binding the two bases, this offers a prefill's blocks to the worker
    at the chunk that closes it rather than at the request's end, so a
    pipeline stage can write its layers while the later stages still run. What
    it offers is the same list the request's end would have offered; only the
    step it is offered on is earlier.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # A stage's write overlaps the stages that come after it, so there is
        # nothing to overlap with when the producer is one stage.
        self._early_push_enabled = (
            connector_option(vllm_config, "push_stream", False)
            and vllm_config.parallel_config.pipeline_parallel_size > 1
        )
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
        # The base tracks a producer's request for the save path only under
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
            # `reqs_to_save` says a request's KV for this rank's layers is
            # complete and ready to leave. Host staging moves it to the host
            # buffer; the direct path has nowhere local to move it to, so the
            # readiness itself is the whole content -- and the base only
            # builds it for host staging.
            self._build_save_meta(meta, scheduler_output)
            self._early_sent.update(meta.reqs_to_save)

        # A preempted request re-prefills into these blocks and one that
        # finished on a non-terminal status hands them straight back, neither
        # of them behind the lease that protects a terminal finish.
        flush = self._early_sent & (
            set(scheduler_output.preempted_req_ids or ()) | meta.reqs_not_processed
        )
        meta.push_early_flush = flush
        self._early_sent -= flush
        meta.valid_tokens = {
            req_id: self._valid_tokens.pop(req_id)
            for req_id in meta.push_finished_blocks
            if req_id in self._valid_tokens
        }
        return meta

    def request_finished(
        self, request: "Request", block_ids: "BlockIds"
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free_blocks, out_params = super().request_finished(request, block_ids)
        if delay_free_blocks:
            # The lease now holds the blocks, so the write no longer needs
            # watching for their reuse.
            self._early_sent.discard(request.request_id)
            # The same count upstream reports as `remote_num_tokens`, taken
            # here because it is the one that matches the handed-over blocks.
            if connector_option(self.vllm_config, "chunk_mode", False):
                self._valid_tokens[request.request_id] = request.num_computed_tokens
        return delay_free_blocks, out_params
