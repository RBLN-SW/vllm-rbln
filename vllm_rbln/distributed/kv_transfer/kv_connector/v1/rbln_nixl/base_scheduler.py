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
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnectorScheduler,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    ReqId,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RblnNixlSchedulerBase(NixlBaseConnectorScheduler):
    """Scheduler-side methods the transfer direction does not decide.

    `_build_save_meta` overrides a hook upstream calls from its own
    `build_connector_meta`, so the override belongs here, not beside a direction.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # NOTE(RBLN): the platform reports device_type "cpu" when device tensors
        # are off, which upstream reads as "no host staging" -- the very setup
        # that needs it. Decide from the requested buffer device instead.
        self.use_host_buffer = vllm_config.kv_transfer_config.kv_buffer_device == "cpu"

        # Blocks collected so far for a prefill that is still being chunked.
        self._block_ids_need_save: dict[ReqId, BlockIds] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Fetch only as much as leaves a prefill starting on a chunk boundary.

        NOTE(RBLN): a prefill chunk must begin on a multiple of the chunk size
        here, and upstream reports whatever the peer holds -- an arbitrary count,
        since a decode node advertises its computed tokens unrounded. Trim to the
        boundary below and let the rest be recomputed, the same trade upstream's
        recompute threshold makes.

        A fetch covering the whole prompt leaves nothing to prefill, so it passes
        through; the remainder decides that, not which base branch ran.
        """
        count, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        resume = num_computed_tokens + count
        if resume >= request.num_prompt_tokens:
            return count, load_async
        overshoot = resume % self.vllm_config.scheduler_config.max_num_batched_tokens
        if not overshoot:
            return count, load_async
        if overshoot >= count:
            # An async load of nothing trips the scheduler's own assertion, so
            # report no match at all rather than zero tokens to load.
            return 0, False
        return count - overshoot, load_async

    def _accumulate_blocks_to_save(
        self,
        req_id: ReqId,
        new_block_id_groups: tuple[list[int], ...] | None,
        resumed: bool,
    ) -> bool:
        """Fold a step's new blocks into what the request still has to save.

        Returns whether the stored list was reseeded rather than extended. A
        resumed request re-sends its whole list instead of a delta, so
        appending would double-count it -- and anything a caller counted off
        the old list is stale.
        """
        if new_block_id_groups is None:
            return False
        if resumed or req_id not in self._block_ids_need_save:
            self._block_ids_need_save[req_id] = tuple(
                list(group) for group in new_block_id_groups
            )
            return True
        for stored_group, new_group in zip(
            self._block_ids_need_save[req_id], new_block_id_groups
        ):
            stored_group.extend(new_group)
        return False

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Stage a prefill's blocks in one save, once every chunk has landed.

        NOTE(RBLN): upstream saves each step's new blocks as they arrive. The
        RBLN host copy moves whole blocks and transfers once, so blocks are
        accumulated here and handed over when the prefill completes.

        What it produces reads as "this request's KV for this rank's layers is
        complete and ready to leave", which is why the write path calls this
        too: host staging moves it to the host buffer, and the direct path,
        having nowhere local to move it to, writes it out.
        """
        for req_id, new_block_id_groups, resumed in yield_req_data(scheduler_output):
            req = self._reqs_need_save.get(req_id)
            if req is None:
                continue

            assert req.kv_transfer_params is not None
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

            has_block_ids_to_save = req_id in self._block_ids_need_save
            has_new_block_ids = new_block_id_groups is not None
            # Blocks follow the chunks, so the closing one often adds none.
            # Neither side holding any would stage a prefill without its KV.
            assert has_block_ids_to_save or has_new_block_ids, (
                "RBLN host-bounce save path reached with no blocks: "
                f"req_id={req_id} resumed={resumed} "
                f"num_computed={req.num_computed_tokens} "
                f"num_prompt={req.num_prompt_tokens} "
                f"num_scheduled={num_scheduled_tokens}"
            )

            self._accumulate_blocks_to_save(req_id, new_block_id_groups, resumed)

            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens

            if not is_partial:
                clipped_block_id_groups = self.get_sw_clipped_blocks(
                    self._block_ids_need_save.pop(req_id)
                )
                meta.add_new_req_to_save(
                    request_id=req_id,
                    local_block_ids=clipped_block_id_groups,
                    kv_transfer_params=req.kv_transfer_params,
                )
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        # NOTE(RBLN): an entry surviving here belongs to a prefill that ended
        # before its closing chunk, so the blocks accumulated for it are stale.
        self._block_ids_need_save.pop(request.request_id, None)
        return super().request_finished(request, block_ids)
