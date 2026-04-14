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
from typing import TYPE_CHECKING, Any

import torch
from rebel.kv_cache import aligned_tensor
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorScheduler,
    NixlConnectorWorker,
    ReqId,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RblnNixlConnector(NixlConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.kv_cache_config = kv_cache_config
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: RblnNixlConnectorScheduler | None = (
                RblnNixlConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: RblnNixlConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = RblnNixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )


class RblnNixlConnectorScheduler(NixlConnectorScheduler):
    """Implementation of Scheduler side methods"""

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        self.use_host_buffer = vllm_config.kv_transfer_config.kv_buffer_device == "cpu"

        self._block_ids_need_save: dict[ReqId, BlockIds] = {}

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self._reqs_need_save:
            # NOTE: For the prefill side, there might be a chance that an early added
            # request is a chunked prefill, so we need to check if new blocks are added
            for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
                req_to_save = self._reqs_need_save.get(req_id)
                if req_to_save is None:
                    continue

                # NOTE(RBLN): RBLN allocates the whole prefill blocks at once
                # and does not resume prefill requests in P/D disaggregation scenario.
                # save_to_host path will be deprecated in the future.
                has_block_ids_to_save = req_id in self._block_ids_need_save
                has_new_block_ids = new_block_id_groups is not None
                assert has_block_ids_to_save ^ has_new_block_ids

                if has_new_block_ids:
                    self._block_ids_need_save[req_id] = new_block_id_groups

                req = req_to_save

                assert req.kv_transfer_params is not None
                assert scheduler_output.num_scheduled_tokens is not None
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                is_partial = (
                    req.num_computed_tokens + num_scheduled_tokens
                ) < req.num_prompt_tokens
                if not is_partial:
                    new_block_id_groups = self._block_ids_need_save.pop(req_id)
                    clipped_block_id_groups = self.get_sw_clipped_blocks(
                        new_block_id_groups
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

        meta.reqs_to_send = self._reqs_need_send  # type: ignore[var-annotated, has-type]
        meta.reqs_in_batch = self._reqs_in_batch  # type: ignore[var-annotated, has-type]
        meta.reqs_not_processed = self._reqs_not_processed  # type: ignore[var-annotated, has-type]

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()  # type: ignore[var-annotated]
        self._reqs_not_processed = set()  # type: ignore[var-annotated]
        self._reqs_need_send = {}  # type: ignore[var-annotated]

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            self._block_ids_need_save.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds "
                "for remote decode to fetch blocks",
                request.request_id,
                envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
            )
            # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
            # trimming down after allocating for the whole sequence length. Empty
            # blocks are always at the start of the list.
            # Here we "unpad" blocks to send the actual remote blocks to be read.
            block_ids = self.get_sw_clipped_blocks(block_ids)

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )


class RblnNixlConnectorWorker(NixlConnectorWorker):
    """Implementation of Worker side methods"""

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        self.use_host_buffer = self.kv_buffer_device == "cpu"

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., RBLN)
        """
        assert self.kv_cache_layout == "HND", (
            "RBLN NIXL Connector only supports HND layout"
        )
        xfer_buffers: dict[str, torch.Tensor] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                xfer_buffers[layer_name] = aligned_tensor(kv_cache.numel()).reshape(
                    kv_cache.shape
                )
        except MemoryError as e:
            logger.error("RblnNixlConnectorWorker gets %s", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        # Set a no-op if the host buffer is not cpu.
        if self.kv_buffer_device != "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation
