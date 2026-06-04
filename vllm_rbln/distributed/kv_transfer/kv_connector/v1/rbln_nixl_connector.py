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

import numpy as np
import torch
from rebel.kv_cache import aligned_tensor
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
    NixlAgentMetadata,
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorScheduler,
    NixlConnectorWorker,
    ReqId,
)
from vllm.v1.core.sched.output import SchedulerOutput

import vllm_rbln.rbln_envs as envs
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
        assert vllm_config.kv_transfer_config.kv_buffer_device != "rbln", (
            "RblnNixlConnector is host-bounce only (kv_buffer_device='cpu'). "
            "For device-to-device (kv_buffer_device='rbln') use "
            "RblnNixlDirectConnector."
        )
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
    """RBLN's KV connector worker.

    The runner filters `kv_caches` to one Full-attention canonical layer
    per HMA pool before `register_kv_caches`, so upstream's
    `cache.shape[0] == num_blocks` invariant holds without a bigger
    override (see `RBLNModelRunner._select_canonical_kv_layers_per_pool`).

    Not supported: pure-SWA single-group with `sliding_window < block_size`
    under a KV connector — the canonical-layer fallback picks the SWA
    layer (kernel granularity), whose `cache.shape[0]` mismatches
    `num_blocks`. Non-disagg serving is unaffected.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        # NOTE: the kv_buffer_device guard now lives on the *connector*
        # __init__ (RblnNixlConnector rejects "rbln", RblnNixlDirectConnector
        # requires it). Keeping it off the worker lets
        # RblnNixlDirectConnectorWorker subclass this worker without
        # inheriting an assert that would reject its own ("rbln") buffer.
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # `RblnPlatform.device_type = "cpu"` makes upstream skip the host
        # buffer; restore it — NIXL cannot register RBLN device memory.
        self.use_host_buffer = self.kv_buffer_device == "cpu"

        # Pin to logical values. Upstream would otherwise multiply by the
        # attention backend's kernel ratio, which doesn't reflect per-spec
        # ratios in hybrid models.
        self.num_blocks = self.kv_cache_config.num_blocks
        self.block_size = self.vllm_config.cache_config.block_size
        self._physical_blocks_per_logical_kv_block = 1
        self._logical_num_blocks = self.num_blocks
        self._block_size[self.engine_id] = self.block_size

        # Emulation toggles short-circuit actual KV data movement, so any
        # generated tokens are not the result of a real prefill→decode KV
        # transfer. Warn loudly at worker startup so a stale `1` left in the
        # env doesn't silently produce garbage outputs in a real run.
        if envs.VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP:
            logger.warning(
                "VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP=1: h2d/d2h host xfer "
                "copies are stubbed out. KV data is not moved between host "
                "buffers and device memory — inference outputs will be "
                "incorrect. Dev-only; unset for real runs."
            )
        if envs.VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP:
            logger.warning(
                "VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP=1: NIXL RDMA `READ` "
                "is skipped. KV blocks are not fetched from the remote peer "
                "— inference outputs will be incorrect. Dev-only; unset for "
                "real runs."
            )

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Allocate one rebel-aligned host buffer per layer.

        Under `VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP=1` or
        `VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP=1`, all layers share a single
        allocation: buffer content is never read/written in emulation
        (copy_blocks and RDMA both no-op), and upstream NIXL's
        `register_kv_caches` dedups same-`data_ptr` views via its HMA
        path, so registration completes without errors.

        The shared-allocation path asserts uniform shape across all
        layers, which is the only case the canonical-layer filter
        currently produces. Any future deviation will surface here
        rather than silently waste/leak memory.
        """
        assert self.kv_cache_layout == "HND", (
            "RBLN NIXL Connector only supports HND layout"
        )
        xfer_buffers: dict[str, torch.Tensor] = {}

        emulate = (
            envs.VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP
            or envs.VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP
        )
        try:
            if emulate and kv_caches:
                # All filtered layers must share the same shape — assert
                # rather than handle heterogeneous shapes implicitly.
                items = list(kv_caches.items())
                first_name, first_cache = items[0]
                first_shape = first_cache.shape
                for name, kv in items[1:]:
                    assert kv.shape == first_shape, (
                        "Emulation expects uniform host_xfer_buffer shape "
                        "across layers; got "
                        f"{tuple(first_shape)} for {first_name!r} vs "
                        f"{tuple(kv.shape)} for {name!r}"
                    )
                shared = aligned_tensor(first_cache.numel()).reshape(first_shape)
                for layer_name, _ in items:
                    xfer_buffers[layer_name] = shared
                logger.info(
                    "Emulation: host xfer buffers share one %.1f MB "
                    "allocation across %d layer(s).",
                    first_cache.numel() * shared.element_size() / (1024 * 1024),
                    len(kv_caches),
                )
            else:
                for layer_name, kv_cache in kv_caches.items():
                    xfer_buffers[layer_name] = aligned_tensor(
                        kv_cache.numel()
                    ).reshape(kv_cache.shape)
        except MemoryError as e:
            logger.error("RblnNixlConnectorWorker gets %s", e)
            raise

        keys_preview = list(xfer_buffers.keys())
        if len(keys_preview) > 8:
            keys_preview = keys_preview[:4] + ["..."] + keys_preview[-4:]
        logger.info(
            "Host xfer buffers allocated: %d pool(s) (keys e.g. %s)",
            len(xfer_buffers),
            keys_preview,
        )

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used.

        Under `VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP=1`, the caller-supplied
        `copy_operation` is replaced with a no-op so upstream's
        `sync_recved_kv_to_device` / `save_kv_to_host` become free.
        """
        if self.kv_buffer_device != "cpu":
            return
        assert self.use_host_buffer
        if envs.VLLM_RBLN_NIXL_EMULATE_HOST_XFER_NOOP:
            self.copy_blocks = self._noop_copy_blocks
        else:
            self.copy_blocks = copy_operation

    @staticmethod
    def _noop_copy_blocks(*args, **kwargs) -> None:
        """No-op stand-in for the h2d/d2h callback under emulation."""
        return None

    def _read_blocks(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        remote_rank: int,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> None:
        """Override to optionally short-circuit the NIXL RDMA `READ` for
        cost-isolation measurements.

        Under `VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP=1`:
          * Skip `make_prepped_xfer` + `transfer` (no data movement).
          * `send_notif` to P-side so its sender blocks release normally
            (avoids waiting on `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`).
          * Touch `self._recving_transfers[request_id]` so the next
            `_pop_done_transfers` poll reports this request as done.

        Otherwise delegate to upstream as a normal RDMA-backed transfer.
        """
        if not envs.VLLM_RBLN_NIXL_EMULATE_REMOTE_XFER_NOOP:
            return super()._read_blocks(
                local_block_ids,
                remote_block_ids,
                dst_engine_id,
                request_id,
                remote_request_id,
                remote_rank,
                local_xfer_side_handle,
                remote_xfer_side_handle,
            )

        # Match upstream notif_id format so P-side's `_get_new_notifs`
        # correlates the completion against its in-flight send list.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()
        agent_name = self._remote_agents[dst_engine_id][remote_rank]
        try:
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
        except Exception as e:
            self._log_failure(
                failure_type="notification_failed",
                msg="P worker blocks will be freed after timeout.",
                req_id=request_id,
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
                remote_agent_name=agent_name,
            )
            self.xfer_stats.record_failed_notification()

        # `_recving_transfers` is a defaultdict[list]; touching the key
        # creates an empty handle list, which `_pop_done_transfers` reports
        # as completed on the next `get_finished` poll.
        _ = self._recving_transfers[request_id]
