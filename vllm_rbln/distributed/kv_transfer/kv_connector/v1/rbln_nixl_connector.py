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
from vllm.v1.kv_cache_interface import SlidingWindowSpec

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

        # Per-group spec cache used by `_get_block_descs_ids` to route
        # SWA groups to the shorter SWA desc range. Indexed by the same
        # group index the scheduler hands us in `local_block_ids` /
        # `remote_block_ids`.
        self._group_specs = [
            g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
        ]

        # Ratio of physical block_size to SWA's sliding_window — drives
        # the SWA-side desc length (`block_len // _sw_ratio`). All SWA
        # groups share the same ratio in RBLN's hybrid layout
        # (physical block_size is uniform across groups, and the SWA
        # kernel always uses `sliding_window` for its slot 0). `None`
        # signals "no SWA group present" or "ratio == 1" (degenerate
        # case where SWA descs would equal Full descs); both collapse
        # to the Full-only desc layout. `VLLM_RBLN_NIXL_SWA_VIEW_OPT=0`
        # forces this collapse even when SWA groups are present, so the
        # SWA group transports full block_size bytes per block — a
        # baseline for measuring the optimization's payoff.
        self._sw_ratio: int | None = None
        if envs.VLLM_RBLN_NIXL_SWA_VIEW_OPT:
            for spec in self._group_specs:
                if isinstance(spec, SlidingWindowSpec):
                    assert spec.block_size % spec.sliding_window == 0
                    ratio = spec.block_size // spec.sliding_window
                    if ratio == 1:
                        continue
                    if self._sw_ratio is None:
                        self._sw_ratio = ratio
                    else:
                        assert self._sw_ratio == ratio, (
                            "RBLN NIXL connector assumes a single SWA ratio "
                            f"across groups, got {self._sw_ratio} vs {ratio}"
                        )
        else:
            logger.info(
                "VLLM_RBLN_NIXL_SWA_VIEW_OPT is disabled; SWA groups will "
                "transport block_size bytes per block (baseline mode)."
            )

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

    # ------------------------------------------------------------------
    # Per-group host-xfer with view_block_size
    # ------------------------------------------------------------------
    #
    # Upstream iterates `local_block_ids` per group and calls
    # `self.copy_blocks(...)` without group context. For an HMA-shared pool
    # canonicalised to Full, the SWA group only needs `sliding_window` bytes
    # per block but would otherwise transport the full `block_size`. Override
    # both call sites to pass a per-group `view_block_size` so the runtime
    # can rescale block strides at transport time (see
    # `runtime._view_and_*_kv_cache`).

    def _group_view_block_size(self, group_idx: int) -> int:
        """SWA group -> `sliding_window`, otherwise the spec's `block_size`.

        Gated on `_sw_ratio is not None` so it stays consistent with the
        desc-range layout chosen at `__init__`: when the SWA view opt is
        off (or no SWA spec exists), descs are registered Full-sized and
        the host xfer must transport the same Full-sized blocks.
        """
        spec = self._group_specs[group_idx]
        if self._sw_ratio is not None and isinstance(spec, SlidingWindowSpec):
            return spec.sliding_window
        return spec.block_size

    def sync_recved_kv_to_device(self, req_id: str, meta: "ReqMeta") -> None:
        assert self.use_host_buffer
        assert self.copy_blocks is not None
        for group_idx, group_block_ids in enumerate(meta.local_physical_block_ids):
            self.copy_blocks(
                self.host_xfer_buffers,
                self.device_kv_caches,
                group_block_ids,
                group_block_ids,
                "h2d",
                self._group_view_block_size(group_idx),
            )

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> None:
        assert self.use_host_buffer
        assert self.copy_blocks is not None
        for req_id, meta in metadata.reqs_to_save.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            for group_idx, group_block_ids in enumerate(meta.local_physical_block_ids):
                self.copy_blocks(
                    self.device_kv_caches,
                    self.host_xfer_buffers,
                    group_block_ids,
                    group_block_ids,
                    "d2h",
                    self._group_view_block_size(group_idx),
                )

    # ------------------------------------------------------------------
    # Hybrid Full + SWA desc layout
    # ------------------------------------------------------------------
    #
    # The runner registers one canonical Full-attention layer per HMA
    # pool, so the underlying NIXL memory regions are Full-attention
    # sized (block_size bytes per region per block). On top of those
    # regions we publish two desc ranges that share the same base
    # addresses:
    #
    #   [0, num_full_descs):
    #       Full-size descriptors — one per region per block, length
    #       block_size. Read by Full-attention groups.
    #
    #   [num_full_descs, 2 * num_full_descs):
    #       SWA-size descriptors — same base addresses, length
    #       sliding_window. Read by SWA groups, which only need the
    #       first `sliding_window` bytes (the in-place ring buffer
    #       always pins kernel slot 0 at the block's base offset, so
    #       the SWA payload is a contiguous prefix of the Full block).
    #
    # `_get_block_descs_ids` dispatches per group: SWA groups pick from
    # the upper range, Full groups from the lower range. One
    # `make_prepped_xfer` call per request, total bytes match what the
    # kernel actually needs.

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        assert self.kv_topo is not None
        assert not self.kv_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout (K and V in "
            "separate regions), not FlashInfer."
        )
        assert not self._has_mamba, (
            "RBLN NIXL connector does not support Mamba layers."
        )

        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        num_blocks = self.num_blocks * block_size_ratio
        blocks_data: list[tuple[int, int, int]] = []

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses but `sliding_window`-sized.
        length_divisors = [1] if self._sw_ratio is None else [1, self._sw_ratio]
        for divisor in length_divisors:
            for i, base_addr in enumerate(local_base_addresses):
                kv_block_len = (
                    self.get_backend_aware_kv_block_len(
                        layer_idx=i, first_split=True, mamba_view=False
                    )
                    // block_size_ratio
                    // divisor
                )
                stride = self.block_len_per_layer[i] // block_size_ratio
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * stride
                    blocks_data.append((addr, kv_block_len, self.device_id))

        logger.debug(
            "Created %s local blocks (Full + SWA) for engine %s rank %s",
            len(blocks_data),
            self.engine_id,
            self.tp_rank,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs), blocks_data

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank %s already "
                "exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        if engine_id not in self._block_size:
            self._block_size[engine_id] = nixl_agent_meta.block_size

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        assert self.kv_topo is not None
        kv_topo = self.kv_topo
        assert not kv_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout."
        )
        assert not self.use_mla, "RBLN NIXL connector does not support MLA."

        block_size_ratio = kv_topo.block_size_ratio_from_engine_id(engine_id)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(engine_id)
        indexes_into_remote = (
            not self.kv_topo.replicates_kv_cache(engine_id) and tp_ratio > 0
        )

        # Heterogeneous TP path (P TP > D TP): logically split own regions
        # into |tp_ratio| chunks. Mirrors upstream; preserved verbatim
        # because RBLN may run heterogeneous TP in the future.
        if (
            tp_ratio < 0
            and not self.use_mla
            and tp_ratio not in self.src_xfer_handles_by_tp_ratio
        ):
            self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
            for i in range(-tp_ratio):
                split_blocks_data = []
                for memory_region in self.src_blocks_data:
                    addr, local_block_len, own_tp_rank = memory_region
                    remote_block_len = local_block_len // (-tp_ratio)
                    addr = addr + i * remote_block_len
                    split_blocks_data.append((addr, remote_block_len, own_tp_rank))
                descs = self.nixl_wrapper.get_xfer_descs(
                    split_blocks_data, self.nixl_memory_type
                )
                handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
                self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)

        blocks_data: list[tuple[int, int, int]] = []
        num_blocks = nixl_agent_meta.num_blocks

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses (same `page_size` stride — the remote
        # tensor's physical block stride is still Full-sized), shorter
        # desc length.
        length_divisors = [1] if self._sw_ratio is None else [1, self._sw_ratio]
        for divisor in length_divisors:
            for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
                local_block_len = self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=True, mamba_view=False
                )
                remote_kv_block_len = local_block_len // block_size_ratio
                if block_size_ratio > 1:
                    local_block_len = remote_kv_block_len
                if tp_ratio < 0 and not self.use_mla:
                    local_block_len = local_block_len // (-tp_ratio)
                desc_len = local_block_len // divisor
                rank_offset = (
                    self.tp_rank % tp_ratio * remote_kv_block_len
                    if indexes_into_remote
                    else 0
                )
                page_size = nixl_agent_meta.block_lens[i]
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * page_size + rank_offset
                    blocks_data.append(
                        (addr, desc_len, nixl_agent_meta.device_id)
                    )

        logger.debug(
            "Created %s remote blocks (Full + SWA) for dst engine %s "
            "remote rank %s local rank %s",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

        if block_size_ratio > 1:
            self.src_xfer_handles_by_block_size[nixl_agent_meta.block_size] = (
                self.register_local_xfer_handler(nixl_agent_meta.block_size)[0]
            )

        return remote_agent_name

    def _get_block_descs_ids(
        self,
        engine_id: str,
        block_ids: BlockIds,
        block_size_ratio: float | None = None,
    ) -> np.ndarray:
        num_blocks = self.dst_num_blocks[engine_id]
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)

        region_ids = np.arange(self.num_regions)[:, None]

        if self._sw_ratio is None:
            # Pure-Full layout: single desc range, matches upstream.
            ids = np.concatenate(block_ids)[None, :]
            return (region_ids * num_blocks + ids).flatten()

        num_full_descs = self.num_regions * num_blocks
        all_descs: list[np.ndarray] = []
        for g, group in enumerate(block_ids):
            if not group:
                continue
            is_sw = isinstance(self._group_specs[g], SlidingWindowSpec)
            offset = num_full_descs if is_sw else 0
            group_arr = np.asarray(group)[None, :]
            all_descs.append(
                (region_ids * num_blocks + group_arr + offset).flatten()
            )
        if not all_descs:
            return np.array([], dtype=np.int64)
        return np.concatenate(all_descs)

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
