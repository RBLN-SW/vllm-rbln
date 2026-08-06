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

import functools
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import numpy as np
import rebel
import torch
import zmq
from rebel.kv_cache import aligned_tensor
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.tracing import extract_trace_context, instrument_manual
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlAgentMetadata,
    rbln_pp_compat_hash,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

#: Bounds for the transfer-issue bookkeeping used by nixl.wait_for_transfer.
#: Pruning only kicks in past the soft limit so the common path stays a dict pop.
_XFER_TRACKING_SOFT_LIMIT = 4096
_XFER_TRACKING_MAX_AGE_S = 600.0


class _XferRecord(NamedTuple):
    """What a request's in-flight KV receive needs for its span.

    ``issued_at`` is wall-clock: it is only ever subtracted from
    ``time.time()`` in :meth:`_emit_transfer_wait_spans` and handed to a span as
    an absolute timestamp, never mixed with a monotonic clock.

    ``trace_headers`` is the request's inbound ``traceparent``, carried here from
    the scheduler so the span lands in the request's trace. ``None`` when the
    request arrived untraced.
    """

    issued_at: float
    trace_headers: Mapping[str, str] | None


def _step_span_links() -> list | None:
    """Link to the engine step that is issuing this read, if there is one.

    Reparenting ``remote_fetch`` into the request's trace costs the step
    relationship it used to get from being nested under ``nixl.kv_transfer``. A
    link keeps it: the read belongs to one request *and* to one step, which is
    exactly the two-context case links exist for.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return None

    step_context = otel_trace.get_current_span().get_span_context()
    if not step_context.is_valid:
        return None
    return [otel_trace.Link(step_context)]


def _traced_remote_fetch(func: Callable) -> Callable:
    """Name a per-request NIXL read and put it in that request's trace.

    ``start_load_kv`` issues the reads for every request scheduled in the step,
    so the step-level ``nixl.kv_transfer`` span cannot say which request's
    transfer was slow.

    A decorator rather than a ``with`` block inside the method because the
    wrapped body is the PP-aware read path — this keeps the span concern out of
    it entirely. It lives here rather than in ``vllm_rbln.utils.tracing`` because
    it reads worker state (the step's trace headers, the in-flight bookkeeping).

    ``extract_trace_context`` returns ``None`` for an untraced request, which
    leaves OTel's default parent in place — the step's ``nixl.kv_transfer`` span,
    the right answer when there is no request trace to join.
    """

    @functools.wraps(func)
    def wrapper(self, req_id: str, meta, *args: Any, **kwargs: Any):
        trace_headers = self._rbln_step_trace_headers.get(req_id)
        self._rbln_xfer_tracking[req_id] = _XferRecord(time.time(), trace_headers)

        try:
            from opentelemetry import trace as otel_trace
        except ImportError:
            return func(self, req_id, meta, *args, **kwargs)

        tracer = otel_trace.get_tracer(func.__module__)
        with tracer.start_as_current_span(
            "remote_fetch",
            context=extract_trace_context(trace_headers),
            links=_step_span_links(),
        ) as span:
            span.set_attribute("ca.request.id", req_id)
            remote = getattr(meta, "remote", None)
            block_ids = getattr(remote, "block_ids", None)
            if block_ids is not None:
                span.set_attribute("nixl.block_count", len(block_ids))
            engine_id = getattr(remote, "engine_id", None)
            if engine_id is not None:
                span.set_attribute("nixl.remote_engine_id", str(engine_id))
            return func(self, req_id, meta, *args, **kwargs)

    return wrapper


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

    compat_hash: str | None
    xfer_handshake_metadata: NixlHandshakePayload | None

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        #: req_id → 발행 시각 + trace context. get_finished 가 완료 시각과 짝지어
        #: nixl.wait_for_transfer span 을 만든다.
        self._rbln_xfer_tracking: dict[str, _XferRecord] = {}
        #: 이번 step 에 스케줄된 req_id → trace headers. start_load_kv 가
        #: connector metadata 에서 받아 채우고, 다음 step 에 통째로 교체된다.
        self._rbln_step_trace_headers: dict[str, Mapping[str, str]] = {}

        # Pick the NIXL transport backend.
        #   * nixl-rbln installed  → use RBLN backend on both paths
        #     (host-bounce DRAM_SEG / D2D VRAM_SEG via ibv_reg{,_dmabuf}_mr).
        #   * nixl-rbln absent     → upstream defaults stand (UCX backend,
        #     DRAM only). D2D (`kv_buffer_device="rbln"`) requires the
        #     RBLN backend and is rejected here.
        try:
            import nixl_rbln  # noqa: F401

            self._use_rbln_nixl_backend = True
        except ImportError:
            self._use_rbln_nixl_backend = False

        if self._use_rbln_nixl_backend:
            self.nixl_backends = ["RBLN"]
            # D2D registers VRAM (device dmabuf); host-bounce keeps DRAM.
            if self.kv_buffer_device == "rbln":
                self.nixl_memory_type = "VRAM"
        elif self.kv_buffer_device == "rbln":
            raise RuntimeError(
                "kv_buffer_device='rbln' (D2D) requires the 'nixl-rbln' "
                "adapter package; install it or set kv_buffer_device='cpu' "
                "to fall back to the upstream NIXL (UCX) host-bounce path."
            )
        else:
            logger.info(
                "RblnNixlConnectorWorker: nixl-rbln not available — "
                "using upstream NIXL (UCX) on the host-bounce path."
            )

        # `RblnPlatform.device_type = "cpu"` makes upstream skip the host
        # buffer; restore it — NIXL cannot register RBLN device memory.
        self.use_host_buffer = self.kv_buffer_device == "cpu"

        self._pending_kv_caches: dict[str, torch.Tensor] | None = None

        # --- Pipeline-parallel (PP) P/D state (empty / inert for pp_size == 1) ---
        # Per remote producer shard, the ordered KV-cache layer names it owns,
        # keyed by engine_id -> global_rank (= pp_rank * tp_size + tp_rank,
        # == pp_rank when tensor parallelism is off).
        self._remote_shard_layer_names: defaultdict[str, dict[int, tuple[str, ...]]] = (
            defaultdict(dict)
        )
        # engine_id -> producer pp_size (discovered at handshake).
        self._remote_pp_size: dict[str, int] = {}
        # engine_id -> the producer stages (flat global ranks) whose layers this
        # rank owns; drives _read_blocks_for_req.
        self._overlapping_ranks: defaultdict[str, list[int]] = defaultdict(list)
        # Per producer shard, a local xfer dlist scoped to that shard's local
        # region subset, keyed by (engine_id, global_rank, block_size); and the
        # shard's per-region KV-group ids, keyed by (engine_id, global_rank).
        self.src_xfer_handles_by_remote: dict[tuple[str, int, int], int] = {}
        self._shard_region_group_ids: dict[tuple[str, int], tuple[int, ...]] = {}
        # Ordered local KV-cache layer names (one per layer), captured at
        # register_kv_caches.
        self.local_seen_layer_names: list[str] = []

        # Pin to logical values. Upstream would otherwise multiply by the
        # attention backend's kernel ratio, which doesn't reflect per-spec
        # ratios in hybrid models.
        self.num_blocks = self.kv_cache_config.num_blocks
        self.block_size = self.vllm_config.cache_config.block_size
        self._physical_blocks_per_logical_kv_block = 1
        self._logical_num_blocks = self.num_blocks

        # SWA-side desc layout: publish a second `sliding_window`-length
        # descriptor range alongside the Full-length range at the same
        # NIXL base addresses, so SWA groups transport only the actually-
        # populated prefix over RDMA (the runtime always pins SWA's
        # kernel slot 0 at the block's base offset). Storage stays Full-
        # tiled, host h2d/d2h still moves the full block — only
        # `register_local_xfer_handler` / `add_remote_agent` /
        # `_compute_desc_ids` consult `_sw_ratio` / `_group_specs`.
        # `_sw_ratio is None` collapses every override to upstream's
        # Full-only desc layout.
        self._group_specs: list[Any] = []
        self._sw_ratio: int | None = None
        if envs.VLLM_RBLN_NIXL_SWA_VIEW_OPT:
            self._group_specs = [
                g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
            ]
            for spec in self._group_specs:
                if not isinstance(spec, SlidingWindowSpec):
                    continue
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
            if self._sw_ratio is not None:
                logger.info(
                    "VLLM_RBLN_NIXL_SWA_VIEW_OPT=1: trimming SWA-group "
                    "RDMA payload by 1/%d (sliding_window-sized descs "
                    "alongside Full descs at shared base addrs).",
                    self._sw_ratio,
                )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Wire KV caches into NIXL.

        D2D (`kv_buffer_device="rbln"`): stash and defer; backing memory
        isn't materialized until warm-up. Backend creation happens in
        `_register_kv_caches_impl` via `nixl_rbln.register_kv_regions`.

        Host-bounce (`kv_buffer_device="cpu"`): when nixl-rbln is
        available, create the RBLN backend on the agent so upstream's
        `register_memory(..., backends=["RBLN"])` resolves. Otherwise
        fall straight through to upstream's UCX-backed registration —
        the host xfer buffers are plain DRAM in either case.
        """
        # Capture the ordered local layer names before any deferral so the PP
        # metadata publish (and the consumer-side name->region matching) can
        # use them; the D2D path re-uses these at finalize time.
        self.local_seen_layer_names = list(kv_caches.keys())
        if self.kv_buffer_device == "rbln":
            self._pending_kv_caches = kv_caches
            logger.info(
                "RblnNixlConnectorWorker (D2D): deferring registration of "
                "%d KV cache layer(s) until after warm-up.",
                len(kv_caches),
            )
            return
        if self._use_rbln_nixl_backend:
            import nixl_rbln

            nixl_rbln.ensure_rbln_backend(self.nixl_wrapper, device_id=0)
        super().register_kv_caches(kv_caches)
        # Re-wrap the base-published handshake metadata with this stage's PP
        # identity + owned layer names (no-op degrade for pp_size == 1).
        if self.xfer_handshake_metadata is not None:
            base_agent_metadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
                self.xfer_handshake_metadata.agent_metadata_bytes
            )
            self._publish_pp_handshake_metadata(base_agent_metadata, kv_caches.keys())

    def finalize_kv_cache_registration(self) -> None:
        """Run the deferred D2D registration. No-op on host-bounce and
        on re-entry (idempotent via `_pending_kv_caches`)."""
        if self._pending_kv_caches is None:
            return
        pending = self._pending_kv_caches
        self._pending_kv_caches = None
        self._register_kv_caches_impl(pending)

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Allocate one rebel-aligned host buffer per layer."""
        assert self.kv_cache_layout == "HND", (
            "RBLN NIXL Connector only supports HND layout"
        )
        xfer_buffers: dict[str, torch.Tensor] = {}

        def _aligned_like(kv_cache: torch.Tensor) -> torch.Tensor:
            """Page-aligned host buffer with `kv_cache`'s shape and dtype.
            `aligned_tensor` only knows fp16 (numpy has no bfloat16), so
            we size by byte count and view-cast to the target dtype."""
            bytes_needed = kv_cache.numel() * kv_cache.element_size()
            assert bytes_needed % 2 == 0, (
                "kv_cache byte footprint must be a multiple of 2 "
                f"(aligned_tensor backing dtype), got {bytes_needed}"
            )
            raw_fp16 = aligned_tensor(bytes_needed // 2)
            return raw_fp16.view(kv_cache.dtype).view(kv_cache.shape)

        try:
            for layer_name, kv_cache in kv_caches.items():
                xfer_buffers[layer_name] = _aligned_like(kv_cache)
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

        Overrides upstream only to drop its `device_type == "cpu"` early
        return: RblnPlatform reports `device_type == "cpu"` yet still needs
        the host-buffer copies wired up on the host-bounce path.
        """
        if self.kv_buffer_device != "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _register_kv_caches_impl(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Direct variant of NixlConnectorWorker.register_kv_caches:
        build the upstream topology, hand the logical K/V regions to
        `nixl_rbln.register_kv_regions` (address translation, sharding,
        MR reg), and feed the returned transfer tables into the base
        transfer state.
        """
        import nixl_rbln

        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )

        # Device id for the RBLN backend's RblnContext.
        sample_kv_cache = next(iter(kv_caches.values()))
        device_id = sample_kv_cache.get_device()
        assert device_id >= 0, (
            "RblnNixlConnectorWorker (D2D): KV cache is not an 'rbln' "
            "device tensor (is VLLM_RBLN_USE_DEVICE_TENSOR=1 set?)."
        )

        # Direct path never stages through a host buffer.
        assert not self.use_host_buffer
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, (
            "host_xfer_buffer should not be initialized when "
            f"kv_buffer_device is {self.kv_buffer_device}"
        )

        logger.info(
            "Registering KV_Caches (direct). use_mla: %s, "
            "kv_buffer_device: %s, device_id: %d",
            self.use_mla,
            self.kv_buffer_device,
            device_id,
        )

        tensor_size_bytes = None
        # Logical K/V regions (entry_tensor, byte_offset, full_block_len)
        # for nixl-rbln.
        regions: list[tuple[Any, int, int]] = []
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs[layer_name]
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate
            # regions.
            physical_page_size = physical_page_size // len(cache_list)
            if self.transfer_topo._cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            curr_tensor_size_bytes = num_blocks * physical_page_size
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes

            # Materialize the backing memory of kv_cache.
            cache_or_caches.zero_()

            # Collect this entry's logical K/V regions.
            entry_base_addr = cache_or_caches.data_ptr()
            for cache in cache_list:
                region_offset = cache.data_ptr() - entry_base_addr
                if isinstance(layer_spec, MambaSpec):
                    full_block_len = (
                        physical_page_size // self._physical_blocks_per_logical_kv_block
                    )
                else:
                    full_block_len = physical_page_size

                assert cache.shape[0] == num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                if not self.use_mla:
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                regions.append((cache_or_caches, region_offset, full_block_len))

        rbln_ctx_ptr = rebel.context_of(sample_kv_cache).rbln_ctx_ptr

        # Delegate sharding and MR registration to nixl-rbln. It registers
        # one whole-entry MR per shard and returns the transfer tables
        # (base addrs + block lens), already shard-expanded so the base
        # connector's descriptor math is correct without this connector
        # knowing the shard count.
        xfer = nixl_rbln.register_kv_regions(
            self.nixl_wrapper,
            regions,
            device_id,
            mem=self.nixl_memory_type,
            rbln_ctx_ptr=rbln_ctx_ptr,
        )
        self.device_id = device_id
        self.block_len_per_layer = list(xfer.block_lens)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = xfer.base_addrs
        self._registered_descs.append(xfer.reg_handle)
        assert len(self.block_len_per_layer) == len(xfer.base_addrs)

        self.num_regions = len(xfer.base_addrs)
        if self.transfer_topo.is_kv_layout_blocks_first:
            self.num_regions *= 2
        self.num_descs = self.num_regions * self.num_blocks

        logger.info(
            "RblnNixlConnectorWorker (D2D): registered %d transfer "
            "region(s) across %d shard(s) (K/V split).",
            self.num_regions,
            xfer.n_shards,
        )

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
        )
        # Publish the handshake metadata wrapped with this stage's PP identity
        # and owned layer names (degrades to no-PP for pp_size == 1).
        self._publish_pp_handshake_metadata(
            agent_metadata, self.device_kv_caches.keys()
        )

    # ------------------------------------------------------------------
    # Hybrid Full + SWA desc layout (RDMA payload only)
    # ------------------------------------------------------------------
    #
    # The runner registers one canonical Full-attention layer per HMA
    # pool, so the underlying NIXL memory regions are Full-sized
    # (block_size bytes per region per block). When
    # `VLLM_RBLN_NIXL_SWA_VIEW_OPT=1` is set and at least one group is
    # SWA, two desc ranges are published at the same base addresses:
    #
    #   [0, num_full_descs):
    #       Full-size descriptors (length `block_size`).  Read by
    #       Full-attention groups.
    #
    #   [num_full_descs, 2 * num_full_descs):
    #       SWA-size descriptors at the same base addresses (length
    #       `sliding_window`).  Read by SWA groups, which only need the
    #       first `sliding_window` bytes — the runtime always pins
    #       SWA's kernel slot 0 at the block's base offset, so the SWA
    #       payload is a contiguous prefix of the Full block.
    #
    # `_compute_desc_ids` routes per-group block lists into the
    # right range. When `_sw_ratio is None` (env off, or no SWA group,
    # or degenerate ratio == 1) every method below collapses to
    # upstream's single Full-range layout.
    #
    # Host-side h2d/d2h still copies the full block — only the over-
    # the-wire RDMA payload is trimmed.  The garbage tail SWA receives
    # back into the SWA-layer block is never read by the kernel
    # (attention reads only `[0, sliding_window)`), and the canonical
    # filter guarantees the storage's Full alias is never co-allocated
    # to the same block id (scheduler keeps Full/SWA block-id pools
    # disjoint).

    def register_local_xfer_handler(
        self,
        block_size: int,
        *,
        registered_layer_names: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        if self._sw_ratio is None:
            if registered_layer_names is None:
                # No SWA view opt, no PP shard: upstream's Full-only layout.
                return super().register_local_xfer_handler(block_size)
            # PP shard — register only this producer stage's local regions.
            return self._register_shard_local_xfer_handler(
                block_size, registered_layer_names
            )
        assert registered_layer_names is None, (
            "RBLN NIXL: SWA view-opt is not supported with pipeline parallelism"
        )
        assert self.transfer_topo is not None
        assert not self.transfer_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout (K and V in "
            "separate regions), not FlashInfer."
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba layers."

        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        num_blocks = self.num_blocks * block_size_ratio
        blocks_data: list[tuple[int, int, int]] = []

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses but `sliding_window`-sized.
        # _sw_ratio is not None here (the None case returned early above).
        length_divisors = [1, self._sw_ratio]
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
            "Created %s local blocks (%s) for engine %s rank %s",
            len(blocks_data),
            "Full + SWA" if self._sw_ratio is not None else "Full",
            self.engine_id,
            self.tp_rank,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        if self._sw_ratio is None:
            # No SWA view opt: upstream handles Full-only remote descs.
            return super().add_remote_agent(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )
        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank %s already "
                "exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][remote_tp_rank]

        # vLLM 0.22 base.add_remote_agent registers the remote engine in the
        # TransferTopology and builds its TPMapping BEFORE any
        # block_size_ratio / tp_ratio / get_engine_info lookup (used below and
        # in _validate_remote_agent_handshake). The _sw_ratio-is-None path
        # delegates to super() which does this; the SWA-view-opt path must
        # replicate the prelude or get_engine_info() raises KeyError.
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=nixl_agent_meta.block_size,
                remote_block_len=nixl_agent_meta.block_lens[0],
                remote_physical_blocks_per_logical=(
                    nixl_agent_meta.physical_blocks_per_logical_kv_block
                ),
            ),
        )
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        kv_topo = self.transfer_topo
        assert not kv_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout."
        )
        assert not self.use_mla, "RBLN NIXL connector does not support MLA."

        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        indexes_into_remote = (
            not self.transfer_topo.is_kv_replicated(engine_id) and tp_ratio > 0
        )

        # RBLN runs homogeneous TP (tp_ratio == 1) or local_tp >= remote_tp
        # (tp_ratio > 0). The remote_tp > local_tp case (tp_ratio < 0) would
        # need to logically split own regions; it is unsupported and untested,
        # so reject it instead of carrying the branch.
        assert tp_ratio >= 0, (
            "RBLN NIXL connector does not support remote TP > local TP "
            f"(tp_ratio={tp_ratio})."
        )

        blocks_data: list[tuple[int, int, int]] = []
        num_blocks = nixl_agent_meta.num_blocks

        # Two passes when SWA is present: Full descs first, then SWA descs
        # at the same base addresses (same `page_size` stride — the
        # remote tensor's physical block stride is still Full-sized),
        # shorter desc length.
        # _sw_ratio is not None here (the None case returned early above).
        length_divisors = [1, self._sw_ratio]
        for divisor in length_divisors:
            for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
                local_block_len = self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=True, mamba_view=False
                )
                remote_kv_block_len = local_block_len // block_size_ratio
                if block_size_ratio > 1:
                    local_block_len = remote_kv_block_len
                desc_len = local_block_len // divisor
                rank_offset = (
                    self.tp_rank % tp_ratio * remote_kv_block_len
                    if indexes_into_remote
                    else 0
                )
                page_size = nixl_agent_meta.block_lens[i]
                for block_id in range(num_blocks):
                    addr = base_addr + block_id * page_size + rank_offset
                    blocks_data.append((addr, desc_len, nixl_agent_meta.device_id))

        logger.debug(
            "Created %s remote blocks (%s) for dst engine %s "
            "remote rank %s local rank %s",
            len(blocks_data),
            "Full + SWA" if self._sw_ratio is not None else "Full",
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

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        if self._sw_ratio is None:
            # No SWA view opt: upstream's Full/SSM desc layout applies.
            return super()._compute_desc_ids(
                block_ids,
                dst_num_blocks,
                block_size_ratio,
                physical_blocks_per_logical,
            )

        # The SWA desc formula below indexes physical blocks directly; the
        # connector pins one physical block per logical block, so the
        # physical_blocks_per_logical argument does not apply here.
        assert physical_blocks_per_logical == 1, (
            "RBLN NIXL connector assumes physical_blocks_per_logical == 1"
        )

        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)

        region_ids = np.arange(self.num_regions)[:, None]
        num_full_descs = self.num_regions * num_blocks
        all_descs: list[np.ndarray] = []
        for g, group in enumerate(block_ids):
            if not group:
                continue
            is_sw = isinstance(self._group_specs[g], SlidingWindowSpec)
            offset = num_full_descs if is_sw else 0
            group_arr = np.asarray(group)[None, :]
            all_descs.append((region_ids * num_blocks + group_arr + offset).flatten())
        return np.concatenate(all_descs) if all_descs else np.empty(0, dtype=int)

    # ------------------------------------------------------------------
    # Pipeline-parallel (PP) P/D over NIXL
    # ------------------------------------------------------------------
    #
    # Under sync-scheduling PP the producer runs pipeline_parallel_size
    # stages, each owning a contiguous band of layers. Every stage
    # advertises its (pp_rank, tp_rank) shard and the KV-cache layer names
    # it registered; the consumer matches each shard to its own local KV
    # regions BY NAME (the owned layer range is derived locally on each
    # side, never sent over the wire) and reads each overlapping shard with
    # a shard-scoped local/remote descriptor pair. All of this collapses to
    # the upstream single-stage path for pp_size == 1.

    def _check_pp_constraints(self) -> None:
        if self.vllm_config.parallel_config.pipeline_parallel_size <= 1:
            return
        assert self.transfer_topo is not None
        if self.transfer_topo.cross_layers_blocks:
            raise RuntimeError(
                "RBLN NIXL: cross-layer-blocks mode is not supported with "
                "pipeline_parallel_size > 1."
            )
        if self._has_mamba:
            raise RuntimeError(
                "RBLN NIXL: hybrid (Mamba/SSM) models are not supported with "
                "pipeline_parallel_size > 1 over NIXL P/D."
            )
        if self._sw_ratio is not None:
            raise RuntimeError(
                "RBLN NIXL: sliding-window view-opt (VLLM_RBLN_NIXL_SWA_VIEW_OPT) "
                "is not supported with pipeline_parallel_size > 1."
            )

    def _publish_pp_handshake_metadata(
        self, base_meta: NixlAgentMetadata, registered_layer_names
    ) -> None:
        self._check_pp_constraints()
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        pp_rank = get_pp_group().rank_in_group if pp_size > 1 else 0
        pp_meta = RblnNixlAgentMetadata(
            engine_id=base_meta.engine_id,
            agent_metadata=base_meta.agent_metadata,
            device_id=base_meta.device_id,
            kv_caches_base_addr=base_meta.kv_caches_base_addr,
            num_blocks=base_meta.num_blocks,
            block_lens=base_meta.block_lens,
            kv_cache_layout=base_meta.kv_cache_layout,
            block_size=base_meta.block_size,
            ssm_sizes=base_meta.ssm_sizes,
            attn_backend_name=base_meta.attn_backend_name,
            physical_blocks_per_logical_kv_block=(
                base_meta.physical_blocks_per_logical_kv_block
            ),
            pp_rank=pp_rank,
            pp_size=pp_size,
            registered_layer_names=list(registered_layer_names),
        )
        base_hash = self.compat_hash
        assert base_hash is not None
        self.compat_hash = rbln_pp_compat_hash(base_hash)
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.Encoder().encode(pp_meta),
        )

    def _query_agent_meta(
        self, sock: "zmq.Socket", remote_rank: int, expected_engine_id: str
    ) -> RblnNixlAgentMetadata:
        sock.send(msgspec.msgpack.encode((GET_META_MSG, remote_rank)))
        handshake_payload = msgspec.msgpack.Decoder(NixlHandshakePayload).decode(
            sock.recv()
        )
        assert self.compat_hash is not None
        if (
            self.enforce_compat_hash
            and handshake_payload.compatibility_hash != self.compat_hash
        ):
            raise RuntimeError(
                "NIXL compatibility hash mismatch "
                f"(local={self.compat_hash}, "
                f"remote={handshake_payload.compatibility_hash}). Prefill and "
                "decode instances have incompatible configurations."
            )
        metadata = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            handshake_payload.agent_metadata_bytes
        )
        if metadata.engine_id != expected_engine_id:
            raise RuntimeError(
                "Remote NIXL agent engine ID mismatch. "
                f"Expected {expected_engine_id}, received {metadata.engine_id}."
            )
        return metadata

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        # Background thread needs a device context (see base _nixl_handshake).
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        assert self.transfer_topo is not None
        p_remote_tp_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        path = make_zmq_path("tcp", host, port)
        remote_rank_to_agent_name: dict[int, str] = {}

        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # ms; avoid hang on dead server

            # Bootstrap: the first shard (pp_rank 0) advertises pp_size.
            first_rank = p_remote_tp_ranks[0]
            metas = {
                first_rank: self._query_agent_meta(sock, first_rank, expected_engine_id)
            }
            pp_size = metas[first_rank].pp_size

            if pp_size > 1:
                if self._sw_ratio is not None:
                    raise RuntimeError(
                        "RBLN NIXL: sliding-window attention combined with "
                        "pipeline-parallel P/D is not supported."
                    )
                if remote_tp_size > 1:
                    raise RuntimeError(
                        "RBLN NIXL: tensor parallelism (remote_tp_size>1) "
                        "combined with pipeline-parallel P/D is not supported."
                    )

            for pp_rank in range(pp_size):
                for remote_tp_rank in p_remote_tp_ranks:
                    global_rank = pp_rank * remote_tp_size + remote_tp_rank
                    if global_rank in metas:
                        metadata = metas[global_rank]
                    else:
                        metadata = self._query_agent_meta(
                            sock, global_rank, expected_engine_id
                        )
                    names = tuple(metadata.registered_layer_names)
                    self._remote_shard_layer_names[expected_engine_id][global_rank] = (
                        names
                    )
                    if pp_size == 1:
                        # No pipeline parallelism: single-stage registration,
                        # exactly as the non-PP path (read path also delegates to
                        # base for pp_size == 1).
                        remote_rank_to_agent_name[global_rank] = self.add_remote_agent(
                            metadata, global_rank, remote_tp_size
                        )
                        continue

                    # PP: only build/register producer stages whose layers this
                    # rank actually owns. base.add_remote_agent -> _build_fa_remote
                    # indexes the *local* block_len_per_layer by the *remote*
                    # region position, so it is only safe when n_remote <= n_local.
                    # Overlapping stages satisfy that (symmetric same-stage: equal;
                    # asymmetric fan-in: remote is a sub-band of this larger band);
                    # a non-overlapping peer stage may be larger under an uneven
                    # split (e.g. 62 layers / 4 = 16/16/15/15) and would index out
                    # of range -- and this rank never reads it -- so skip it.
                    indices = self._local_region_indices_for_layer_names(names)
                    if 0 < len(indices) < len(names):
                        raise RuntimeError(
                            f"RBLN NIXL PP: producer stage {global_rank} "
                            f"({len(names)} layers) partially overlaps this "
                            f"decode rank's band ({len(indices)} owned). "
                            "The prefill pipeline-parallel size must be >= "
                            "the decode pipeline-parallel size and an integer "
                            "multiple of it."
                        )
                    if not indices:
                        continue
                    remote_rank_to_agent_name[global_rank] = self.add_remote_agent(
                        metadata, global_rank, remote_tp_size
                    )
                    self._register_shard_read_state(
                        expected_engine_id,
                        global_rank,
                        metadata.block_size,
                        names,
                    )
                    self._overlapping_ranks[expected_engine_id].append(global_rank)
        self._remote_pp_size[expected_engine_id] = pp_size
        return remote_rank_to_agent_name

    def _register_shard_read_state(
        self,
        engine_id: str,
        global_rank: int,
        block_size: int,
        registered_layer_names: tuple[str, ...],
    ) -> None:
        handle, _ = self.register_local_xfer_handler(
            block_size, registered_layer_names=registered_layer_names
        )
        self.src_xfer_handles_by_remote[(engine_id, global_rank, block_size)] = handle
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "RBLN NIXL PP currently supports a single KV-cache group"
        )
        region_ids = self._shard_local_region_ids(registered_layer_names)
        self._shard_region_group_ids[(engine_id, global_rank)] = tuple(
            0 for _ in region_ids
        )

    def _local_region_indices_for_layer_names(
        self, registered_layer_names: tuple[str, ...] | list[str]
    ) -> list[int]:
        positions_by_name: dict[str, list[int]] = defaultdict(list)
        for local_idx, layer_name in enumerate(self.local_seen_layer_names):
            positions_by_name[layer_name].append(local_idx)

        occurrences_by_name: dict[str, int] = defaultdict(int)
        local_indices: list[int] = []
        for layer_name in registered_layer_names:
            occurrence = occurrences_by_name[layer_name]
            occurrences_by_name[layer_name] += 1
            matches = positions_by_name.get(layer_name, [])
            if occurrence >= len(matches):
                continue
            local_indices.append(matches[occurrence])
        return local_indices

    def _regions_per_layer(self) -> int:
        num_layers = len(self.local_seen_layer_names)
        assert num_layers > 0 and self.num_regions % num_layers == 0, (
            f"num_regions={self.num_regions} not divisible by num_layers={num_layers}"
        )
        return self.num_regions // num_layers

    def _shard_local_region_ids(
        self, registered_layer_names: tuple[str, ...] | list[str]
    ) -> list[int]:
        rpl = self._regions_per_layer()
        layer_indices = self._local_region_indices_for_layer_names(
            registered_layer_names
        )
        return [layer_idx * rpl + k for layer_idx in layer_indices for k in range(rpl)]

    def _register_shard_local_xfer_handler(
        self, block_size: int, registered_layer_names: tuple[str, ...] | list[str]
    ) -> tuple[int, list[tuple[int, int, int]]]:
        assert self.transfer_topo is not None
        assert not self.transfer_topo.is_kv_layout_blocks_first, (
            "RBLN NIXL connector only supports FA layout (K and V in separate "
            "regions), not FlashInfer."
        )
        assert not self._has_mamba, "RBLN NIXL connector does not support Mamba."

        block_size_ratio = self.block_size // block_size
        num_blocks = self.num_blocks * block_size_ratio
        all_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        region_ids = self._shard_local_region_ids(registered_layer_names)

        blocks_data: list[tuple[int, int, int]] = []
        for region_id in region_ids:
            base_addr = all_base_addrs[region_id]
            kv_block_len = (
                self.get_backend_aware_kv_block_len(
                    layer_idx=region_id, first_split=True, mamba_view=False
                )
                // block_size_ratio
            )
            stride = self.block_len_per_layer[region_id] // block_size_ratio
            for block_id in range(num_blocks):
                blocks_data.append(
                    (base_addr + block_id * stride, kv_block_len, self.device_id)
                )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return (
            self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs),
            blocks_data,
        )

    def _get_block_descs_ids_for_shard(
        self,
        engine_id: str,
        global_rank: int,
        num_blocks: int,
        block_ids: BlockIds,
    ) -> np.ndarray:
        region_group_ids = self._shard_region_group_ids[(engine_id, global_rank)]
        desc_ids: list[np.ndarray] = []
        for region_id, group_id in enumerate(region_group_ids):
            group_arr = np.asarray(block_ids[group_id], dtype=np.int64)
            if group_arr.size == 0:
                continue
            desc_ids.append(region_id * num_blocks + group_arr)
        if not desc_ids:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(desc_ids)

    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        if getattr(nixl_agent_meta, "pp_size", 1) <= 1:
            super()._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)
            return

        assert self.transfer_topo is not None
        remote_engine_id = nixl_agent_meta.engine_id
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size == 1, (
            "PP over NIXL P/D requires TP=1 on both sides."
        )
        assert self.transfer_topo.block_size_ratio(nixl_agent_meta.block_size) == 1, (
            "PP over NIXL P/D requires equal P/D block sizes."
        )
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
        rpl = self._regions_per_layer()
        n_remote = len(nixl_agent_meta.kv_caches_base_addr)
        assert (
            n_remote > 0
            and n_remote % rpl == 0
            and n_remote <= len(self.block_len_per_layer)
        ), (
            f"PP shard advertised {n_remote} KV regions, not a valid "
            f"sub-multiple of this consumer's {len(self.block_len_per_layer)} "
            f"regions (regions/layer={rpl})."
        )

    def start_load_kv(self, metadata) -> None:
        """Pick up the trace context the scheduler attached, then load as usual.

        ``getattr`` rather than an attribute access: upstream's plain
        ``NixlConnectorMetadata`` is what ``start_load_kv`` is typed against and
        what failure-recovery paths may hand us, and a missing trace context must
        cost a span, not a KV transfer.
        """
        self._rbln_step_trace_headers = getattr(metadata, "trace_headers", None) or {}
        return super().start_load_kv(metadata)

    @_traced_remote_fetch
    def _read_blocks_for_req(self, req_id: str, meta: "ReqMeta") -> None:
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        pp_size = self._remote_pp_size.get(engine_id, 1)
        if pp_size == 1:
            return super()._read_blocks_for_req(req_id, meta)

        remote_info = self.transfer_topo.get_engine_info(engine_id)
        assert self.transfer_topo.tp_ratio(remote_info.remote_tp_size) == 1, (
            "RBLN NIXL PP read path supports TP=1 only"
        )
        assert (
            self.transfer_topo.block_size_ratio(remote_info.remote_block_size) == 1
        ), "RBLN NIXL PP read path requires equal P/D block sizes"
        remote_block_size = remote_info.remote_block_size

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids, remote_info.remote_physical_blocks_per_logical
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
        prefix_hit = len(local_block_ids) == 0
        n_prompt_blocks = sum(len(g) for g in remote_block_ids)

        if not prefix_hit:
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                local_block_ids,
                remote_block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )

        n_read_blocks = sum(len(g) for g in local_block_ids)
        logger.info(
            "PP read req %s: pp_size=%d prompt_blocks=%d read_blocks=%d "
            "prefix_skipped=%d%s",
            req_id,
            pp_size,
            n_prompt_blocks,
            n_read_blocks,
            n_prompt_blocks - n_read_blocks,
            " (full prefix hit, notif only)" if prefix_hit else "",
        )

        for global_rank in self._overlapping_ranks[engine_id]:
            if prefix_hit:
                agent_name = self._remote_agents[engine_id][global_rank]
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
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
                self._recving_transfers[req_id].append(handle)
            except Exception as e:
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=req_id,
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=engine_id,
                    remote_pp_rank=global_rank,
                )
                self._handle_failed_transfer(req_id, handle)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending, done_recving = super().get_finished()
        if self._rbln_xfer_tracking and done_recving:
            self._emit_transfer_wait_spans(done_recving)
        return done_sending, done_recving

    def _emit_transfer_wait_spans(self, done_recving: set[str]) -> None:
        """One ``nixl.wait_for_transfer`` span per completed receive.

        ``nixl.kv_transfer`` on the connector only covers the *issue* of the
        reads — measured at 0.05ms on ca2, because ``start_load_kv`` hands the
        transfer to NIXL and returns. The elapsed time is the transfer
        completing, which upstream discovers by polling ``_pop_done_transfers``
        from ``get_finished`` every engine step. Neither end is a span by itself,
        so the interval is reconstructed from issue → completion.

        A span per poll would emit thousands of empty spans and still not show
        the wait, which is why this keys off the requests that actually finished.

        The span is parented to the request's own trace context rather than to
        whatever step happens to be polling — the wait spans however many steps
        it takes, so the polling step is not its parent in any meaningful sense.

        Tracing failures are swallowed: a missing span is a lost measurement, a
        raised exception is a lost request.
        """
        finished_at = time.time()
        # A request cancelled mid-transfer never shows up in done_recving, so its
        # entry would sit here forever. Drop anything older than any plausible
        # transfer — this dict is a measurement aid, not state the transfer needs.
        if len(self._rbln_xfer_tracking) > _XFER_TRACKING_SOFT_LIMIT:
            cutoff = finished_at - _XFER_TRACKING_MAX_AGE_S
            self._rbln_xfer_tracking = {
                rid: record
                for rid, record in self._rbln_xfer_tracking.items()
                if record.issued_at >= cutoff
            }
        for req_id in done_recving:
            record = self._rbln_xfer_tracking.pop(req_id, None)
            if record is None:
                # Issued before tracing started, or a failed-setup request that
                # never went through _read_blocks_for_req.
                continue
            try:
                instrument_manual(
                    span_name="nixl.wait_for_transfer",
                    start_time=int(record.issued_at * 1e9),
                    end_time=int(finished_at * 1e9),
                    attributes={"ca.request.id": req_id},
                    context=extract_trace_context(record.trace_headers),
                )
            except Exception:  # noqa: BLE001 - tracing must not break KV transfer
                logger.debug("nixl.wait_for_transfer span emit failed", exc_info=True)
