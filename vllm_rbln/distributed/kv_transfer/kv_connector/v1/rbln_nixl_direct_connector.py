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
"""SR-122 Design-2 connector — direct vmem-backed NIXL registration.

This is a sibling of `RblnNixlConnector` that **does not** create a CPU
host-staging buffer and **does not** drive D2H/H2D bounce copies on
every block transfer. Instead it asks the `nixl_rbln` adapter to
register the KV cache tensors directly with NIXL's RBLN backend, so
the NIC's dmabuf MR points straight at NPU memory.

It is registered alongside `RblnNixlConnector` rather than replacing
it, so existing deployments keep working and only consumers that opt
into the direct path see the new behavior. Switch via
`kv_transfer_config.kv_connector="RblnNixlDirectConnector"`.

Requirements layered on top of `RblnNixlConnector`:

* `nixl-rbln` (this adapter) installed in the same venv. If the
  package is missing the connector still imports — the worker
  initializer is the late binding point that errors with a clear
  message, so a misconfigured environment shows up at startup, not
  inside the first transfer.
* NIC capable of `ibv_reg_dmabuf_mr` (Mellanox CX-5+ on the firmware
  levels we've seen; Intel `irdma` and Broadcom Thor decline with
  EOPNOTSUPP). See `nixl-rbln/docs/dmabuf-fd-handoff.md` for the
  design point that would remove this requirement.
* KV cache tensors must already be `PHYSICAL_VIEW_IS_LATEST` or
  `SYNCED` by the time the connector tries to register them. The
  default vllm-rbln allocation path allocates KV cache via
  `rebel::torch::rbln_v_malloc_eager`, which is exactly that state,
  so this is satisfied for the standard P/D-disaggregation flow.

What this connector deliberately does NOT do, compared to
`RblnNixlConnector`:

* No `host_xfer_buffers` allocation — no
  `rebel.kv_cache.aligned_tensor` per layer.
* No `copy_blocks` callback wiring — there's nothing to copy because
  NIXL talks to the NPU buffer directly.
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    nixl_connector as _up,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorRole,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
    RblnNixlConnector,
    RblnNixlConnectorScheduler,
    RblnNixlConnectorWorker,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlDirectConnector(RblnNixlConnector):
    """Same scheduler-side behavior as RblnNixlConnector; the
    difference is the worker, which registers KV caches directly
    instead of bouncing through a CPU staging buffer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # Do NOT call RblnNixlConnector.__init__: it asserts
        # kv_buffer_device != "rbln". The direct path *requires* "rbln",
        # so we replicate the small base body here, swapping in the
        # direct worker and the opposite assert. Scheduler-side behavior
        # is identical to the host-bounce connector, so the scheduler
        # class is reused as-is.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        assert vllm_config.kv_transfer_config.kv_buffer_device == "rbln", (
            "RblnNixlDirectConnector is device-to-device only "
            "(kv_buffer_device='rbln'). For the host-bounce path "
            "(kv_buffer_device='cpu') use RblnNixlConnector."
        )
        self.kv_cache_config = kv_cache_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = RblnNixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = RblnNixlDirectConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    def finalize_kv_cache_registration(self) -> None:
        """Run the worker's deferred NIXL registration. Invoked by
        RBLNWorker.compile_or_warm_up_model after warm-up has allocated
        the KV cache physical views (see
        RblnNixlDirectConnectorWorker.register_kv_caches for why this is
        deferred). No-op on the scheduler side."""
        if self.connector_worker is not None:
            self.connector_worker.finalize_kv_cache_registration()


class RblnNixlDirectConnectorWorker(RblnNixlConnectorWorker):
    """Worker that registers KV caches via the `nixl_rbln` adapter
    rather than allocating per-layer CPU staging buffers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        assert vllm_config.kv_transfer_config.kv_buffer_device == "rbln", (
            "RblnNixlDirectConnectorWorker requires kv_buffer_device='rbln'."
        )
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # Late import — if nixl-rbln isn't installed we want the
        # failure surface to be obviously a missing dependency, not a
        # cryptic AttributeError deep inside register_kv_caches.
        try:
            import nixl_rbln  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "RblnNixlDirectConnector requires the 'nixl-rbln' "
                "adapter package to be installed in the same venv. "
                "Install with: uv pip install -e "
                "~/codebase/nixl-rbln  "
                f"(import failed: {e})"
            ) from e

        # Direct path registers NPU memory, not host DRAM. The platform
        # reports "DRAM" (host-bounce default) so we override here.
        self.nixl_memory_type = "VRAM"
        # Register/transfer over the RBLN backend. super().__init__ has
        # already built the NixlWrapper with the *previous* backend list
        # (default ["UCX"]), so RBLN was NOT auto-created with empty
        # params — register_kv_caches creates it with the right
        # rbln_ctx_ptr via nixl_rbln.ensure_rbln_backend before use.
        self.nixl_backends = ["RBLN"]

    # ---- host-bounce removal ------------------------------------------

    def initialize_host_xfer_buffer(
        self, kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Direct path: there is no host xfer buffer. NIXL talks to
        NPU memory through the dmabuf MR the RBLN plugin built; no
        staging needed."""
        # Intentionally leave self.host_xfer_buffers empty / unset.
        # The base class default (host bounce) is what we're opting
        # out of.
        if self.use_host_buffer:
            logger.info(
                "RblnNixlDirectConnector: ignoring kv_buffer_device='cpu' "
                "— direct-vmem path registers NPU memory with NIXL "
                "directly, no host staging buffer needed.",
            )

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp) -> None:
        """No copy ops to wire — see `initialize_host_xfer_buffer`."""
        return

    # ---- device-id resolution -----------------------------------------

    def _resolve_ctx_device_id(self, kv_caches: dict[str, torch.Tensor]) -> int:
        """Find the NPU device id whose global RblnContext is live.

        The KV cache tensors are vmem-backed but appear as CPU tensors
        (device_type="cpu"), so ``tensor.get_device()`` is -1 and useless
        here. The model compile created a global RblnContext on the
        worker's physical NPU (the RBLN_DEVICES entry, which
        rbln_worker._init_device_env does NOT remap to 0). Pick the first
        candidate whose ``Context.from_key(global_key_at_device(d))`` is
        non-None.
        """
        import os

        import rebel

        _C = rebel._C

        def _ctx_at(d: int):
            try:
                return _C.Context.from_key(_C.Context.global_key_at_device(d))
            except Exception:  # noqa: BLE001
                return None

        t0 = next(iter(kv_caches.values()))
        candidates: list[int] = []
        idx = getattr(t0.device, "index", None)
        if idx is not None:
            candidates.append(int(idx))
        rbln_devices = os.environ.get("RBLN_DEVICES", "")
        if rbln_devices:
            try:
                candidates.append(int(rbln_devices.split(",")[0]))
            except ValueError:
                pass
        candidates.append(0)
        seen: set[int] = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        chosen = next((c for c in candidates if _ctx_at(c) is not None), None)
        logger.info(
            "RblnNixlDirectConnector: ctx device_id candidates=%s chosen=%s "
            "(t.device=%s get_device=%d RBLN_DEVICES=%r)",
            candidates,
            chosen,
            t0.device,
            t0.get_device(),
            rbln_devices,
        )
        if chosen is None:
            # Debug safety net: scan so the log shows which device ids
            # actually own a live context.
            live = [d for d in range(32) if _ctx_at(d) is not None]
            logger.warning(
                "RblnNixlDirectConnector: no candidate had a live "
                "RblnContext; scan found live contexts at device ids %s",
                live,
            )
            chosen = live[0] if live else None
        assert chosen is not None, (
            "RblnNixlDirectConnector: could not find a live RblnContext for "
            "any NPU device id; cannot create the RBLN NIXL backend."
        )
        return chosen

    # ---- deferred registration (post-warmup) --------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Defer the real registration until after model warm-up.

        register_kv_caches is called from initialize_kv_cache, *before*
        the warm-up forward. At that point the KV cache vmem entries are
        still EMPTY (torch.empty(device="rbln") allocates lazily) — they
        have no physical view, so the vaddr->dva translation would fail
        in ensure_synced_on_device. The physical view is only allocated
        when the compiled model runs (warm-up), which RBLNWorker drives
        in compile_or_warm_up_model *after* this call.

        So we stash the caches here and do the actual NIXL registration
        in finalize_kv_cache_registration(), which RBLNWorker invokes
        once warm-up has materialized the physical views. The KV cache
        vmem entries (hence vaddr/dva) are stable for the worker's
        lifetime, so registering post-warm-up still describes the memory
        the model reads/writes on every subsequent forward.
        """
        self._pending_kv_caches = kv_caches
        logger.info(
            "RblnNixlDirectConnector: deferring registration of %d KV "
            "cache layer(s) until after warm-up (physical views are not "
            "allocated yet).",
            len(kv_caches),
        )

    def finalize_kv_cache_registration(self) -> None:
        """Run the deferred NIXL registration. Idempotent; a no-op if
        registration already ran or there is nothing pending. Called by
        RBLNWorker after warm_up_model() has allocated the KV cache
        physical views."""
        pending = getattr(self, "_pending_kv_caches", None)
        if pending is None:
            return
        self._pending_kv_caches = None
        self._register_kv_caches_impl(pending)

    # ---- the actual direct register path ------------------------------

    def _register_kv_caches_impl(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Direct-vmem variant of NixlConnectorWorker.register_kv_caches.

        This is a faithful copy of the upstream method body (pinned to
        the installed vllm version), changed only where device-to-device
        RBLN registration needs it:

        * the RBLN NIXL backend is created with the right RblnContext
          pointer (via ``nixl_rbln.ensure_rbln_backend``) before any
          ``register_memory`` call;
        * the single base-address derivation (``cache.data_ptr()``
          upstream) is translated from a vmem vaddr to the NPU device
          address (dva), with per-region byte-offset handling for the
          K/V-split case.

        Everything downstream — ``seen_base_addresses`` ->
        ``kv_caches_base_addr``, ``caches_data`` -> ``register_memory``,
        and the transfer-side descriptor math in ``_read_blocks`` /
        ``register_local_xfer_handler`` — derives from that one point,
        so converting it once keeps register + transfer dva-consistent.

        ``nixl_memory_type`` ("VRAM") and ``nixl_backends`` (["RBLN"])
        are set on the worker in ``__init__``.
        """
        import nixl_rbln

        self.kv_topo = _up.TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,
            remote_block_size=self._block_size,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = _up.compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.kv_topo.cross_layers_blocks
        )

        # --- delta: create the RBLN backend on the RblnContext that owns
        # the KV cache vmem before registering.
        #
        # device_id resolution: under device_type="cpu" the vmem-backed
        # KV cache tensors report get_device()==-1, so we can't rely on
        # it. The global RblnContext was created by the model compile on
        # the worker's physical NPU (RBLN_DEVICES, not remapped to 0).
        # Pick the candidate device id whose global RblnContext actually
        # resolves.
        device_id = self._resolve_ctx_device_id(kv_caches)

        # Direct path never stages through a host buffer.
        assert not self.use_host_buffer
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, (
            "host_xfer_buffer should not be initialized when "
            f"kv_buffer_device is {self.kv_buffer_device}"
        )

        logger.info(
            "Registering KV_Caches (direct vmem). use_mla: %s, "
            "kv_buffer_device: %s, device_id: %d",
            self.use_mla,
            self.kv_buffer_device,
            device_id,
        )

        tensor_size_bytes = None
        # Logical K/V regions handed to nixl-rbln: each is (entry_tensor,
        # byte_offset_within_entry, full_block_len). nixl-rbln does the
        # vmem-vaddr -> NPU-dva translation, the chiplet sharding and the MR
        # registration, and returns the physical transfer tables. This
        # connector therefore never sees NPU dvas or the chiplet count — it
        # only ever deals in torch tensors and logical byte offsets.
        regions: list[tuple[Any, int, int]] = []
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs[layer_name]
            if isinstance(layer_spec, _up.UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.kv_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, _up.MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate regions.
            physical_page_size = physical_page_size // len(cache_list)
            if self.kv_topo._cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, _up.MambaSpec)
                else self.num_blocks
            )
            curr_tensor_size_bytes = num_blocks * physical_page_size
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes

            # --- delta: materialize the vmem physical view. Under
            # VLLM_RBLN_USE_DEVICE_TENSOR the KV cache is created with
            # torch.empty(device="rbln") (RBLNModelRunner.
            # _allocate_kv_cache_tensors), which leaves the vmem entry in
            # the EMPTY sub-state with no physical view until the first
            # device write. NIXL has to register real NPU memory, and
            # vaddr_to_dva -> ensure_synced_on_device requires the
            # physical view to exist. A one-time in-place zero_() forces
            # the entry to allocate it (EMPTY -> PHYSICAL_VIEW_IS_LATEST)
            # and zero-inits the cache, which is harmless. The vmem entry
            # (hence vaddr/dva) is stable for the worker's lifetime, so
            # the model's later writes land in the same registered memory.
            cache_or_caches.zero_()

            # Collect this entry's logical K/V regions. K and V are views into
            # one vmem entry (V at entry_base + K_size); pass the entry tensor
            # plus the view's byte offset so nixl-rbln converts the entry base
            # (and its chiplet shards) once.
            entry_base_vaddr = cache_or_caches.data_ptr()
            for cache in cache_list:
                region_offset = cache.data_ptr() - entry_base_vaddr
                if isinstance(layer_spec, _up.MambaSpec):
                    full_block_len = (
                        physical_page_size
                        // self._physical_blocks_per_logical_kv_block
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

        # Delegate vmem-vaddr -> NPU-dva, chiplet sharding and MR registration
        # to nixl-rbln. It registers one whole-entry MR per chiplet shard and
        # returns the physical transfer tables (base addrs + block lens),
        # already chiplet-expanded so the base connector's descriptor math is
        # correct without this connector knowing the chiplet count.
        xfer = nixl_rbln.register_kv_regions(
            self.nixl_wrapper, regions, device_id, mem=self.nixl_memory_type
        )
        self.device_id = device_id
        self.block_len_per_layer = list(xfer.block_lens)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = xfer.base_addrs
        self._registered_descs.append(xfer.reg_handle)
        assert len(self.block_len_per_layer) == len(xfer.base_addrs)

        self.num_regions = len(xfer.base_addrs)
        if self.kv_topo.is_kv_layout_blocks_first:
            self.num_regions *= 2
        self.num_descs = self.num_regions * self.num_blocks

        logger.info(
            "RblnNixlDirectConnector: registered %d transfer region(s) "
            "across %d chiplet shard(s) (K/V split).",
            self.num_regions, xfer.n_shards,
        )

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = _up.NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
        )
        assert self.compat_hash is not None
        encoder = _up.msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = _up.NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )
