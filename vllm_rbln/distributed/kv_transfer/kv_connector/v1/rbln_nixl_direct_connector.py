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

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorRole,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
    RblnNixlConnector,
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
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            # Replace the host-bounce worker with the direct one.
            self.connector_worker = RblnNixlDirectConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config,
            )


class RblnNixlDirectConnectorWorker(RblnNixlConnectorWorker):
    """Worker that registers KV caches via the `nixl_rbln` adapter
    rather than allocating per-layer CPU staging buffers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
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

    # ---- the actual direct register path ------------------------------

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register every KV cache tensor with NIXL via `nixl_rbln`.

        Called once at worker init time by the base class. Each
        tensor's `data_ptr()` is the vmem vaddr produced by
        torch_rbln; `nixl_rbln.register` translates it to the NPU
        dva, looks up the matching RblnContext, and hands NIXL a
        dmabuf-backed MR.

        This method intentionally keeps any handles it gets back on
        `self` so they live as long as the worker does. NIXL holds
        them via its own rkey table; we just need to keep the Python
        reference alive to prevent GC.
        """
        import nixl_rbln  # already imported in __init__, just for the name

        agent = self._get_or_create_agent()  # base-class helper
        device_id = self._get_device_id()    # base-class helper

        self._direct_regs: list[Any] = []
        for layer_name, tensor in kv_caches.items():
            reg = nixl_rbln.register(
                agent,
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                device_id=device_id,
                mem="VRAM",
            )
            self._direct_regs.append(reg)
            logger.debug(
                "RblnNixlDirectConnector: registered KV cache layer "
                "%s (vaddr=0x%x size=%dB) directly with NIXL",
                layer_name,
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
            )

        logger.info(
            "RblnNixlDirectConnector: %d KV cache layer(s) registered "
            "directly (no host staging)",
            len(self._direct_regs),
        )

    def deregister_kv_caches(self) -> None:
        import nixl_rbln
        agent = self._get_or_create_agent()
        for reg in getattr(self, "_direct_regs", []):
            try:
                nixl_rbln.deregister(agent, reg)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "RblnNixlDirectConnector: deregister failed: %s", e,
                )
        self._direct_regs = []

    # ---- placeholders for base-class helpers used above --------------
    #
    # `_get_or_create_agent` and `_get_device_id` are not actually
    # methods on the upstream `NixlConnectorWorker` we inherit from
    # (the upstream worker owns its agent privately). This connector
    # is checked in as a *design template* — the integration glue to
    # actually call it from inside RblnModelRunner is the next ticket,
    # and that's where these helpers will be wired up. For now, the
    # import + class structure compiles and the failure path
    # (no nixl_rbln installed) is exercised at worker init.

    def _get_or_create_agent(self):
        # upstream NixlConnectorWorker.__init__ creates a `nixl_agent`
        # and stashes it as `self.nixl_wrapper`. Reuse if present.
        agent = getattr(self, "nixl_wrapper", None)
        if agent is None:
            raise RuntimeError(
                "RblnNixlDirectConnector: no nixl_agent on the "
                "worker. Upstream NixlConnectorWorker normally owns "
                "one; this code path expects the same. If you see "
                "this in production, the upstream layout has shifted "
                "and we need to refresh the assumption.",
            )
        return agent

    def _get_device_id(self) -> int:
        # vllm-rbln currently only supports a single NPU device per
        # worker. This will need to grow when we support multi-NPU
        # workers, at which point this connector would have to register
        # each KV cache to the worker's local NPU.
        return 0
