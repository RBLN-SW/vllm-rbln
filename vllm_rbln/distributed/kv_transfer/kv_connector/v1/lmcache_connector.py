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
"""LMCache KV connector for RBLN NPU.

Uses lmcache_rbln's clean connector integration (no monkeypatching).
RBLNLMCacheConnectorV1Impl uses RBLNLMCacheManager internally, which
injects RBLNConnector for NPU <-> CPU KV cache transfer via the rebel
runtime API.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from lmcache_rbln.integration.vllm.connector import (
    RBLNLMCacheConnectorV1Impl,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RBLNLMCacheConnector(KVConnectorBase_V1):
    """LMCache connector for RBLN NPU.

    Uses RBLNLMCacheConnectorV1Impl from lmcache_rbln which internally
    creates RBLNLMCacheManager -> RBLNConnector. No monkeypatching needed.

    The rebel runtime must be injected via set_runtime() after model
    compilation (called from rbln_model_runner).
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._lmcache_engine = RBLNLMCacheConnectorV1Impl(
            vllm_config, role, self
        )
        logger.info("RBLNLMCacheConnector initialized (role=%s)", role.name)

    # ==============================
    # RBLN-specific methods
    # ==============================
    def set_runtime_holder(self, runtime_holder: list) -> None:
        """Pass the runtime_holder list reference for lazy runtime access.

        Following the PR #477 pattern: the RBLN compile backend populates
        runtime_holder after compilation.  RBLNConnector accesses
        runtime_holder[0] lazily at transfer time.

        Args:
            runtime_holder: Mutable list that will contain the rebel
                runtime after compilation.
        """
        self._lmcache_engine.set_runtime_holder(runtime_holder)

    def set_runtime(self, runtime) -> None:
        """Eagerly set the rebel runtime (legacy fallback).

        Args:
            runtime: The rebel runtime object.
        """
        self._lmcache_engine.set_runtime(runtime)

    def initialize_host_xfer_buffer(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Initialize per-layer aligned transfer buffers for D2H/H2D.

        Called by model runner after KV caches are allocated. Creates
        aligned buffers matching each layer's kv_cache shape, following
        the same pattern as RblnNixlConnectorWorker.
        """
        self._lmcache_engine.register_kv_caches(kv_caches)

    def set_host_xfer_buffer_ops(self, copy_operation) -> None:
        """Assign copy (d2h, h2d) operations when host buffer is used.

        For LMCache, the copy operations are handled internally by
        RBLNConnector via fetch_kv_cache/update_kv_cache, so this is
        a no-op. Provided for interface compatibility with
        RblnNixlConnectorWorker.
        """
        pass

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches and extract layer names for runtime API."""
        self._lmcache_engine.register_kv_caches(kv_caches)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs
    ) -> None:
        metadata = self._lmcache_engine._parent._get_connector_metadata()
        for req in metadata.requests:
            load_spec = req.load_spec
            logger.info(
                "[start_load_kv] req=%s, num_tokens=%d, "
                "load_spec=%s, save_spec=%s",
                req.req_id,
                len(req.token_ids),
                (
                    f"LoadSpec(vllm={load_spec.vllm_cached_tokens}, "
                    f"lmcache={load_spec.lmcache_cached_tokens}, "
                    f"can_load={load_spec.can_load})"
                    if load_spec
                    else "None"
                ),
                (
                    f"SaveSpec(skip={req.save_spec.skip_leading_tokens}, "
                    f"can_save={req.save_spec.can_save})"
                    if req.save_spec
                    else "None"
                ),
            )
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._lmcache_engine.get_block_ids_with_load_errors()

    def shutdown(self):
        return self._lmcache_engine.shutdown()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        num_matched = self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        logger.info(
            "[get_num_new_matched_tokens] req=%s, "
            "num_prompt_tokens=%d, num_computed_tokens=%d, "
            "num_matched=%s",
            request.request_id,
            request.num_prompt_tokens,
            num_computed_tokens,
            num_matched,
        )
        return num_matched, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        logger.info(
            "[update_state_after_alloc] req=%s, "
            "num_external_tokens=%d",
            request.request_id,
            num_external_tokens,
        )
        self._lmcache_engine.update_state_after_alloc(
            request, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return self._lmcache_engine.request_finished(request, block_ids)
