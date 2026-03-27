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

import atexit
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.patch import (
    apply_lmcache_patches,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RBLNLMCacheConnectorV1(KVConnectorBase_V1):
    """Repo-local LMCache wrapper for the RBLN runtime.

    This wrapper reuses LMCache's vLLM adapter while constraining the runtime to
    the initial CPU compatibility mode required by the current RBLN stack.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._impl: Any | None = None
        self._kv_cache_events: Any | None = None
        self._copy_blocks_op: Any | None = None
        self._shutdown = False

        self._validate_supported_config(vllm_config)
        apply_lmcache_patches()
        self._impl = self._create_lmcache_impl(vllm_config, role)
        self._register_shutdown_hook()

    def _register_shutdown_hook(self) -> None:
        threading_atexit = getattr(threading, "_register_atexit", None)
        if callable(threading_atexit):
            threading_atexit(self.shutdown)
        else:  # pragma: no cover - fallback for unexpected runtimes
            atexit.register(self.shutdown)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._get_impl().register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type["AttentionBackend"],
    ) -> None:
        raise NotImplementedError(
            "RBLNLMCacheConnectorV1 does not support cross-layer KV caches yet"
        )

    def set_host_xfer_buffer_ops(self, copy_operation) -> None:
        self._copy_blocks_op = copy_operation

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._get_impl().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._get_impl().wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        self._get_impl().save_kv_layer(
            layer_name,
            kv_layer,
            attn_metadata,
            **kwargs,
        )

    def wait_for_save(self) -> None:
        self._get_impl().wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._get_impl().get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        method = getattr(self._get_impl(), "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()
        return set()

    def get_kv_connector_kv_cache_events(self) -> Any | None:
        from vllm.distributed.kv_events import BlockStored
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheKVEvents,
        )

        events = self._get_impl().get_kv_events()
        if not events:
            return None

        blocks = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                lora_id=e.lora_id,
                block_size=e.block_size,
                medium=e.medium,
            )
            for e in events
        ]

        lmcache_kv_events = LMCacheKVEvents(num_workers=1)
        lmcache_kv_events.add_events(blocks)
        return lmcache_kv_events

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        impl = self._impl
        self._impl = None
        if impl is not None:
            impl.shutdown()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return (
            self._get_impl().get_num_new_matched_tokens(
                request,
                num_computed_tokens,
            ),
            False,
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del blocks
        self._get_impl().update_state_after_alloc(
            request,
            num_external_tokens,
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return self._get_impl().build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheKVEvents,
        )

        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, LMCacheKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._get_impl().request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None

    def _get_impl(self) -> Any:
        assert self._impl is not None, "LMCache wrapper is not initialized"
        return self._impl

    def _create_lmcache_impl(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
    ) -> Any:
        from lmcache.integration.vllm.utils import lmcache_get_or_create_config
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        config = lmcache_get_or_create_config()
        logger.debug(
            "Initializing RBLN LMCache connector in CPU compatibility mode "
            "(chunk_size=%s, save_unfull_chunk=%s)",
            getattr(config, "chunk_size", None),
            getattr(config, "save_unfull_chunk", None),
        )
        return LMCacheConnectorV1Impl(vllm_config, role, self)

    def _validate_supported_config(self, vllm_config: "VllmConfig") -> None:
        from lmcache.integration.vllm.utils import (
            lmcache_get_or_create_config,
            mla_enabled,
        )

        config = lmcache_get_or_create_config()
        unsupported: list[str] = []

        if vllm_config.parallel_config.world_size != 1:
            unsupported.append("world_size != 1")
        if getattr(config, "enable_async_loading", False):
            unsupported.append("enable_async_loading=True")
        if getattr(config, "use_layerwise", False):
            unsupported.append("use_layerwise=True")
        if getattr(config, "enable_blending", False):
            unsupported.append("enable_blending=True")
        if mla_enabled(vllm_config.model_config):
            unsupported.append("MLA model")

        if unsupported:
            raise ValueError(
                "RBLNLMCacheConnectorV1 currently supports only single-rank, "
                "sync, non-layerwise, non-blending, non-MLA mode; got: "
                + ", ".join(unsupported)
            )

        if not getattr(config, "save_unfull_chunk", False):
            logger.warning(
                "LMCache save_unfull_chunk is disabled. Prompts shorter than "
                "lmcache.chunk_size=%s will not be stored, so early RBLN LMCache "
                "tests may show zero hits until the prompt reaches that size.",
                getattr(config, "chunk_size", None),
            )
