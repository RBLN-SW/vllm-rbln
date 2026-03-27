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

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_ORIGINAL_CREATE_GPU_CONNECTOR: Any | None = None


def _patched_get_vllm_torch_dev() -> tuple[Any, str]:
    return torch.cpu, "cpu"


def _patched_is_cuda_worker(metadata) -> bool:
    del metadata
    return False


def _patched_kv_layer_group_hidden_dim_size(self) -> int:
    shape = self.shape
    if len(shape) == 6:
        if shape[0] == 2 and shape[3] == 1:
            return int(shape[2] * shape[5])
        raise ValueError(f"Invalid RBLN KV shape: {shape}")
    if len(shape) == 5:
        return int(shape[3] * shape[4])
    if len(shape) == 3:
        return int(shape[2])
    raise ValueError(f"Invalid shape: {shape}")


def _patched_create_gpu_connector(config, metadata, engine):
    from lmcache.utils import EngineType
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.connector import (
        RBLNLMCacheTensorConnector,
    )

    if engine != EngineType.VLLM:
        assert _ORIGINAL_CREATE_GPU_CONNECTOR is not None
        return _ORIGINAL_CREATE_GPU_CONNECTOR(config, metadata, engine)

    unsupported: list[str] = []
    if getattr(config, "use_layerwise", False):
        unsupported.append("use_layerwise=True")
    if getattr(config, "enable_blending", False):
        unsupported.append("enable_blending=True")
    if getattr(config, "use_gpu_connector_v3", False):
        unsupported.append("use_gpu_connector_v3=True")
    if getattr(metadata, "use_mla", False):
        unsupported.append("use_mla=True")

    if unsupported:
        raise NotImplementedError(
            "RBLN LMCache CPU compatibility mode does not support: "
            + ", ".join(unsupported)
        )

    return RBLNLMCacheTensorConnector.from_metadata(
        metadata,
        use_gpu=False,
        device=torch.device("cpu"),
    )


def _run_with_timeout(
    name: str,
    target,
    timeout: float,
    errors: list[tuple[str, Any]],
) -> None:
    failure: list[Exception] = []

    def runner() -> None:
        try:
            target()
        except Exception as exc:  # pragma: no cover - defensive shutdown path
            failure.append(exc)

    logger.debug("Closing %s...", name)
    thread = threading.Thread(
        target=runner,
        name=f"rbln-lmcache-stop-{name}",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(
            "%s close operation timed out after %ss. Continuing with shutdown...",
            name,
            timeout,
        )
        errors.append((name, "Timeout"))
        return

    if failure:
        logger.error("Error closing %s: %s", name, failure[0])
        errors.append((name, failure[0]))
        return

    logger.debug("%s closed successfully", name)


def _patched_zmq_offload_server_close(self) -> None:
    from lmcache.logging import init_logger

    offload_logger = init_logger(__name__)
    offload_logger.debug("Closing ZMQOffloadServer...")
    self.running = False

    thread = getattr(self, "thread", None)
    if thread is not None and thread.is_alive():
        offload_logger.debug(
            "Skipping offload server socket close/join in RBLN CPU compatibility "
            "mode to avoid libzmq shutdown assertions; daemon thread will exit "
            "with process teardown"
        )
        return

    socket = getattr(self, "socket", None)
    if socket is not None:
        try:
            socket.close(linger=0)
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            offload_logger.warning("Error closing offload server socket: %s", exc)


def _patched_zmq_reqrep_client_transport_close(self) -> None:
    for socket in getattr(self, "sockets", []):
        try:
            socket.close(linger=0)
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            logger.warning("Error closing lookup client socket: %s", exc)
    self.sockets = []
    logger.debug(
        "Skipped terminating shared ZMQ context for lookup client in RBLN "
        "CPU compatibility mode"
    )


def _patched_initialize_usage_context(config, metadata, local_log=None):
    del config, metadata, local_log
    logger.debug(
        "Skipping LMCache usage tracking context in RBLN CPU compatibility mode"
    )
    return None


def _patched_init_scheduler_components(self) -> None:
    from lmcache.integration.vllm.utils import create_lmcache_metadata
    from lmcache.v1.lookup_client.factory import LookupClientFactory

    assert self._vllm_config is not None, "vllm_config required for vLLM mode"

    if self._config.enable_scheduler_bypass_lookup:
        self._lmcache_engine = self._create_lmcache_engine(role="scheduler")
        self._lmcache_engine_metadata = self._lmcache_engine.metadata
    else:
        self._lmcache_engine = None
        self._lmcache_engine_metadata, _ = create_lmcache_metadata(
            self._vllm_config,
            role="scheduler",
        )
        logger.debug(
            "Skipping scheduler PrometheusLogger initialization in RBLN CPU "
            "compatibility mode"
        )

    self._lookup_client = LookupClientFactory.create_lookup_client(
        self._config,
        self._lmcache_engine_metadata,
        self._lmcache_engine,
    )


def _patched_stop_services(self) -> None:
    logger.debug("Stopping LMCacheManager services...")
    start_time = time.time()
    errors: list[tuple[str, Any]] = []

    if self._health_monitor is not None:
        _run_with_timeout("health_monitor", self._health_monitor.stop, 5.0, errors)

    if self._offload_server is not None:
        _run_with_timeout("offload_server", self._offload_server.close, 10.0, errors)

    if self._runtime_plugin_launcher is not None:
        _run_with_timeout(
            "runtime_plugin_launcher",
            self._runtime_plugin_launcher.stop_plugins,
            10.0,
            errors,
        )

    if self._api_server is not None:
        _run_with_timeout("api_server", self._api_server.stop, 10.0, errors)

    if self._lookup_server is not None:
        _run_with_timeout("lookup_server", self._lookup_server.close, 10.0, errors)

    if self._lookup_client is not None:
        _run_with_timeout("lookup_client", self._lookup_client.close, 10.0, errors)

    if getattr(self, "_lmcache_engine", None) is not None:
        manager_mod = importlib.import_module("lmcache.v1.manager")
        utils_mod = importlib.import_module("lmcache.integration.vllm.utils")

        def destroy_cache_engine() -> None:
            manager_mod.LMCacheEngineBuilder.destroy(utils_mod.ENGINE_NAME)
            self._lmcache_engine = None
            self._lmcache_engine_metadata = None

        _run_with_timeout(
            "cache_engine",
            destroy_cache_engine,
            15.0,
            errors,
        )

    elapsed = time.time() - start_time
    if errors:
        logger.warning(
            "Shutdown completed with %d errors in %.2fs: %s",
            len(errors),
            elapsed,
            errors,
        )
    else:
        logger.debug("LMCacheManager services stopped successfully in %.2fs", elapsed)


def apply_lmcache_patches() -> None:
    """Patch LMCache runtime hooks for the RBLN CPU compatibility path."""
    global _PATCHED, _ORIGINAL_CREATE_GPU_CONNECTOR

    if _PATCHED:
        return

    utils = importlib.import_module("lmcache.integration.vllm.utils")
    gpu_connector = importlib.import_module("lmcache.v1.gpu_connector")
    manager = importlib.import_module("lmcache.v1.manager")
    storage_backend = importlib.import_module("lmcache.v1.storage_backend")
    storage_manager = importlib.import_module(
        "lmcache.v1.storage_backend.storage_manager"
    )
    kv_layer_groups = importlib.import_module("lmcache.v1.kv_layer_groups")
    cache_engine = importlib.import_module("lmcache.v1.cache_engine")
    usage_context = importlib.import_module("lmcache.usage_context")
    zmq_transport = importlib.import_module("lmcache.v1.rpc.zmq_transport")
    offload_server = importlib.import_module("lmcache.v1.offload_server.zmq_server")

    _ORIGINAL_CREATE_GPU_CONNECTOR = gpu_connector.CreateGPUConnector

    utils.get_vllm_torch_dev = _patched_get_vllm_torch_dev
    gpu_connector.CreateGPUConnector = _patched_create_gpu_connector
    manager.CreateGPUConnector = _patched_create_gpu_connector
    manager.LMCacheManager._init_scheduler_components = (
        _patched_init_scheduler_components
    )
    manager.LMCacheManager.stop_services = _patched_stop_services
    cache_engine.InitializeUsageContext = _patched_initialize_usage_context
    usage_context.InitializeUsageContext = _patched_initialize_usage_context
    zmq_transport.ZmqReqRepClientTransport.close = (
        _patched_zmq_reqrep_client_transport_close
    )
    offload_server.ZMQOffloadServer.close = _patched_zmq_offload_server_close
    kv_layer_groups.KVLayerGroupInfo.hidden_dim_size = property(
        _patched_kv_layer_group_hidden_dim_size
    )
    storage_backend.is_cuda_worker = _patched_is_cuda_worker
    storage_manager.is_cuda_worker = _patched_is_cuda_worker

    _PATCHED = True
