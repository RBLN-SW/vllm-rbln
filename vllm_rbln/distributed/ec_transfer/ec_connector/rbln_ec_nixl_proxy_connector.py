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

"""
RblnECNixlProxyConnector: Proxy-mediated NIXL EC connector for N:M topology.

Architecture
------------
Supports N encoders and M llms via a proxy that routes NIXL metadata.
Actual tensor data is transferred via NIXL (CPU-to-CPU); only lightweight
metadata (~hundreds of bytes) flows through the proxy.

  Encoder  -> runs the vision encoder
            -> registers encoder output tensors with NIXL
            -> returns NIXL metadata via request_finished() → HTTP response

  Proxy     -> receives NIXL metadata from encoder HTTP response
            -> selects a llm (random / round-robin / etc.)
            -> deposits metadata to the chosen llm's ZMQ PULL socket

  LLM  -> binds a ZMQ PULL socket (receives metadata deposits from proxy)
            -> background thread queues incoming metadata
            -> main thread: add_remote_agent, initiates NIXL pull

Transfer flow per request
-------------------------
  1. Proxy sends encode request to Encoder P_i
  2. Encoder encodes image, registers tensors with NIXL
  3. Encoder returns HTTP response containing ec_transfer_params
     (serialised NIXL metadata: agent_metadata + tensor addresses)
  4. Proxy extracts ec_transfer_params, selects LLM C_j
  5. Proxy sends original request to LLM C_j (HTTP)
  6. Proxy pushes NIXL metadata to LLM C_j's PULL socket (ZMQ)
  7. LLM background PULL thread receives metadata, queues it
  8. LLM main thread: drains queue, add_remote_agent, NIXL pull
  9. NIXL transfers tensor data (CPU -> CPU)
  10. LLM runs prefill + decode

Comparison with other connectors
---------------------------------
  RblnECNixlConnector:  ZMQ PUSH/PULL direct, NIXL data   (N:1, ~0ms)
  RblnECNixlProxyConnector: Proxy-routed metadata, NIXL data  (N:M, ~3-15ms)

Port layout
-----------
  Each llm binds:
    - pull_port: ZMQ PULL (receives NIXL metadata deposits from proxy)

  Encoder has no ZMQ sockets (metadata returned via HTTP).

Configuration example
---------------------
  # Encoder (no ZMQ config needed):
  ECTransferConfig(
      ec_connector="RblnECNixlProxyConnector",
      ec_role="ec_producer",
      ec_connector_extra_config={
          "backends": ["UCX"],
      },
  )

  # LLM (binds PULL port for metadata deposits):
  ECTransferConfig(
      ec_connector="RblnECNixlProxyConnector",
      ec_role="ec_consumer",
      ec_connector_extra_config={
          "pull_host": "0.0.0.0",
          "pull_port": 16200,
          "backends": ["UCX"],
      },
  )

Proxy integration
-----------------
  The proxy must:
    1. Extract ``ec_transfer_params`` from the encoder's HTTP response.
    2. Create a ZMQ PUSH socket connected to the chosen llm's PULL port.
    3. Forward the raw bytes from ``ec_transfer_params`` via zmq.send().

  The ``ec_transfer_params`` value is a msgpack-encoded ``ECNixlProxyMetadata``
  struct that the llm connector knows how to decode.  The proxy does NOT
  need to parse it — just forward the opaque bytes.

  Example proxy pseudocode::

      # After receiving encoder response:
      ec_params = encoder_response["ec_transfer_params"]

      # Select llm and forward:
      llm = random.choice(llms)
      push_sock = get_push_socket(llm)  # cached per llm
      push_sock.send(ec_params)              # forward opaque bytes

      # Send request to llm via HTTP as usual:
      llm.submit(request)
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from rebel.kv_cache import aligned_tensor
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.example_connector import MMMeta
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PULL_HOST = "0.0.0.0"
_DEFAULT_PULL_PORT = 16200
_CACHE_WAIT_TIMEOUT_S = 30.0
_XFER_POLL_INTERVAL_S = 0.001

# ---------------------------------------------------------------------------
# Wire protocol (msgspec) — same struct used for HTTP response and ZMQ deposit
# ---------------------------------------------------------------------------


class ECNixlProxyTensorInfo(msgspec.Struct):
    """Per-tensor metadata."""
    key: str
    base_addr: int
    nbytes: int
    device_id: int
    shape: list[int]
    dtype_str: str


class ECNixlProxyMetadata(msgspec.Struct):
    """NIXL metadata produced by the encoder, forwarded by the proxy.

    This struct is:
      - Serialised by the encoder and returned as ``ec_transfer_params``
      - Forwarded as opaque bytes by the proxy
      - Deserialised by the llm to initiate NIXL pull
    """
    engine_id: str
    mm_hash: str
    agent_metadata: bytes
    tensors: list[ECNixlProxyTensorInfo]


# ---------------------------------------------------------------------------
# Connector metadata (scheduler -> worker)
# ---------------------------------------------------------------------------


@dataclass
class ECNixlProxyConnectorMetadata(ECConnectorMetadata):
    """Metadata passed from scheduler to worker each step."""
    mm_datas_to_load: list[MMMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler-side implementation
# ---------------------------------------------------------------------------


class RblnECNixlProxyConnectorScheduler(ECConnectorBase):
    """Scheduler-side: tracks which mm_hashes the llm needs to load."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._mm_datas_need_loads: dict[str, int] = {}

    def has_cache_item(self, identifier: str) -> bool:
        return self.is_consumer

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        if not self.is_consumer:
            return
        mm_hash = request.mm_features[index].identifier
        num_tokens = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_tokens

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> ECNixlProxyConnectorMetadata:
        meta = ECNixlProxyConnectorMetadata()
        for mm_hash, num_tokens in self._mm_datas_need_loads.items():
            meta.mm_datas_to_load.append(
                MMMeta.make_meta(mm_hash, num_tokens)
            )
        self._mm_datas_need_loads.clear()
        return meta

    # -- Not used on scheduler side --

    def start_load_caches(self, encoder_cache: dict[str, Any], **kwargs) -> None:
        raise RuntimeError("start_load_caches must be called on the worker")

    def save_caches(self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs) -> None:
        raise RuntimeError("save_caches must be called on the worker")


# ---------------------------------------------------------------------------
# Worker-side implementation
# ---------------------------------------------------------------------------


class RblnECNixlProxyConnectorWorker(ECConnectorBase):
    """Worker-side EC connector for proxy-mediated N:M topology.

    Encoder:
      - Registers tensors with NIXL on save_caches().
      - Stores serialised metadata per mm_hash.
      - Returns metadata via request_finished() for the proxy to forward.
      - No ZMQ sockets.

    LLM:
      - Binds a ZMQ PULL socket for metadata deposits from the proxy.
      - Background thread receives deposits, queues them.
      - Main thread drains queue, registers NIXL agents, initiates pulls.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        try:
            from nixl._api import nixl_agent as NixlAgent
        except ImportError as e:
            raise RuntimeError(
                "NIXL is not available. Install the nixl package to use "
                "RblnECNixlProxyConnector."
            ) from e

        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None

        self._backends: list[str] = ec_cfg.get_from_extra_config(
            "backends", ["UCX"]
        )
        self._engine_id: str = str(uuid.uuid4())
        self._nixl_agent = NixlAgent(self._engine_id, None)
        self._stop_event = threading.Event()
        self._encoder = msgspec.msgpack.Encoder()

        # -- Encoder state --
        self._registered_caches: dict[str, dict[str, torch.Tensor]] = {}
        self._registered_descs: dict[str, Any] = {}
        # mm_hash -> serialised ECNixlProxyMetadata bytes (for HTTP response)
        self._pending_ec_params: dict[str, bytes] = {}

        # -- LLM state --
        self._incoming_metadata: queue.Queue[ECNixlProxyMetadata] = queue.Queue()
        self._cache_events: dict[str, threading.Event] = {}
        self._cache_events_lock = threading.Lock()
        self._remote_agents: dict[str, str] = {}
        self._tensor_registry: dict[
            str, tuple[str, list[ECNixlProxyTensorInfo]]
        ] = {}
        self._pending_loads: dict[
            str, tuple[Any, dict[str, torch.Tensor], Any]
        ] = {}
        self._encoder_cache: dict[str, Any] | None = None

        if self.is_consumer:
            pull_host: str = ec_cfg.get_from_extra_config(
                "pull_host", _DEFAULT_PULL_HOST
            )
            pull_port: int = ec_cfg.get_from_extra_config(
                "pull_port", _DEFAULT_PULL_PORT
            )
            self._pull_addr = f"tcp://{pull_host}:{pull_port}"
            self._zmq_ctx = zmq.Context()
            self._pull_sock = self._zmq_ctx.socket(zmq.PULL)
            self._pull_sock.setsockopt(zmq.RCVHWM, 64)
            self._pull_sock.bind(self._pull_addr)

            self._receiver_thread = threading.Thread(
                target=self._receiver_loop,
                daemon=True,
                name="ec_nixl_proxy_receiver",
            )
            self._receiver_thread.start()
            logger.info(
                "RblnECNixlProxyConnector (llm): PULL bound on %s",
                self._pull_addr,
            )

        if self.is_producer:
            logger.info(
                "RblnECNixlProxyConnector (encoder): no ZMQ sockets "
                "(metadata returned via ec_transfer_params)"
            )

    # ------------------------------------------------------------------
    # LLM: background PULL receiver (receives deposits from proxy)
    # ------------------------------------------------------------------

    def _receiver_loop(self) -> None:
        """Background thread: receive NIXL metadata deposits from proxy."""
        poller = zmq.Poller()
        poller.register(self._pull_sock, zmq.POLLIN)
        decoder = msgspec.msgpack.Decoder(ECNixlProxyMetadata)

        while not self._stop_event.is_set():
            events = dict(poller.poll(200))
            if self._pull_sock not in events:
                continue
            try:
                raw = self._pull_sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue

            try:
                meta = decoder.decode(raw)
                self._incoming_metadata.put(meta)

                with self._cache_events_lock:
                    evt = self._cache_events.get(meta.mm_hash)
                    if evt is not None:
                        evt.set()

                logger.debug(
                    "EC NixlProxy: received metadata deposit for "
                    "mm_hash=%s from engine=%s (%d tensors)",
                    meta.mm_hash, meta.engine_id, len(meta.tensors),
                )
            except Exception as exc:
                logger.warning(
                    "EC NixlProxy: failed to decode deposited metadata: %s",
                    exc,
                )

    def _process_pending_metadata(self) -> set[str]:
        """Drain metadata queue, register NIXL remote agents.

        Must be called from main thread (NIXL ops are not thread-safe).
        """
        new_mm_hashes: set[str] = set()

        while True:
            try:
                meta = self._incoming_metadata.get_nowait()
            except queue.Empty:
                break

            engine_id = meta.engine_id
            mm_hash = meta.mm_hash

            if engine_id in self._remote_agents:
                self._nixl_agent.remove_remote_agent(
                    self._remote_agents[engine_id]
                )
            self._remote_agents[engine_id] = (
                self._nixl_agent.add_remote_agent(meta.agent_metadata)
            )

            self._tensor_registry[mm_hash] = (engine_id, meta.tensors)
            new_mm_hashes.add(mm_hash)

            logger.debug(
                "EC NixlProxy: registered remote agent for engine=%s, "
                "mm_hash=%s",
                engine_id, mm_hash,
            )

        return new_mm_hashes

    def _get_or_create_event(self, mm_hash: str) -> threading.Event:
        with self._cache_events_lock:
            evt = self._cache_events.get(mm_hash)
            if evt is None:
                evt = threading.Event()
                self._cache_events[mm_hash] = evt
            return evt

    def _wait_for_cache(self, mm_hash: str) -> bool:
        """Wait until metadata for mm_hash arrives via proxy deposit."""
        self._process_pending_metadata()
        if mm_hash in self._tensor_registry:
            return True

        evt = self._get_or_create_event(mm_hash)
        deadline = time.monotonic() + _CACHE_WAIT_TIMEOUT_S

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            evt.clear()
            evt.wait(timeout=min(remaining, 1.0))
            self._process_pending_metadata()
            if mm_hash in self._tensor_registry:
                return True

        logger.warning(
            "EC NixlProxy: timed out waiting for mm_hash=%s (%.1fs)",
            mm_hash, _CACHE_WAIT_TIMEOUT_S,
        )
        return False

    # ------------------------------------------------------------------
    # NIXL pull (llm)
    # ------------------------------------------------------------------

    def _initiate_pull(
        self, mm_hash: str,
    ) -> tuple[Any, dict[str, torch.Tensor], Any]:
        """Allocate local buffers and start async NIXL pull."""
        engine_id, tensor_infos = self._tensor_registry[mm_hash]
        remote_agent_name = self._remote_agents[engine_id]

        local_bufs: dict[str, torch.Tensor] = {}
        local_reg_data: list[tuple[int, int, int, str]] = []
        local_xfer_data: list[tuple[int, int, int]] = []
        remote_xfer_data: list[tuple[int, int, int]] = []

        for tinfo in tensor_infos:
            numel = tinfo.nbytes // _dtype_size(tinfo.dtype_str)
            buf = aligned_tensor(numel)
            if tinfo.shape:
                buf = buf.reshape(tinfo.shape)
            local_bufs[tinfo.key] = buf
            local_reg_data.append((buf.data_ptr(), tinfo.nbytes, 0, ""))
            local_xfer_data.append((buf.data_ptr(), tinfo.nbytes, 0))
            remote_xfer_data.append((
                tinfo.base_addr, tinfo.nbytes, tinfo.device_id,
            ))

        num_descs = len(local_xfer_data)

        local_descs = self._nixl_agent.get_reg_descs(local_reg_data, "DRAM")
        self._nixl_agent.register_memory(local_descs, backends=self._backends)

        local_prepped = self._nixl_agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_xfer_data, "DRAM"
        )
        remote_prepped = self._nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_data, "DRAM"
        )

        indices = list(range(num_descs))
        handle = self._nixl_agent.make_prepped_xfer(
            "READ", local_prepped, indices, remote_prepped, indices,
            notif_msg=b"",
        )
        status = self._nixl_agent.transfer(handle)
        if status not in ("DONE", "PROC"):
            raise RuntimeError(
                f"EC NixlProxy: transfer initiation failed for "
                f"mm_hash={mm_hash}"
            )

        return handle, local_bufs, local_descs

    # ------------------------------------------------------------------
    # ECConnectorBase interface — worker side
    # ------------------------------------------------------------------

    def save_caches(
        self,
        encoder_cache: dict[str, Any],
        mm_hash: str,
        **kwargs,
    ) -> None:
        """Encoder: register tensors with NIXL and store metadata.

        No ZMQ send — metadata is returned via request_finished() for
        the proxy to pick up and forward.
        """
        if not self.is_producer:
            return
        if mm_hash in self._registered_caches:
            return

        raw: dict[str, torch.Tensor] = encoder_cache[mm_hash]
        if not isinstance(raw, dict):
            raw = {"inputs_embeds": raw}

        aligned: dict[str, torch.Tensor] = {}
        caches_data: list[tuple[int, int, int, str]] = []
        tensor_infos: list[ECNixlProxyTensorInfo] = []

        for key, value in raw.items():
            if not isinstance(value, torch.Tensor):
                aligned[key] = value
                continue

            t = value.detach().cpu()
            buf = aligned_tensor(t.numel()).reshape(t.shape)
            buf.copy_(t)
            aligned[key] = buf

            nbytes = buf.numel() * buf.element_size()
            caches_data.append((buf.data_ptr(), nbytes, 0, ""))
            tensor_infos.append(ECNixlProxyTensorInfo(
                key=key,
                base_addr=buf.data_ptr(),
                nbytes=nbytes,
                device_id=0,
                shape=list(buf.shape),
                dtype_str=str(buf.dtype),
            ))

        descs = self._nixl_agent.get_reg_descs(caches_data, "DRAM")
        self._nixl_agent.register_memory(descs, backends=self._backends)

        self._registered_caches[mm_hash] = aligned
        self._registered_descs[mm_hash] = descs

        # Store serialised metadata for proxy to pick up
        proxy_meta = ECNixlProxyMetadata(
            engine_id=self._engine_id,
            mm_hash=mm_hash,
            agent_metadata=self._nixl_agent.get_agent_metadata(),
            tensors=tensor_infos,
        )
        self._pending_ec_params[mm_hash] = self._encoder.encode(proxy_meta)

        logger.debug(
            "EC NixlProxy: registered mm_hash=%s (%d tensors, "
            "metadata stored for proxy)",
            mm_hash, len(tensor_infos),
        )

    def start_load_caches(
        self,
        encoder_cache: dict[str, Any],
        blocking: bool = True,
        **kwargs,
    ) -> None:
        """LLM: drain metadata queue and initiate NIXL pulls."""
        if self.is_producer:
            return

        self._encoder_cache = encoder_cache
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECNixlProxyConnectorMetadata)

        self._process_pending_metadata()

        for mm_data in metadata.mm_datas_to_load:
            mm_hash = mm_data.mm_hash

            if mm_hash in encoder_cache:
                continue
            if mm_hash in self._pending_loads:
                continue

            if mm_hash not in self._tensor_registry:
                if not self._wait_for_cache(mm_hash):
                    logger.error(
                        "EC NixlProxy: mm_hash=%s not available after timeout",
                        mm_hash,
                    )
                    continue

            handle, local_bufs, local_descs = self._initiate_pull(mm_hash)
            self._pending_loads[mm_hash] = (handle, local_bufs, local_descs)
            logger.debug(
                "EC NixlProxy: initiated pull for mm_hash=%s", mm_hash
            )

        if self._pending_loads and blocking:
            self._wait_for_pulls(encoder_cache)

    def _wait_for_pulls(self, encoder_cache: dict[str, Any]) -> None:
        """Block until all pending NIXL pulls complete."""
        deadline = time.monotonic() + _CACHE_WAIT_TIMEOUT_S
        while self._pending_loads and time.monotonic() < deadline:
            for mm_hash, (handle, local_bufs, local_descs) in list(
                self._pending_loads.items()
            ):
                status = self._nixl_agent.check_xfer_state(handle)
                if status == "DONE":
                    encoder_cache[mm_hash] = local_bufs
                    self._nixl_agent.release_xfer_handle(handle)
                    del self._pending_loads[mm_hash]
                    logger.debug(
                        "EC NixlProxy: pull complete for mm_hash=%s", mm_hash
                    )
                elif status not in ("DONE", "PROC"):
                    logger.error(
                        "EC NixlProxy: transfer failed for mm_hash=%s",
                        mm_hash,
                    )
                    self._nixl_agent.release_xfer_handle(handle)
                    del self._pending_loads[mm_hash]
            if self._pending_loads:
                time.sleep(_XFER_POLL_INTERVAL_S)

        if self._pending_loads:
            logger.warning(
                "EC NixlProxy: %d pulls did not complete within timeout",
                len(self._pending_loads),
            )
            for mm_hash, (handle, _, _) in list(self._pending_loads.items()):
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Poll pending NIXL transfers."""
        if not self._pending_loads or self._encoder_cache is None:
            return None, None

        self._process_pending_metadata()

        completed: set[str] = set()
        for mm_hash, (handle, local_bufs, local_descs) in list(
            self._pending_loads.items()
        ):
            status = self._nixl_agent.check_xfer_state(handle)
            if status == "DONE":
                self._encoder_cache[mm_hash] = local_bufs
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]
                completed.add(mm_hash)
            elif status not in ("DONE", "PROC"):
                logger.error(
                    "EC NixlProxy: transfer failed for mm_hash=%s", mm_hash
                )
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]

        return None, completed if completed else None

    def request_finished(
        self, request: "Request",
    ) -> tuple[bool, dict[str, Any] | None]:
        """Encoder: return NIXL metadata for proxy to forward.
        LLM: clean up state.
        """
        if self.is_producer:
            ec_params: dict[str, bytes] = {}
            for feature in request.mm_features:
                mm_hash = feature.identifier
                params = self._pending_ec_params.pop(mm_hash, None)
                if params is not None:
                    ec_params[mm_hash] = params
                # Deregister NIXL memory
                if mm_hash in self._registered_descs:
                    try:
                        self._nixl_agent.deregister_memory(
                            self._registered_descs.pop(mm_hash)
                        )
                    except Exception as exc:
                        logger.warning(
                            "EC NixlProxy: failed to deregister "
                            "mm_hash=%s: %s", mm_hash, exc,
                        )
                    self._registered_caches.pop(mm_hash, None)

            if ec_params:
                return False, {"ec_transfer_params": ec_params}
            return False, None

        if self.is_consumer:
            for feature in request.mm_features:
                mm_hash = feature.identifier
                with self._cache_events_lock:
                    self._cache_events.pop(mm_hash, None)
                self._tensor_registry.pop(mm_hash, None)
                desc_key = f"_llm_{mm_hash}"
                if desc_key in self._registered_descs:
                    try:
                        self._nixl_agent.deregister_memory(
                            self._registered_descs.pop(desc_key)
                        )
                    except Exception:
                        pass

        return False, None

    def shutdown(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_zmq_ctx"):
            self._zmq_ctx.destroy(linger=1000)

    # -- Not used on worker side --

    def has_cache_item(self, identifier: str) -> bool:
        raise RuntimeError("has_cache_item must be called on the scheduler")

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        raise RuntimeError(
            "update_state_after_alloc must be called on the scheduler"
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        raise RuntimeError(
            "build_connector_meta must be called on the scheduler"
        )


# ---------------------------------------------------------------------------
# Top-level connector
# ---------------------------------------------------------------------------


class RblnECNixlProxyConnector(ECConnectorBase):
    """Entry point registered with ECConnectorFactory."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        if role == ECConnectorRole.SCHEDULER:
            self._impl: ECConnectorBase = RblnECNixlProxyConnectorScheduler(
                vllm_config, role
            )
        elif role == ECConnectorRole.WORKER:
            self._impl = RblnECNixlProxyConnectorWorker(vllm_config, role)
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

        logger.info("RblnECNixlProxyConnector created (role=%s)", role.name)

    def has_cache_item(self, identifier: str) -> bool:
        return self._impl.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        self._impl.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        return self._impl.build_connector_meta(scheduler_output)

    def start_load_caches(
        self, encoder_cache: dict[str, Any], **kwargs
    ) -> None:
        self._impl.start_load_caches(encoder_cache, **kwargs)

    def save_caches(
        self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs
    ) -> None:
        self._impl.save_caches(encoder_cache, mm_hash, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._impl.get_finished(finished_req_ids)

    def request_finished(
        self, request: "Request",
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._impl.request_finished(request)

    def bind_connector_metadata(self, metadata: ECConnectorMetadata) -> None:
        self._impl.bind_connector_metadata(metadata)

    def clear_connector_metadata(self) -> None:
        self._impl.clear_connector_metadata()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPE_SIZES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.int32": 4,
}


def _dtype_size(dtype_str: str) -> int:
    size = _DTYPE_SIZES.get(dtype_str)
    if size is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return size
