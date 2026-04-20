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
RblnECNixlConnector: NIXL data transfer + ZMQ PUSH/PULL notification.

Architecture
------------
Uses NIXL for CPU-to-CPU tensor data transfer, with direct ZMQ PUSH/PULL
for metadata notification — encoder pushes NIXL agent metadata and tensor
registry entries straight to the llm's PULL socket, so the llm
can initiate the NIXL pull as soon as the encoder finishes.

  Encoder  -> runs the vision encoder
            -> registers encoder output tensors with NIXL
            -> pushes NIXL metadata (agent_metadata + tensor addresses)
               directly to llm via ZMQ PUSH

  LLM  -> binds a ZMQ PULL socket at init (fan-in from N encoders)
            -> background thread receives metadata, queues it
            -> main thread drains queue, registers NIXL remote agent,
               initiates NIXL pull to fetch actual tensor data

Transfer flow per request
-------------------------
  1. Client sends the same request to both encoder and llm
  2. Encoder encodes image -> encoder outputs ready
  3. Encoder registers tensors with NIXL, gets agent_metadata
  4. Encoder pushes metadata via ZMQ PUSH to llm
  5. LLM background PULL thread receives, queues metadata, sets event
  6. LLM main thread: drains queue, add_remote_agent, initiates NIXL pull
  7. NIXL transfers tensor data (CPU -> CPU)
  8. LLM populates encoder_cache, runs prefill + decode

Design decisions
----------------
- ZMQ PUSH/PULL for metadata notification: direct push from encoder to
  llm with zero polling.  N:1 fan-in is natively supported.
- NIXL for data transfer: efficient CPU memory transfer for large tensors
  via UCX backend.  Only metadata (~hundreds of bytes) flows over ZMQ;
  tensor data (~45 MB) flows over NIXL.
- Event-based waiting: llm waits on threading.Event per mm_hash,
  near-zero latency once metadata arrives.
- No ROUTER fallback needed: PUSH/PULL is reliable (ZMQ buffers messages
  until the llm connects, unlike PUB which drops).

Port layout
-----------
  LLM binds on a single port:
    - pull_port: ZMQ PULL (receives NIXL metadata from all encoders)

  Each encoder connects to the llm's PULL port via ZMQ PUSH.

Configuration example
---------------------
  # Encoder:
  ECTransferConfig(
      ec_connector="RblnECNixlConnector",
      ec_role="ec_producer",
      ec_connector_extra_config={
          "llm_host": "127.0.0.1",
          "llm_pull_port": 16100,
          "backends": ["UCX"],
      },
  )

  # LLM:
  ECTransferConfig(
      ec_connector="RblnECNixlConnector",
      ec_role="ec_consumer",
      ec_connector_extra_config={
          "pull_host": "0.0.0.0",
          "pull_port": 16100,
          "backends": ["UCX"],
      },
  )
"""

from __future__ import annotations

import contextlib
import queue
import socket
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
_DEFAULT_PULL_PORT = 16100
_DEFAULT_LLM_HOST = "127.0.0.1"
_CACHE_WAIT_TIMEOUT_S = 30.0
_XFER_POLL_INTERVAL_S = 0.001
_LLM_PROBE_TIMEOUT_S = 1.0


def _probe_tcp(host: str, port: int, timeout: float = _LLM_PROBE_TIMEOUT_S) -> bool:
    """Return True if *host:port* is accepting TCP connections.

    Used at encoder startup to check whether the LLM's ZMQ PULL socket
    is already listening. ZMQ's connect() itself is non-blocking and
    succeeds even when the peer isn't up yet, so we do this one-shot
    probe to give the operator a clear "start the LLM first" signal.
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Wire protocol (msgspec)
# ---------------------------------------------------------------------------


class ECNixlTensorInfo(msgspec.Struct):
    """Per-tensor metadata sent from encoder to llm."""

    key: str
    base_addr: int
    nbytes: int
    device_id: int
    shape: list[int]
    dtype_str: str


class ECNixlMetadata(msgspec.Struct):
    """Metadata pushed from encoder to llm via ZMQ.

    Contains everything the llm needs to register the remote NIXL
    agent and initiate a pull for this mm_hash.
    """

    engine_id: str
    mm_hash: str
    agent_metadata: bytes
    tensors: list[ECNixlTensorInfo]
    # Non-tensor values (e.g. second_per_grid_ts) serialised as
    # {key: value} — llm restores these alongside pulled tensors.
    non_tensor_data: dict = {}


# ---------------------------------------------------------------------------
# Connector metadata (scheduler -> worker)
# ---------------------------------------------------------------------------


@dataclass
class ECNixlConnectorMetadata(ECConnectorMetadata):
    """Metadata passed from scheduler to worker each step."""

    mm_datas_to_load: list[MMMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler-side implementation
# ---------------------------------------------------------------------------


class RblnECNixlConnectorScheduler(ECConnectorBase):
    """Scheduler-side: tracks which mm_hashes the llm needs to load."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._mm_datas_need_loads: dict[str, int] = {}

    def has_cache_item(self, identifier: str) -> bool:
        return self.is_consumer

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        if not self.is_consumer:
            return
        mm_hash = request.mm_features[index].identifier
        num_tokens = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_tokens

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECNixlConnectorMetadata:
        meta = ECNixlConnectorMetadata()
        for mm_hash, num_tokens in self._mm_datas_need_loads.items():
            meta.mm_datas_to_load.append(MMMeta.make_meta(mm_hash, num_tokens))
        self._mm_datas_need_loads.clear()
        return meta

    # -- Not used on scheduler side --

    def start_load_caches(self, encoder_cache: dict[str, Any], **kwargs) -> None:
        raise RuntimeError("start_load_caches must be called on the worker")

    def save_caches(
        self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs
    ) -> None:
        raise RuntimeError("save_caches must be called on the worker")


# ---------------------------------------------------------------------------
# Worker-side implementation
# ---------------------------------------------------------------------------


class RblnECNixlConnectorWorker(ECConnectorBase):
    """Worker-side EC connector: NIXL data transfer + ZMQ PUSH/PULL metadata.

    Encoder:
      - Creates NIXL agent and ZMQ PUSH socket at init.
      - On save_caches(): registers tensors with NIXL, pushes metadata
        via ZMQ directly to llm.

    LLM:
      - Creates NIXL agent and binds ZMQ PULL socket at init.
      - Background thread receives metadata from all encoders into a queue.
      - Main thread drains queue, registers NIXL remote agents, and
        initiates NIXL pulls.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        try:
            from nixl._api import nixl_agent as NixlAgent
        except ImportError as e:
            raise RuntimeError(
                "NIXL is not available. Install the nixl package to use "
                "RblnECNixlConnector."
            ) from e

        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None

        self._backends: list[str] = ec_cfg.get_from_extra_config("backends", ["UCX"])
        self._engine_id: str = str(uuid.uuid4())
        self._nixl_agent = NixlAgent(self._engine_id, None)
        self._stop_event = threading.Event()

        # -- Encoder state --
        # mm_hash -> dict[key, aligned CPU tensor]
        self._registered_caches: dict[str, dict[str, torch.Tensor]] = {}
        # mm_hash -> NIXL descriptor (for deregistration)
        self._registered_descs: dict[str, Any] = {}

        # -- LLM state --
        # Incoming metadata queue (filled by background PULL thread)
        self._incoming_metadata: queue.Queue[ECNixlMetadata] = queue.Queue()
        # Per-mm_hash event for waiters
        self._cache_events: dict[str, threading.Event] = {}
        self._cache_events_lock = threading.Lock()
        # engine_id -> remote NIXL agent name
        self._remote_agents: dict[str, str] = {}
        # mm_hash -> (engine_id, list[ECNixlTensorInfo])
        self._tensor_registry: dict[str, tuple[str, list[ECNixlTensorInfo]]] = {}
        # mm_hash -> non-tensor data dict
        self._non_tensor_registry: dict[str, dict] = {}
        # Pending async NIXL transfers: mm_hash -> (handle, local_bufs, local_descs)
        self._pending_loads: dict[str, tuple[Any, dict[str, torch.Tensor], Any]] = {}
        self._encoder_cache: dict[str, Any] | None = None

        if self.is_producer:
            llm_host: str = ec_cfg.get_from_extra_config("llm_host", _DEFAULT_LLM_HOST)
            llm_pull_port: int = ec_cfg.get_from_extra_config(
                "llm_pull_port", _DEFAULT_PULL_PORT
            )
            self._push_addr = f"tcp://{llm_host}:{llm_pull_port}"
            if not _probe_tcp(llm_host, llm_pull_port):
                logger.warning(
                    "RblnECNixlConnector (encoder): LLM PULL port %s:%d is not "
                    "accepting connections yet. Start the LLM first with "
                    "`bash examples/optimum/serve_ec_llm.sh` — ZMQ will still "
                    "buffer messages and reconnect once the LLM is up, but "
                    "tail latency for the earliest requests may be poor.",
                    llm_host,
                    llm_pull_port,
                )
            self._zmq_ctx = zmq.Context()
            self._push_sock = self._zmq_ctx.socket(zmq.PUSH)
            self._push_sock.setsockopt(zmq.SNDHWM, 64)
            self._push_sock.setsockopt(zmq.LINGER, 5000)
            self._push_sock.connect(self._push_addr)
            logger.info(
                "RblnECNixlConnector (encoder): PUSH connected to %s",
                self._push_addr,
            )

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
                name="ec_nixl_push_receiver",
            )
            self._receiver_thread.start()
            logger.info(
                "RblnECNixlConnector (llm): PULL bound on %s",
                self._pull_addr,
            )

    # ------------------------------------------------------------------
    # LLM: background PULL receiver
    # ------------------------------------------------------------------

    def _receiver_loop(self) -> None:
        """Background thread: receive NIXL metadata from all encoders."""
        poller = zmq.Poller()
        poller.register(self._pull_sock, zmq.POLLIN)
        decoder = msgspec.msgpack.Decoder(ECNixlMetadata)

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

                # Signal waiter for this mm_hash
                with self._cache_events_lock:
                    evt = self._cache_events.get(meta.mm_hash)
                    if evt is not None:
                        evt.set()

                logger.debug(
                    "EC Nixl: received metadata for mm_hash=%s from engine=%s "
                    "(%d tensors)",
                    meta.mm_hash,
                    meta.engine_id,
                    len(meta.tensors),
                )
            except Exception as exc:
                logger.warning("EC Nixl: failed to decode received metadata: %s", exc)

    def _process_pending_metadata(self) -> set[str]:
        """Drain metadata queue, register NIXL remote agents.

        Returns set of newly available mm_hashes.
        Must be called from the main thread (NIXL ops are not thread-safe).
        """
        new_mm_hashes: set[str] = set()

        while True:
            try:
                meta = self._incoming_metadata.get_nowait()
            except queue.Empty:
                break

            engine_id = meta.engine_id
            mm_hash = meta.mm_hash

            # Register or update remote NIXL agent for this encoder
            if engine_id in self._remote_agents:
                self._nixl_agent.remove_remote_agent(self._remote_agents[engine_id])
            self._remote_agents[engine_id] = self._nixl_agent.add_remote_agent(
                meta.agent_metadata
            )

            # Store tensor registry and non-tensor data for this mm_hash
            self._tensor_registry[mm_hash] = (engine_id, meta.tensors)
            if meta.non_tensor_data:
                self._non_tensor_registry[mm_hash] = meta.non_tensor_data
            new_mm_hashes.add(mm_hash)

            logger.debug(
                "EC Nixl: registered remote agent for engine=%s, mm_hash=%s",
                engine_id,
                mm_hash,
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
        """Wait until metadata for mm_hash arrives via PULL."""
        # Check if already in registry
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

            # Drain queue on main thread (NIXL not thread-safe)
            self._process_pending_metadata()
            if mm_hash in self._tensor_registry:
                logger.debug(
                    "EC Nixl: mm_hash=%s discovered via PUSH notification",
                    mm_hash,
                )
                return True

        logger.warning(
            "EC Nixl: timed out waiting for mm_hash=%s (%.1fs)",
            mm_hash,
            _CACHE_WAIT_TIMEOUT_S,
        )
        return False

    # ------------------------------------------------------------------
    # NIXL pull
    # ------------------------------------------------------------------

    def _initiate_pull(
        self,
        mm_hash: str,
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
            remote_xfer_data.append((tinfo.base_addr, tinfo.nbytes, tinfo.device_id))

        num_descs = len(local_xfer_data)

        # Register local destination buffers with NIXL
        local_descs = self._nixl_agent.get_reg_descs(local_reg_data, "DRAM")
        self._nixl_agent.register_memory(local_descs, backends=self._backends)

        # Prepare transfer descriptors
        local_prepped = self._nixl_agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_xfer_data, "DRAM"
        )
        remote_prepped = self._nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_data, "DRAM"
        )

        # Initiate async READ (llm pulls from encoder)
        indices = list(range(num_descs))
        handle = self._nixl_agent.make_prepped_xfer(
            "READ",
            local_prepped,
            indices,
            remote_prepped,
            indices,
            notif_msg=b"",
        )
        status = self._nixl_agent.transfer(handle)
        if status not in ("DONE", "PROC"):
            raise RuntimeError(
                f"EC Nixl: transfer initiation failed for mm_hash={mm_hash}"
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
        """Encoder: register tensors with NIXL, push metadata to llm."""
        if not self.is_producer:
            return
        if mm_hash in self._registered_caches:
            return

        raw: dict[str, torch.Tensor] = encoder_cache[mm_hash]
        if not isinstance(raw, dict):
            raw = {"inputs_embeds": raw}

        # Allocate aligned CPU tensors and register with NIXL
        aligned: dict[str, Any] = {}
        caches_data: list[tuple[int, int, int, str]] = []
        tensor_infos: list[ECNixlTensorInfo] = []
        non_tensor_data: dict = {}

        def _register_tensor(key: str, tensor: torch.Tensor) -> None:
            t = tensor.detach().cpu()
            buf = aligned_tensor(t.numel()).reshape(t.shape)
            buf.copy_(t)
            aligned[key] = buf
            nbytes = buf.numel() * buf.element_size()
            caches_data.append((buf.data_ptr(), nbytes, 0, ""))
            tensor_infos.append(
                ECNixlTensorInfo(
                    key=key,
                    base_addr=buf.data_ptr(),
                    nbytes=nbytes,
                    device_id=0,
                    shape=list(buf.shape),
                    dtype_str=str(buf.dtype),
                )
            )

        def _collect(key: str, value: Any) -> None:
            """Recursively register tensors and collect non-tensor data."""
            if isinstance(value, torch.Tensor):
                _register_tensor(key, value)
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    _collect(f"{key}.{i}", item)
                non_tensor_data[f"_seq_meta.{key}"] = {
                    "length": len(value),
                    "is_tuple": isinstance(value, tuple),
                }
            else:
                # Primitive value (int, float, str, None, etc.)
                non_tensor_data[key] = value

        for key, value in raw.items():
            _collect(key, value)

        if not caches_data:
            logger.warning(
                "EC Nixl: no tensors to register for mm_hash=%s "
                "(all values are non-tensor)",
                mm_hash,
            )
            return

        descs = self._nixl_agent.get_reg_descs(caches_data, "DRAM")
        self._nixl_agent.register_memory(descs, backends=self._backends)

        self._registered_caches[mm_hash] = aligned
        self._registered_descs[mm_hash] = descs

        # Push metadata to llm via ZMQ
        push_meta = ECNixlMetadata(
            engine_id=self._engine_id,
            mm_hash=mm_hash,
            agent_metadata=self._nixl_agent.get_agent_metadata(),
            tensors=tensor_infos,
            non_tensor_data=non_tensor_data,
        )
        encoded = msgspec.msgpack.Encoder().encode(push_meta)
        self._push_sock.send(encoded)

        logger.debug(
            "EC Nixl: registered + pushed mm_hash=%s (%d tensors)",
            mm_hash,
            len(tensor_infos),
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
        assert isinstance(metadata, ECNixlConnectorMetadata)

        # Process any pending metadata from encoders
        self._process_pending_metadata()

        for mm_data in metadata.mm_datas_to_load:
            mm_hash = mm_data.mm_hash

            if mm_hash in encoder_cache:
                continue
            if mm_hash in self._pending_loads:
                continue

            # Wait for metadata if not yet received
            if mm_hash not in self._tensor_registry and not self._wait_for_cache(
                mm_hash
            ):
                logger.error(
                    "EC Nixl: mm_hash=%s not available after timeout",
                    mm_hash,
                )
                continue

            handle, local_bufs, local_descs = self._initiate_pull(mm_hash)
            self._pending_loads[mm_hash] = (handle, local_bufs, local_descs)
            logger.debug("EC Nixl: initiated pull for mm_hash=%s", mm_hash)

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
                    non_tensor = self._non_tensor_registry.pop(mm_hash, None)
                    encoder_cache[mm_hash] = _merge_pull_result(local_bufs, non_tensor)
                    self._nixl_agent.release_xfer_handle(handle)
                    del self._pending_loads[mm_hash]
                    logger.debug("EC Nixl: pull complete for mm_hash=%s", mm_hash)
                elif status not in ("DONE", "PROC"):
                    logger.error("EC Nixl: transfer failed for mm_hash=%s", mm_hash)
                    self._nixl_agent.release_xfer_handle(handle)
                    del self._pending_loads[mm_hash]
            if self._pending_loads:
                time.sleep(_XFER_POLL_INTERVAL_S)

        if self._pending_loads:
            logger.warning(
                "EC Nixl: %d pulls did not complete within timeout",
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

        # Drain metadata for pre-fetch benefit
        self._process_pending_metadata()

        completed: set[str] = set()
        for mm_hash, (handle, local_bufs, local_descs) in list(
            self._pending_loads.items()
        ):
            status = self._nixl_agent.check_xfer_state(handle)
            if status == "DONE":
                result = dict(local_bufs)
                non_tensor = self._non_tensor_registry.pop(mm_hash, None)
                if non_tensor:
                    result.update(non_tensor)
                self._encoder_cache[mm_hash] = result
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]
                completed.add(mm_hash)
                logger.debug("EC Nixl: pull complete for mm_hash=%s", mm_hash)
            elif status not in ("DONE", "PROC"):
                logger.error("EC Nixl: transfer failed for mm_hash=%s", mm_hash)
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]

        return None, completed if completed else None

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        """Deregister NIXL memory for completed requests (encoder only)."""
        if self.is_producer:
            for feature in request.mm_features:
                mm_hash = feature.identifier
                if mm_hash in self._registered_descs:
                    try:
                        self._nixl_agent.deregister_memory(
                            self._registered_descs.pop(mm_hash)
                        )
                    except Exception as exc:
                        logger.warning(
                            "EC Nixl: failed to deregister mm_hash=%s: %s",
                            mm_hash,
                            exc,
                        )
                    self._registered_caches.pop(mm_hash, None)

        if self.is_consumer:
            for feature in request.mm_features:
                mm_hash = feature.identifier
                with self._cache_events_lock:
                    self._cache_events.pop(mm_hash, None)
                self._tensor_registry.pop(mm_hash, None)
                # Deregister local llm buffers
                desc_key = f"_llm_{mm_hash}"
                if desc_key in self._registered_descs:
                    with contextlib.suppress(Exception):
                        self._nixl_agent.deregister_memory(
                            self._registered_descs.pop(desc_key)
                        )

        return False, None

    def shutdown(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_zmq_ctx"):
            self._zmq_ctx.destroy(linger=1000)

    # -- Not used on worker side --

    def has_cache_item(self, identifier: str) -> bool:
        raise RuntimeError("has_cache_item must be called on the scheduler")

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        raise RuntimeError("update_state_after_alloc must be called on the scheduler")

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        raise RuntimeError("build_connector_meta must be called on the scheduler")


# ---------------------------------------------------------------------------
# Top-level connector
# ---------------------------------------------------------------------------


class RblnECNixlConnector(ECConnectorBase):
    """Entry point registered with ECConnectorFactory."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        if role == ECConnectorRole.SCHEDULER:
            self._impl: ECConnectorBase = RblnECNixlConnectorScheduler(
                vllm_config, role
            )
        elif role == ECConnectorRole.WORKER:
            self._impl = RblnECNixlConnectorWorker(vllm_config, role)
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

        logger.info("RblnECNixlConnector created (role=%s)", role.name)

    def has_cache_item(self, identifier: str) -> bool:
        return self._impl.has_cache_item(identifier)

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        self._impl.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        return self._impl.build_connector_meta(scheduler_output)

    def start_load_caches(self, encoder_cache: dict[str, Any], **kwargs) -> None:
        self._impl.start_load_caches(encoder_cache, **kwargs)

    def save_caches(
        self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs
    ) -> None:
        self._impl.save_caches(encoder_cache, mm_hash, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._impl.get_finished(finished_req_ids)

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
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


def _merge_pull_result(
    local_bufs: dict[str, torch.Tensor],
    non_tensor: dict | None,
) -> dict[str, Any]:
    """Merge NIXL-pulled tensors with non-tensor metadata.

    Reconstructs tuple/list values that were flattened during
    save_caches (e.g. "image_embeds.0", "image_embeds.1" → tuple).
    """
    if non_tensor is None:
        return dict(local_bufs)

    # Pool all values (tensors + non-tensors) into a flat lookup
    pool: dict[str, Any] = dict(local_bufs)
    seq_metas: dict[str, dict] = {}
    for k, v in non_tensor.items():
        if k.startswith("_seq_meta."):
            seq_metas[k[len("_seq_meta.") :]] = v
        else:
            pool[k] = v

    def _reconstruct(key: str) -> Any:
        """Recursively reconstruct a value from the flat pool."""
        if key in seq_metas:
            meta = seq_metas[key]
            items = [_reconstruct(f"{key}.{i}") for i in range(meta["length"])]
            return tuple(items) if meta.get("is_tuple", False) else items
        return pool.pop(key, None)

    # Reconstruct all sequences first (deepest-first via recursion)
    result: dict[str, Any] = {}
    for key in sorted(seq_metas, key=lambda k: k.count("."), reverse=True):
        if key.count(".") == 0:
            # Top-level sequence
            result[key] = _reconstruct(key)

    # Add remaining flat values (tensors and primitives)
    for k, v in pool.items():
        # Skip sub-keys already consumed by reconstruction
        if "." not in k:
            result[k] = v

    return result
