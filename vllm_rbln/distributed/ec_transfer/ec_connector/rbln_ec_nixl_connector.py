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

# Stage 2 defaults (tunable via ec_connector_extra_config)
_DEFAULT_METADATA_QUEUE_MAX = 256
_DEFAULT_MAX_CONCURRENT_PULLS = 32
_DEFAULT_PRODUCER_CACHE_CAPACITY = 128
_DEFAULT_PRODUCER_CACHE_TTL_S = 60.0
# How long the receiver thread blocks on queue.put before re-checking
# the stop_event — keeps shutdown responsive when the queue is saturated.
_QUEUE_PUT_TIMEOUT_S = 0.2

# Stage 3a defaults (ACK channel)
_DEFAULT_ACK_HOST = "127.0.0.1"
_DEFAULT_ACK_PORT = 0  # 0 = bind ephemeral, advertise actual port
_DEFAULT_ACK_QUEUE_MAX = 256

# Stage 3b defaults (heartbeat / liveness)
_DEFAULT_PING_INTERVAL_S = 60.0
_DEFAULT_DEAD_THRESHOLD_S = 180.0  # ~3 × ping interval, tolerates missed pings


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
    # ACK channel the llm should PUSH completion/fail notifications to.
    # Empty host means the producer did not enable ACKs (backward compat).
    ack_host: str = ""
    ack_port: int = 0


class ECNixlAck(msgspec.Struct):
    """Completion / liveness notification pushed from llm back to encoder.

    status is one of:
      - "ok":   NIXL pull finished cleanly; producer may deregister.
      - "fail": transfer failed, timed out, or request was torn down;
                producer may deregister.
      - "ping": idle liveness beacon; carries no mm_hash, the producer
                uses it only to refresh the consumer's last_seen.

    consumer_engine_id identifies which consumer sent the message; the
    producer tracks liveness per consumer on that key. Empty means the
    sender did not enable liveness (backward compat).
    """

    mm_hash: str
    engine_id: str
    status: str = "ok"
    consumer_engine_id: str = ""


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
            raise ImportError(
                "The 'nixl' package is required to use 'RblnECNixlConnector'. "
                "Please install it by running: pip install nixl"
            ) from e

        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None, (
            "RblnECNixlConnector requires `ec_transfer_config` to be set."
        )

        self._backends: list[str] = ec_cfg.get_from_extra_config("backends", ["UCX"])
        self._engine_id: str = str(uuid.uuid4())
        self._nixl_agent = NixlAgent(self._engine_id, None)
        self._stop_event = threading.Event()

        # -- Tunable knobs --
        self._metadata_queue_max: int = ec_cfg.get_from_extra_config(
            "metadata_queue_max", _DEFAULT_METADATA_QUEUE_MAX
        )
        self._max_concurrent_pulls: int = ec_cfg.get_from_extra_config(
            "max_concurrent_pulls", _DEFAULT_MAX_CONCURRENT_PULLS
        )
        self._producer_cache_capacity: int = ec_cfg.get_from_extra_config(
            "producer_cache_capacity", _DEFAULT_PRODUCER_CACHE_CAPACITY
        )
        self._producer_cache_ttl_s: float = ec_cfg.get_from_extra_config(
            "producer_cache_ttl_s", _DEFAULT_PRODUCER_CACHE_TTL_S
        )
        self._ack_host: str = ec_cfg.get_from_extra_config(
            "ack_host", _DEFAULT_ACK_HOST
        )
        self._ack_port_configured: int = ec_cfg.get_from_extra_config(
            "ack_port", _DEFAULT_ACK_PORT
        )
        self._ack_queue_max: int = ec_cfg.get_from_extra_config(
            "ack_queue_max", _DEFAULT_ACK_QUEUE_MAX
        )
        self._ping_interval_s: float = ec_cfg.get_from_extra_config(
            "ping_interval_s", _DEFAULT_PING_INTERVAL_S
        )
        self._dead_threshold_s: float = ec_cfg.get_from_extra_config(
            "dead_threshold_s", _DEFAULT_DEAD_THRESHOLD_S
        )

        # -- Encoder state --
        # Insertion order is the LRU order; we never move entries, so a plain
        # dict (Python 3.7+) is sufficient.
        # mm_hash -> dict[key, aligned CPU tensor]
        self._registered_caches: dict[str, dict[str, torch.Tensor]] = {}
        # mm_hash -> NIXL descriptor (for deregistration)
        self._registered_descs: dict[str, Any] = {}
        # mm_hash -> monotonic timestamp of registration (for TTL)
        self._registered_timestamps: dict[str, float] = {}
        # Liveness tracking keyed by consumer_engine_id. Updated by the ACK
        # receiver thread on every incoming message (including pings).
        self._last_seen: dict[str, float] = {}
        self._engine_alive: dict[str, bool] = {}

        # -- LLM state --
        # Incoming metadata queue (filled by background PULL thread). Bounded
        # so receiver.put() blocks when the main thread is slow, which in
        # turn fills the ZMQ PULL HWM and back-pressures encoders via PUSH.
        self._incoming_metadata: queue.Queue[ECNixlMetadata] = queue.Queue(
            maxsize=self._metadata_queue_max
        )
        # mm_hashes the scheduler asked us to load but for which we deferred
        # initiating the pull because _pending_loads hit the concurrency cap.
        # Retried on the next start_load_caches() call.
        self._deferred_loads: set[str] = set()
        # Per-encoder PUSH socket for ACKs, keyed by (host, port). Created
        # lazily the first time we see metadata advertising that address.
        # Owned by the main thread (used from _send_ack_if_unsent).
        self._ack_push_sockets: dict[tuple[str, int], Any] = {}
        # mm_hash -> (ack_host, ack_port) — where to send ACK for this hash.
        self._mm_hash_ack_addr: dict[str, tuple[str, int]] = {}
        # Used to guarantee at most one ACK per mm_hash per request lifetime.
        self._ack_sent: set[str] = set()
        # Known encoder endpoints for liveness pings (engine_id -> address).
        # Written by main thread, read by ping thread — shared under a lock
        # so the ping thread sees a consistent snapshot.
        self._known_ack_addrs: dict[str, tuple[str, int]] = {}
        self._known_ack_addrs_lock = threading.Lock()
        # Dedicated PUSH sockets owned by the ping thread. ZMQ sockets are
        # not thread-safe, so the ping thread must not reuse _ack_push_sockets.
        self._ping_push_sockets: dict[tuple[str, int], Any] = {}
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
                    "`bash examples/optimum/ec_disagg/serve_ec_llm.sh` - "
                    "ZMQ will still buffer messages and reconnect"
                    "once the LLM is up, but tail latency for the earliest"
                    "requests may be poor.",
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

            # ACK PULL socket — receives completion notifications from LLM so
            # we can deregister NIXL memory immediately instead of relying on
            # TTL / request_finished as the only drain path.
            self._ack_pull_sock = self._zmq_ctx.socket(zmq.PULL)
            self._ack_pull_sock.setsockopt(zmq.RCVHWM, self._ack_queue_max)
            if self._ack_port_configured == 0:
                self._ack_pull_sock.bind(f"tcp://{self._ack_host}:*")
                endpoint = self._ack_pull_sock.getsockopt(zmq.LAST_ENDPOINT).decode()
                self._ack_port = int(endpoint.rsplit(":", 1)[1])
            else:
                self._ack_pull_sock.bind(
                    f"tcp://{self._ack_host}:{self._ack_port_configured}"
                )
                self._ack_port = self._ack_port_configured
            self._incoming_acks: queue.Queue[ECNixlAck] = queue.Queue(
                maxsize=self._ack_queue_max
            )
            self._ack_receiver_thread = threading.Thread(
                target=self._ack_receiver_loop,
                daemon=True,
                name="ec_nixl_ack_receiver",
            )
            self._ack_receiver_thread.start()
            logger.info(
                "RblnECNixlConnector (encoder): ACK PULL bound on %s:%d",
                self._ack_host,
                self._ack_port,
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

            # Idle-ping thread: keeps producer's liveness view of this
            # consumer warm even when no real ACKs are flowing.
            self._ping_thread = threading.Thread(
                target=self._ping_loop,
                daemon=True,
                name="ec_nixl_ping",
            )
            self._ping_thread.start()

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
            except Exception as exc:
                logger.warning("EC Nixl: failed to decode received metadata: %s", exc)
                continue

            # Block on put() when the queue is full so back-pressure propagates
            # through ZMQ's PULL HWM (and then PUSH HWM) to encoders. Poll
            # stop_event so shutdown is never stuck behind a saturated queue.
            while not self._stop_event.is_set():
                try:
                    self._incoming_metadata.put(meta, timeout=_QUEUE_PUT_TIMEOUT_S)
                    break
                except queue.Full:
                    continue
            else:
                return

            # Signal waiter for this mm_hash
            with self._cache_events_lock:
                evt = self._cache_events.get(meta.mm_hash)
                if evt is not None:
                    evt.set()

            logger.debug(
                "EC Nixl: received metadata for mm_hash=%s from engine=%s (%d tensors)",
                meta.mm_hash,
                meta.engine_id,
                len(meta.tensors),
            )

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

            # Re-register the remote NIXL agent on every incoming metadata.
            # agent_metadata is NOT stable across mm_hashes: the encoder
            # allocates fresh aligned buffers per save_caches (new base
            # addresses), so NIXL needs to drop the stale view of this
            # agent and re-add with the latest memory map before any
            # prep_xfer_dlist can resolve the new addresses. Skipping this
            # (the earlier "once per engine_id" shortcut) crashes the
            # consumer with NIXL_ERR_NOT_FOUND on the encoder's 2nd+
            # request. remove+add must be paired — add-only without
            # remove triggers NIXL_ERR_REMOTE_DISCONNECT when UCX state
            # gets confused.
            if engine_id in self._remote_agents:
                self._nixl_agent.remove_remote_agent(self._remote_agents[engine_id])
            self._remote_agents[engine_id] = self._nixl_agent.add_remote_agent(
                meta.agent_metadata
            )

            # Store tensor registry and non-tensor data for this mm_hash
            self._tensor_registry[mm_hash] = (engine_id, meta.tensors)
            if meta.non_tensor_data:
                self._non_tensor_registry[mm_hash] = meta.non_tensor_data
            if meta.ack_host and meta.ack_port:
                addr = (meta.ack_host, meta.ack_port)
                self._mm_hash_ack_addr[mm_hash] = addr
                # Track this encoder for liveness pings. Log when we see an
                # engine_id for the first time so operators can observe the
                # fan-in topology as it converges.
                with self._known_ack_addrs_lock:
                    is_new = engine_id not in self._known_ack_addrs
                    self._known_ack_addrs[engine_id] = addr
                if is_new:
                    logger.info(
                        "EC Nixl: discovered encoder engine=%s at %s:%d",
                        engine_id,
                        addr[0],
                        addr[1],
                    )
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

        # Close race: the receiver thread may have enqueued metadata between
        # our drain above and the event creation here. In that window, the
        # receiver's `_cache_events.get(mm_hash)` returns None and no signal
        # fires, so drain once more before sleeping on the event.
        self._process_pending_metadata()
        if mm_hash in self._tensor_registry:
            return True

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
    # LLM: ACK sender
    # ------------------------------------------------------------------

    def _send_ack_if_unsent(self, mm_hash: str, status: str) -> None:
        """Best-effort ACK the producer for this mm_hash.

        Idempotent per mm_hash via self._ack_sent. Silently skips if the
        producer did not advertise an ACK address (older encoder) or if a
        prior ACK was already sent for this mm_hash.
        """
        if mm_hash in self._ack_sent:
            return
        addr = self._mm_hash_ack_addr.get(mm_hash)
        if addr is None:
            return
        engine_id = self._tensor_registry.get(mm_hash, (None,))[0]
        if engine_id is None:
            return

        sock = self._ack_push_sockets.get(addr)
        if sock is None:
            sock = self._zmq_ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, self._ack_queue_max)
            sock.setsockopt(zmq.LINGER, 1000)
            sock.connect(f"tcp://{addr[0]}:{addr[1]}")
            self._ack_push_sockets[addr] = sock

        ack = ECNixlAck(
            mm_hash=mm_hash,
            engine_id=engine_id,
            status=status,
            consumer_engine_id=self._engine_id,
        )
        try:
            sock.send(msgspec.msgpack.Encoder().encode(ack), zmq.NOBLOCK)
            self._ack_sent.add(mm_hash)
        except zmq.Again:
            logger.warning(
                "EC Nixl: ACK send would block for mm_hash=%s (encoder ACK queue full)",
                mm_hash,
            )

    # ------------------------------------------------------------------
    # LLM: idle-ping loop
    # ------------------------------------------------------------------

    def _ping_loop(self) -> None:
        """Consumer: periodically PUSH a liveness ping to each known encoder.

        Runs on its own thread with its own PUSH sockets (ZMQ sockets are
        not thread-safe, so we cannot reuse _ack_push_sockets). Ping is
        encoded as an ECNixlAck with empty mm_hash and status="ping" — the
        producer refreshes last_seen and ignores it for eviction.
        """
        encoder = msgspec.msgpack.Encoder()
        while not self._stop_event.wait(self._ping_interval_s):
            with self._known_ack_addrs_lock:
                snapshot = list(self._known_ack_addrs.items())
            for engine_id, addr in snapshot:
                sock = self._ping_push_sockets.get(addr)
                if sock is None:
                    sock = self._zmq_ctx.socket(zmq.PUSH)
                    sock.setsockopt(zmq.SNDHWM, self._ack_queue_max)
                    sock.setsockopt(zmq.LINGER, 1000)
                    sock.connect(f"tcp://{addr[0]}:{addr[1]}")
                    self._ping_push_sockets[addr] = sock
                ping = ECNixlAck(
                    mm_hash="",
                    engine_id=engine_id,
                    status="ping",
                    consumer_engine_id=self._engine_id,
                )
                # Encoder ACK queue full: skip this round, next ping cycle
                # will retry. Not worth logging.
                with contextlib.suppress(zmq.Again):
                    sock.send(encoder.encode(ping), zmq.NOBLOCK)

    # ------------------------------------------------------------------
    # Encoder: ACK receiver + liveness tracking
    # ------------------------------------------------------------------

    def _record_liveness(self, consumer_engine_id: str) -> None:
        """Refresh last_seen and log a transition if the consumer recovered."""
        if not consumer_engine_id:
            return  # older consumer without liveness support
        self._last_seen[consumer_engine_id] = time.monotonic()
        prev = self._engine_alive.get(consumer_engine_id)
        self._engine_alive[consumer_engine_id] = True
        if prev is None:
            logger.info(
                "EC Nixl: first contact from consumer engine=%s",
                consumer_engine_id,
            )
        elif prev is False:
            logger.warning(
                "EC Nixl: consumer engine=%s RECOVERED (ACK/ping received "
                "after dead threshold)",
                consumer_engine_id,
            )

    def _check_engine_liveness(self) -> None:
        """Transition alive->dead consumers past the threshold; log once."""
        now = time.monotonic()
        for consumer_engine_id, ts in list(self._last_seen.items()):
            if (
                self._engine_alive.get(consumer_engine_id, False)
                and (now - ts) > self._dead_threshold_s
            ):
                self._engine_alive[consumer_engine_id] = False
                logger.warning(
                    "EC Nixl: consumer engine=%s DEAD (no ACK/ping for %.1fs, "
                    "threshold=%.1fs) — logging only, registered caches "
                    "left to TTL backstop",
                    consumer_engine_id,
                    now - ts,
                    self._dead_threshold_s,
                )

    def _ack_receiver_loop(self) -> None:
        """Background thread on producer: receive ACKs from llm."""
        poller = zmq.Poller()
        poller.register(self._ack_pull_sock, zmq.POLLIN)
        decoder = msgspec.msgpack.Decoder(ECNixlAck)

        while not self._stop_event.is_set():
            events = dict(poller.poll(200))
            # Scan for alive->dead transitions every poll tick so the
            # threshold triggers promptly even when no ACKs are arriving.
            self._check_engine_liveness()
            if self._ack_pull_sock not in events:
                continue
            try:
                raw = self._ack_pull_sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue

            try:
                ack = decoder.decode(raw)
            except Exception as exc:
                logger.warning("EC Nixl: failed to decode ACK: %s", exc)
                continue

            # Any message — ok / fail / ping — counts as liveness evidence.
            self._record_liveness(ack.consumer_engine_id)

            if ack.status == "ping":
                # Liveness-only; nothing to evict.
                continue

            while not self._stop_event.is_set():
                try:
                    self._incoming_acks.put(ack, timeout=_QUEUE_PUT_TIMEOUT_S)
                    break
                except queue.Full:
                    continue
            else:
                return

    def _drain_acks(self) -> None:
        """Evict any ACKed producer entries. Main-thread only (NIXL)."""
        while True:
            try:
                ack = self._incoming_acks.get_nowait()
            except queue.Empty:
                break
            if ack.mm_hash in self._registered_caches:
                logger.debug(
                    "EC Nixl: ACK status=%s for mm_hash=%s, evicting",
                    ack.status,
                    ack.mm_hash,
                )
                self._evict_producer_entry(ack.mm_hash)

    # ------------------------------------------------------------------
    # Producer-side cache admission (LRU + TTL)
    # ------------------------------------------------------------------

    def _evict_producer_entry(self, mm_hash: str) -> None:
        """Deregister NIXL memory and drop the producer-side cache entry."""
        descs = self._registered_descs.pop(mm_hash, None)
        if descs is not None:
            try:
                self._nixl_agent.deregister_memory(descs)
            except Exception as exc:
                # Never let cleanup failures abort eviction — the local
                # dict pops below must still run. Log so descriptor-pool
                # leaks don't go silent.
                logger.warning(
                    "EC Nixl: deregister_memory failed for mm_hash=%s: %s",
                    mm_hash,
                    exc,
                )
        self._registered_caches.pop(mm_hash, None)
        self._registered_timestamps.pop(mm_hash, None)

    def _sweep_expired_producer_entries(self) -> None:
        """Drop any registered producer entries older than the TTL."""
        now = time.monotonic()
        expired = [
            h
            for h, ts in self._registered_timestamps.items()
            if now - ts > self._producer_cache_ttl_s
        ]
        for h in expired:
            logger.warning(
                "EC Nixl: TTL-evicting producer cache for mm_hash=%s "
                "(ttl=%.1fs, consumer never pulled or request_finished "
                "never fired)",
                h,
                self._producer_cache_ttl_s,
            )
            self._evict_producer_entry(h)

    def _admit_producer_entry(self) -> None:
        """Make room before inserting a new producer cache entry.

        Runs a lazy TTL sweep first, then evicts LRU entries (insertion
        order) until the capacity budget allows one more insertion.
        """
        self._sweep_expired_producer_entries()
        while len(self._registered_caches) >= self._producer_cache_capacity:
            oldest = next(iter(self._registered_caches))
            logger.warning(
                "EC Nixl: LRU-evicting producer cache for mm_hash=%s "
                "(capacity=%d reached)",
                oldest,
                self._producer_cache_capacity,
            )
            self._evict_producer_entry(oldest)

    # ------------------------------------------------------------------
    # NIXL pull
    # ------------------------------------------------------------------

    def _release_pull(self, handle: Any, local_descs: Any) -> None:
        """Release NIXL handle and deregister local destination memory.

        Safe to call on both success and failure paths. Failures are
        logged but never raised so the caller's cleanup sequence runs
        to completion.
        """
        try:
            self._nixl_agent.release_xfer_handle(handle)
        except Exception as exc:
            logger.warning("EC Nixl: release_xfer_handle failed: %s", exc)
        try:
            self._nixl_agent.deregister_memory(local_descs)
        except Exception as exc:
            logger.warning("EC Nixl: deregister_memory failed: %s", exc)

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

        # Drain any ACKs from completed llm-side pulls first so evictions
        # make room before we potentially admit a new entry.
        self._drain_acks()

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

        # Enforce LRU capacity + TTL before registering a new entry.
        self._admit_producer_entry()

        descs = self._nixl_agent.get_reg_descs(caches_data, "DRAM")
        self._nixl_agent.register_memory(descs, backends=self._backends)

        self._registered_caches[mm_hash] = aligned
        self._registered_descs[mm_hash] = descs
        self._registered_timestamps[mm_hash] = time.monotonic()

        # Push metadata to llm via ZMQ
        push_meta = ECNixlMetadata(
            engine_id=self._engine_id,
            mm_hash=mm_hash,
            agent_metadata=self._nixl_agent.get_agent_metadata(),
            tensors=tensor_infos,
            non_tensor_data=non_tensor_data,
            ack_host=self._ack_host,
            ack_port=self._ack_port,
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

        # Merge fresh scheduler asks with hashes deferred from previous steps
        # due to the concurrent-pull cap. Dedup via set so the scheduler's
        # new list takes priority on ordering but deferred entries aren't lost.
        todo: list[str] = list(self._deferred_loads)
        self._deferred_loads.clear()
        seen = set(todo)
        for mm_data in metadata.mm_datas_to_load:
            if mm_data.mm_hash not in seen:
                todo.append(mm_data.mm_hash)
                seen.add(mm_data.mm_hash)

        for mm_hash in todo:
            if mm_hash in encoder_cache:
                continue
            if mm_hash in self._pending_loads:
                continue

            # Enforce concurrent-pull cap: defer the rest to a later step
            # so we don't flood NIXL/local memory with in-flight transfers.
            if len(self._pending_loads) >= self._max_concurrent_pulls:
                self._deferred_loads.add(mm_hash)
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
                    self._release_pull(handle, local_descs)
                    del self._pending_loads[mm_hash]
                    self._send_ack_if_unsent(mm_hash, "ok")
                    logger.debug("EC Nixl: pull complete for mm_hash=%s", mm_hash)
                elif status not in ("DONE", "PROC"):
                    logger.error("EC Nixl: transfer failed for mm_hash=%s", mm_hash)
                    self._release_pull(handle, local_descs)
                    self._non_tensor_registry.pop(mm_hash, None)
                    del self._pending_loads[mm_hash]
                    self._send_ack_if_unsent(mm_hash, "fail")
            if self._pending_loads:
                time.sleep(_XFER_POLL_INTERVAL_S)

        if self._pending_loads:
            logger.warning(
                "EC Nixl: %d pulls did not complete within timeout",
                len(self._pending_loads),
            )
            for mm_hash, (handle, _, local_descs) in list(self._pending_loads.items()):
                self._release_pull(handle, local_descs)
                self._non_tensor_registry.pop(mm_hash, None)
                del self._pending_loads[mm_hash]
                self._send_ack_if_unsent(mm_hash, "fail")

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
                self._release_pull(handle, local_descs)
                del self._pending_loads[mm_hash]
                completed.add(mm_hash)
                self._send_ack_if_unsent(mm_hash, "ok")
                logger.debug("EC Nixl: pull complete for mm_hash=%s", mm_hash)
            elif status not in ("DONE", "PROC"):
                logger.error("EC Nixl: transfer failed for mm_hash=%s", mm_hash)
                self._release_pull(handle, local_descs)
                self._non_tensor_registry.pop(mm_hash, None)
                del self._pending_loads[mm_hash]
                self._send_ack_if_unsent(mm_hash, "fail")

        return None, completed if completed else None

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        """Deregister NIXL memory for completed requests (encoder only)."""
        if self.is_producer:
            # Drain any late ACKs so eviction state stays current.
            self._drain_acks()
            for feature in request.mm_features:
                mm_hash = feature.identifier
                self._evict_producer_entry(mm_hash)

        if self.is_consumer:
            for feature in request.mm_features:
                mm_hash = feature.identifier

                # Send fail ACK first (before popping tensor_registry, which
                # holds engine_id), so producer can evict even when the pull
                # never finished. _send_ack_if_unsent is idempotent, so the
                # ok case from _wait_for_pulls / get_finished is preserved.
                self._send_ack_if_unsent(mm_hash, "fail")

                with self._cache_events_lock:
                    self._cache_events.pop(mm_hash, None)
                self._tensor_registry.pop(mm_hash, None)
                self._non_tensor_registry.pop(mm_hash, None)
                self._mm_hash_ack_addr.pop(mm_hash, None)
                self._ack_sent.discard(mm_hash)
                self._deferred_loads.discard(mm_hash)
                pending = self._pending_loads.pop(mm_hash, None)
                if pending is not None:
                    handle, _, local_descs = pending
                    self._release_pull(handle, local_descs)

        return False, None

    def shutdown(self) -> None:
        self._stop_event.set()

        # Join receiver threads so they stop touching ZMQ/NIXL before we tear
        # down the underlying resources.
        for attr in ("_receiver_thread", "_ack_receiver_thread", "_ping_thread"):
            thread = getattr(self, attr, None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        # Consumer: release any in-flight pulls.
        for mm_hash, (handle, _, local_descs) in list(self._pending_loads.items()):
            self._release_pull(handle, local_descs)
        self._pending_loads.clear()
        self._ack_push_sockets.clear()  # closed by zmq_ctx.destroy below
        self._ping_push_sockets.clear()

        # Producer: deregister any remaining NIXL memory.
        for mm_hash in list(self._registered_descs.keys()):
            self._evict_producer_entry(mm_hash)

        self._deferred_loads.clear()

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
