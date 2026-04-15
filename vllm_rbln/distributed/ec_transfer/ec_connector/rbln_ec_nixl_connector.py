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
RblnECNixlConnector: NIXL-based EC (Encoder Cache) disaggregation connector.

Architecture
------------
EC disaggregation splits vision-language model execution across two processes:

  Producer  → runs the vision encoder (_preprocess_prefill)
            → saves encoder outputs (inputs_embeds, position_embed, rope_deltas)
              to an EC cache keyed by mm_hash
            → exposes the cache via NIXL over RDMA (UCX backend)

  Consumer  → receives requests with the same mm_hash
            → uses NIXL to pull encoder cache from producer
            → runs prefill_decoder + decoder phases only

Transfer flow per request
-------------------------
  1. Producer encodes image → encoder_cache[mm_hash] populated
  2. Producer registers tensor memory with NIXL (dynamic, per mm_hash)
  3. Producer's side-channel (ZMQ ROUTER) publishes agent metadata
  4. Consumer (on first request) fetches producer's agent metadata via ZMQ REQ
  5. Consumer calls start_load_caches → initiates async NIXL pull (xfer)
  6. Consumer polls get_finished until transfer completes
  7. Consumer proceeds with prefill_decoder using the pulled tensors

Design decisions
----------------
- Dynamic NIXL registration: EC tensors are variable-size (~45 MB typical for
  Qwen2-VL-7B with 6400 visual tokens), making pre-allocated fixed buffers
  impractical. Tensors are registered on first save and deregistered when the
  request finishes.
- Consumer Pull: consumer initiates the NIXL transfer; producer is passive
  after registering memory. This matches the sequential flow where the
  orchestrator ensures the producer finishes before sending the consumer's
  request.
- Separate side-channel port: configurable via ec_connector_extra_config to
  avoid collision with the KV transfer side-channel used in PD disaggregation.
- UCX backend: same as feat_pd_disag's RblnNixlConnector.
- aligned_tensor: all CPU tensors are allocated via rebel.kv_cache.aligned_tensor
  for efficient NPU DMA.

Configuration example
---------------------
  # Single producer (backward-compatible):
  ECTransferConfig(
      ec_connector="RblnECNixlConnector",
      ec_role="ec_producer",  # or "ec_consumer"
      ec_buffer_device="cpu",
      ec_connector_extra_config={
          "side_channel_host": "127.0.0.1",
          "side_channel_port": 15100,
          "backends": ["UCX"],
      },
  )

  # Multi-producer (consumer connects to 8 producers):
  ECTransferConfig(
      ec_connector="RblnECNixlConnector",
      ec_role="ec_consumer",
      ec_buffer_device="cpu",
      ec_connector_extra_config={
          "producer_endpoints": [
              {"host": "127.0.0.1", "port": 15100},
              {"host": "127.0.0.1", "port": 15101},
              ...
          ],
          "backends": ["UCX"],
      },
  )
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

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
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Side-channel constants
# ---------------------------------------------------------------------------

_GET_META_MSG = b"get_ec_meta_msg"
_DEFAULT_SIDE_CHANNEL_HOST = "127.0.0.1"
_DEFAULT_SIDE_CHANNEL_PORT = 15100  # separate from VLLM_NIXL_SIDE_CHANNEL_PORT
_HANDSHAKE_TIMEOUT_MS = 10_000  # ZMQ recv timeout (ms)
_XFER_POLL_INTERVAL_S = 0.001   # poll interval for async transfer completion
_CACHE_WAIT_TIMEOUT_S = 30.0    # max time to wait for producer cache availability
_CACHE_WAIT_POLL_S = 0.05       # poll interval when waiting for producer cache

# ---------------------------------------------------------------------------
# Wire-protocol data classes (msgspec for zero-copy serialisation)
# ---------------------------------------------------------------------------


class ECNixlAgentMetadata(msgspec.Struct):
    """Metadata sent from producer to consumer during NIXL handshake."""
    engine_id: str
    agent_metadata: bytes
    # list of (mm_hash, key, base_addr, nbytes, device_id, shape, dtype_str)
    registered_tensors: list


class ECNixlHandshakePayload(msgspec.Struct):
    """Wire payload sent from producer's ZMQ ROUTER to consumer's ZMQ REQ."""
    agent_metadata_bytes: bytes


# ---------------------------------------------------------------------------
# Connector metadata (scheduler → worker)
# ---------------------------------------------------------------------------


@dataclass
class ECNixlConnectorMetadata(ECConnectorMetadata):
    """Metadata passed from scheduler to worker each step."""
    # mm_hashes the consumer worker needs to pull from producer
    mm_datas_to_load: list[MMMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler-side implementation
# ---------------------------------------------------------------------------


class RblnECNixlConnectorScheduler(ECConnectorBase):
    """Scheduler-side EC connector using NIXL for inter-process transfer.

    Tracks which mm_hashes need to be loaded by the consumer worker and
    runs a ZMQ ROUTER side-channel thread (producer only) to serve handshake
    requests from the consumer.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None

        self._side_channel_host: str = ec_cfg.get_from_extra_config(
            "side_channel_host", _DEFAULT_SIDE_CHANNEL_HOST
        )
        self._side_channel_port: int = ec_cfg.get_from_extra_config(
            "side_channel_port", _DEFAULT_SIDE_CHANNEL_PORT
        )

        # mm_hash → num_encoder_tokens (consumer only: tracks what to load)
        self._mm_datas_need_loads: dict[str, int] = {}

        # Side-channel listener thread (producer only)
        self._stop_event = threading.Event()
        self._listener_thread: threading.Thread | None = None
        # Populated by worker via update_agent_metadata()
        self._encoded_agent_metadata: bytes | None = None

        logger.info(
            "RblnECNixlConnectorScheduler initialised (role=%s, side_channel=%s:%d)",
            "producer" if self.is_producer else "consumer",
            self._side_channel_host,
            self._side_channel_port,
        )

    # ------------------------------------------------------------------
    # Called by worker once it has completed NIXL registration
    # ------------------------------------------------------------------

    def update_agent_metadata(self, encoded_metadata: bytes) -> None:
        """Start the side-channel listener with fresh agent metadata.

        Called by the worker after it has registered EC tensors with NIXL
        and encoded the ECNixlHandshakePayload.
        """
        if not self.is_producer:
            return
        self._encoded_agent_metadata = encoded_metadata
        if self._listener_thread is None:
            ready = threading.Event()
            self._listener_thread = threading.Thread(
                target=self._side_channel_listener,
                args=(encoded_metadata, ready, self._stop_event,
                      self._side_channel_host, self._side_channel_port),
                daemon=True,
                name="ec_nixl_handshake_listener",
            )
            self._listener_thread.start()
            ready.wait()
            logger.info(
                "EC NIXL side-channel listener started on %s:%d",
                self._side_channel_host, self._side_channel_port,
            )

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5)
            self._listener_thread = None

    # ------------------------------------------------------------------
    # ECConnectorBase interface - scheduler side
    # ------------------------------------------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        """Consumer returns True so the scheduler tracks mm_hashes to load.

        The consumer will pull encoder caches from the producer via NIXL.
        Returning True tells the scheduler to call update_state_after_alloc,
        which populates mm_datas_need_loads for the worker.
        """
        return self.is_consumer

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        if not self.is_consumer:
            return
        num_tokens = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_tokens

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECNixlConnectorMetadata:
        meta = ECNixlConnectorMetadata()
        for mm_hash, num_tokens in self._mm_datas_need_loads.items():
            meta.mm_datas_to_load.append(
                MMMeta.make_meta(mm_hash, num_tokens)
            )
        self._mm_datas_need_loads.clear()
        return meta

    # ------------------------------------------------------------------
    # ECConnectorBase interface - worker side (not used on scheduler)
    # ------------------------------------------------------------------

    def start_load_caches(self, encoder_cache: dict[str, Any], **kwargs) -> None:
        raise RuntimeError("start_load_caches must be called on the worker connector")

    def save_caches(self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs) -> None:
        raise RuntimeError("save_caches must be called on the worker connector")

    # ------------------------------------------------------------------
    # Static side-channel listener (ZMQ ROUTER)
    # ------------------------------------------------------------------

    @staticmethod
    def _side_channel_listener(
        metadata_holder: list[bytes],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
    ) -> None:
        """ZMQ ROUTER that serves the latest agent metadata to consumers.

        ``metadata_holder`` is a 1-element list whose contents are
        replaced in-place by ``_publish_agent_metadata`` whenever the
        tensor registry changes — no thread restart needed.
        """
        path = make_zmq_path("tcp", host, port)
        logger.debug("EC NIXL side-channel listening on %s", path)
        with _zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while not stop_event.is_set():
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    continue
                if msg == _GET_META_MSG:
                    sock.send_multipart((identity, b"", metadata_holder[0]))
                else:
                    logger.warning("EC side-channel got unexpected message: %s", msg)


# ---------------------------------------------------------------------------
# Worker-side implementation
# ---------------------------------------------------------------------------


class RblnECNixlConnectorWorker(ECConnectorBase):
    """Worker-side EC connector using NIXL for RDMA-based encoder cache transfer.

    Producer side:
      - After save_caches() is called, registers the EC tensors with NIXL
        and notifies the scheduler to start the side-channel listener.

    Consumer side:
      - On first request, performs a ZMQ handshake with the producer to
        obtain NIXL agent metadata.
      - start_load_caches() initiates async NIXL pulls.
      - get_finished() polls for completion and copies results into
        encoder_cache.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        try:
            from nixl._api import nixl_agent as NixlAgent  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "NIXL is not available. Install the nixl package to use "
                "RblnECNixlConnector."
            ) from e

        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None

        self._backends: list[str] = ec_cfg.get_from_extra_config(
            "backends", ["UCX"]
        )

        # Build producer endpoint list (consumer uses this to connect).
        # Supports both legacy single-endpoint and new multi-endpoint config.
        producer_endpoints = ec_cfg.get_from_extra_config(
            "producer_endpoints", None
        )
        if producer_endpoints is not None:
            self._producer_endpoints: list[dict[str, Any]] = list(
                producer_endpoints
            )
        else:
            # Backward-compatible: single side_channel_host/port
            host = ec_cfg.get_from_extra_config(
                "side_channel_host", _DEFAULT_SIDE_CHANNEL_HOST
            )
            port = ec_cfg.get_from_extra_config(
                "side_channel_port", _DEFAULT_SIDE_CHANNEL_PORT
            )
            self._producer_endpoints = [{"host": host, "port": port}]

        # Producer still uses its own side_channel_host/port for listening.
        self._side_channel_host: str = ec_cfg.get_from_extra_config(
            "side_channel_host", _DEFAULT_SIDE_CHANNEL_HOST
        )
        self._side_channel_port: int = ec_cfg.get_from_extra_config(
            "side_channel_port", _DEFAULT_SIDE_CHANNEL_PORT
        )

        self._engine_id: str = str(uuid.uuid4())
        self._nixl_agent = NixlAgent(self._engine_id, None)

        # mm_hash → dict[tensor_key, aligned CPU tensor]
        self._registered_caches: dict[str, dict[str, torch.Tensor]] = {}
        # mm_hash → nixl descriptor list (for deregistration)
        self._registered_descs: dict[str, Any] = {}

        # Pending async NIXL transfer handles (consumer only)
        # mm_hash → (handle, local_tensor_dict)
        self._pending_loads: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}

        # Pointer to the shared encoder_cache dict (set in start_load_caches)
        self._encoder_cache: dict[str, Any] | None = None

        # Multi-producer state (consumer only, populated after handshake)
        # producer_idx → remote NIXL agent name
        self._remote_agent_names: dict[int, str] = {}
        # producer_idx → {mm_hash → {key: (base_addr, size, device_id, shape, dtype_str)}}
        self._remote_tensor_registries: dict[
            int, dict[str, dict[str, tuple[int, int, int, list, str]]]
        ] = {}
        # Reverse index: mm_hash → producer_idx (for quick pull lookup)
        self._mm_hash_to_producer: dict[str, int] = {}

        # Side-channel listener (producer only)
        self._stop_event = threading.Event()
        self._listener_thread: threading.Thread | None = None

        # Start the side-channel listener immediately so consumers can
        # connect before any request arrives. Initial metadata has an
        # empty tensor registry; _publish_agent_metadata() will restart
        # it with actual tensors after the first save_caches() call.
        if self.is_producer:
            self._publish_agent_metadata()

        logger.info(
            "RblnECNixlConnectorWorker initialised "
            "(engine_id=%s, role=%s, backends=%s, num_producers=%d)",
            self._engine_id,
            "producer" if self.is_producer else "consumer",
            self._backends,
            len(self._producer_endpoints),
        )

    # ------------------------------------------------------------------
    # ECConnectorBase interface - worker side
    # ------------------------------------------------------------------

    def save_caches(
        self,
        encoder_cache: dict[str, Any],
        mm_hash: str,
        **kwargs,
    ) -> None:
        """Producer: register EC tensors with NIXL and update side-channel.

        Called by the model runner after preprocess_prefill completes.
        Tensors are dynamically registered on the first save for each mm_hash.
        """
        if not self.is_producer:
            return
        if mm_hash in self._registered_caches:
            logger.debug("EC NIXL: mm_hash=%s already registered, skipping", mm_hash)
            return

        raw: dict[str, torch.Tensor] = encoder_cache[mm_hash]
        if not isinstance(raw, dict):
            raw = {"inputs_embeds": raw}

        # Allocate aligned CPU tensors and copy data
        aligned: dict[str, torch.Tensor] = {}
        caches_data: list[tuple[int, int, int, str]] = []

        for key, tensor in raw.items():
            t = tensor.detach().cpu()
            buf = aligned_tensor(t.numel()).reshape(t.shape)
            buf.copy_(t)
            aligned[key] = buf
            caches_data.append((
                buf.data_ptr(),        # base address
                buf.numel() * buf.element_size(),  # size in bytes
                0,                     # device_id 0 = CPU for NIXL DRAM
                "",                    # label (unused)
            ))

        # Register with NIXL
        descs = self._nixl_agent.get_reg_descs(caches_data, "DRAM")
        self._nixl_agent.register_memory(descs, backends=self._backends)

        self._registered_caches[mm_hash] = aligned
        self._registered_descs[mm_hash] = descs

        logger.debug(
            "EC NIXL: registered %d tensors for mm_hash=%s (keys=%s)",
            len(aligned), mm_hash, list(aligned.keys()),
        )

        # Notify scheduler to publish updated agent metadata
        self._publish_agent_metadata()

    def start_load_caches(
        self,
        encoder_cache: dict[str, Any],
        blocking: bool = True,
        **kwargs,
    ) -> None:
        """Consumer: initiate async NIXL pull for each pending mm_hash.

        Args:
            encoder_cache: Shared encoder cache dict to populate.
            blocking: If True, wait for all pulls to complete before
                returning. If False, only initiate pulls — caller must
                poll via ``get_finished()`` or call
                ``wait_for_pending_pulls()`` later.
        """
        if self.is_producer:
            return

        self._encoder_cache = encoder_cache
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECNixlConnectorMetadata)

        # Ensure we have connections to all producers
        if not self._remote_agent_names:
            self._do_handshake_all()

        for mm_data in metadata.mm_datas_to_load:
            mm_hash = mm_data.mm_hash
            if mm_hash in self._pending_loads:
                continue

            # Refresh producers' metadata to discover the mm_hash.
            if mm_hash not in self._mm_hash_to_producer:
                self._wait_for_cache(mm_hash)
            if mm_hash not in self._mm_hash_to_producer:
                logger.error(
                    "EC NIXL: mm_hash=%s not available from any producer "
                    "after timeout, skipping",
                    mm_hash,
                )
                continue

            handle, local_bufs = self._initiate_pull(mm_hash)
            self._pending_loads[mm_hash] = (handle, local_bufs)
            logger.debug("EC NIXL: initiated pull for mm_hash=%s", mm_hash)

        if self._pending_loads and blocking:
            self._wait_for_pulls(encoder_cache)

    def wait_for_pending_pulls(self) -> None:
        """Block until all pending NIXL pulls complete.

        Convenience method for callers that used
        ``start_load_caches(blocking=False)`` and now need the data.
        """
        if self._pending_loads and self._encoder_cache is not None:
            self._wait_for_pulls(self._encoder_cache)

    def _wait_for_pulls(self, encoder_cache: dict[str, Any]) -> None:
        """Synchronously wait for all pending NIXL pulls to complete."""
        deadline = time.monotonic() + _CACHE_WAIT_TIMEOUT_S
        while self._pending_loads and time.monotonic() < deadline:
            for mm_hash, (handle, local_bufs) in list(
                self._pending_loads.items()
            ):
                status = self._nixl_agent.check_xfer_state(handle)
                if status == "DONE":
                    # Use local_bufs directly — no clone needed since
                    # they are allocated per-pull and not reused.
                    encoder_cache[mm_hash] = local_bufs
                    self._nixl_agent.release_xfer_handle(handle)
                    del self._pending_loads[mm_hash]
                    logger.debug(
                        "EC NIXL: pull complete for mm_hash=%s", mm_hash
                    )
                elif status not in ("DONE", "PROC"):
                    logger.error(
                        "EC NIXL: transfer failed for mm_hash=%s", mm_hash
                    )
                    self._nixl_agent.release_xfer_handle(handle)
                    del self._pending_loads[mm_hash]
            if self._pending_loads:
                time.sleep(_XFER_POLL_INTERVAL_S)

        if self._pending_loads:
            logger.warning(
                "EC NIXL: %d pulls did not complete within timeout, "
                "consumer will fall back to local encoding",
                len(self._pending_loads),
            )
            # Clean up timed-out transfers
            for mm_hash, (handle, _) in list(self._pending_loads.items()):
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Poll pending NIXL transfers and move completed ones to encoder_cache."""
        if not self._pending_loads or self._encoder_cache is None:
            return None, None

        completed: set[str] = set()
        for mm_hash, (handle, local_bufs) in list(self._pending_loads.items()):
            status = self._nixl_agent.check_xfer_state(handle)
            if status == "DONE":
                self._encoder_cache[mm_hash] = local_bufs
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]
                completed.add(mm_hash)
                logger.debug("EC NIXL: pull complete for mm_hash=%s", mm_hash)
            elif status not in ("DONE", "PROC"):
                logger.error("EC NIXL: transfer failed for mm_hash=%s", mm_hash)
                self._nixl_agent.release_xfer_handle(handle)
                del self._pending_loads[mm_hash]

        return None, completed if completed else None

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Deregister NIXL memory for completed requests (producer only)."""
        if not self.is_producer:
            return False, None
        for feature in request.mm_features:
            mm_hash = feature.identifier
            self._deregister_mm_hash(mm_hash)
        return False, None

    # ------------------------------------------------------------------
    # NIXL helpers
    # ------------------------------------------------------------------

    def _refresh_metadata(self, producer_idx: int) -> None:
        """Re-fetch a specific producer's agent metadata and update the
        tensor registry.  Only re-registers the NIXL remote agent when
        new mm_hashes are discovered (avoids expensive remove/re-add on
        every poll)."""
        endpoint = self._producer_endpoints[producer_idx]
        path = make_zmq_path("tcp", endpoint["host"], endpoint["port"])
        try:
            with _zmq_ctx(zmq.REQ, path) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 500)
                sock.send(_GET_META_MSG)
                raw = sock.recv()
            decoder = msgspec.msgpack.Decoder(ECNixlHandshakePayload)
            payload = decoder.decode(raw)
            meta_decoder = msgspec.msgpack.Decoder(ECNixlAgentMetadata)
            agent_meta = meta_decoder.decode(payload.agent_metadata_bytes)

            # Build the new tensor registry to compare with the old one.
            new_registry: dict[str, dict[str, tuple[int, int, int, list, str]]] = {}
            for entry in agent_meta.registered_tensors:
                mm_hash, key, base_addr, nbytes, device_id = entry[:5]
                shape = entry[5] if len(entry) > 5 else []
                dtype_str = entry[6] if len(entry) > 6 else "torch.bfloat16"
                if mm_hash not in new_registry:
                    new_registry[mm_hash] = {}
                new_registry[mm_hash][key] = (
                    base_addr, nbytes, device_id, shape, dtype_str
                )

            old_registry = self._remote_tensor_registries.get(producer_idx, {})
            new_mm_hashes = set(new_registry) - set(old_registry)

            # Only re-register NIXL agent when new tensors appear
            # (remote memory regions have changed).
            if new_mm_hashes:
                if producer_idx in self._remote_agent_names:
                    self._nixl_agent.remove_remote_agent(
                        self._remote_agent_names[producer_idx]
                    )
                self._remote_agent_names[producer_idx] = (
                    self._nixl_agent.add_remote_agent(agent_meta.agent_metadata)
                )

            self._remote_tensor_registries[producer_idx] = new_registry
            for mm_hash in new_registry:
                self._mm_hash_to_producer[mm_hash] = producer_idx

            if new_mm_hashes:
                logger.info(
                    "EC NIXL: producer %d — %d new mm_hash(es) discovered "
                    "(%d total across all producers)",
                    producer_idx,
                    len(new_mm_hashes),
                    len(self._mm_hash_to_producer),
                )
            else:
                logger.info(
                    "EC NIXL: producer %d refresh — %d mm_hashes (no new)",
                    producer_idx,
                    len(new_registry),
                )
        except (zmq.Again, zmq.ZMQError, msgspec.DecodeError) as exc:
            logger.info(
                "EC NIXL: metadata refresh for producer %d failed (%s)",
                producer_idx, exc,
            )

    def _wait_for_cache(self, mm_hash: str) -> None:
        """Poll all producers' side-channels until *mm_hash* appears in
        any producer's registry or timeout is reached."""
        deadline = time.monotonic() + _CACHE_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            for idx in range(len(self._producer_endpoints)):
                if idx not in self._remote_agent_names:
                    # Try connecting to producers that failed initial handshake
                    endpoint = self._producer_endpoints[idx]
                    try:
                        self._do_handshake_single(
                            idx, endpoint["host"], endpoint["port"]
                        )
                    except RuntimeError:
                        continue
                self._refresh_metadata(idx)
                if mm_hash in self._mm_hash_to_producer:
                    logger.debug(
                        "EC NIXL: mm_hash=%s discovered on producer %d",
                        mm_hash, idx,
                    )
                    return
            time.sleep(_CACHE_WAIT_POLL_S)
        logger.warning(
            "EC NIXL: timed out waiting for mm_hash=%s (%.1fs)",
            mm_hash, _CACHE_WAIT_TIMEOUT_S,
        )

    def _publish_agent_metadata(self) -> None:
        """Encode current agent metadata and update the side-channel listener.

        On first call, starts the listener thread. On subsequent calls,
        updates the metadata in-place (no thread restart needed).
        """
        if not self.is_producer:
            return

        # Build the registered tensor registry for the consumer
        registered_tensors: list[tuple[str, str, int, int, int, list, str]] = []
        for mm_hash, tensor_dict in self._registered_caches.items():
            for key, buf in tensor_dict.items():
                registered_tensors.append((
                    mm_hash,
                    key,
                    buf.data_ptr(),
                    buf.numel() * buf.element_size(),
                    0,  # CPU device_id
                    list(buf.shape),
                    str(buf.dtype),
                ))

        agent_meta = ECNixlAgentMetadata(
            engine_id=self._engine_id,
            agent_metadata=self._nixl_agent.get_agent_metadata(),
            registered_tensors=registered_tensors,
        )
        encoder = msgspec.msgpack.Encoder()
        payload = ECNixlHandshakePayload(
            agent_metadata_bytes=encoder.encode(agent_meta)
        )
        encoded = encoder.encode(payload)

        if self._listener_thread is not None:
            # Listener already running — update metadata in-place.
            self._metadata_holder[0] = encoded
            logger.info(
                "EC NIXL: metadata updated in-place (%d mm_hashes, %d tensors)",
                len(self._registered_caches),
                len(registered_tensors),
            )
        else:
            # First call — start the listener thread.
            self._metadata_holder = [encoded]
            ready = threading.Event()
            self._listener_thread = threading.Thread(
                target=RblnECNixlConnectorScheduler._side_channel_listener,
                args=(self._metadata_holder, ready, self._stop_event,
                      self._side_channel_host, self._side_channel_port),
                daemon=True,
                name="ec_nixl_handshake_listener",
            )
            self._listener_thread.start()
            ready.wait()
            logger.info(
                "EC NIXL side-channel listener started on %s:%d",
                self._side_channel_host, self._side_channel_port,
            )

    def _do_handshake_all(self, fast: bool = False) -> None:
        """Consumer: connect to all configured producers sequentially.

        Args:
            fast: If True, use a short timeout per producer (1s) so startup
                isn't blocked waiting for producers that aren't ready yet.
        """
        for idx, endpoint in enumerate(self._producer_endpoints):
            if idx in self._remote_agent_names:
                continue  # already connected
            try:
                self._do_handshake_single(
                    idx, endpoint["host"], endpoint["port"],
                    timeout_ms=1000 if fast else _HANDSHAKE_TIMEOUT_MS,
                )
            except RuntimeError:
                logger.warning(
                    "EC NIXL: handshake with producer %d (%s:%d) failed, "
                    "will retry on demand",
                    idx, endpoint["host"], endpoint["port"],
                )
        logger.info(
            "EC NIXL: connected to %d / %d producers after initial handshake",
            len(self._remote_agent_names),
            len(self._producer_endpoints),
        )

    def _do_handshake_single(
        self, producer_idx: int, host: str, port: int,
        timeout_ms: int = _HANDSHAKE_TIMEOUT_MS,
    ) -> None:
        """Consumer: fetch a single producer's NIXL agent metadata via ZMQ."""
        path = make_zmq_path("tcp", host, port)
        logger.info(
            "EC NIXL: performing handshake with producer %d at %s",
            producer_idx, path,
        )

        deadline = time.monotonic() + timeout_ms / 1000
        payload: ECNixlHandshakePayload | None = None

        while time.monotonic() < deadline:
            try:
                with _zmq_ctx(zmq.REQ, path) as sock:
                    sock.setsockopt(zmq.RCVTIMEO, 2000)
                    sock.send(_GET_META_MSG)
                    raw = sock.recv()
                decoder = msgspec.msgpack.Decoder(ECNixlHandshakePayload)
                payload = decoder.decode(raw)
                break
            except (zmq.Again, zmq.ZMQError, msgspec.DecodeError) as exc:
                logger.debug(
                    "EC NIXL: handshake attempt with producer %d failed (%s), "
                    "retrying ...",
                    producer_idx, exc,
                )
                time.sleep(0.5)

        if payload is None:
            raise RuntimeError(
                f"EC NIXL handshake with producer {producer_idx} at "
                f"{host}:{port} timed out after {_HANDSHAKE_TIMEOUT_MS}ms"
            )

        meta_decoder = msgspec.msgpack.Decoder(ECNixlAgentMetadata)
        agent_meta: ECNixlAgentMetadata = meta_decoder.decode(
            payload.agent_metadata_bytes
        )

        # Register the remote producer as a NIXL agent
        self._remote_agent_names[producer_idx] = (
            self._nixl_agent.add_remote_agent(agent_meta.agent_metadata)
        )
        logger.info(
            "EC NIXL: registered remote producer %d "
            "(engine_id=%s, agent=%s)",
            producer_idx,
            agent_meta.engine_id,
            self._remote_agent_names[producer_idx],
        )

        # Build per-producer registry and reverse index
        self._remote_tensor_registries[producer_idx] = {}
        for entry in agent_meta.registered_tensors:
            mm_hash, key, base_addr, nbytes, device_id = entry[:5]
            shape = entry[5] if len(entry) > 5 else []
            dtype_str = entry[6] if len(entry) > 6 else "torch.bfloat16"
            registry = self._remote_tensor_registries[producer_idx]
            if mm_hash not in registry:
                registry[mm_hash] = {}
            registry[mm_hash][key] = (
                base_addr, nbytes, device_id, shape, dtype_str
            )
            self._mm_hash_to_producer[mm_hash] = producer_idx

        logger.info(
            "EC NIXL: handshake with producer %d complete — "
            "%d mm_hashes available",
            producer_idx,
            len(self._remote_tensor_registries[producer_idx]),
        )

    def _initiate_pull(
        self, mm_hash: str
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Prepare local buffers and start an async NIXL pull from the
        producer that owns *mm_hash*."""
        producer_idx = self._mm_hash_to_producer[mm_hash]
        remote_tensors = self._remote_tensor_registries[producer_idx][mm_hash]
        remote_agent_name = self._remote_agent_names[producer_idx]

        local_bufs: dict[str, torch.Tensor] = {}
        # 4-tuples for reg_descs: (addr, size, device_id, label)
        local_reg_data: list[tuple[int, int, int, str]] = []
        # 3-tuples for xfer_descs: (addr, size, device_id)
        local_xfer_data: list[tuple[int, int, int]] = []
        remote_xfer_data: list[tuple[int, int, int]] = []

        for key, (base_addr, nbytes, device_id, shape, _dtype_str) in remote_tensors.items():
            # Allocate aligned local buffer matching remote tensor size.
            # Producer uses aligned_tensor(numel) which defaults to float16
            # (2 bytes/element), so we do the same here.
            numel = nbytes // 2  # float16 = 2 bytes per element
            buf = aligned_tensor(numel)
            if shape:
                buf = buf.reshape(shape)
            local_bufs[key] = buf
            local_reg_data.append((buf.data_ptr(), nbytes, 0, ""))
            local_xfer_data.append((buf.data_ptr(), nbytes, 0))
            remote_xfer_data.append((base_addr, nbytes, device_id))

        num_descs = len(local_xfer_data)

        # Register local destination buffers
        local_reg_descs = self._nixl_agent.get_reg_descs(local_reg_data, "DRAM")
        self._nixl_agent.register_memory(local_reg_descs, backends=self._backends)
        self._registered_descs[f"_consumer_{mm_hash}"] = local_reg_descs

        # Prep transfer descriptor lists
        local_prepped = self._nixl_agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_xfer_data, "DRAM"
        )
        remote_prepped = self._nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_data, "DRAM"
        )

        # Initiate async NIXL READ (consumer pulls from producer)
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
                f"EC NIXL: transfer initiation failed for mm_hash={mm_hash} "
                f"from producer {producer_idx}"
            )

        return handle, local_bufs

    def _deregister_mm_hash(self, mm_hash: str) -> None:
        """Deregister NIXL memory for a completed mm_hash (producer only)."""
        if mm_hash in self._registered_descs:
            try:
                self._nixl_agent.deregister_memory(self._registered_descs.pop(mm_hash))
            except Exception as exc:
                logger.warning(
                    "EC NIXL: failed to deregister mm_hash=%s: %s", mm_hash, exc
                )
        self._registered_caches.pop(mm_hash, None)

    # ------------------------------------------------------------------
    # Unused scheduler-side stubs (required by abstract base)
    # ------------------------------------------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        raise RuntimeError("has_cache_item must be called on the scheduler connector")

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        raise RuntimeError("update_state_after_alloc must be called on the scheduler connector")

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ECConnectorMetadata:
        raise RuntimeError("build_connector_meta must be called on the scheduler connector")


# ---------------------------------------------------------------------------
# Top-level connector: instantiates scheduler or worker based on role
# ---------------------------------------------------------------------------


class RblnECNixlConnector(ECConnectorBase):
    """Entry point registered with ECConnectorFactory.

    Delegates to RblnECNixlConnectorScheduler or RblnECNixlConnectorWorker
    depending on the role assigned by vLLM's engine infrastructure.
    """

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

        logger.info(
            "RblnECNixlConnector created (role=%s)", role.name
        )

    # ------------------------------------------------------------------
    # Delegate all interface methods to the appropriate impl
    # ------------------------------------------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        return self._impl.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        self._impl.update_state_after_alloc(request, index)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ECConnectorMetadata:
        return self._impl.build_connector_meta(scheduler_output)

    def start_load_caches(self, encoder_cache: dict[str, Any], **kwargs) -> None:
        self._impl.start_load_caches(encoder_cache, **kwargs)

    def save_caches(self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs) -> None:
        self._impl.save_caches(encoder_cache, mm_hash, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._impl.get_finished(finished_req_ids)

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._impl.request_finished(request)

    def bind_connector_metadata(self, metadata: ECConnectorMetadata) -> None:
        self._impl.bind_connector_metadata(metadata)

    def clear_connector_metadata(self) -> None:
        self._impl.clear_connector_metadata()


# ---------------------------------------------------------------------------
# ZMQ context manager (mirrors nixl_connector.py's zmq_ctx helper)
# ---------------------------------------------------------------------------


@contextmanager
def _zmq_ctx(socket_type: int, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket (ROUTER binds, REQ connects)."""
    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()
        yield make_zmq_socket(
            ctx=ctx,
            path=addr,
            socket_type=socket_type,
            bind=socket_type == zmq.ROUTER,
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
