# Copyright 2026 Rebellions Inc. All rights reserved.
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

"""Encoder cache transfer between RBLN device buffers over NIXL VRAM.

ZMQ carries descriptors and completion acknowledgements. Encoder tensor bytes
remain on the NPU, including the snapshot held until every consumer TP rank
has completed its read.
"""

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import msgspec
import torch
import zmq
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

from vllm_rbln.distributed.nixl_device import register_device_regions


@dataclass
class DeviceECMetadata(ECConnectorMetadata):
    saves: set[str] = field(default_factory=set)
    loads: dict[str, int] = field(default_factory=dict)


@dataclass
class DeviceBuffer:
    tensor: torch.Tensor
    address: int
    nbytes: int
    registration: Any


@dataclass
class DeviceLease:
    mm_hash: str
    buffer: DeviceBuffer
    metadata: dict
    readers: int | None = None
    consumer_id: str | None = None
    offered: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)


class DeviceECTransport:
    """Owns flat, dtype-preserving VRAM registrations and synchronous reads."""

    def __init__(self, device: torch.device, listen_ip: str, timeout: float):
        import nixl_rbln
        from vllm.distributed.nixl_utils import NixlWrapper

        device = torch.device(device)
        if device.type != "rbln":
            raise ValueError("Device EC requires RBLN device tensors")
        if not hasattr(torch.rbln, "bind_device_memory"):
            raise RuntimeError("Device EC requires torch-rbln bind_device_memory")
        self.device = torch.device("rbln", device.index or 0)
        self.timeout = timeout
        self.adapter = nixl_rbln
        if NixlWrapper is None:
            raise ImportError("Device EC requires NIXL and nixl-rbln")
        self.agent = NixlWrapper(f"rbln-ec-{uuid.uuid4().hex}")
        self.listen_ip = listen_ip
        # Keep registrations and tensors alive if an asynchronous read times out.
        self.inflight: list[tuple[Any, DeviceBuffer, str]] = []

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceBuffer:
        import rebel

        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        torch.rbln.bind_device_memory(tensor)
        areas = self.adapter.entry_areas(tensor.data_ptr())
        nbytes = tensor.numel() * tensor.element_size()
        if len(areas) != 1 or areas[0].size != nbytes:
            raise RuntimeError("EC transfer requires one flat device allocation")
        address = int(areas[0].device_addr)
        registration = register_device_regions(
            self.agent,
            [(address, nbytes)],
            device_id=tensor.get_device(),
            rbln_ctx_ptr=rebel.context_of(tensor).rbln_ctx_ptr,
            backend_options={"listen_ip": self.listen_ip},
        )
        return DeviceBuffer(tensor, address, nbytes, registration)

    def snapshot(self, source: torch.Tensor) -> DeviceBuffer:
        if source.device != self.device:
            raise ValueError(f"EC source must be on {self.device}, got {source.device}")
        buffer = self.allocate(tuple(source.shape), source.dtype)
        try:
            buffer.tensor.copy_(source)
            torch.rbln.synchronize(self.device)
        except BaseException:
            self.release(buffer)
            raise
        return buffer

    def release(self, buffer: DeviceBuffer) -> None:
        self.agent.deregister_memory(buffer.registration)

    def read(self, metadata: dict, shape: tuple[int, ...], dtype: torch.dtype):
        destination = self.allocate(shape, dtype)
        remote = None
        try:
            remote = self.agent.add_remote_agent(metadata["agent"])
            local_descs = self.agent.get_xfer_descs(
                [
                    (
                        destination.address,
                        destination.nbytes,
                        destination.tensor.get_device(),
                    )
                ],
                "VRAM",
            )
            remote_descs = self.agent.get_xfer_descs(
                [(metadata["address"], destination.nbytes, metadata["device_id"])],
                "VRAM",
            )
            handle = self.agent.initialize_xfer(
                "READ", local_descs, remote_descs, remote, backends=["RBLN"]
            )
        except BaseException:
            self.release(destination)
            if remote is not None:
                self.agent.remove_remote_agent(remote)
            raise
        self.inflight.append((handle, destination, remote))
        status = self.agent.transfer(handle)
        deadline = time.monotonic() + self.timeout
        while status == "PROC" and time.monotonic() < deadline:
            time.sleep(0.001)
            status = self.agent.check_xfer_state(handle)
        if status == "PROC":
            raise TimeoutError("EC VRAM read timed out; buffers remain registered")
        if status != "DONE":
            raise RuntimeError(f"EC VRAM read failed: {status}")
        # External DMA bypasses torch's writer tracking.
        from rebel import _C

        _C.vmem.mark_device_memory_is_updated(destination.tensor.data_ptr())
        self.agent.release_xfer_handle(handle)
        self.release(destination)
        self.agent.remove_remote_agent(remote)
        self.inflight.pop()
        return destination.tensor


class RblnECDeviceConnector(ECConnectorBase):
    """Pulls producer snapshots into each consumer TP rank's device cache."""

    def __init__(self, vllm_config, role: ECConnectorRole):
        super().__init__(vllm_config, role)
        if self.is_producer == self.is_consumer:
            raise ValueError("Device EC requires ec_producer or ec_consumer")
        parallel = vllm_config.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
            raise ValueError("Device EC supports TP; use PP=1 and DP=1")
        cfg = vllm_config.ec_transfer_config
        self.host = cfg.get_from_extra_config("producer_host", "127.0.0.1")
        self.port = int(cfg.get_from_extra_config("producer_port", 16100))
        self.timeout = float(cfg.get_from_extra_config("transfer_timeout", 60))
        self.capacity = int(cfg.get_from_extra_config("device_buffer_bytes", 1 << 30))
        if self.timeout <= 0 or self.capacity <= 0:
            raise ValueError("EC timeout and device buffer capacity must be positive")
        self.dtype = vllm_config.model_config.dtype
        self.consumer_id = cfg.engine_id
        self.hidden_size = vllm_config.model_config.get_inputs_embeds_size()
        self.model_signature = (
            vllm_config.model_config.model,
            vllm_config.model_config.revision,
            str(self.dtype),
            self.hidden_size,
        )
        self._saves: set[str] = set()
        self._loads: dict[str, int] = {}
        if role == ECConnectorRole.SCHEDULER:
            return
        self.rank = get_tensor_model_parallel_rank()
        self.readers = parallel.tensor_parallel_size
        self.transport = DeviceECTransport(
            torch.device(vllm_config.device_config.device),
            cfg.get_from_extra_config("listen_ip", "auto"),
            self.timeout,
        )
        self._lock = threading.Lock()
        self._leases: dict[str, DeviceLease] = {}
        self._current: dict[str, str] = {}
        self._stop = threading.Event()
        self._context = zmq.Context()
        self._socket = None
        self._thread = None
        self._service_ready = threading.Event()
        self._service_error: BaseException | None = None
        if self.is_producer and self.rank == 0:
            bind_host = cfg.get_from_extra_config("bind_host", self.host)
            self._thread = threading.Thread(
                target=self._serve, args=(bind_host,), daemon=True
            )
            self._thread.start()
            if not self._service_ready.wait(self.timeout):
                raise TimeoutError("EC descriptor service did not start")
            if self._service_error is not None:
                raise RuntimeError(
                    "EC descriptor service failed"
                ) from self._service_error
        if self.is_consumer:
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(f"tcp://{self.host}:{self.port}")

    def has_cache_item(self, identifier: str) -> bool:
        return self.is_consumer

    def update_state_after_alloc(self, request, index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        if self.is_producer:
            self._saves.add(mm_hash)
        else:
            self._loads[mm_hash] = request.get_num_encoder_embeds(index)

    def update_state_after_cache_hit(self, request, index: int) -> None:
        self.update_state_after_alloc(request, index)

    def build_connector_meta(self, scheduler_output) -> DeviceECMetadata:
        metadata = DeviceECMetadata(self._saves, self._loads)
        self._saves, self._loads = set(), {}
        return metadata

    def _reply(self, request: dict) -> dict:
        with self._lock:
            if self._stop.is_set():
                return {"status": "error", "message": "EC producer is stopping"}
            rank = request["rank"]
            if request["op"] == "get":
                generation = self._current.get(request["mm_hash"])
                if generation is None:
                    return {"status": "cached" if request.get("cached") else "pending"}
                lease = self._leases[generation]
                readers = request["readers"]
                consumer_id = request["consumer_id"]
                if readers <= 0 or not 0 <= rank < readers:
                    return {"status": "error", "message": "invalid reader rank"}
                if lease.readers is not None and lease.readers != readers:
                    return {"status": "error", "message": "reader count changed"}
                if lease.consumer_id not in (None, consumer_id):
                    return {"status": "error", "message": "consumer engine changed"}
                if rank in lease.completed:
                    return {"status": "cached" if request.get("cached") else "pending"}
                lease.readers = readers
                lease.consumer_id = consumer_id
                lease.offered.add(rank)
                return {"status": "ready", **lease.metadata}
            if request["op"] != "ack":
                return {"status": "error", "message": "unknown EC operation"}
            acknowledged = self._leases.get(request["generation"])
            if acknowledged is not None:
                if (
                    rank not in acknowledged.offered
                    or acknowledged.consumer_id != request["consumer_id"]
                ):
                    return {"status": "error", "message": "reader was not offered"}
                acknowledged.completed.add(rank)
            return {"status": "ok"}

    def _serve(self, host: str) -> None:
        socket = self._context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(f"tcp://{host}:{self.port}")
            self._service_ready.set()
            while not self._stop.is_set():
                if socket.poll(100):
                    request = msgspec.msgpack.decode(socket.recv())
                    socket.send(msgspec.msgpack.encode(self._reply(request)))
        except BaseException as error:
            self._service_error = error
        finally:
            self._service_ready.set()
            socket.close()

    def _request(self, request: dict) -> dict:
        assert self._socket is not None
        self._socket.send(
            msgspec.msgpack.encode(
                {
                    **request,
                    "rank": self.rank,
                    "consumer_id": self.consumer_id,
                }
            )
        )
        if not self._socket.poll(int(self.timeout * 1000)):
            raise TimeoutError("EC producer control request timed out")
        response = msgspec.msgpack.decode(self._socket.recv())
        if response["status"] == "error":
            raise RuntimeError(response["message"])
        return response

    def _reap(self) -> None:
        if self._service_error is not None:
            raise RuntimeError("EC descriptor service failed") from self._service_error
        with self._lock:
            done = [
                key
                for key, lease in self._leases.items()
                if lease.readers is not None and len(lease.completed) == lease.readers
            ]
            for generation in done:
                lease = self._leases[generation]
                self.transport.release(lease.buffer)
                del self._leases[generation]
                if self._current.get(lease.mm_hash) == generation:
                    del self._current[lease.mm_hash]

    def save_caches(self, encoder_cache, mm_hash: str, **kwargs) -> None:
        if not self.is_producer or self.rank != 0:
            return
        self._reap()
        source = encoder_cache[mm_hash]
        if (
            source.dtype != self.dtype
            or source.dim() != 2
            or source.shape[1] != self.hidden_size
        ):
            raise ValueError("EC output shape or dtype does not match the model")
        with self._lock:
            if mm_hash in self._current:
                return
            occupied = sum(lease.buffer.nbytes for lease in self._leases.values())
        nbytes = source.numel() * source.element_size()
        if nbytes <= 0 or occupied + nbytes > self.capacity:
            raise RuntimeError("EC device buffer capacity exceeded by pending reads")
        buffer = self.transport.snapshot(source)
        generation = uuid.uuid4().hex
        try:
            agent_metadata = self.transport.agent.get_agent_metadata()
        except BaseException:
            self.transport.release(buffer)
            raise
        metadata = {
            "generation": generation,
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "nbytes": nbytes,
            "address": buffer.address,
            "device_id": buffer.tensor.get_device(),
            "agent": agent_metadata,
            "model": self.model_signature,
        }
        with self._lock:
            self._leases[generation] = DeviceLease(mm_hash, buffer, metadata)
            self._current[mm_hash] = generation

    def start_save_caches(self, encoder_cache=None, **kwargs) -> None:
        if encoder_cache is None:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, DeviceECMetadata)
        for mm_hash in metadata.saves:
            if mm_hash in encoder_cache:
                self.save_caches(encoder_cache, mm_hash)

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, DeviceECMetadata)
        for mm_hash, num_tokens in metadata.loads.items():
            deadline = time.monotonic() + self.timeout
            while True:
                remote = self._request(
                    {
                        "op": "get",
                        "mm_hash": mm_hash,
                        "readers": self.readers,
                        "cached": mm_hash in encoder_cache,
                    }
                )
                if remote["status"] in ("ready", "cached"):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"EC output {mm_hash} was not published")
                time.sleep(0.01)
            if remote["status"] == "cached":
                continue
            shape = tuple(remote["shape"])
            if (
                shape != (num_tokens, self.hidden_size)
                or remote["dtype"] != str(self.dtype)
                or remote["nbytes"]
                != math.prod(shape) * torch.empty((), dtype=self.dtype).element_size()
                or remote["nbytes"] > self.capacity
                or tuple(remote["model"]) != self.model_signature
            ):
                raise ValueError("Remote EC shape, dtype, or size/model does not match")
            if mm_hash not in encoder_cache:
                tensor = self.transport.read(remote, shape, self.dtype)
                encoder_cache[mm_hash] = tensor
            self._request({"op": "ack", "generation": remote["generation"]})

    def get_finished(self, finished_req_ids):
        if self.is_producer and self.rank == 0:
            self._reap()
        return None, None

    def shutdown(self) -> None:
        if self.role == ECConnectorRole.SCHEDULER:
            return
        if self.transport.inflight:
            raise RuntimeError("EC reads are still in flight; retaining registrations")
        if self.is_producer and self.rank == 0:
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                self._reap()
                with self._lock:
                    active = any(
                        lease.readers is not None
                        and len(lease.completed) < lease.readers
                        for lease in self._leases.values()
                    )
                if not active:
                    break
                time.sleep(0.01)
            with self._lock:
                if any(
                    lease.readers is not None and len(lease.completed) < lease.readers
                    for lease in self._leases.values()
                ):
                    raise TimeoutError("EC consumers still hold device buffer leases")
                self._stop.set()
            assert self._thread is not None
            self._thread.join()
            for lease in self._leases.values():
                self.transport.release(lease.buffer)
            self._leases.clear()
        if self._socket is not None:
            self._socket.close()
        self._context.term()
        del self.transport.agent
