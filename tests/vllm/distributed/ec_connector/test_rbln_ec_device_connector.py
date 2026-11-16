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

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

import vllm_rbln.distributed.ec_transfer.ec_connector.rbln_ec_device_connector as ec
from tests.vllm.v1.core.utils import create_rbln_scheduler, image_at, make_request


def _config(role, port, *, device="rbln:0", readers=1, timeout=1):
    return SimpleNamespace(
        ec_transfer_config=ECTransferConfig(
            ec_connector="RblnECDeviceConnector",
            ec_role=role,
            ec_connector_extra_config={
                "producer_port": port,
                "transfer_timeout": timeout,
                "listen_ip": os.environ.get("RBLN_RDMA_IP", "auto"),
            },
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=readers,
        ),
        model_config=SimpleNamespace(
            model="ec-model",
            revision="main",
            dtype=torch.bfloat16,
            get_inputs_embeds_size=lambda: 64,
        ),
        device_config=SimpleNamespace(device=torch.device(device)),
    )


@pytest.fixture
def connectors(monkeypatch):
    buffers = {}

    class Transport:
        def __init__(self, *args):
            self.inflight = []
            self.agent = Mock()
            self.agent.get_agent_metadata.return_value = b"metadata"

        def snapshot(self, source):
            tensor = source.clone()
            address = tensor.data_ptr()
            buffers[address] = tensor
            return ec.DeviceBuffer(
                tensor, address, tensor.numel() * tensor.element_size(), address
            )

        def release(self, buffer):
            del buffers[buffer.address]

        def read(self, metadata, shape, dtype):
            return buffers[metadata["address"]].clone()

    port = 16100
    created: list[ec.RblnECDeviceConnector] = []

    def build(role, *, rank=0, readers=1):
        connector = ec.RblnECDeviceConnector(
            _config(role, port, readers=readers), ECConnectorRole.SCHEDULER
        )
        connector._role = ECConnectorRole.WORKER
        connector.rank, connector.readers = rank, readers
        connector.transport = Transport()
        connector._lock = threading.Lock()
        connector._leases, connector._current = {}, {}
        connector._stop = threading.Event()
        connector._service_error = None
        connector._thread = Mock()
        connector._socket = None
        connector._context = Mock()
        if connector.is_consumer:
            connector._request = lambda request: created[0]._reply(
                {
                    **request,
                    "rank": rank,
                    "consumer_id": "consumer",
                }
            )
        created.append(connector)
        return connector

    yield build
    for connector in reversed(created):
        connector.shutdown()


def _load(consumer, cache):
    consumer.bind_connector_metadata(ec.DeviceECMetadata(loads={"image": 2}))
    consumer.start_load_caches(cache)


def test_snapshot_survives_encoder_output_reuse(connectors):
    producer = connectors("ec_producer")
    consumer = connectors("ec_consumer")
    source = torch.full((2, 64), 65536, dtype=torch.bfloat16)
    producer.save_caches({"image": source}, "image")
    source.zero_()
    cache: dict[str, torch.Tensor] = {}
    _load(consumer, cache)
    assert cache["image"].dtype == torch.bfloat16
    assert torch.equal(cache["image"], torch.full_like(source, 65536))
    producer._reap()
    assert not producer._leases


def test_snapshot_is_retained_until_all_tp_readers_finish(connectors):
    producer = connectors("ec_producer")
    first = connectors("ec_consumer", readers=2)
    second = connectors("ec_consumer", rank=1, readers=2)
    source = torch.ones(2, 64, dtype=torch.bfloat16)
    producer.save_caches({"image": source}, "image")
    _load(first, {})
    producer._reap()
    assert len(producer._leases) == 1
    _load(second, {})
    producer._reap()
    assert not producer._leases


def test_cached_consumer_acknowledges_republished_snapshot(connectors):
    producer = connectors("ec_producer")
    consumer = connectors("ec_consumer")
    source = torch.ones(2, 64, dtype=torch.bfloat16)
    cache: dict[str, torch.Tensor] = {}
    for _ in range(2):
        producer.save_caches({"image": source}, "image")
        _load(consumer, cache)
        producer._reap()
        assert not producer._leases


@pytest.mark.parametrize("reap", [False, True])
def test_later_prefill_chunk_uses_cache_without_waiting_for_publication(
    connectors, reap
):
    producer = connectors("ec_producer")
    consumer = connectors("ec_consumer")
    source = torch.ones(2, 64, dtype=torch.bfloat16)
    producer.save_caches({"image": source}, "image")
    cache: dict[str, torch.Tensor] = {}
    _load(consumer, cache)
    first = cache["image"]
    if reap:
        producer._reap()
    _load(consumer, cache)
    assert cache["image"] is first
    producer._reap()
    assert not producer._leases


def test_remote_dtype_is_rejected_before_read(connectors):
    consumer = connectors("ec_consumer")
    consumer._request = Mock(
        return_value={
            "status": "ready",
            "shape": [2, 64],
            "dtype": "torch.float16",
            "nbytes": 256,
        }
    )
    consumer.transport.read = Mock()
    with pytest.raises(ValueError, match="shape, dtype, or size"):
        _load(consumer, {})
    consumer.transport.read.assert_not_called()


def test_unoffered_ack_cannot_release_snapshot(connectors):
    producer = connectors("ec_producer")
    producer.save_caches({"image": torch.ones(2, 64, dtype=torch.bfloat16)}, "image")
    generation = producer._current["image"]
    response = producer._reply(
        {
            "op": "ack",
            "rank": 0,
            "generation": generation,
            "consumer_id": "consumer",
        }
    )
    assert response["status"] == "error"
    producer._reap()
    assert generation in producer._leases


def test_other_consumer_engine_cannot_share_rank_leases(connectors):
    producer = connectors("ec_producer")
    producer.save_caches({"image": torch.ones(2, 64, dtype=torch.bfloat16)}, "image")
    request = {
        "op": "get",
        "mm_hash": "image",
        "rank": 0,
        "readers": 1,
        "consumer_id": "first",
    }
    offered = producer._reply(request)
    assert offered["status"] == "ready"
    rejected = producer._reply({**request, "consumer_id": "second"})
    assert rejected["status"] == "error"
    producer._reply(
        {
            "op": "ack",
            "rank": 0,
            "consumer_id": "first",
            "generation": offered["generation"],
        }
    )
    producer._reap()


def test_shutdown_waits_for_tp_rank_not_yet_offered(connectors):
    producer = connectors("ec_producer")
    first = connectors("ec_consumer", readers=2)
    second = connectors("ec_consumer", rank=1, readers=2)
    producer.save_caches({"image": torch.ones(2, 64, dtype=torch.bfloat16)}, "image")
    _load(first, {})
    producer.timeout = 0
    with pytest.raises(TimeoutError, match="consumers still hold"):
        producer.shutdown()
    assert len(producer._leases) == 1
    _load(second, {})
    producer._reap()


def test_remote_model_is_rejected_before_read(connectors):
    consumer = connectors("ec_consumer")
    consumer._request = Mock(
        return_value={
            "status": "ready",
            "shape": [2, 64],
            "dtype": "torch.bfloat16",
            "nbytes": 256,
            "model": ("other-model", "main", "torch.bfloat16", 64),
        }
    )
    consumer.transport.read = Mock()
    with pytest.raises(ValueError, match="size/model"):
        _load(consumer, {})
    consumer.transport.read.assert_not_called()


def test_pending_snapshots_are_not_evicted_to_meet_capacity(connectors):
    producer = connectors("ec_producer")
    producer.capacity = 256
    source = torch.ones(2, 64, dtype=torch.bfloat16)
    producer.save_caches({"image": source}, "image")
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        producer.save_caches({"other": source}, "other")
    assert list(producer._current) == ["image"]


def test_timed_out_read_keeps_device_allocation_registered():
    transport = ec.DeviceECTransport.__new__(ec.DeviceECTransport)
    transport.timeout = 0
    transport.inflight = []
    transport.agent = Mock()
    transport.agent.transfer.return_value = "PROC"
    tensor = Mock()
    tensor.get_device.return_value = 0
    buffer = ec.DeviceBuffer(tensor, 100, 256, object())
    transport.allocate = Mock(return_value=buffer)
    with pytest.raises(TimeoutError, match="buffers remain registered"):
        transport.read(
            {"agent": b"metadata", "address": 200, "device_id": 0},
            (2, 64),
            torch.bfloat16,
        )
    assert transport.inflight[0][1] is buffer
    transport.agent.deregister_memory.assert_not_called()
    transport.agent.release_xfer_handle.assert_not_called()


@pytest.mark.parametrize("role", ["ec_producer", "ec_consumer"])
def test_runner_cache_hit_still_schedules_device_lease_handshake(
    role, tmp_path, monkeypatch
):
    from transformers import OPTConfig

    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(tmp_path)
    monkeypatch.setattr(
        "tests.vllm.v1.core.utils.local_model_path", lambda model: str(tmp_path)
    )
    connector = ec.RblnECDeviceConnector(
        _config(role, 16100), ECConnectorRole.SCHEDULER
    )
    scheduler = create_rbln_scheduler(
        model=str(tmp_path), max_num_seqs=4, max_num_batched_tokens=8, max_model_len=64
    )
    scheduler.ec_connector = connector
    scheduler.max_num_encoder_input_tokens = 16
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    request = make_request("cached", list(range(8)), 16, mm_features=[image_at(0, 3)])
    scheduler.encoder_cache_manager.allocate(request, 0)
    scheduler.encoder_cache_manager.free(request)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs == {}
    metadata = output.ec_connector_metadata
    mm_hash = request.mm_features[0].identifier
    if role == "ec_producer":
        assert mm_hash in metadata.saves
    else:
        assert metadata.loads == {mm_hash: 3}


@pytest.mark.use_device
def test_npu_transfer_preserves_bf16_without_d2h(monkeypatch):
    from rebel.core.torch_compile import rbln_backend

    assert torch.rbln.device_count() >= 2, "This test requires two visible NPUs"
    port = get_open_port()
    monkeypatch.setattr(ec, "get_tensor_model_parallel_rank", lambda: 0)
    producer = ec.RblnECDeviceConnector(
        _config("ec_producer", port), ECConnectorRole.WORKER
    )
    consumer = ec.RblnECDeviceConnector(
        _config("ec_consumer", port, device="rbln:1"), ECConnectorRole.WORKER
    )
    try:
        expected = torch.full((2, 64), 65536, dtype=torch.bfloat16)
        matmul = torch.compile(
            torch.matmul,
            backend=rbln_backend,
            fullgraph=True,
            options={"dtype": "bfloat16", "use_static_output": True},
        )
        lhs = torch.eye(2, dtype=torch.bfloat16, device="rbln:0")
        rhs = expected.to("rbln:0")
        source = matmul(lhs, rhs)
        with torch.rbln.explain(with_stack=True) as trace:
            producer.save_caches({"image": source}, "image")
            source.zero_()
            cache: dict[str, torch.Tensor] = {}
            _load(consumer, cache)
            producer._reap()
        residency = trace.dump()["runtime_residency"]
        print(trace.report(), residency, flush=True)
        assert residency["available"]
        assert residency["real_host_sync_d2h"] == {"count": 0, "bytes": 0}
        assert cache["image"].device == torch.device("rbln:1")
        assert cache["image"].dtype == torch.bfloat16
        torch.testing.assert_close(cache["image"].cpu(), expected, atol=0, rtol=0)
        assert not producer._leases
    finally:
        consumer.shutdown()
        producer.shutdown()
