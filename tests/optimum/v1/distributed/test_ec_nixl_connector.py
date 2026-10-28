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

import ctypes
import queue
import socket
import sys
import threading
from types import SimpleNamespace

import msgspec
import pytest
import torch
from vllm.config import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole

import vllm_rbln.distributed.ec_transfer.ec_connector.rbln_ec_nixl_connector as conn
from vllm_rbln.distributed.ec_transfer.ec_connector.rbln_ec_nixl_connector import (
    _DTYPE_SIZES,
    ECNixlAck,
    ECNixlMetadata,
    RblnECNixlConnector,
    RblnECNixlConnectorWorker,
    _dtype_size,
)


def test_dtype_size_known_dtypes():
    assert _dtype_size("torch.float16") == 2
    assert _dtype_size("torch.bfloat16") == 2
    assert _dtype_size("torch.float32") == 4
    assert _dtype_size("torch.int64") == 8
    assert _dtype_size("torch.int32") == 4


def test_dtype_size_covers_registered_table():
    # Keep _DTYPE_SIZES and _dtype_size in lockstep — if a new dtype is
    # added to the table, this test forces the lookup to stay wired up.
    for dtype_str, expected in _DTYPE_SIZES.items():
        assert _dtype_size(dtype_str) == expected


def test_dtype_size_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _dtype_size("torch.float64")


def test_ec_both_is_rejected():
    vllm_config = SimpleNamespace(
        ec_transfer_config=ECTransferConfig(
            ec_connector="RblnECNixlConnector", ec_role="ec_both"
        )
    )

    with pytest.raises(ValueError, match="not 'ec_both'"):
        RblnECNixlConnector(vllm_config, ECConnectorRole.SCHEDULER)


class _FakeNixlAgent:
    """Enough of the NIXL agent for one CPU-to-CPU pull: descriptors are the
    (addr, nbytes, ...) tuples themselves and transfer() is a memmove."""

    def get_reg_descs(self, data, _mem_type):
        return data

    def register_memory(self, descs, backends):
        pass

    def deregister_memory(self, descs):
        pass

    def get_agent_metadata(self):
        return b"encoder-agent"

    def add_remote_agent(self, agent_metadata):
        return "encoder"

    def remove_remote_agent(self, name):
        pass

    def prep_xfer_dlist(self, _agent, xfer_data, _mem_type):
        return xfer_data

    def make_prepped_xfer(self, _op, local, _li, remote, _ri, notif_msg):
        return (local, remote)

    def transfer(self, handle):
        local, remote = handle
        for (dst, nbytes, _), (src, _, _) in zip(local, remote):
            ctypes.memmove(dst, src, nbytes)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        pass


class _Sink:
    def __init__(self):
        self.messages = []

    def send(self, data):
        self.messages.append(data)


def _producer(agent, llm_tp_size=1):
    w = object.__new__(RblnECNixlConnectorWorker)
    w._is_producer = True
    w._is_consumer = False
    w._is_sending_rank = True
    w._nixl_agent = agent
    w._backends = ["UCX"]
    w._engine_id = "encoder"
    w._incoming_acks = queue.Queue()
    w._registered_caches = {}
    w._registered_descs = {}
    w._registered_timestamps = {}
    w._unacked = {}
    w._producer_cache_capacity = 4
    w._producer_cache_ttl_s = 60.0
    w._ack_host = ""
    w._ack_port = 0
    w._push_socks = [_Sink() for _ in range(llm_tp_size)]
    return w


def _consumer(agent):
    w = object.__new__(RblnECNixlConnectorWorker)
    w._is_producer = False
    w._is_consumer = True
    w._is_receiving_rank = True
    w._nixl_agent = agent
    w._backends = ["UCX"]
    w._incoming_metadata = queue.Queue()
    w._remote_agents = {}
    w._tensor_registry = {}
    w._mm_hash_ack_addr = {}
    w._known_ack_addrs = {}
    w._known_ack_addrs_lock = threading.Lock()
    w._cache_events = {}
    w._cache_events_lock = threading.Lock()
    w._pending_loads = {}
    w._ack_sent = set()
    return w


def test_pull_delivers_the_cached_tensor_as_is():
    # The runner caches one tensor per mm_hash; the consumer must find the
    # same tensor (shape, dtype, values) in its encoder cache, not a wrapper.
    agent = _FakeNixlAgent()
    producer, consumer = _producer(agent), _consumer(agent)
    embeds = torch.arange(12, dtype=torch.float16).reshape(3, 4)

    producer.save_caches({"img": embeds}, "img")

    (pushed,) = producer._push_socks[0].messages
    meta = msgspec.msgpack.Decoder(ECNixlMetadata).decode(pushed)
    assert meta.mm_hash == "img"
    assert meta.tensor.shape == [3, 4]
    assert meta.tensor.dtype_str == "torch.float16"

    consumer._incoming_metadata.put(meta)
    assert consumer._process_pending_metadata() == {"img"}
    consumer._pending_loads["img"] = consumer._initiate_pull("img")
    encoder_cache: dict[str, torch.Tensor] = {}
    consumer._wait_for_pulls(encoder_cache)

    assert not consumer._pending_loads
    assert torch.equal(encoder_cache["img"], embeds)


def test_only_encoder_tp_rank_0_sends():
    producer = _producer(_FakeNixlAgent())
    producer._is_sending_rank = False

    producer.save_caches({"img": torch.ones(2, 2)}, "img")

    assert producer._push_socks[0].messages == []
    assert not producer._registered_caches


def test_every_llm_rank_is_sent_to_and_acks_before_eviction():
    producer = _producer(_FakeNixlAgent(), llm_tp_size=2)

    producer.save_caches({"img": torch.ones(2, 2)}, "img")

    assert [len(sock.messages) for sock in producer._push_socks] == [1, 1]
    producer._incoming_acks.put(ECNixlAck(mm_hash="img", engine_id="encoder"))
    producer._drain_acks()
    assert "img" in producer._registered_caches
    producer._incoming_acks.put(ECNixlAck(mm_hash="img", engine_id="encoder"))
    producer._drain_acks()
    assert "img" not in producer._registered_caches


def test_llm_ranks_of_one_instance_bind_without_conflict(monkeypatch):
    # Every worker of a TP > 1 or PP > 1 LLM instance builds its own connector.
    fake_nixl = SimpleNamespace(nixl_agent=lambda name, config: _FakeNixlAgent())
    monkeypatch.setitem(sys.modules, "nixl", SimpleNamespace(_api=fake_nixl))
    monkeypatch.setitem(sys.modules, "nixl._api", fake_nixl)
    monkeypatch.setattr(conn, "model_parallel_is_initialized", lambda: True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        pull_port = probe.getsockname()[1]
    vllm_config = SimpleNamespace(
        ec_transfer_config=ECTransferConfig(
            ec_connector="RblnECNixlConnector",
            ec_role="ec_consumer",
            ec_connector_extra_config={
                "pull_host": "127.0.0.1",
                "pull_port": pull_port,
            },
        )
    )

    workers = []
    try:
        for tp_rank, first_pp_rank in [(0, True), (1, True), (0, False)]:
            monkeypatch.setattr(
                conn,
                "get_tp_group",
                lambda r=tp_rank: SimpleNamespace(rank_in_group=r),
            )
            monkeypatch.setattr(
                conn,
                "get_pp_group",
                lambda f=first_pp_rank: SimpleNamespace(is_first_rank=f),
            )
            workers.append(
                RblnECNixlConnectorWorker(vllm_config, ECConnectorRole.WORKER)
            )
    finally:
        for worker in workers:
            worker.shutdown()

    assert [getattr(w, "_pull_addr", None) for w in workers] == [
        f"tcp://127.0.0.1:{pull_port}",
        f"tcp://127.0.0.1:{pull_port + 1}",
        None,
    ]
