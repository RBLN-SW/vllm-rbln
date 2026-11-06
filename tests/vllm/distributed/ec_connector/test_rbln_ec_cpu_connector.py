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

import uuid
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
import vllm.distributed.ec_transfer.ec_connector.cpu.worker as cpu_worker
from vllm.config import ECTransferConfig
from vllm.config.ec_manager_config import EncoderCacheManagerConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.outputs import ECConnectorOutput

from tests.vllm.v1.core.utils import (
    create_rbln_scheduler,
    image_at,
    make_model_runner_output,
    make_request,
)
from vllm_rbln.distributed.ec_transfer.ec_connector.rbln_ec_cpu_connector import (
    RblnECCPUConnector,
    RblnECCPUWorker,
)

HIDDEN = 4


@pytest.fixture
def instance_id():
    return uuid.uuid4().hex


@pytest.fixture
def connect(monkeypatch, instance_id):
    """Build connectors that share one region, and clean it up."""
    connectors = []

    def build(*, ec_role, tp_rank=0, role=ECConnectorRole.WORKER):
        monkeypatch.setattr(
            cpu_worker, "get_tensor_model_parallel_rank", lambda: tp_rank
        )
        monkeypatch.setattr(
            cpu_worker, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0)
        )
        vllm_config = SimpleNamespace(
            instance_id=instance_id,
            use_v2_model_runner=False,
            parallel_config=SimpleNamespace(
                data_parallel_index=0,
                tensor_parallel_size=1,
                prefill_context_parallel_size=1,
                distributed_executor_backend="uni",
            ),
            model_config=SimpleNamespace(
                model="ec-test-model",
                dtype=torch.bfloat16,
                hf_config=SimpleNamespace(),
                get_inputs_embeds_size=lambda: HIDDEN,
            ),
            ec_transfer_config=ECTransferConfig(
                ec_connector="RblnECCPUConnector",
                ec_role=ec_role,
                ec_connector_extra_config={"ec_cpu_bytes": 8 * HIDDEN * 2},
            ),
        )
        connector = RblnECCPUConnector(vllm_config, role)
        connectors.append(connector)
        return connector

    yield build
    for connector in connectors:
        connector.shutdown()


@pytest.fixture
def ec_consumer(connect, monkeypatch):
    import vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq as control
    import vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl as data
    import vllm.distributed.nixl_utils as nixl

    transport = Mock()
    transport.poll.return_value = {}
    transport.poll_dead.return_value = []
    transport.connect.side_effect = RuntimeError("producer unavailable")
    monkeypatch.setattr(nixl, "NixlWrapper", object())
    monkeypatch.setattr(nixl, "nixl_agent_config", object())
    monkeypatch.setattr(data, "NixlDataTransport", lambda **kwargs: Mock())
    monkeypatch.setattr(control, "ZmqClientTransport", lambda: transport)
    connector = connect(ec_role="ec_consumer", role=ECConnectorRole.SCHEDULER)
    monkeypatch.setattr(
        ECConnectorFactory, "create_connector", lambda config, role: connector
    )
    scheduler = create_rbln_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=8,
        max_model_len=64,
        ec_transfer_config=ECTransferConfig(
            ec_connector="RblnECCPUConnector", ec_role="ec_consumer"
        ),
    )
    scheduler.max_num_encoder_input_tokens = 16
    scheduler.encoder_cache_manager = type(scheduler.encoder_cache_manager)(
        cache_size=8
    )
    return scheduler, transport


def _remote_request(features):
    for feature in features:
        feature.data = None
    request = make_request("remote", list(range(24)), 16, mm_features=features)
    request.ec_transfer_params = {
        feature.identifier: {
            "peer_host": feature.identifier,
            "peer_port": 1234,
            "size_bytes": feature.mm_position.length * HIDDEN * 2,
        }
        for feature in features
    }
    return request


def _step(connector, metadata, run):
    connector.bind_connector_metadata(metadata)
    run()
    connector.get_finished(set())
    reported = connector.build_connector_worker_meta()
    connector.clear_connector_metadata()
    return reported


def test_the_rbln_runner_gets_the_host_copy_worker(connect):
    assert isinstance(connect(ec_role="ec_producer").connector_worker, RblnECCPUWorker)


@pytest.mark.parametrize(
    ("ec_role", "nixl"),
    [("ec_producer", True), ("ec_consumer", True), ("ec_both", False)],
)
def test_nixl_without_ec_enable_nixl(connect, monkeypatch, ec_role, nixl):
    setups = []
    monkeypatch.setattr(
        ECCPUScheduler, "_setup_nixl", lambda self, cfg: setups.append(cfg)
    )

    connect(ec_role=ec_role, role=ECConnectorRole.SCHEDULER)

    assert len(setups) == int(nixl)


def test_a_saved_output_loads_back_as_is(connect):
    producer, consumer = connect(ec_role="ec_producer"), connect(ec_role="ec_consumer")
    embeds = torch.arange(3 * HIDDEN, dtype=torch.bfloat16).reshape(3, HIDDEN)
    # One block per placeholder, which can outnumber the embeddings.
    block_ids = [5, 2, 3, 7]
    encoder_cache: dict[str, torch.Tensor] = {}

    saved = _step(
        producer,
        ECCPUConnectorMetadata(saves={"img": block_ids}),
        lambda: producer.save_caches({"img": embeds}, "img"),
    )
    loaded = _step(
        consumer,
        ECCPUConnectorMetadata(loads={"img": (9, block_ids)}),
        lambda: consumer.start_load_caches(encoder_cache),
    )

    assert torch.equal(encoder_cache["img"][:3], embeds)
    assert saved == ECCPUWorkerMetadata(completed_saves=["img"])
    assert loaded == ECCPUWorkerMetadata(completed_loads=[9])
    assert producer.build_connector_worker_meta() is None


def test_only_tp_rank_0_saves(connect):
    producer = connect(ec_role="ec_producer", tp_rank=1)
    embeds = torch.ones(1, HIDDEN, dtype=torch.bfloat16)

    saved = _step(
        producer,
        ECCPUConnectorMetadata(saves={"img": [0]}),
        lambda: producer.save_caches({"img": embeds}, "img"),
    )

    assert saved is None
    assert not producer.connector_worker._region.blocks.any()


def test_a_cached_output_is_saved_at_step_entry(connect):
    producer, consumer = connect(ec_role="ec_producer"), connect(ec_role="ec_consumer")
    embeds = torch.arange(3 * HIDDEN, dtype=torch.bfloat16).reshape(3, HIDDEN)
    saved = _step(
        producer,
        ECCPUConnectorMetadata(saves={"img": [0, 1, 2], "new": [3]}),
        lambda: producer.start_save_caches(encoder_cache={"img": embeds}),
    )
    cache: dict[str, torch.Tensor] = {}
    _step(
        consumer,
        ECCPUConnectorMetadata(loads={"img": (1, [0, 1, 2])}),
        lambda: consumer.start_load_caches(cache),
    )

    assert torch.equal(cache["img"], embeds)
    assert saved == ECCPUWorkerMetadata(completed_saves=["img"])


def test_a_runner_hit_is_republished_after_shared_memory_eviction(connect):
    connector = connect(ec_role="ec_both", role=ECConnectorRole.SCHEDULER)
    producer, consumer = connect(ec_role="ec_both"), connect(ec_role="ec_consumer")
    request = make_request("repeat", list(range(8)), 16, mm_features=[image_at(0, 3)])
    replacement = make_request(
        "replacement", list(range(8)), 16, mm_features=[image_at(0, 8)]
    )
    replacement.mm_features[0].identifier = "replacement"
    mm_hash = request.mm_features[0].identifier
    embeds = torch.arange(3 * HIDDEN, dtype=torch.bfloat16).reshape(3, HIDDEN)
    runner_cache = {mm_hash: embeds}

    for req, tensor in (
        (request, embeds),
        (replacement, torch.full((8, HIDDEN), 7, dtype=torch.bfloat16)),
    ):
        key = req.mm_features[0].identifier
        connector.update_state_after_alloc(req, 0)
        metadata = connector.build_connector_meta(None)
        reported = _step(
            producer,
            metadata,
            lambda key=key, tensor=tensor: producer.save_caches({key: tensor}, key),
        )
        connector.update_connector_output(
            ECConnectorOutput(ec_connector_worker_meta=reported)
        )
    assert connector.connector_scheduler._cache.get(mm_hash) is None

    scheduler = create_rbln_scheduler(
        max_num_seqs=4, max_num_batched_tokens=8, max_model_len=64
    )
    scheduler.ec_connector = connector
    scheduler.max_num_encoder_input_tokens = 16
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    scheduler.encoder_cache_manager.allocate(request, 0)
    scheduler.encoder_cache_manager.free(request)
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 8}
    assert output.scheduled_encoder_inputs == {}
    metadata = output.ec_connector_metadata
    assert mm_hash in metadata.saves
    assert metadata.loads == {}
    connector.update_state_after_cache_hit(request, 0)
    assert connector.build_connector_meta(None).saves == {}
    reported = _step(
        producer,
        metadata,
        lambda: producer.start_save_caches(encoder_cache=runner_cache),
    )
    connector.update_connector_output(
        ECConnectorOutput(ec_connector_worker_meta=reported)
    )
    assert connector.connector_scheduler._cache.get(mm_hash).ready
    connector.update_state_after_cache_hit(request, 0)
    repeated = connector.build_connector_meta(None)
    assert repeated.saves == {}
    assert repeated.loads == {}

    cache: dict[str, torch.Tensor] = {}
    _step(
        consumer,
        ECCPUConnectorMetadata(loads={mm_hash: (1, metadata.saves[mm_hash])}),
        lambda: consumer.start_load_caches(cache),
    )
    assert torch.equal(cache[mm_hash], embeds)


@pytest.mark.parametrize("running", [False, True])
def test_consumer_reuses_runner_hit_without_contacting_producer(ec_consumer, running):
    scheduler, transport = ec_consumer
    request = _remote_request([image_at(8, 3)])
    params = request.ec_transfer_params
    request.ec_transfer_params = None
    scheduler.add_request(request)
    if running:
        first = scheduler.schedule()
        scheduler.update_from_output(first, make_model_runner_output(first))
    request.ec_transfer_params = params
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    manager.free(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 8}
    assert output.scheduled_encoder_inputs == {}
    assert transport.connect.call_count == 0
    if not running:
        assert (
            request.request_id not in manager.cached[request.mm_features[0].identifier]
        )


def test_consumer_still_fetches_uncached_inputs_in_a_mixed_request(ec_consumer):
    scheduler, transport = ec_consumer
    request = _remote_request([image_at(0, 2), image_at(2, 2)])
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    manager.free(request)

    assert scheduler._ec_transfer_pending(request, 0)
    assert transport.connect.call_args_list == [call(("image@2", 1234))]
    assert scheduler.ec_connector.take_unavailable_requests() == {request.request_id}


def test_consumer_preserves_explicit_manager_and_uses_connector_admission(ec_consumer):
    original, transport = ec_consumer
    scheduler = create_rbln_scheduler(
        ec_transfer_config=original.vllm_config.ec_transfer_config,
        ec_manager_config=EncoderCacheManagerConfig(
            encoder_cache_manager_cls=(
                "vllm.v1.core.encoder_cache_manager.EncoderCacheManager"
            )
        ),
    )
    assert type(scheduler.encoder_cache_manager) is EncoderCacheManager
    scheduler.encoder_cache_manager = type(scheduler.encoder_cache_manager)(
        cache_size=8
    )
    request = _remote_request([image_at(0, 3)])
    scheduler.encoder_cache_manager.allocate(request, 0)
    scheduler.encoder_cache_manager.free(request)

    assert scheduler._ec_transfer_pending(request, 0)
    assert transport.connect.call_count == 1


def test_consumer_refetches_a_runner_hit_evicted_before_readmission(ec_consumer):
    scheduler, transport = ec_consumer
    request = _remote_request([image_at(0, 3)])
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    manager.free(request)
    assert not scheduler._ec_transfer_pending(request, 0)
    assert transport.connect.call_count == 0

    replacement = make_request(
        "replacement", list(range(24)), 16, mm_features=[image_at(8, 8)]
    )
    assert manager.can_allocate(replacement, 0, 16, 0)
    manager.allocate(replacement, 0)
    assert scheduler._ec_transfer_pending(request, 0)
    assert transport.connect.call_count == 1


def test_consumer_rechecks_embedding_count_before_reusing_a_runner_hit(ec_consumer):
    scheduler, transport = ec_consumer
    request = _remote_request([image_at(0, 3)])
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    manager.free(request)
    assert not scheduler._ec_transfer_pending(request, 0)
    assert transport.connect.call_count == 0

    request = _remote_request([image_at(0, 4)])
    assert scheduler._ec_transfer_pending(request, 0)
    assert transport.connect.call_count == 1


def test_consumer_defers_a_runner_hit_evicted_during_scheduling(ec_consumer):
    scheduler, transport = ec_consumer
    request = _remote_request([image_at(0, 2), image_at(2, 2)])
    other = make_request("other", list(range(8)), 16, mm_features=[image_at(8, 2)])
    manager = type(scheduler.encoder_cache_manager)(cache_size=4)
    scheduler.encoder_cache_manager = manager
    manager.allocate(request, 1)
    manager.free(request)
    manager.allocate(other, 0)
    manager.free(other)
    cpu_cache = scheduler.ec_connector.connector_scheduler._cache
    first_hash = request.mm_features[0].identifier
    cpu_cache.alloc(first_hash, 2)
    cpu_cache.mark_ready(first_hash)
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 2}
    assert output.scheduled_encoder_inputs == {}
    assert request.request_id not in scheduler.encoder_cache_mismatch_reqs
    assert transport.connect.call_count == 0
    assert manager.can_allocate(request, 1, 16, 0)
    assert scheduler._ec_transfer_pending(request, 2)
    assert transport.connect.call_count == 1


def test_an_output_in_another_dtype_is_refused(connect):
    # A load reads the bytes back in the model dtype.
    producer = connect(ec_role="ec_producer")
    embeds = torch.ones(1, HIDDEN // 2, dtype=torch.float32)
    producer.bind_connector_metadata(ECCPUConnectorMetadata(saves={"img": [0]}))

    with pytest.raises(AssertionError, match="does not fit"):
        producer.save_caches({"img": embeds}, "img")


def test_the_vllm_path_registers_the_connector():
    cls = ECConnectorFactory._registry["RblnECCPUConnector"]()
    assert cls is RblnECCPUConnector
