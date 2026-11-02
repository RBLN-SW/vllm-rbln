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

import pytest
import torch
import vllm.distributed.ec_transfer.ec_connector.cpu.worker as cpu_worker
from vllm.config import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.outputs import ECConnectorOutput

from tests.vllm.v1.core.utils import create_rbln_scheduler, image_at, make_request
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
