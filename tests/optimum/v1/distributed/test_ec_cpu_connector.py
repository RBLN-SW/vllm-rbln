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

"""Optimum scheduler/runner integration with real CPU cache storage.

Compiled model execution and NIXL transport are replaced; connector allocation,
copies, runner contexts, and scheduler completion handling run unchanged.
"""

import uuid
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.ec_transfer.ec_connector.cpu.worker as cpu_worker
import vllm.distributed.ec_transfer.ec_transfer_state as ec_state
from transformers import PretrainedConfig
from vllm.config import ECTransferConfig, MultiModalConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.request import RequestStatus

import vllm_rbln.v1.worker.optimum_model_runner as runner_module
from vllm_rbln import register_ops
from vllm_rbln.model_executor.models.optimum.model_base import (
    RBLNOptimumMultimodalMixin,
)
from vllm_rbln.v1.core.optimum_scheduler import RBLNOptimumScheduler
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

from ..core.utils import create_requests, create_scheduler
from ..worker.utils import MockModelWrapper, fake_load_model


class _MultimodalModel(MockModelWrapper, RBLNOptimumMultimodalMixin):
    def __init__(self, dtype, vocab_size):
        super().__init__(max_num_seqs=4, dtype=dtype)
        self.vocab_size = vocab_size
        self.embeddings = torch.arange(24, dtype=dtype).reshape(2, 12)
        self.encoded = 0
        self.forward_inputs = []

    def embed_multimodal(self, **kwargs):
        self.encoded += 1
        return [self.embeddings.clone()]

    def build_prefill_forward_inputs(self, model_input):
        return model_input

    def build_decode_forward_inputs(self, model_input):
        return model_input

    def forward(self, model_input):
        self.forward_inputs.append(model_input)
        return torch.zeros(
            model_input.padded_batch_size, 1, self.vocab_size, dtype=self.dtype
        )


@pytest.fixture
def instances(monkeypatch):
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25")
    register_ops()
    monkeypatch.setattr(
        "vllm_rbln.utils.optimum.converter.dispatch._resolve_rbln_config",
        lambda _: None,
    )
    monkeypatch.setattr(cpu_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        cpu_worker, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(
        runner_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    monkeypatch.setattr(
        runner_module,
        "group_and_batch_mm_kwargs",
        lambda items, **kwargs: (("image", 1, {}) for _ in items),
    )
    # These tests start at the local CPU storage boundary, after a remote read.
    monkeypatch.setattr(
        ECCPUScheduler,
        "_setup_nixl",
        lambda self, config: setattr(self, "_nixl_enabled", False),
    )
    monkeypatch.setattr(ec_state, "_EC_CONNECTOR_AGENT", None)
    connectors = []

    def create(role):
        template = create_scheduler()
        config = template.vllm_config
        config.instance_id = uuid.uuid4().hex
        config.model_config.dtype = torch.bfloat16
        config.model_config.multimodal_config = MultiModalConfig(
            mm_encoder_only=role == "ec_producer"
        )
        config.model_config.hf_config.vision_config = PretrainedConfig(
            out_hidden_size=4, deepstack_visual_indexes=[0, 1]
        )
        config.additional_config.use_custom_sampler = False
        config.ec_transfer_config = ECTransferConfig(
            ec_connector="RblnECCPUConnector",
            ec_role=role,
            ec_connector_extra_config={"ec_cpu_bytes": 16 * 12 * 2},
        )
        scheduler = RBLNOptimumScheduler(
            config,
            template.kv_cache_config,
            template.structured_output_manager,
            block_size=template.block_size,
        )
        connectors.append(scheduler.ec_connector)
        worker = ECConnectorFactory.create_connector(config, ECConnectorRole.WORKER)
        connectors.append(worker)
        runner = RBLNOptimumModelRunner(config, "cpu")
        fake_load_model(runner)
        runner.model = _MultimodalModel(
            config.model_config.dtype, config.model_config.get_vocab_size()
        )
        runner.supports_mm_inputs = True
        monkeypatch.setattr(ec_state, "_EC_CONNECTOR_AGENT", worker)
        return scheduler, runner

    yield create
    for connector in reversed(connectors):
        connector.shutdown()


def _request():
    return create_requests(
        num_requests=1,
        num_tokens=4,
        max_tokens=2,
        mm_hashes_list=[["img"]],
        mm_positions=[[PlaceholderRange(offset=1, length=2)]],
    )[0]


def test_producer_save_becomes_readable_after_optimum_runner_report(instances):
    scheduler, runner = instances("ec_producer")
    request = _request()
    scheduler.add_request(request)

    scheduled = scheduler.schedule()
    assert len(scheduled.ec_connector_metadata.saves["img"]) == 2
    entry = scheduler.ec_connector.connector_scheduler._cache.get("img")
    assert not entry.ready

    output = runner.execute_model(scheduled)
    assert output.ec_connector_output.ec_connector_worker_meta.completed_saves == [
        "img"
    ]
    assert output.sampled_token_ids == [[]]
    assert runner.model.encoded == 1
    assert not runner.model.forward_inputs
    scheduler.update_from_output(scheduled, output)

    assert entry.ready
    assert request.status == RequestStatus.FINISHED_STOPPED
    stored = scheduler.ec_connector.connector_scheduler._region.blocks[
        list(entry.block_ids)
    ]
    torch.testing.assert_close(stored.view(torch.bfloat16), runner.model.embeddings)


def test_consumer_loads_cpu_outputs_before_optimum_prefill(instances):
    scheduler, runner = instances("ec_consumer")
    cache = scheduler.ec_connector.connector_scheduler
    entry = cache._cache.alloc("img", 2)
    cache._region.blocks[list(entry.block_ids)] = runner.model.embeddings.view(
        torch.int8
    )
    cache._cache.mark_ready("img")
    scheduler.add_request(_request())

    scheduled = scheduler.schedule()
    transfer_id, _ = scheduled.ec_connector_metadata.loads["img"]
    assert runner.execute_model(scheduled) is None
    output = runner.sample_tokens(None)
    scheduler.update_from_output(scheduled, output)

    assert runner.model.encoded == 0
    torch.testing.assert_close(
        runner.model.forward_inputs[0].mm_embeds[0], runner.model.embeddings
    )
    assert runner.model.forward_inputs[0].is_mm_embed.tolist() == [
        [False, True, True, False]
    ]
    assert output.ec_connector_output.ec_connector_worker_meta.completed_loads == [
        transfer_id
    ]
    assert not cache._load_acks


def test_ec_both_reloads_evicted_outputs_and_still_runs_language_model(instances):
    scheduler, runner = instances("ec_both")
    first = _request()
    scheduler.add_request(first)
    scheduled = scheduler.schedule()
    assert runner.execute_model(scheduled) is None
    output = runner.sample_tokens(None)
    scheduler.update_from_output(scheduled, output)
    scheduler.finish_requests(first.request_id, RequestStatus.FINISHED_ABORTED)
    runner.encoder_cache.clear()

    second = _request()
    second.request_id = "second"
    scheduler.add_request(second)
    scheduled = scheduler.schedule()
    assert "img" in scheduled.ec_connector_metadata.loads
    assert not scheduled.ec_connector_metadata.saves
    assert runner.execute_model(scheduled) is None
    output = runner.sample_tokens(None)
    scheduler.update_from_output(scheduled, output)

    assert runner.model.encoded == 1
    assert len(runner.model.forward_inputs) == 2
    torch.testing.assert_close(
        runner.model.forward_inputs[-1].mm_embeds[0], runner.model.embeddings
    )
    assert output.sampled_token_ids and len(output.sampled_token_ids[0]) == 1
    assert not scheduler.ec_connector.connector_scheduler._load_acks
