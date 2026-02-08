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
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
from vllm import SamplingParams, TextPrompt
from vllm.config import (CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig,
                         VllmConfig, set_current_vllm_config)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.utils import merge_async_iterators

from vllm_rbln.model_executor.models.optimum import ModelInputForRBLN
from vllm_rbln.model_executor.models.optimum.model_base import (
    KVCacheBlockAdapter)

NUM_LORAS = 5
BLOCK_SIZE = 16
NUM_BLOCKS = 8
BATCH_SIZE = 4
MAX_LORA_RANK = 8
MAX_MODEL_LEN = 128
MODEL_PATH = "facebook/opt-125m"
VOCAB_SIZE = 32000

V0_PATH = "vllm_rbln.worker.optimum_model_runner.RBLNOptimumModelRunner.load_model"  # noqa
V1_PATH = "vllm_rbln.v1.worker.optimum_model_runner.RBLNOptimumModelRunner.load_model"  # noqa

result = []
golden = []


def get_vllm_config(async_scheduling=False):
    model_config = ModelConfig(
        MODEL_PATH,
        dtype=torch.float,
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=BATCH_SIZE,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        async_scheduling=async_scheduling,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        swap_space=0,
        cache_dtype="auto",
    )
    lora_config = LoRAConfig(max_lora_rank=MAX_LORA_RANK,
                             max_cpu_loras=NUM_LORAS,
                             max_loras=NUM_LORAS)
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=lora_config,
    )
    return vllm_config


def get_lora_requests():
    lora_requests = [
        LoRARequest(str(i + 1), i + 1, "/path/adapter" + str(i + 1))
        for i in range(NUM_LORAS)
    ]
    return lora_requests


def parse_lora_int_ids(running_requests_ids):
    lora_ids = []
    for running_request in running_requests_ids:
        lora_ids.append(int(running_request.split("-")[1]))
    return lora_ids


async def add_lora_request(llm, lora_int_ids):
    lora_requests = [
        LoRARequest(str(lora_int_id), lora_int_id,
                    "/path/adapter" + str(lora_int_id))
        for lora_int_id in lora_int_ids
    ]
    sampling_params = SamplingParams(n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     ignore_eos=True,
                                     max_tokens=2)

    generators = []

    for i, lora_request in enumerate(lora_requests):
        lora_int_id = lora_request.lora_int_id
        generator = llm.generate(prompt=TextPrompt(
            prompt=f"hello {lora_int_id}", multi_modal_data=None),
                                 sampling_params=sampling_params,
                                 lora_request=lora_request,
                                 request_id=f"REQ{i}:LORA-{lora_int_id}")
        generators.append(generator)

    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        pass


class MockModelWrapper(nn.Module):

    class MockModel:

        def __init__(self):
            self.rbln_config = SimpleNamespace(lora_config=SimpleNamespace(
                adapters=[
                    type("RBLNLoRAAdapterConfig", (), {"lora_int_id": i + 1})()
                    for i in range(NUM_LORAS)
                ]))

        def set_lora_int_ids(self, lora_int_ids):
            self.lora_int_ids = lora_int_ids

    def __init__(self):
        super().__init__()
        self.model = self.MockModel()
        self.logits_processor = LogitsProcessor(VOCAB_SIZE,
                                                logits_as_input=True)
        self.sampler = Sampler()

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        request_nums = input_ids.shape[0]
        fake_logits = torch.zeros(request_nums, 1, VOCAB_SIZE)

        running_requests_ids = model_input.running_requests_ids
        parsed_lora_int_ids = parse_lora_int_ids(running_requests_ids)
        result.append(parsed_lora_int_ids)
        golden.append(self.model.lora_int_ids)

        return fake_logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.logits_processor(None, hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


def fake_load_model(self):
    with set_current_vllm_config(self.vllm_config, check_compile=False):
        self.model = MockModelWrapper()
        self.use_optimum_lora = True
        self.valid_lora_ids = list(range(NUM_LORAS + 1))
        self.model.kv_block_adapter = KVCacheBlockAdapter(
            vllm_config=get_vllm_config(),
            estimated_kvcache_num_blocks=NUM_BLOCKS + 1,
        )
        self.valid_lora_ids = list(range(NUM_LORAS + 1))


def clear_global_vars():
    golden.clear()
    result.clear()


def validate_vars():
    for r, g in zip(result, golden):
        for i in range(len(r)):
            assert g[i] == r[i]


async def add_lora():
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=NUM_LORAS,
        max_lora_rank=MAX_LORA_RANK,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )
    lora_int_ids = [1, 2, 3, 0, 1, 2]
    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing=True) as llm:
        await add_lora_request(llm, lora_int_ids)


async def list_loras():
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=NUM_LORAS,
        max_lora_rank=MAX_LORA_RANK,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing=True) as llm:
        lora_ids = await llm.list_loras()

    return lora_ids


@pytest.mark.asyncio
async def test_add_lora_v0(monkeypatch):
    clear_global_vars()
    monkeypatch.setenv("VLLM_USE_V1", "0")
    monkeypatch.setattr(V0_PATH, fake_load_model)
    await add_lora()
    validate_vars()


@pytest.mark.asyncio
async def test_add_lora_v1(monkeypatch):
    clear_global_vars()
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setattr(V1_PATH, fake_load_model)

    await add_lora()
    validate_vars()


@pytest.mark.asyncio
async def test_list_lora_v1(monkeypatch):
    clear_global_vars()
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setattr(V1_PATH, fake_load_model)

    lora_ids = await list_loras()
    assert set(lora_ids) == {1, 2, 3, 4, 5}