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

import tempfile

import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.core.optimum_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

from .utils import _schedule_new_request, fake_load_model

BLOCK_SIZE = 16
NUM_BLOCKS = 8
DEVICE = current_platform.device_type


# TODO add tests for both `enable_prefix_caching = True` and `False`
def get_vllm_config(async_scheduling=False):
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=128,
        max_model_len=128,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype=torch.float,
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        cache_dtype="auto",
    )
    vllm_config = VllmConfig(
        cache_config=cache_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        additional_config={
            "prefix_block_size": 4,
            "rbln_config": {
                "prefill_chunk_size": 4,
            },
        },
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config, check_compile=False):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            1,
            1,
        )
    runner = RBLNOptimumModelRunner(vllm_config, DEVICE)
    fake_load_model(runner)
    return runner


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(
    model_runner, sampling_metadata_before: SamplingMetadata
):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]

    num_block_of_runner = block_table.num_blocks_per_row[req_index]
    num_block_of_req_state = len(req_state.block_ids[0])
    if num_block_of_runner != num_block_of_req_state:
        return False
    return (
        block_table.block_table.np[req_index, :num_block_of_runner]
        == req_state.block_ids[0]
    ).all()


def test_update_states_new_request(model_runner):
    req_id = "req_0"

    # schedule new request
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )
    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner):
    req_id = "req_0"

    # schedule new request
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish request
    scheduler_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner):
    req_id = "req_0"

    # schedule new request
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule request
    scheduler_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume request
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[([0],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    scheduler_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner):
    req_id = "req_0"

    # schedule req0
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    new_req_id = "req_1"

    # schedule req1
    # scheduling new request(req1)
    # prevent req0 from being scheduled
    scheduler_output = _schedule_new_request(
        new_req_id, block_ids=([1],), outer_block_ids=torch.tensor([[1]])
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    assert _is_req_added(model_runner, new_req_id)
    assert _is_req_scheduled(model_runner, new_req_id)


def _wire_mega_cache(model_runner, monkeypatch) -> list[tuple]:
    """Record the order of bundle load/save and sampler compiles."""
    from vllm_rbln.v1.worker import mega_cache

    steps: list[tuple] = []
    model_runner.use_rbln_sampler = True
    monkeypatch.setattr(
        model_runner, "sampler", lambda **kwargs: steps.append(("compile",))
    )
    monkeypatch.setattr(
        mega_cache, "load", lambda model, sig: steps.append(("load", model, sig))
    )
    monkeypatch.setattr(
        mega_cache, "save", lambda model, sig: steps.append(("save", model, sig))
    )
    return steps


def test_dummy_sampler_run_loads_bundle_first_and_saves_last(model_runner, monkeypatch):
    steps = _wire_mega_cache(model_runner, monkeypatch)

    model_runner.dummy_sampler_run()

    kinds = [step[0] for step in steps]
    assert kinds[0] == "load"
    assert kinds[-1] == "save"
    assert "compile" in kinds
    # Same (model, sig) on both ends, or the run writes a bundle it can never
    # load back.
    assert steps[0][1:] == steps[-1][1:]
    assert steps[0][1] == model_runner.model_config.model


def test_failed_dummy_sampler_run_writes_no_bundle(model_runner, monkeypatch):
    steps = _wire_mega_cache(model_runner, monkeypatch)

    def failing_sampler(**kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(model_runner, "sampler", failing_sampler)

    with pytest.raises(RuntimeError, match="compile failed"):
        model_runner.dummy_sampler_run()
    assert [step[0] for step in steps] == ["load"]


def _prepare_sampler_with_buckets(model_runner, monkeypatch, bucket_sizes):
    monkeypatch.setattr(
        RBLNOptimumModelRunner,
        "get_bucket_sizes",
        staticmethod(lambda max_num_seqs: list(bucket_sizes)),
    )
    # The pooled logits are (bucket, vocab); keep them small for many buckets.
    monkeypatch.setattr(model_runner.model_config, "get_vocab_size", lambda: 8)
    model_runner.prepare_rbln_sampler()


def test_prepare_rbln_sampler_fits_every_graph_under_both_recompile_limits(
    model_runner, monkeypatch
):
    from vllm_rbln.v1.sample import SAMPLER_GRAPHS_PER_BATCH_SIZE

    dynamo_config = torch._dynamo.config
    monkeypatch.setattr(dynamo_config, "recompile_limit", 8)
    monkeypatch.setattr(dynamo_config, "accumulated_recompile_limit", 256)
    # Enough batch sizes that the sampler graphs alone exceed the default
    # process-wide limit.
    bucket_sizes = range(1, 256 // SAMPLER_GRAPHS_PER_BATCH_SIZE + 2)

    _prepare_sampler_with_buckets(model_runner, monkeypatch, bucket_sizes)

    num_graphs = SAMPLER_GRAPHS_PER_BATCH_SIZE * len(bucket_sizes)
    assert num_graphs > 256
    assert dynamo_config.recompile_limit >= num_graphs
    assert dynamo_config.accumulated_recompile_limit >= num_graphs


def test_prepare_rbln_sampler_never_lowers_recompile_limits(model_runner, monkeypatch):
    dynamo_config = torch._dynamo.config
    monkeypatch.setattr(dynamo_config, "recompile_limit", 1000)
    monkeypatch.setattr(dynamo_config, "accumulated_recompile_limit", 2000)

    _prepare_sampler_with_buckets(model_runner, monkeypatch, [1, 2, 4, 8])

    assert dynamo_config.recompile_limit == 1000
    assert dynamo_config.accumulated_recompile_limit == 2000
