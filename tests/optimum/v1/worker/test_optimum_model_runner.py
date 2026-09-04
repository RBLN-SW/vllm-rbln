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
from types import SimpleNamespace

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
        new_req_id, block_ids=([1],), outer_block_ids=[1]
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    assert _is_req_added(model_runner, new_req_id)
    assert _is_req_scheduled(model_runner, new_req_id)


def _decode_step(req_ids: list[str], cache_slot_ids: list[int], dummy_block=None):
    # Only the scheduler fields _prepare_decode reads.
    return SimpleNamespace(
        block_table_dict={
            req_id: torch.tensor([index + 1]) for index, req_id in enumerate(req_ids)
        },
        cache_slot_id_dict=dict(zip(req_ids, cache_slot_ids)),
        dummy_block=dummy_block,
    )


def _two_running_requests(model_runner):
    model_runner._update_states(
        _schedule_new_request("r0", "r1", block_ids=([1],), outer_block_ids=[0])
    )
    # Advance r1 so the two rows differ: r1 decodes token 3 at position 2.
    r1 = model_runner.input_batch.req_id_to_index["r1"]
    model_runner.input_batch.num_computed_tokens_cpu[r1] = 2


def test_prepare_decode_pads_running_order_to_the_decoder_batch(model_runner):
    _two_running_requests(model_runner)
    batch = model_runner.model.decoder_batch_size

    model_input = model_runner._prepare_decode(_decode_step(["r0", "r1"], [0, 3]))

    assert model_input.padded_batch_size == batch
    assert model_input.batch_rows is None
    assert model_input.input_tokens.dtype == torch.int64
    assert model_input.input_tokens[:2, 0].tolist() == [1, 3]
    assert model_input.input_positions.dtype == torch.int32
    assert model_input.input_positions[:2, 0].tolist() == [0, 2]
    assert model_input.block_tables.dtype == torch.int16
    assert model_input.block_tables.shape[0] == batch
    # Padding rows share one block that no running request uses...
    real_blocks = model_input.block_tables[:2]
    pad_blocks = model_input.block_tables[2:]
    assert (pad_blocks == pad_blocks[0, 0]).all()
    assert not torch.isin(pad_blocks, real_blocks).any()
    # ...and a cache slot that no running request owns.
    slots = model_input.cache_slot_ids
    assert slots.shape == (batch, 1) and slots.dtype == torch.int16
    assert slots[:2, 0].tolist() == [0, 3]
    assert not torch.isin(slots[2:], slots[:2]).any()


def test_prepare_decode_pads_with_the_scheduler_scratch_block(model_runner):
    _two_running_requests(model_runner)

    model_input = model_runner._prepare_decode(
        _decode_step(["r0", "r1"], [0, 1], dummy_block=7)
    )

    assert (model_input.block_tables[2:] == 7).all()


def test_prepare_decode_pins_rows_the_model_names(model_runner):
    _two_running_requests(model_runner)
    # A model with per-row on-device state pins each request to its cache slot.
    model_runner.model.decode_batch_rows = lambda slots, block_tables: slots.to(
        torch.long
    )

    model_input = model_runner._prepare_decode(_decode_step(["r0", "r1"], [3, 0]))

    assert model_input.batch_rows.tolist() == [3, 0]
    assert model_input.padded_batch_size == model_runner.model.decoder_batch_size
    # r0 lands on row 3 and r1 on row 0; the other rows are padding.
    assert model_input.input_tokens[3, 0] == 1 and model_input.input_tokens[0, 0] == 3
    assert model_input.input_positions[3, 0] == 0
    assert model_input.input_positions[0, 0] == 2
    assert model_input.cache_slot_ids[3, 0] == 3
    assert model_input.cache_slot_ids[0, 0] == 0
    assert model_input.block_tables[3, 0] != model_input.block_tables[1, 0]
