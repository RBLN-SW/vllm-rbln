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

import pytest
import torch
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.request import Request, RequestStatus

from .utils import create_model_runner_output, create_scheduler

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
OB_SIZE = 16
IB_SIZE = 4
NUM_BLOCKS = MAX_MODEL_LEN // IB_SIZE * MAX_NUM_SEQ + 1  # 9
DEVICE = current_platform.device_type
HASH_FN = sha256

# checkout output scheduler, runner in prefill, decode tests
# not preempt + eviction test
# preempt + eviction test


@pytest.fixture
def scheduler():
    scheduler = create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=NUM_BLOCKS,
        block_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        outer_block_size=OB_SIZE,
        enable_prefix_caching=True,
    )
    return scheduler


@pytest.fixture
def limited_4blocks_scheduler(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_NPU_NUM_BLOCKS", "4")
    scheduler = create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=4 * (OB_SIZE // IB_SIZE) + 1,
        block_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        outer_block_size=OB_SIZE,
        enable_prefix_caching=True,
    )
    return scheduler


@pytest.fixture
def limited_6blocks_scheduler(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_NPU_NUM_BLOCKS", "6")
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=6 * (OB_SIZE // IB_SIZE) + 1,
        block_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        outer_block_size=OB_SIZE,
        enable_prefix_caching=True,
    )
    return scheduler


def create_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn: callable,
) -> Request:
    block_hasher = get_request_block_hasher(block_size, hash_fn)
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_tokens=MAX_MODEL_LEN,
            temperature=0.0,
        ),
        block_hasher=block_hasher,
        pooling_params=None,
        eos_token_id=100,
    )
    return request


@pytest.mark.parametrize(
    "token_length, cached_block_table, cached_length, ",
    [
        pytest.param(
            50,
            [0, 1, 2],
            [16, 16, 16],
            id="50_tokens",
        ),
        pytest.param(
            54,
            [0, 1, 2, 3],
            [16, 16, 16, 4],
            id="54_tokens",
        ),
    ],
)
def test_prefix_cache_hit_same_prompt(scheduler, token_length: int,
                                      cached_block_table: list[int],
                                      cached_length: list[int]):
    init_none_hash(HASH_FN)
    """
    Check the prefix caching works as expected
    between two requests with the same prompt.
    Note that if the prompt is fully cached,
    the last inner block is excluded from cache hit
    to return the logits.
    """
    # 1. Generate req0: 42 tokens -> 11 inner blocks
    common_token_ids = [i for i in range(token_length)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    req1_id = "req1"
    req1 = create_request(req1_id, common_token_ids, IB_SIZE, HASH_FN)

    requests = [req0, req1]
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    answer = torch.tensor([0, 1, 2, 3], dtype=torch.int16)
    assert torch.allclose(output.block_table_dict[req0_id], answer)
    assert output.cached_block_table == []
    assert output.cached_length == []

    # Model output of the first request.
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    answer = torch.tensor([4, 5, 6, 7], dtype=torch.int16)
    assert torch.allclose(output.block_table_dict[req1_id], answer)
    assert output.cached_block_table == cached_block_table
    assert output.cached_length == cached_length


@pytest.mark.parametrize("token_length, decode_steps, allocated_blocks", [
    pytest.param(
        31,
        1,
        [0, 1, 2, -1],
        id="decode_1_step_and_eviction",
    ),
    pytest.param(
        25,
        7,
        [0, 1, 2, -1],
        id="decode_7_steps_eviction",
    ),
])
def test_eviction(limited_4blocks_scheduler, token_length: int,
                  decode_steps: int, allocated_blocks: list[int]):
    init_none_hash(HASH_FN)
    """
    Check the eviction works as expected.

    Scenario:
    0. There are 4 blocks in total.
    1. Generate req0 which spends 2 blocks when it is scheduled.
    2. Generate req1 which spends 2 blocks when it is scheduled.
    3. req0 requires 1 more block, and req1 is preempted.
    4. After req0 is done, req1 resumes.
    We check whether req1 can resume correctly
    by evicting req0's blocks.
    5. Finally, we check whether req1's prefix caching works correctly.
    """
    # 1. Generate req0: 31 tokens -> 12 inner blocks
    common_token_ids = [i for i in range(token_length)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)
    req1_id = "req1"
    req1 = create_request(req1_id, common_token_ids, IB_SIZE, HASH_FN)

    requests = [req0, req1]
    for request in requests:
        limited_4blocks_scheduler.add_request(request)

    until_to_preemption = len(requests) + decode_steps
    for _ in range(until_to_preemption):
        output = limited_4blocks_scheduler.schedule()
        model_runner_output = create_model_runner_output(output)
        limited_4blocks_scheduler.update_from_output(output,
                                                     model_runner_output)
    assert req1_id in output.block_table_dict
    # Now, req1 is preempted, and req0 continues.
    # req0 requires 1 more block.
    output = limited_4blocks_scheduler.schedule()
    assert req1_id not in output.block_table_dict
    answer = torch.tensor(allocated_blocks, dtype=torch.int16)
    assert torch.allclose(output.block_table_dict[req0_id], answer)


def test_cache_hit_child_block(limited_6blocks_scheduler):
    """
    This test verifies the difference between
    our prefix caching implementation and vLLM's approach.

    Unlike vLLM, which directly reuses cache-hit blocks,
    our implementation creates new blocks and copies the contents
    of the cache-hit blocks into them.

    In other words:
    - vLLM returns the original (root) cache-hit blocks.
    - Our implementation returns newly allocated blocks that contain copies
    of those cache-hit blocks.

    This test ensures that this behavior works as intended.

    Test scenario:
    0. There are 6 blocks in total.
    1. Generate req0 which spends 2 blocks when it is scheduled.
    => [0, 1]
    2. Generate req1 which spends 2 blocks when it is scheduled.
    => [2, 3]
    3. Generate req2 with a new prompt, which spends 2 blocks when scheduled.
    => [4, 5]
    4. Finish req0 and generate req3 with a different prompt,
       which spends 2 blocks when scheduled.
    => [0, 1]
    5. Finish req2 and generate req4 with the same prompt as req1,
       which spends 2 blocks when scheduled.
       Here, we check whether req4 correctly reuses req1's cached blocks.
    => [4, 5] (reuse req1's cached blocks [2, 3])
    """
    init_none_hash(HASH_FN)
    # 1. Generate req0: 32 tokens -> 8 inner blocks
    common_token_ids = [i for i in range(21)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    req1_id = "req1"
    req1 = create_request(req1_id, common_token_ids, IB_SIZE, HASH_FN)

    req2_id = "req2"
    req2 = create_request(req2_id, [i + 100 for i in common_token_ids],
                          IB_SIZE, HASH_FN)
    req3_id = "req3"
    req3 = create_request(req3_id, [i + 50 for i in common_token_ids], IB_SIZE,
                          HASH_FN)

    req4_id = "req4"
    req4 = create_request(req4_id, common_token_ids, IB_SIZE, HASH_FN)

    requests = [req0, req1, req2, req3, req4]
    for request in requests:
        limited_6blocks_scheduler.add_request(request)

    # Schedule req0 [0, 1]
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)

    # Schedule req1 [2, 3]
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert req2_id not in output.block_table_dict

    # Schedule req2 [4, 5] with new prompt
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert req2_id in output.block_table_dict

    # Finish req0 and schedule req3 [0, 1]
    # To remove the cache, the prompt of req3 is different from req0/req1
    limited_6blocks_scheduler.finish_requests(req0_id,
                                              RequestStatus.FINISHED_ABORTED)
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert req3_id in output.block_table_dict

    # Finish req2 and schedule req4 [4, 5]
    # Reuse the req1's blocks [2, 3] for req4
    limited_6blocks_scheduler.finish_requests(req2_id,
                                              RequestStatus.FINISHED_ABORTED)
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    answer = torch.tensor([4, 5, -1, -1], dtype=torch.int16)
    assert torch.allclose(output.block_table_dict[req4_id], answer)
    assert output.cached_block_table == [2, 3]


def test_allocated_blocks_excluded_from_cache_hit(limited_6blocks_scheduler):
    """
    If the allocated blocks are excluded when calculating
    the cached blocks for prefix caching.
    0. There are 4 blocks in total.
    1. Generate req0 which spends 2 blocks when it is scheduled.
    2. Generate req1 which spends 2 blocks when it is scheduled.
    3. Generate req2 which spends 2 blocks when it is scheduled.
        - It is for getting enough freed inner blocks.
    4. Finish req0 and req1, and generate req3 with the same prompt
        - Here, req3's prompt is the same as req0's prompt.
        - req0 is allocated [0, 1]
        - So the cached blocks become [2, 3], not [0, 1].
    """
    init_none_hash(HASH_FN)
    # 1. Generate req0: 32 tokens -> 8 inner blocks
    common_token_ids = [i for i in range(31)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    req1_id = "req1"
    req1 = create_request(req1_id, common_token_ids, IB_SIZE, HASH_FN)

    req2_id = "req2"
    req2 = create_request(req2_id, common_token_ids, IB_SIZE, HASH_FN)

    req3_id = "req3"
    req3 = create_request(req3_id, common_token_ids, IB_SIZE, HASH_FN)

    requests = [req0, req1, req2, req3]
    for request in requests:
        limited_6blocks_scheduler.add_request(request)

    # Schedule req0 [0, 1]
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert output.cached_block_table == []
    assert output.block_table_dict[req0_id].tolist() == [0, 1, -1, -1]

    # Schedule req1 [2, 3]
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert output.cached_block_table == [0, 1]
    assert output.block_table_dict[req1_id].tolist() == [2, 3, -1, -1]

    # Schedule req2 [4, 5]
    limited_6blocks_scheduler.finish_requests(req0_id,
                                              RequestStatus.FINISHED_ABORTED)
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert output.cached_block_table == [2, 3]
    assert output.block_table_dict[req2_id].tolist() == [4, 5, -1, -1]

    # Finish Schedule req0, req1 and schedule req3 [0, 1]
    # 1. To get enough freed inner blocks
    # this pytest finishes req0 and req1
    # 2. req3 is allocated [0, 1] again
    # 3. So the cached blocks become [2, 3]
    # to exclude the allocated blocks [0, 1].
    limited_6blocks_scheduler.finish_requests(req0_id,
                                              RequestStatus.FINISHED_ABORTED)
    limited_6blocks_scheduler.finish_requests(req1_id,
                                              RequestStatus.FINISHED_ABORTED)
    output = limited_6blocks_scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    limited_6blocks_scheduler.update_from_output(output, model_runner_output)
    assert output.cached_block_table == [4, 5]
    assert output.block_table_dict[req3_id].tolist() == [0, 1, -1, -1]


def test_finish_request(scheduler):
    """
    Check that finishing a request correctly frees its allocated blocks
    while retaining the mapping for future reuse.
    """
    init_none_hash(HASH_FN)
    # 1. Generate req0: 10 tokens -> 3 inner blocks
    common_token_ids = [i for i in range(10)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    scheduler.add_request(req0)

    output = scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    # Finish req0
    scheduler.finish_requests(req0_id, RequestStatus.FINISHED_ABORTED)
    prefix_cache_manager = scheduler.kv_cache_manager.prefix_cache_manager
    # The allocated blocks for req0 is not freed yet
    allocated_blocks = prefix_cache_manager._allocator._allocated_blocks
    assert len(allocated_blocks) == 1

    # Mark req0 as finished and free its blocks
    mapping_manager = prefix_cache_manager._mapping_manager
    mapping = mapping_manager.get_mapping(0)
    assert not mapping_manager.is_request_registered(req0_id)
    assert mapping.request_id is None
    assert mapping.is_active is False

    # Keep the mapping between outer and inner blocks
    # to reuse them for future requests.
    assert mapping.outer_block_id == 0
    assert mapping.inner_block_ids == [1, 2, 3]
    assert len(mapping_manager._inner_to_outer) == 3


def test_free_outer_blocks(scheduler):
    """
    Check the evicted blocks are correctly freed
    from the prefix cache manager.
    """
    init_none_hash(HASH_FN)
    # 1. Generate req0: 10 tokens -> 3 inner blocks
    common_token_ids = [i for i in range(10)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    scheduler.add_request(req0)

    output = scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    # Evict the outer block 0
    prefix_cache_manager = scheduler.kv_cache_manager.prefix_cache_manager
    mapping_manager = prefix_cache_manager._mapping_manager
    prefix_cache_manager.free_request(req0_id, preemption=True)
    assert not mapping_manager.is_request_registered(req0_id)
    assert mapping_manager.get_mapping(0) is None

    assert len(mapping_manager._inner_to_outer) == 0