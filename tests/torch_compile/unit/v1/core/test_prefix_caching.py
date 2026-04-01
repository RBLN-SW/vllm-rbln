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

from .utils import create_requests, create_runner_output, create_scheduler


def test_basic():
    # test basic prefix caching functionality with requests with same contents

    block_size = 16
    num_requests = 4
    num_blocks_per_request = 4

    scheduler = create_scheduler(
        block_size=block_size,
        num_blocks=num_blocks_per_request + num_requests,
        max_num_batched_tokens=num_blocks_per_request * block_size * 2,
        enable_prefix_caching=True,
        max_model_len=num_blocks_per_request * block_size * 2,
    )

    same_requests = create_requests(
        num_requests,
        num_tokens=num_blocks_per_request * block_size,
        max_tokens=1,
        same_prompt=True,
    )

    for request in same_requests:
        scheduler.add_request(request)

    cached_block_ids = list(range(1, num_blocks_per_request))
    for req_index in range(num_requests):
        scheduler_output = scheduler.schedule()
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        scheduled_new_reqs = scheduler_output.scheduled_new_reqs

        # prefill batch size fixed to 1
        assert len(req_ids) == 1
        assert req_ids[0] == str(req_index)
        assert len(scheduled_new_reqs) == 1

        # assume single kv cache group
        assert len(scheduled_new_reqs[0].block_ids) == 1

        # check if prefix blocks are properly cached and allocated
        allocated_block_ids = scheduled_new_reqs[0].block_ids[0]
        num_cached_tokens = len(cached_block_ids) * block_size
        num_computed_tokens = scheduled_new_reqs[0].num_computed_tokens
        assert allocated_block_ids[:-1] == cached_block_ids
        assert req_index == 0 or num_computed_tokens == num_cached_tokens

        # check if ref count of blocks are properly counted
        assert all(
            block.ref_cnt == 1
            for block in scheduler.kv_cache_manager.get_blocks(req_ids[0]).blocks[0]
        )

        model_runner_output = create_runner_output(scheduler_output, 0)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # check if every request drained
    assert not scheduler.has_unfinished_requests()

    # check if blocks are properly cached
    block_pool = scheduler.kv_cache_manager.block_pool
    cache = block_pool.cached_block_hash_to_block._cache.values()
    entry_with_multiple_blocks = [
        blks for blks in cache if isinstance(blks, dict) and len(blks) != 1
    ]
    assert len(cache) == num_blocks_per_request
    assert len(entry_with_multiple_blocks) == 1
    assert len(entry_with_multiple_blocks[0]) == num_requests


def test_preallocation_in_prefill():
    # test that block preallocation during the prefill phase
    # does not break prefix caching functionality

    block_size = 16
    num_blocks_per_request = 2

    scheduler = create_scheduler(
        block_size=block_size,
        max_num_batched_tokens=block_size,
        enable_prefix_caching=True,
        max_model_len=block_size * num_blocks_per_request * 2,
    )
    kv_cache_manager = scheduler.kv_cache_manager

    req = create_requests(
        1,
        num_tokens=num_blocks_per_request * block_size,
        max_tokens=1,
        same_prompt=True,
    )[0]

    scheduler.add_request(req)

    scheduler_output = scheduler.schedule()

    assert len(scheduler_output.scheduled_new_reqs) == 1
    scheduled_new_req = scheduler_output.scheduled_new_reqs[0]

    # assume single kv cache group
    assert len(scheduled_new_req.block_ids) == 1

    # check if all blocks are preallocated
    block_ids = scheduled_new_req.block_ids[0]
    assert len(block_ids) == num_blocks_per_request

    # check if ref count of blocks are properly counted
    blocks = kv_cache_manager.get_blocks(req.request_id).blocks[0]
    assert len(blocks) == len(block_ids)
    assert [block.block_id for block in blocks] == block_ids
    assert all(block.ref_cnt == 1 for block in blocks)

    # check if only the first block is cached
    first_block_id = block_ids[0]
    cache = kv_cache_manager.block_pool.cached_block_hash_to_block._cache
    entry = next(iter(cache.values()))
    assert all(block.block_hash is None for block in blocks[1:])
    assert len(cache) == 1
    assert not isinstance(entry, dict) or len(entry) == 1
    assert (isinstance(entry, dict) and first_block_id in entry) or (
        not isinstance(entry, dict) and first_block_id == entry.block_id
    )


def test_preallocation_in_decode():
    # test that block preallocation during the decode phase
    # does not break prefix caching functionality

    block_size = 16

    scheduler = create_scheduler(
        block_size=block_size,
        max_num_batched_tokens=block_size,
        enable_prefix_caching=True,
        max_model_len=block_size * 2,
    )
    kv_cache_manager = scheduler.kv_cache_manager

    req_a, req_b = tuple(
        create_requests(
            2,
            num_tokens=block_size,
            max_tokens=1,
            same_prompt=True,
        )
    )
    assert req_a.prompt_token_ids
    req_a.prompt_token_ids.pop()
    req_a._all_token_ids.pop()
    req_a.num_prompt_tokens = len(req_a.prompt_token_ids)
    req_a.max_tokens = 2

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    # first iteration
    scheduler_output = scheduler.schedule()

    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    scheduled_new_reqs = scheduler_output.scheduled_new_reqs

    # prefill batch size fixed to 1
    assert len(req_ids) == 1
    assert req_ids[0] == "0"
    assert len(scheduled_new_reqs) == 1

    # assume single kv cache group
    assert len(scheduled_new_reqs[0].block_ids) == 1

    # check if block allocated to request a is not cached yet
    req_a_blocks = kv_cache_manager.get_blocks(req_ids[0]).blocks[0]
    req_a_block = req_a_blocks[0]
    assert len(req_a_blocks) == 1
    assert req_a_block.block_hash is None
    assert len(kv_cache_manager.block_pool.cached_block_hash_to_block) == 0

    model_runner_output = create_runner_output(scheduler_output, 0)
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # second iteration
    scheduler_output = scheduler.schedule()

    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    scheduled_new_reqs = scheduler_output.scheduled_new_reqs

    # prefill batch size fixed to 1
    assert len(req_ids) == 1
    assert req_ids[0] == "1"
    assert len(scheduled_new_reqs) == 1

    # assume single kv cache group
    assert len(scheduled_new_reqs[0].block_ids) == 1

    # check if block allocated to request a is not cached yet
    assert req_a_block.block_hash is None

    model_runner_output = create_runner_output(scheduler_output, 0)
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # third iteration
    scheduler_output = scheduler.schedule()

    # check if block allocated to request a is properly cached
    assert req_a_block.block_hash is not None
