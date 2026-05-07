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
