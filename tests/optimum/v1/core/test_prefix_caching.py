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

from collections.abc import Callable
from typing import Any

import pytest
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request, RequestStatus

from .utils import create_model_runner_output, create_scheduler

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
BLOCK_SIZE = 16
PREFILL_CHUNK_SIZE = 4
# 8 usable blocks (a full batch) + the null block + the pinned dummy block.
NUM_BLOCKS = MAX_MODEL_LEN // BLOCK_SIZE * MAX_NUM_SEQ + 2
HASH_FN = sha256


@pytest.fixture
def scheduler():
    return create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
    )


@pytest.fixture
def limited_4blocks_scheduler():
    # 4 usable blocks + null + dummy.
    return create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=4 + 2,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
    )


def create_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn: Callable[[Any], str],
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
    )
    return request


def run_one_step(scheduler):
    output = scheduler.schedule()
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)
    return output


@pytest.mark.parametrize(
    "token_length, expected_num_cached_tokens, expected_copy_op",
    [
        # 50 tokens: 3 full blocks (48 tokens) are cached; the 2-token
        # remainder is smaller than a sub-block, so no sub-block match.
        pytest.param(50, 48, None, id="50_tokens"),
        # 62 tokens: 3 full blocks (48 tokens) plus a sub-block extension of
        # 12 tokens (3 sub-blocks of req0's partial block 4, copied into
        # req1's own block 5). The hit cap at num_tokens - 1 = 61 leaves the
        # last sub-block out.
        pytest.param(62, 60, (4, 5, 12), id="62_tokens"),
    ],
)
def test_prefix_cache_hit_shares_blocks(
    scheduler,
    token_length: int,
    expected_num_cached_tokens: int,
    expected_copy_op: tuple[int, int, int] | None,
):
    """A request with a cached prefix reuses the very blocks that hold it:
    the full-block hit appears in the new request's block table as shared
    blocks, a sub-block hit becomes a copy op, and only the remainder of the
    prompt is scheduled for computation."""
    init_none_hash(HASH_FN)
    common_token_ids = list(range(token_length))
    req0 = create_request("req0", common_token_ids, BLOCK_SIZE, HASH_FN)
    req1 = create_request("req1", common_token_ids, BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    output = run_one_step(scheduler)
    new_req = output.scheduled_new_reqs[0]
    assert new_req.req_id == "req0"
    assert new_req.block_ids[0] == [1, 2, 3, 4]
    assert new_req.num_computed_tokens == 0
    assert output.num_scheduled_tokens["req0"] == token_length
    assert output.kv_cache_copy_ops == []

    output = scheduler.schedule()
    new_req = output.scheduled_new_reqs[0]
    assert new_req.req_id == "req1"
    # The full blocks are shared with req0; only the last block is new.
    assert new_req.block_ids[0] == [1, 2, 3, 5]
    assert new_req.num_computed_tokens == expected_num_cached_tokens
    assert (
        output.num_scheduled_tokens["req1"] == token_length - expected_num_cached_tokens
    )

    # The shared blocks are referenced by both requests, not copied.
    block_pool = scheduler.kv_cache_manager.block_pool
    for block_id in (1, 2, 3):
        assert block_pool.blocks[block_id].ref_cnt == 2

    if expected_copy_op is None:
        assert output.kv_cache_copy_ops == []
        return
    src, dst, num_tokens = expected_copy_op
    (op,) = output.kv_cache_copy_ops
    assert (op.src_block_id, op.dst_block_id, op.num_tokens) == (src, dst, num_tokens)
    # The pending copy op holds a ref on the source block (req0's own ref
    # plus the copy op's) until update_from_output releases it.
    assert block_pool.blocks[src].ref_cnt == 2
    scheduler.update_from_output(output, create_model_runner_output(output))
    assert block_pool.blocks[src].ref_cnt == 1


def test_sub_block_hit_after_finish(scheduler):
    """A finished request's partial block keeps serving sub-block hits: the
    synthetic block hash keeps it in the LRU, and the match is realized as a
    copy op into the new request's own block."""
    init_none_hash(HASH_FN)
    # 30 tokens: 1 full block + a partial block with 3 complete sub-blocks.
    common_token_ids = list(range(30))
    req0 = create_request("req0", common_token_ids, BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    run_one_step(scheduler)
    scheduler.finish_requests("req0", RequestStatus.FINISHED_ABORTED)

    req1 = create_request("req1", common_token_ids, BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req1)
    output = scheduler.schedule()
    new_req = output.scheduled_new_reqs[0]
    # Full hit on block 1 (16 tokens) + 3 sub-blocks (12 tokens) copied from
    # req0's freed partial block 2 into req1's own block 3.
    assert new_req.block_ids[0] == [1, 3]
    assert new_req.num_computed_tokens == 28
    assert output.num_scheduled_tokens["req1"] == 2
    (op,) = output.kv_cache_copy_ops
    assert (op.src_block_id, op.dst_block_id, op.num_tokens) == (2, 3, 12)


def test_sub_block_disabled_when_chunk_covers_block():
    """When the prefill chunk size equals the block size, sub-block caching
    adds nothing and stays off; full-block sharing still works."""
    init_none_hash(HASH_FN)
    scheduler = create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        prefill_chunk_size=BLOCK_SIZE,
    )
    assert scheduler.kv_cache_manager.sub_block_size is None

    common_token_ids = list(range(30))
    req0 = create_request("req0", common_token_ids, BLOCK_SIZE, HASH_FN)
    req1 = create_request("req1", common_token_ids, BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    scheduler.add_request(req1)
    run_one_step(scheduler)
    output = scheduler.schedule()
    new_req = output.scheduled_new_reqs[0]
    assert new_req.num_computed_tokens == BLOCK_SIZE
    assert output.kv_cache_copy_ops == []


def test_evicted_block_leaves_sub_block_index(limited_4blocks_scheduler):
    """Evicting a block removes it from the sub-block index, so a later
    request cannot match KV that no longer exists."""
    scheduler = limited_4blocks_scheduler
    init_none_hash(HASH_FN)
    req0 = create_request("req0", list(range(30)), BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    run_one_step(scheduler)  # blocks [1, 2], both indexed
    scheduler.finish_requests("req0", RequestStatus.FINISHED_ABORTED)

    # A 62-token request takes all 4 blocks, evicting req0's cached ones.
    reqx = create_request("reqx", [i + 100 for i in range(62)], BLOCK_SIZE, HASH_FN)
    scheduler.add_request(reqx)
    run_one_step(scheduler)
    scheduler.finish_requests("reqx", RequestStatus.FINISHED_STOPPED)

    # req0's prompt must now miss entirely: a stale sub-block index entry
    # would emit a copy op reading reqx's KV.
    req1 = create_request("req1", list(range(30)), BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req1)
    output = scheduler.schedule()
    new_req = output.scheduled_new_reqs[0]
    assert new_req.num_computed_tokens == 0
    assert output.kv_cache_copy_ops == []


def test_finished_request_blocks_are_reused(scheduler):
    """Blocks of a finished request stay cached and serve later hits."""
    init_none_hash(HASH_FN)
    common_token_ids = list(range(50))
    req0 = create_request("req0", common_token_ids, BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    run_one_step(scheduler)
    scheduler.finish_requests("req0", RequestStatus.FINISHED_ABORTED)

    req1 = create_request("req1", common_token_ids, BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req1)
    output = scheduler.schedule()
    new_req = output.scheduled_new_reqs[0]
    # Blocks 1-3 are req0's cached blocks. The new last block is 4: req0's
    # freed unhashed block goes back to the head of the free queue.
    assert new_req.block_ids[0] == [1, 2, 3, 4]
    assert new_req.num_computed_tokens == 48
    assert output.num_scheduled_tokens["req1"] == 2


def test_cached_blocks_evicted_under_pressure(limited_4blocks_scheduler):
    """Cached blocks of finished requests are evicted when a new request
    needs the space."""
    scheduler = limited_4blocks_scheduler
    init_none_hash(HASH_FN)
    req0 = create_request("req0", list(range(40)), BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    run_one_step(scheduler)
    assert scheduler.kv_cache_manager.get_block_ids("req0")[0] == [1, 2, 3]
    scheduler.finish_requests("req0", RequestStatus.FINISHED_ABORTED)

    # A different prompt cannot hit req0's cache, so its blocks are evicted
    # to make room.
    req1 = create_request("req1", [i + 100 for i in range(40)], BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req1)
    output = scheduler.schedule()
    new_req = output.scheduled_new_reqs[0]
    assert new_req.num_computed_tokens == 0
    assert len(new_req.block_ids[0]) == 3


def test_preemption_and_resume_with_cache_hit(limited_4blocks_scheduler):
    """Preemption frees blocks under pressure; the preempted request later
    resumes and re-hits its own cached prefix."""
    scheduler = limited_4blocks_scheduler
    init_none_hash(HASH_FN)
    req0 = create_request("req0", list(range(30)), BLOCK_SIZE, HASH_FN)
    req1 = create_request("req1", [i + 100 for i in range(30)], BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    run_one_step(scheduler)  # req0 prefill -> [1, 2]
    run_one_step(scheduler)  # req1 prefill -> [3, 4]

    # Decode until req0 needs a third block (33 tokens). The pool is empty,
    # so req1 is preempted to free it.
    for _ in range(4):
        run_one_step(scheduler)
        if req1.status == RequestStatus.PREEMPTED:
            break
    assert req1.status == RequestStatus.PREEMPTED
    assert len(scheduler.kv_cache_manager.get_block_ids("req0")[0]) == 3

    # Finish req0; req1 resumes and re-hits its own cached first block.
    scheduler.finish_requests("req0", RequestStatus.FINISHED_STOPPED)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens.keys() == {"req1"}
    assert req1.status == RequestStatus.RUNNING
    assert req1.num_computed_tokens >= BLOCK_SIZE


def test_dummy_block_for_decode_padding(scheduler):
    """A partially filled decode batch reports the pinned dummy block, which
    is never allocated to a request."""
    init_none_hash(HASH_FN)
    req0 = create_request("req0", list(range(20)), BLOCK_SIZE, HASH_FN)
    scheduler.add_request(req0)
    output = run_one_step(scheduler)
    assert output.dummy_block is None  # prefill step

    output = run_one_step(scheduler)  # decode with 1 < MAX_NUM_SEQ requests
    # The pool pins its last block (vLLM id NUM_BLOCKS - 1) as the dummy;
    # the scheduler reports it in compiler-space (shifted by -1).
    assert output.dummy_block == NUM_BLOCKS - 2
    dummy = scheduler.kv_cache_manager.block_pool.dummy_block
    assert dummy is not None and dummy.ref_cnt == 0
