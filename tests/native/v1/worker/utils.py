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

# Builders for the v1/worker tests: the runner's config, its KV cache config, and
# the scheduler outputs that drive it. Real NewRequestData/CachedRequestData, not
# duck types, so an upstream field rename surfaces here.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)

from vllm_rbln.v1.core.rbln_scheduler import RBLNSchedulerOutput

DEFAULT_PROMPT = [1, 2, 3]

# Only the HF config is read (kv heads / head size); no weights are loaded.
RUNNER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# RBLN needs an explicit block_size; NUM_BLOCKS stays tiny (~20 MiB of KV).
BLOCK_SIZE = 1024
NUM_BLOCKS = 10

MAX_NUM_SEQS = 4
MAX_NUM_BATCHED_TOKENS = 128
MAX_MODEL_LEN = 4096


def make_runner_config(**overrides: Any):
    """VllmConfig for the model runner, via the shared EngineArgs builder."""
    from tests.native.vllm_config import make_vllm_config

    kwargs: dict[str, Any] = dict(
        model=RUNNER_MODEL,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
    )
    kwargs.update(overrides)
    return make_vllm_config(**kwargs)


def make_kv_cache_config(
    runner: Any = None,
    *,
    groups: Sequence[Sequence[str]] = (("layer.0",),),
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    spec: KVCacheSpec | None = None,
) -> KVCacheConfig:
    """One group (and one tensor) per entry in ``groups``; each entry lists the
    layer names sharing that group. The spec comes from ``runner`` unless given."""
    if spec is None:
        assert runner is not None, "pass either a runner or an explicit spec"
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
            head_size=runner.model_config.get_head_size(),
            dtype=runner.kv_cache_dtype,
        )
    tensor_size = spec.page_size_bytes * num_blocks
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=list(layer_names))
            for layer_names in groups
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)
            for layer_names in groups
        ],
    )


def _pick(values, i, default):
    if values is None or i >= len(values) or values[i] is None:
        return default
    return values[i]


def make_scheduler_output(
    *,
    new_reqs: list[NewRequestData] | None = None,
    cached_reqs: CachedRequestData | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    finished_req_ids: set[str] | None = None,
    spec_decode_tokens: dict[str, list[int]] | None = None,
) -> RBLNSchedulerOutput:
    scheduled = num_scheduled_tokens or {}
    return RBLNSchedulerOutput(
        scheduled_new_reqs=new_reqs or [],
        scheduled_cached_reqs=cached_reqs or CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens=spec_decode_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=[],
    )


def schedule_new(
    *req_ids: str,
    prompt_token_ids: list[list[int]] | None = None,
    block_ids: list[tuple] | None = None,
    sampling_params: list[SamplingParams] | None = None,
    num_computed_tokens: list[int] | None = None,
    num_scheduled_tokens: list[int] | None = None,
) -> RBLNSchedulerOutput:
    """Schedule brand-new requests. Each keyword is a list aligned with
    ``req_ids``; a None entry falls back to that field's default, so callers
    only spell out what they vary."""
    new_reqs = []
    scheduled: dict[str, int] = {}
    for i, req_id in enumerate(req_ids):
        prompt = _pick(prompt_token_ids, i, DEFAULT_PROMPT)
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=_pick(sampling_params, i, SamplingParams()),
                pooling_params=None,
                block_ids=_pick(block_ids, i, ([i],)),
                num_computed_tokens=_pick(num_computed_tokens, i, 0),
                lora_request=None,
            )
        )
        scheduled[req_id] = _pick(num_scheduled_tokens, i, len(prompt))
    return make_scheduler_output(new_reqs=new_reqs, num_scheduled_tokens=scheduled)


def schedule_cached(
    *,
    req_ids: list[str],
    num_computed_tokens: list[int],
    num_scheduled_tokens: dict[str, int] | None = None,
    new_block_ids: list[Any] | None = None,
    new_token_ids: list[list[int]] | None = None,
    num_output_tokens: list[int] | None = None,
    resumed_req_ids: set[str] | None = None,
    spec_decode_tokens: dict[str, list[int]] | None = None,
) -> RBLNSchedulerOutput:
    """Schedule already-known requests (the running / resumed path)."""
    n = len(req_ids)
    cached = CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=resumed_req_ids or set(),
        new_token_ids=new_token_ids if new_token_ids is not None else [[]] * n,
        all_token_ids={},
        new_block_ids=new_block_ids if new_block_ids is not None else [None] * n,
        num_computed_tokens=list(num_computed_tokens),
        num_output_tokens=(
            num_output_tokens if num_output_tokens is not None else [0] * n
        ),
    )
    return make_scheduler_output(
        cached_reqs=cached,
        num_scheduled_tokens=(
            num_scheduled_tokens
            if num_scheduled_tokens is not None
            else dict.fromkeys(req_ids, 1)
        ),
        spec_decode_tokens=spec_decode_tokens,
    )


def is_in_batch(runner, req_id: str) -> bool:
    return req_id in runner.input_batch.req_id_to_index


def block_table_matches_state(runner, req_id: str) -> bool:
    """Whether the persistent batch's block table row agrees with the cached
    request state."""
    req_index = runner.input_batch.req_id_to_index[req_id]
    block_table = runner.input_batch.block_table[0]
    req_state = runner.requests[req_id]
    num_blocks = block_table.num_blocks_per_row[req_index]
    if num_blocks != len(req_state.block_ids[0]):
        return False
    row = block_table.block_table.np[req_index, :num_blocks]
    return (row == req_state.block_ids[0]).all()
