# Copyright 2025 Rebellions Inc. All rights reserved.
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

import contextlib
import os
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.sample.metadata import SamplingMetadata

import vllm_rbln.v1.worker.rbln_model_runner as rbln_model_runner_module
from vllm_rbln.v1.core.rbln_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

BLOCK_SIZE = 1024
NUM_BLOCKS = 10
DEVICE_TYPE = current_platform.device_type


# Fixture / Helper
def initialize_kv_cache(runner: RBLNModelRunner):
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS

    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )

    runner.initialize_kv_cache(kv_cache_config)


def get_vllm_config(
    *,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 128,
    max_model_len: int = 4096,
    block_size: int = BLOCK_SIZE,
) -> VllmConfig:
    model_config = ModelConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dtype=torch.float16,
        seed=42,
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    parallel_config = ParallelConfig()

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )


@contextmanager
def ensure_current_vllm_config():
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )

    if get_current_vllm_config_or_none() is not None:
        yield
    else:
        with set_current_vllm_config(VllmConfig()):
            yield


@pytest.fixture
def rbln_model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()

        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            num_heads,
            head_size,
            scale=0.1,
            prefix="layer.0",
        )

        runner = RBLNModelRunner(vllm_config, DEVICE_TYPE)
        initialize_kv_cache(runner)
        yield runner


@pytest.fixture
def dist_init():
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)

    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend=current_platform.dist_backend,
            )
            initialize_model_parallel(1, 1)
            yield
        try:
            cleanup_dist_env_and_memory()
        except RuntimeError as exc:
            if "Cannot access accelerator device when none is available." not in str(
                exc
            ):
                raise
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def _schedule_new_request(
    *req_ids: str,
    prompt_token_ids=None,
    block_ids=None,
    sampling_params=None,
    pooling_params=None,
    num_computed_tokens=None,
    num_scheduled_tokens=None,
    mm_features=None,
    lora_request=None,
) -> SchedulerOutput:
    """Build a SchedulerOutput scheduling one or more new requests.

    Called with only ``req_ids`` it reproduces the original defaults: prompt
    ``[1, 2, 3]``, block ``([i],)``, ``SamplingParams()``, ``num_computed_tokens``
    0 and ``num_scheduled_tokens == len(prompt)``.

    Each keyword is an optional list positionally aligned with ``req_ids``. A
    list shorter than ``req_ids`` (or a ``None`` at position ``i``) falls back to
    the default for that request/field, so callers only specify what they vary.
    """
    for name, values in (
        ("prompt_token_ids", prompt_token_ids),
        ("block_ids", block_ids),
        ("sampling_params", sampling_params),
        ("pooling_params", pooling_params),
        ("num_computed_tokens", num_computed_tokens),
        ("num_scheduled_tokens", num_scheduled_tokens),
        ("mm_features", mm_features),
        ("lora_request", lora_request),
    ):
        assert values is None or len(values) <= len(req_ids), (
            f"{name} override list ({len(values)}) longer than req_ids ({len(req_ids)})"
        )

    def _pick(values, i, default):
        if values is None or i >= len(values) or values[i] is None:
            return default
        return values[i]

    new_reqs = []
    scheduled_tokens = {}
    for i, req_id in enumerate(req_ids):
        prompt = _pick(prompt_token_ids, i, [1, 2, 3])
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=_pick(mm_features, i, []),
                sampling_params=_pick(sampling_params, i, SamplingParams()),
                pooling_params=_pick(pooling_params, i, None),
                block_ids=_pick(block_ids, i, ([i],)),
                num_computed_tokens=_pick(num_computed_tokens, i, 0),
                lora_request=_pick(lora_request, i, None),
            )
        )
        scheduled_tokens[req_id] = _pick(num_scheduled_tokens, i, len(prompt))

    return RBLNSchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_req_scheduled(model_runner: RBLNModelRunner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner: RBLNModelRunner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(
    model_runner: RBLNModelRunner, sampling_metadata_before: SamplingMetadata
):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner: RBLNModelRunner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (
        block_table.block_table.np[req_index, :num_blocks] == req_state.block_ids[0]
    ).all()


def _unexpected_call(message: str):
    def fail(*args, **kwargs):
        raise AssertionError(message)

    return fail


def _sampler_output(token_ids, *, device=None, logprobs_tensors=None):
    if isinstance(token_ids, torch.Tensor):
        sampled_token_ids = token_ids
    else:
        sampled_token_ids = torch.tensor(token_ids, dtype=torch.int32, device=device)

    return rbln_model_runner_module.SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=logprobs_tensors,
    )


def _recording_attention_metadata_builder(calls, common_attn_metadata=None):
    def fake_build_attention_metadata(**kwargs):
        calls.update(kwargs)
        return {}, common_attn_metadata

    return fake_build_attention_metadata


def _recording_model_executable(calls, hidden_states, logits):
    def fake_model_executable(
        *,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        token_indices,
        **kwargs,
    ):
        calls.update(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=token_indices,
            kwargs=kwargs,
        )
        return hidden_states, None, logits

    return fake_model_executable


def _set_execute_model_state(
    runner,
    *,
    scheduler_output,
    logits,
    hidden_states,
    spec_decode_metadata=None,
    common_attn_metadata=None,
    sample_hidden_states=None,
    aux_hidden_states=None,
):
    runner.execute_model_state = rbln_model_runner_module.ExecuteModelState(
        scheduler_output=scheduler_output,
        logits=logits,
        spec_decode_metadata=spec_decode_metadata,
        spec_decode_common_attn_metadata=common_attn_metadata,
        hidden_states=hidden_states,
        sample_hidden_states=hidden_states
        if sample_hidden_states is None
        else sample_hidden_states,
        aux_hidden_states=aux_hidden_states,
    )


def _bookkeeping_return(
    *,
    valid_sampled_token_ids,
    req_ids,
    req_id_to_index,
    num_nans_in_logits=None,
    logprobs_lists=None,
    prompt_logprobs_dict=None,
):
    return (
        num_nans_in_logits or {},
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict or {},
        req_ids,
        req_id_to_index,
    )


class FakeSpecConfig:
    def __init__(self, method: str = "ngram", num_speculative_tokens: int = 3):
        self.method = method
        self.num_speculative_tokens = num_speculative_tokens

    def use_eagle(self):
        return self.method in {"eagle", "eagle3"}


class FakeBucketingManager:
    def __init__(self, *, buckets=None, default=None, fail_message: str | None = None):
        self.buckets = buckets or {}
        self.default = default
        self.fail_message = fail_message
        self.calls: list[Any] = []

    def find_decode_batch_bucket(self, batch_size):
        if self.fail_message is not None:
            raise AssertionError(self.fail_message)

        self.calls.append(batch_size)
        if batch_size in self.buckets:
            return self.buckets[batch_size]
        if self.default is not None:
            return self.default
        raise AssertionError(f"unexpected decode bucket request: {batch_size}")


# ============================================================================
# Tests for Request Lifecycle
# ============================================================================
def test_update_states_new_request(rbln_model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = rbln_model_runner.input_batch.sampling_metadata
    rbln_model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(rbln_model_runner, metadata_before)
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_request_finished(rbln_model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    rbln_model_runner._update_states(scheduler_output)
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = rbln_model_runner.input_batch.sampling_metadata
    rbln_model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(rbln_model_runner, metadata_before)
    assert not _is_req_added(rbln_model_runner, req_id)
    assert not _is_req_scheduled(rbln_model_runner, req_id)


def test_update_states_unscheduled_cached_request_readded(rbln_model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    rbln_model_runner._update_states(scheduler_output)
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    rbln_model_runner._update_states(scheduler_output)
    assert _is_req_added(rbln_model_runner, req_id)
    assert not _is_req_scheduled(rbln_model_runner, req_id)
    assert rbln_model_runner.requests[req_id].block_ids[0] == [0]

    # schedule the cached request again without preemption. This re-adds the
    # existing cached state to the persistent batch; it must not replace the
    # block table like the resumed-from-preemption path does.
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[3],
        num_output_tokens=[0],
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = rbln_model_runner.input_batch.sampling_metadata
    rbln_model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(rbln_model_runner, metadata_before)
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)
    assert rbln_model_runner.requests[req_id].block_ids[0] == [0]
    req_index = rbln_model_runner.input_batch.req_id_to_index[req_id]
    assert rbln_model_runner.input_batch.num_computed_tokens_cpu[req_index] == 3
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_no_changes(rbln_model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    rbln_model_runner._update_states(scheduler_output)
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)

    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    # Keep the same cached request running without new tokens or block deltas.
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = rbln_model_runner.input_batch.sampling_metadata
    rbln_model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(rbln_model_runner, metadata_before)
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_block_table_append_on_running_request(
    rbln_model_runner, dist_init
):
    """Running (non-resumed) cached req: new_block_ids across steps must EXTEND
    req_state.block_ids and grow block_table.num_blocks_per_row (append, not
    replace)."""
    req_id = "req_0"

    # Step 1: new request enters with a single block [0].
    rbln_model_runner._update_states(_schedule_new_request(req_id))
    assert _is_req_scheduled(rbln_model_runner, req_id)

    # req_state is mutated in place by _update_states, so capture it once.
    req_state = rbln_model_runner.requests[req_id]
    req_index = rbln_model_runner.input_batch.req_id_to_index[req_id]
    block_table = rbln_model_runner.input_batch.block_table[0]

    assert req_state.block_ids[0] == [0]
    assert block_table.num_blocks_per_row[req_index] == 1
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)

    # Step 2: same request keeps running (NOT resumed) and is granted a new
    # block [1] as a delta on the next decode step.
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),  # running, not resumed-from-preemption
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([1],)],  # delta: one new block for group 0
        num_computed_tokens=[3],
        num_output_tokens=[0],
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},  # keeps req in the batch (req_index stays)
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    rbln_model_runner._update_states(scheduler_output)

    # The delta must be APPENDED, not replace the table: [0] -> [0, 1].
    assert req_state.block_ids[0] == [0, 1]
    assert rbln_model_runner.input_batch.req_id_to_index[req_id] == req_index
    assert block_table.num_blocks_per_row[req_index] == 2
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_resumed_from_preemption_replaces_block_table(
    rbln_model_runner, dist_init
):
    """resumed_from_preemption: block_ids are REPLACED with the full table (not
    appended), and req_index must be None at that point."""
    req_id = "req_0"

    # Step 1: new request enters the batch with block [0].
    rbln_model_runner._update_states(_schedule_new_request(req_id))
    req_state = rbln_model_runner.requests[req_id]  # same object, mutated in place
    assert req_state.block_ids[0] == [0]
    assert _is_req_scheduled(rbln_model_runner, req_id)

    # Step 2: unschedule it -> removed from the persistent batch (req_index
    # becomes None) but kept in self.requests. This is the precondition for the
    # resumed branch, which asserts req_index is None.
    rbln_model_runner._update_states(
        SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
    )
    assert _is_req_added(rbln_model_runner, req_id)
    assert not _is_req_scheduled(rbln_model_runner, req_id)

    # Step 3: resume from preemption with a brand-new FULL block table [5].
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids={req_id},  # <- resumed-from-preemption path
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([5],)],  # full table, not a delta
        num_computed_tokens=[0],  # preemption reset computed tokens
        num_output_tokens=[0],
    )
    rbln_model_runner._update_states(
        SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={req_id: 3},
            total_num_scheduled_tokens=3,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
    )

    # REPLACE, not append: block_ids must be exactly [5] (NOT [0, 5]).
    assert req_state.block_ids[0] == [5]
    assert _is_req_scheduled(rbln_model_runner, req_id)
    req_index = rbln_model_runner.input_batch.req_id_to_index[req_id]
    block_table = rbln_model_runner.input_batch.block_table[0]
    assert block_table.num_blocks_per_row[req_index] == 1
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_streaming_request_readd(rbln_model_runner, dist_init):
    """A req_id already in self.requests reappearing in scheduled_new_reqs goes
    through _update_streaming_request: output_token_ids cleared, prompt/blocks
    refreshed, removed + re-added to the batch."""
    req_id = "req_0"

    # Step 1: register the request (prompt [1,2,3], block [0]) so a later
    # scheduled_new_reqs entry with the same id hits the streaming branch.
    rbln_model_runner._update_states(_schedule_new_request(req_id))
    req_state = rbln_model_runner.requests[req_id]  # updated in place later
    assert _is_req_scheduled(rbln_model_runner, req_id)

    # Seed prior-turn output so we can verify it gets cleared on re-submit.
    req_state.output_token_ids.extend([9, 9])

    # Step 2: same req_id re-submitted as a streaming turn with a longer prompt
    # (prior output folded in) and a different block table.
    new_prompt = [1, 2, 3, 4, 5]
    scheduler_output = _schedule_new_request(
        req_id,
        prompt_token_ids=[new_prompt],
        block_ids=[([2],)],  # different block table than step 1 ([0])
    )
    rbln_model_runner._update_states(scheduler_output)

    # The cached state object is refreshed in place (not recreated).
    assert rbln_model_runner.requests[req_id] is req_state
    assert req_state.prompt_token_ids == new_prompt
    assert req_state.num_prompt_tokens == len(new_prompt)
    assert req_state.block_ids[0] == [2]  # replaced, NOT [0, 2]
    assert req_state.output_token_ids == []  # prior-turn output cleared

    # Removed then re-added to the persistent batch with the new block table.
    assert _is_req_scheduled(rbln_model_runner, req_id)
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_prompt_logprobs_registration(rbln_model_runner, dist_init):
    """sampling_params.prompt_logprobs set -> num_prompt_logprobs registered
    (-1 -> vocab_size); finished req -> entry popped."""
    vocab_size = rbln_model_runner.input_batch.vocab_size

    # req_id -> prompt_logprobs value
    cases = {
        "req_none": None,  # not requested -> must NOT be registered
        "req_k": 5,  # explicit count -> registered as 5
        "req_all": -1,  # "all" sentinel -> registered as vocab_size
    }

    scheduler_output = _schedule_new_request(
        *cases,
        sampling_params=[SamplingParams(prompt_logprobs=plp) for plp in cases.values()],
    )
    rbln_model_runner._update_states(scheduler_output)

    # Registration: None -> absent; k -> k; -1 -> vocab_size.
    assert "req_none" not in rbln_model_runner.num_prompt_logprobs
    assert rbln_model_runner.num_prompt_logprobs["req_k"] == 5
    assert rbln_model_runner.num_prompt_logprobs["req_all"] == vocab_size

    # Cleanup: finishing the requests pops their prompt-logprobs entries.
    finish_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(cases),
        free_encoder_mm_hashes=[],
    )
    rbln_model_runner._update_states(finish_output)

    for req_id in cases:
        assert req_id not in rbln_model_runner.num_prompt_logprobs


def test_update_states_resumed_while_still_cached_is_recreated(
    rbln_model_runner, dist_init
):
    """Forced-preemption overlap (reset_prefix_cache): a request that is still
    in the persistent batch AND marked resumed in the same step must be cleared
    from the batch first, so the resumed path (assert req_index is None +
    full block-table replace) runs cleanly. Guards the `- resumed_req_ids` term
    in `unscheduled = cached - (scheduled - resumed)`."""
    req_id = "req_0"

    # Step 1: new request enters the batch with block [0] (req_index set).
    rbln_model_runner._update_states(_schedule_new_request(req_id))
    req_state = rbln_model_runner.requests[req_id]  # same object, mutated later
    assert _is_req_scheduled(rbln_model_runner, req_id)
    assert req_state.block_ids[0] == [0]

    # Step 2: SAME step marks req_0 resumed while it is STILL in the batch
    # (no unschedule in between). The `- resumed` term must move it into the
    # unscheduled set so it is removed before the resumed branch re-adds it;
    # without that term the resumed branch would hit `assert req_index is None`.
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids={req_id},  # resumed + still cached (forced preempt)
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([5],)],  # full table, replaces [0]
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    # Must not raise (req_index was cleared, so the resumed-branch assert holds).
    rbln_model_runner._update_states(scheduler_output)

    # Cleared then re-added via the resumed path: block table REPLACED with [5].
    assert req_state.block_ids[0] == [5]  # replaced, NOT [0, 5]
    assert _is_req_scheduled(rbln_model_runner, req_id)
    req_index = rbln_model_runner.input_batch.req_id_to_index[req_id]
    block_table = rbln_model_runner.input_batch.block_table[0]
    assert block_table.num_blocks_per_row[req_index] == 1
    assert _is_req_state_block_table_match(rbln_model_runner, req_id)


def test_update_states_condense_after_gaps(rbln_model_runner, dist_init):
    """Removing a middle request leaves a gap; condense() slides the highest
    request down into it so occupied indices stay contiguous [0, num_reqs), and
    that request's state (block table) moves with it."""
    req_ids = ("req_0", "req_1", "req_2")

    # Step 1: three requests occupy indices 0, 1, 2 (blocks [0], [1], [2]).
    rbln_model_runner._update_states(_schedule_new_request(*req_ids))
    assert rbln_model_runner.input_batch.req_id_to_index == {
        "req_0": 0,
        "req_1": 1,
        "req_2": 2,
    }

    # Step 2: keep req_0 and req_2, drop the MIDDLE one (req_1), add nothing.
    # -> req_1 removed (gap at index 1); with no new request to backfill it,
    #    condense() slides req_2 (index 2) down into index 1.
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 1, "req_2": 1},  # req_1 omitted -> unscheduled
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    rbln_model_runner._update_states(scheduler_output)

    # Batch is condensed: indices contiguous, req_2 moved 2 -> 1.
    assert rbln_model_runner.input_batch.num_reqs == 2
    assert rbln_model_runner.input_batch.req_id_to_index == {"req_0": 0, "req_2": 1}

    # req_1 was unscheduled (not finished): out of the batch, still cached.
    assert not _is_req_scheduled(rbln_model_runner, "req_1")
    assert _is_req_added(rbln_model_runner, "req_1")

    # State moved with the request: index 1's block table is req_2's [2],
    # and req_0 at index 0 is untouched.
    assert rbln_model_runner.requests["req_2"].block_ids[0] == [2]
    assert _is_req_state_block_table_match(rbln_model_runner, "req_2")
    assert _is_req_state_block_table_match(rbln_model_runner, "req_0")


def test_update_states_random_seed_generator(rbln_model_runner, dist_init):
    """SamplingType.RANDOM_SEED (seed set + temperature > 0) -> a torch.Generator
    seeded from sampling_params is attached to the cached state; otherwise the
    generator is None."""
    # req_seed: seed set -> RANDOM_SEED -> seeded generator.
    # req_none: no seed -> RANDOM -> generator is None.
    rbln_model_runner._update_states(
        _schedule_new_request(
            "req_seed",
            "req_none",
            sampling_params=[SamplingParams(seed=42), SamplingParams()],
        )
    )

    gen = rbln_model_runner.requests["req_seed"].generator
    assert isinstance(gen, torch.Generator)
    assert gen.initial_seed() == 42

    assert rbln_model_runner.requests["req_none"].generator is None


# ============================================================================
# Tests for Spec Decode Indexing
# ============================================================================
def test_calc_spec_decode_metadata_golden(rbln_model_runner):
    """Index arithmetic of _calc_spec_decode_metadata."""
    # Fill input_ids with positions (in place, no resize) so draft_token_ids is
    # checkable; every gathered logits index stays < buffer size.
    n = rbln_model_runner.input_ids.shape[0]
    rbln_model_runner.input_ids[:] = torch.arange(
        n,
        dtype=rbln_model_runner.input_ids.dtype,
        device=rbln_model_runner.input_ids.device,
    )

    num_draft_tokens = np.array([3, 0, 2, 0, 1], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 9, 12, 17, 19], dtype=np.int32)

    md = rbln_model_runner._calc_spec_decode_metadata(
        num_draft_tokens, cu_num_scheduled_tokens
    )

    assert md.num_draft_tokens == [3, 0, 2, 0, 1]
    assert md.cu_num_draft_tokens.tolist() == [3, 3, 5, 5, 6]
    assert md.cu_num_sampled_tokens.tolist() == [4, 5, 8, 9, 11]
    assert md.logits_indices.tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18]
    assert md.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]
    assert md.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10]
    # input_ids == positions, so draft_token_ids == gathered[target_logits+1].
    assert md.draft_token_ids.tolist() == [1, 2, 3, 10, 11, 18]


def test_may_reorder_batch_noop_paths(rbln_model_runner, dist_init, monkeypatch):
    """_may_reorder_batch leaves the batch order untouched when sorting is
    disabled, when already sorted, and when there are < 2 requests."""
    envs = rbln_model_runner_module.envs
    ib = rbln_model_runner.input_batch
    sched = _schedule_new_request()  # arg is ignored by the RBLN reorder path

    rbln_model_runner._update_states(_schedule_new_request("a", "b", "c"))

    # (1) Sorting disabled: even an unsorted batch is left untouched.
    monkeypatch.setattr(envs, "VLLM_RBLN_SORT_BATCH", False)
    ib.num_tokens_no_spec[:3] = [1, 5, 3]  # NOT descending
    before = dict(ib.req_id_to_index)
    rbln_model_runner._may_reorder_batch(sched)
    assert dict(ib.req_id_to_index) == before

    # (2) Sorting enabled but batch already descending -> no swaps.
    monkeypatch.setattr(envs, "VLLM_RBLN_SORT_BATCH", True)
    ib.num_tokens_no_spec[:3] = [5, 3, 1]  # already descending
    before = dict(ib.req_id_to_index)
    rbln_model_runner._may_reorder_batch(sched)
    assert dict(ib.req_id_to_index) == before

    # (3) Sorting enabled but fewer than 2 requests -> early return.
    rbln_model_runner._update_states(
        SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"a": 1},  # keep only "a"; b, c unscheduled
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
    )
    assert ib.num_reqs == 1
    before = dict(ib.req_id_to_index)
    rbln_model_runner._may_reorder_batch(sched)
    assert dict(ib.req_id_to_index) == before


def test_may_reorder_batch_sorts_by_seq_len_descending(
    rbln_model_runner, dist_init, monkeypatch
):
    """With sorting enabled and an unsorted batch, _may_reorder_batch does a
    stable descending sort by num_tokens_no_spec via in-place swap_states, and
    every request's state (block table) moves with it."""
    envs = rbln_model_runner_module.envs
    ib = rbln_model_runner.input_batch
    sched = _schedule_new_request()  # arg ignored by the RBLN reorder path

    # a, b, c at indices 0, 1, 2 with blocks [0], [1], [2].
    rbln_model_runner._update_states(_schedule_new_request("a", "b", "c"))
    assert ib.req_id_to_index == {"a": 0, "b": 1, "c": 2}

    monkeypatch.setattr(envs, "VLLM_RBLN_SORT_BATCH", True)
    # a=1, b=5, c=3  -> descending order should be b, c, a.
    ib.num_tokens_no_spec[:3] = [1, 5, 3]

    rbln_model_runner._may_reorder_batch(sched)

    # b(5) -> idx0, c(3) -> idx1, a(1) -> idx2.
    assert ib.req_id_to_index == {"b": 0, "c": 1, "a": 2}
    assert ib.num_tokens_no_spec[:3].tolist() == [5, 3, 1]

    # State moved with each request: block tables still match their owners
    # (a still owns [0], b [1], c [2], just at new indices).
    for req_id in ("a", "b", "c"):
        assert _is_req_state_block_table_match(rbln_model_runner, req_id)


# ============================================================================
# Tests for Input Preparation and Property Boundaries
# ============================================================================
def test_prepare_inputs_decode_path(rbln_model_runner, dist_init):
    """No spec tokens -> logits_indices = query_start_loc[1:]-1, seq_lens =
    num_computed + num_scheduled, discard_mask all False, spec metadata None."""
    # Two fully-prefilled requests in decode (1 token each).
    rbln_model_runner._update_states(_schedule_new_request("a", "b"))
    ib = rbln_model_runner.input_batch
    # num_tokens (prompt [1,2,3]) = 3; set num_computed so seq_lens(=4) >= 3
    # -> nothing discarded.
    ib.num_computed_tokens_cpu[:2] = [3, 3]

    sched = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"a": 1, "b": 1},
        total_num_scheduled_tokens=2,  # must be > 0
        scheduled_spec_decode_tokens={},  # empty -> use_spec_decode False
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits_indices, spec_md, query_lengths, total = rbln_model_runner._prepare_inputs(
        sched, np.array([1, 1], dtype=np.int32)
    )

    # Decode path: no spec metadata, last-token logits, qlen == 1 each.
    assert spec_md is None
    assert logits_indices.tolist() == [0, 1]  # query_start_loc[1:3]-1
    assert query_lengths.tolist() == [1, 1]
    assert total == 2

    # seq_lens = num_computed(3) + scheduled(1); nothing discarded.
    assert rbln_model_runner.seq_lens[:2].tolist() == [4, 4]
    assert rbln_model_runner.discard_request_mask[:2].tolist() == [False, False]


def test_prepare_inputs_chunked_prefill_sets_discard_mask(rbln_model_runner, dist_init):
    """A partial (chunked) prefill request has seq_lens < num_tokens, so its
    discard_request_mask entry is True (its sampled token is dropped); a fully
    processed request stays False."""
    # a, b both have prompt [1,2,3] -> num_tokens == 3.
    rbln_model_runner._update_states(_schedule_new_request("a", "b"))
    ib = rbln_model_runner.input_batch
    # a: mid-prefill (computed 0, will schedule 2 -> seq_lens 2 < 3) -> discard.
    # b: last chunk    (computed 2, will schedule 1 -> seq_lens 3 == 3) -> keep.
    ib.num_computed_tokens_cpu[:2] = [0, 2]

    sched = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"a": 2, "b": 1},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={},  # use_spec_decode False
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits_indices, spec_md, query_lengths, total = rbln_model_runner._prepare_inputs(
        sched, np.array([2, 1], dtype=np.int32)
    )

    # a is partial -> discarded; b is complete -> kept.
    assert rbln_model_runner.discard_request_mask[:2].tolist() == [True, False]
    assert rbln_model_runner.seq_lens[:2].tolist() == [2, 3]  # computed + scheduled
    # Supporting: still the non-spec branch.
    assert spec_md is None
    assert query_lengths.tolist() == [2, 1]
    assert total == 3
    assert logits_indices.tolist() == [1, 2]  # query_start_loc[1:]-1


def test_prepare_inputs_spec_decode_path(rbln_model_runner, dist_init, monkeypatch):
    """Spec decode kept: query_lengths padded to num_spec+1, backfill >= 0,
    positions shifted by -backfill, seq_lens from the *logical* count, and
    spec_decode_metadata produced."""
    rbln_model_runner._update_states(_schedule_new_request("a"))  # prompt [1,2,3]
    ib = rbln_model_runner.input_batch

    # Enable spec decode (fixture has none) and force a decode state.
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 2)  # target qlen = 3
    ib.num_computed_tokens_cpu[0] = 3
    ib.num_tokens_no_spec[0] = 3  # is_prefill: 3 < 3-1 -> False (decode)

    # 1 draft kept -> logical = 1 real + 1 draft = 2 ; backfill = 3 - 2 = 1.
    sched = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"a": 2},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={"a": [11]},  # non-empty -> use_spec_decode
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits_indices, spec_md, query_lengths, total = rbln_model_runner._prepare_inputs(
        sched, np.array([2], dtype=np.int32)
    )

    # Fixed full-spec query, padded to num_spec+1 even though logical was 2.
    assert query_lengths.tolist() == [3]
    assert total == 3

    # positions shifted by -backfill(1): start at num_computed(3) - 1 = 2.
    assert rbln_model_runner.positions_np[:3].tolist() == [2, 3, 4]

    # seq_lens uses the LOGICAL count (2), not query_lengths: 3 + 2 = 5.
    assert rbln_model_runner.seq_lens[:1].tolist() == [5]

    # Spec metadata produced; draft count reflects the scheduled draft tokens.
    assert spec_md is not None
    assert spec_md.num_draft_tokens == [1]
    assert logits_indices.tolist() == spec_md.logits_indices.tolist()


def test_prepare_inputs_num_spec_tokens_without_scheduled_drafts_uses_logical_lengths(
    rbln_model_runner, dist_init, monkeypatch
):
    """num_spec_tokens alone must not force full-spec query padding when the
    scheduler did not keep any draft tokens for the step."""
    req_id = "a"

    rbln_model_runner._update_states(_schedule_new_request(req_id))
    ib = rbln_model_runner.input_batch

    # Spec decode is configured, but this step has no accepted/scheduled draft
    # tokens. This can happen for zero-draft ngram/suffix steps or boundary cases.
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 2)

    # Force decode state. is_prefill must be False.
    ib.num_computed_tokens_cpu[0] = 3
    ib.num_tokens_no_spec[0] = 3
    assert rbln_model_runner.is_prefill is False

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[0],
        ),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},  # key condition: no draft tokens this step
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits_indices, spec_md, query_lengths, total = rbln_model_runner._prepare_inputs(
        scheduler_output,
        np.array([1], dtype=np.int32),
    )

    assert spec_md is None
    assert query_lengths.tolist() == [1]
    assert total == 1
    assert logits_indices.tolist() == [0]

    # No backfill/full-spec padding: position is computed + arange = 3.
    assert rbln_model_runner.positions_np[:1].tolist() == [3]

    # seq_lens uses the logical scheduled token count, not num_spec_tokens + 1.
    assert rbln_model_runner.seq_lens[:1].tolist() == [4]
    assert rbln_model_runner.discard_request_mask[:1].tolist() == [False]


def test_prepare_inputs_pads_query_start_loc_and_seq_lens(rbln_model_runner, dist_init):
    """Entries beyond num_reqs are overwritten: query_start_loc tail = total
    query tokens (cu[-1]), seq_lens tail = 0, regardless of stale leftovers."""
    rbln_model_runner._update_states(_schedule_new_request("a", "b"))
    ib = rbln_model_runner.input_batch
    ib.num_computed_tokens_cpu[:2] = [3, 3]

    # Poison the tail with stale values to prove they get overwritten.
    rbln_model_runner.query_start_loc.fill_(999)
    rbln_model_runner.seq_lens.fill_(999)

    sched = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"a": 1, "b": 1},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    rbln_model_runner._prepare_inputs(sched, np.array([1, 1], dtype=np.int32))

    qsl = rbln_model_runner.query_start_loc
    sl = rbln_model_runner.seq_lens

    # Real part: [0, cu...] = [0, 1, 2]; seq_lens = computed + scheduled = [4, 4].
    assert qsl[:3].tolist() == [0, 1, 2]
    assert sl[:2].tolist() == [4, 4]

    # Padded tail (beyond num_reqs): query_start_loc = total (2), seq_lens = 0.
    assert (qsl[3:] == 2).all()  # cu_num_tokens[
    assert (sl[2:] == 0).all()


def test_prepare_kv_sharing_fast_prefill_pads_with_last_index(rbln_model_runner):
    """Copies logits_indices into the buffer, fills the trailing slots with the
    last index (so stale leftovers can't be out-of-range), and returns the
    [:num_logits] slice. When the buffer is exactly full, no padding is needed."""
    dev = rbln_model_runner.device
    # Buffer normally exists only when kv_sharing_fast_prefill is enabled; set it
    # up with stale values (999) to prove the tail gets overwritten.
    rbln_model_runner.kv_sharing_fast_prefill_logits_indices = torch.full(
        (8,), 999, dtype=torch.int32, device=dev
    )

    logits_indices = torch.tensor([0, 2, 5], dtype=torch.int32, device=dev)
    result = rbln_model_runner._prepare_kv_sharing_fast_prefill(logits_indices)

    buf = rbln_model_runner.kv_sharing_fast_prefill_logits_indices
    # Real part copied in; returned slice is exactly the real part.
    assert result.tolist() == [0, 2, 5]
    assert buf[:3].tolist() == [0, 2, 5]
    # Tail padded with the LAST index (5), stale 999 overwritten.
    assert (buf[3:] == 5).all()

    # Exact fit: no tail exists, so the method should simply expose the copied
    # logits indices without touching any padded region.
    rbln_model_runner.kv_sharing_fast_prefill_logits_indices = torch.full(
        (3,), 999, dtype=torch.int32, device=dev
    )
    logits_indices = torch.tensor([0, 2, 5], dtype=torch.int32, device=dev)
    result = rbln_model_runner._prepare_kv_sharing_fast_prefill(logits_indices)
    assert result.tolist() == [0, 2, 5]


def test_is_prefill_boundary_properties(rbln_model_runner):
    """is_prefill: num_computed < num_tokens_no_spec - 1 (boundary at ==);
    is_intermediate_chunked_prefill = is_prefill AND discard_mask[0];
    use_wrapped_compute_logits = not pooling."""
    ib = rbln_model_runner.input_batch

    def set_state(computed, nts, discard):
        ib.num_computed_tokens_cpu[0] = computed
        ib.num_tokens_no_spec[0] = nts
        rbln_model_runner.discard_request_mask[0] = discard

    # is_prefill: strictly less than nts - 1.
    set_state(1, 5, False)
    assert rbln_model_runner.is_prefill is True  # 1 < 4
    set_state(4, 5, False)
    assert rbln_model_runner.is_prefill is False  # 4 < 4 -> boundary == decode
    set_state(5, 5, False)
    assert rbln_model_runner.is_prefill is False  # past boundary

    # is_intermediate_chunked_prefill = is_prefill AND discard_mask[0].
    set_state(1, 5, True)  # prefill + discarded chunk
    assert rbln_model_runner.is_intermediate_chunked_prefill is True
    set_state(1, 5, False)  # prefill, last chunk kept
    assert rbln_model_runner.is_intermediate_chunked_prefill is False
    set_state(4, 5, True)  # NOT prefill -> False regardless
    assert rbln_model_runner.is_intermediate_chunked_prefill is False

    # Generation model (fixture) is not a pooling model.
    assert rbln_model_runner.is_pooling_model is False
    assert rbln_model_runner.use_wrapped_compute_logits is True


# ============================================================================
# Tests for Sampling and Bookkeeping
# ============================================================================
def test_sample_skips_sampler_for_intermediate_chunked_prefill(
    rbln_model_runner, monkeypatch
):
    """Intermediate chunked prefill returns a dummy -1 token without invoking
    the sampler. This guards the fast path used when sampled tokens are
    discarded by discard_request_mask."""
    ib = rbln_model_runner.input_batch

    # Make is_prefill True:
    ib.num_computed_tokens_cpu[0] = 1
    ib.num_tokens_no_spec[0] = 5

    # Make is_intermediate_chunked_prefill True:
    rbln_model_runner.discard_request_mask[0] = True
    assert rbln_model_runner.is_intermediate_chunked_prefill is True

    monkeypatch.setattr(
        rbln_model_runner,
        "sampler",
        _unexpected_call("regular sampler must not be called"),
    )

    logits = torch.randn(2, 16, dtype=torch.float32, device=rbln_model_runner.device)

    output = rbln_model_runner._sample(
        logits=logits,
        spec_decode_metadata=None,
    )

    assert output.sampled_token_ids.shape == (1, 1)
    assert output.sampled_token_ids.dtype == torch.int32
    assert output.sampled_token_ids.device == logits.device
    assert output.sampled_token_ids.tolist() == [[-1]]
    assert output.logprobs_tensors is None


def test_sample_routes_to_regular_sampler(rbln_model_runner, monkeypatch):
    """Without spec decode metadata and outside intermediate chunked prefill,
    _sample should call the regular sampler with current SamplingMetadata."""
    ib = rbln_model_runner.input_batch

    # Make sure this is not intermediate chunked prefill.
    ib.num_computed_tokens_cpu[0] = 3
    ib.num_tokens_no_spec[0] = 3
    rbln_model_runner.discard_request_mask[0] = False
    assert rbln_model_runner.is_intermediate_chunked_prefill is False

    expected_output = _sampler_output([[7]])

    calls = {}

    def fake_sampler(*, logits, sampling_metadata):
        calls["logits"] = logits
        calls["sampling_metadata"] = sampling_metadata
        return expected_output

    monkeypatch.setattr(rbln_model_runner, "sampler", fake_sampler)
    monkeypatch.setattr(
        rbln_model_runner,
        "rejection_sampler",
        _unexpected_call("rejection sampler must not be called"),
        raising=False,
    )

    logits = torch.randn(1, 16, dtype=torch.float32, device=rbln_model_runner.device)

    output = rbln_model_runner._sample(
        logits=logits,
        spec_decode_metadata=None,
    )

    assert output is expected_output
    assert calls["logits"] is logits
    assert calls["sampling_metadata"] is rbln_model_runner.input_batch.sampling_metadata


def test_sample_routes_to_rejection_sampler_for_spec_decode(
    rbln_model_runner, monkeypatch
):
    """When spec_decode_metadata is present, _sample must call the rejection
    sampler rather than the regular sampler."""
    expected_output = _sampler_output([[10, 11]])

    calls = {}

    def fake_rejection_sampler(
        spec_decode_metadata,
        draft_probs,
        logits,
        sampling_metadata,
    ):
        calls["spec_decode_metadata"] = spec_decode_metadata
        calls["draft_probs"] = draft_probs
        calls["logits"] = logits
        calls["sampling_metadata"] = sampling_metadata
        return expected_output

    monkeypatch.setattr(
        rbln_model_runner,
        "sampler",
        _unexpected_call("regular sampler must not be called"),
    )
    monkeypatch.setattr(
        rbln_model_runner, "rejection_sampler", fake_rejection_sampler, raising=False
    )

    logits = torch.randn(2, 16, dtype=torch.float32, device=rbln_model_runner.device)
    spec_decode_metadata = object()

    output = rbln_model_runner._sample(
        logits=logits,
        spec_decode_metadata=spec_decode_metadata,
    )

    assert output is expected_output
    assert calls["spec_decode_metadata"] is spec_decode_metadata
    assert calls["draft_probs"] is None
    assert calls["logits"] is logits
    assert calls["sampling_metadata"] is rbln_model_runner.input_batch.sampling_metadata


def test_bookkeeping_sync_caches_sampled_tokens(rbln_model_runner, dist_init):
    """Accepted sampled tokens are written to token_ids_cpu / is_token_ids,
    num_tokens_no_spec is advanced, and req_state.output_token_ids is extended."""
    req_ids = ("req_0", "req_1")

    rbln_model_runner._update_states(_schedule_new_request(*req_ids))

    ib = rbln_model_runner.input_batch
    assert ib.req_id_to_index == {"req_0": 0, "req_1": 1}

    # Default prompt is [1, 2, 3], so sampled decode tokens should be appended
    # at position 3 for each request.
    ib.num_tokens_no_spec[:2] = [3, 3]
    rbln_model_runner.discard_request_mask[:2] = False

    sampler_output = _sampler_output(
        [[101], [202]],
        device=rbln_model_runner.device,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 1, "req_1": 1},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    hidden_states = torch.empty(
        (2, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    (
        num_nans_in_logits,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
    ) = rbln_model_runner._bookkeeping_sync(
        scheduler_output=scheduler_output,
        sampler_output=sampler_output,
        logits=None,
        hidden_states=hidden_states,
        num_scheduled_tokens=2,
    )

    assert num_nans_in_logits == {}
    assert logprobs_lists is None
    assert prompt_logprobs_dict == {}

    assert valid_sampled_token_ids == [[101], [202]]
    assert req_ids_output_copy == ["req_0", "req_1"]
    assert req_id_to_index_output_copy == {"req_0": 0, "req_1": 1}

    # Tokens are appended at the old num_tokens_no_spec position.
    assert ib.token_ids_cpu[0, 3].item() == 101
    assert ib.token_ids_cpu[1, 3].item() == 202
    assert ib.is_token_ids[0, 3].item() is True
    assert ib.is_token_ids[1, 3].item() is True

    # Cursor advanced by one token for each request.
    assert ib.num_tokens_no_spec[:2].tolist() == [4, 4]

    # Cached request state is updated for scheduler-side reuse.
    assert rbln_model_runner.requests["req_0"].output_token_ids == [101]
    assert rbln_model_runner.requests["req_1"].output_token_ids == [202]


def test_bookkeeping_sync_converts_logprobs_tensors(rbln_model_runner, dist_init):
    """Non-spec sampling should convert sampler logprobs tensors via tolists()
    and propagate the converted lists to ModelRunnerOutput."""
    req_id = "req_0"

    rbln_model_runner._update_states(_schedule_new_request(req_id))

    ib = rbln_model_runner.input_batch
    ib.num_tokens_no_spec[0] = 3
    rbln_model_runner.discard_request_mask[0] = False

    class FakeLogprobsTensors:
        def __init__(self):
            self.calls = 0

        def tolists(self):
            self.calls += 1
            return {
                "token_ids": [[[101, 102]]],
                "logprobs": [[[-0.1, -0.2]]],
                "ranks": [[[1, 2]]],
            }

    fake_logprobs_tensors = FakeLogprobsTensors()

    sampler_output = _sampler_output(
        [[101]],
        device=rbln_model_runner.device,
        logprobs_tensors=fake_logprobs_tensors,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[0],
        ),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    hidden_states = torch.empty(
        (1, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    (
        num_nans_in_logits,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
    ) = rbln_model_runner._bookkeeping_sync(
        scheduler_output=scheduler_output,
        sampler_output=sampler_output,
        logits=None,
        hidden_states=hidden_states,
        num_scheduled_tokens=1,
    )

    assert num_nans_in_logits == {}
    assert fake_logprobs_tensors.calls == 1
    assert logprobs_lists == {
        "token_ids": [[[101, 102]]],
        "logprobs": [[[-0.1, -0.2]]],
        "ranks": [[[1, 2]]],
    }
    assert valid_sampled_token_ids == [[101]]
    assert prompt_logprobs_dict == {}
    assert req_ids_output_copy == [req_id]
    assert req_id_to_index_output_copy == {req_id: 0}

    assert ib.token_ids_cpu[0, 3].item() == 101
    assert ib.is_token_ids[0, 3].item() is True
    assert ib.num_tokens_no_spec[0].item() == 4
    assert rbln_model_runner.requests[req_id].output_token_ids == [101]


def test_bookkeeping_sync_discards_chunked_prefill_samples(
    rbln_model_runner, dist_init
):
    """Requests marked by discard_request_mask must not expose or cache sampled
    tokens, even if the sampler returned a token for that row."""
    req_ids = ("req_0", "req_1")

    rbln_model_runner._update_states(_schedule_new_request(*req_ids))

    ib = rbln_model_runner.input_batch
    assert ib.req_id_to_index == {"req_0": 0, "req_1": 1}

    # Default prompt is [1, 2, 3]. Sampling would append at position 3.
    ib.num_tokens_no_spec[:2] = [3, 3]

    # req_0 is an intermediate chunked prefill row and must discard its sample.
    # req_1 is a normal row and should cache its sample.
    rbln_model_runner.discard_request_mask[:2] = torch.tensor([True, False])

    # Poison the append slot so we can prove req_0 was not overwritten.
    ib.token_ids_cpu[0, 3] = 9001
    ib.token_ids_cpu[1, 3] = 9002
    ib.is_token_ids[0, 3] = False
    ib.is_token_ids[1, 3] = False

    sampler_output = _sampler_output(
        [[101], [202]],
        device=rbln_model_runner.device,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 1, "req_1": 1},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    hidden_states = torch.empty(
        (2, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    (
        num_nans_in_logits,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
    ) = rbln_model_runner._bookkeeping_sync(
        scheduler_output=scheduler_output,
        sampler_output=sampler_output,
        logits=None,
        hidden_states=hidden_states,
        num_scheduled_tokens=2,
    )

    assert num_nans_in_logits == {}
    assert logprobs_lists is None
    assert prompt_logprobs_dict == {}

    assert valid_sampled_token_ids == [[], [202]]
    assert req_ids_output_copy == ["req_0", "req_1"]
    assert req_id_to_index_output_copy == {"req_0": 0, "req_1": 1}

    # Discarded row: sampler returned 101, but it must not be cached.
    assert ib.token_ids_cpu[0, 3].item() == 9001
    assert ib.is_token_ids[0, 3].item() is False
    assert ib.num_tokens_no_spec[0].item() == 3
    assert rbln_model_runner.requests["req_0"].output_token_ids == []

    # Normal row: sampler returned 202 and it is cached.
    assert ib.token_ids_cpu[1, 3].item() == 202
    assert ib.is_token_ids[1, 3].item() is True
    assert ib.num_tokens_no_spec[1].item() == 4
    assert rbln_model_runner.requests["req_1"].output_token_ids == [202]


def test_bookkeeping_sync_parses_spec_decode_output(rbln_model_runner, dist_init):
    """Speculative sampler output is parsed into accepted tokens and only those
    accepted tokens are cached in the request state."""
    req_ids = ("req_0", "req_1")

    rbln_model_runner._update_states(_schedule_new_request(*req_ids))

    ib = rbln_model_runner.input_batch
    assert ib.req_id_to_index == {"req_0": 0, "req_1": 1}

    # Default prompt is [1, 2, 3]. Accepted spec tokens should be appended from
    # position 3 for each request.
    ib.num_tokens_no_spec[:2] = [3, 3]
    rbln_model_runner.discard_request_mask[:2] = False

    # Spec decode output format: [batch, max_spec_len + 1].
    # - req_0 accepted two tokens: 101, 102
    # - req_1 accepted one token: 201
    # - -1 means rejected / placeholder and must be filtered out.
    sampler_output = _sampler_output(
        [
            [101, 102, -1, -1],
            [201, -1, -1, -1],
        ],
        device=rbln_model_runner.device,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 3, "req_1": 3},
        total_num_scheduled_tokens=6,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    hidden_states = torch.empty(
        (6, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    (
        num_nans_in_logits,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
    ) = rbln_model_runner._bookkeeping_sync(
        scheduler_output=scheduler_output,
        sampler_output=sampler_output,
        logits=None,
        hidden_states=hidden_states,
        num_scheduled_tokens=6,
    )

    assert num_nans_in_logits == {}
    assert logprobs_lists is None
    assert prompt_logprobs_dict == {}

    # Placeholder -1 entries are removed by RBLNRejectionSampler.parse_output().
    assert valid_sampled_token_ids == [[101, 102], [201]]
    assert req_ids_output_copy == ["req_0", "req_1"]
    assert req_id_to_index_output_copy == {"req_0": 0, "req_1": 1}

    # req_0 accepted two tokens, appended at positions 3 and 4.
    assert ib.token_ids_cpu[0, 3:5].tolist() == [101, 102]
    assert ib.is_token_ids[0, 3:5].tolist() == [True, True]
    assert ib.num_tokens_no_spec[0].item() == 5
    assert rbln_model_runner.requests["req_0"].output_token_ids == [101, 102]

    # req_1 accepted one token, appended at position 3.
    assert ib.token_ids_cpu[1, 3].item() == 201
    assert ib.is_token_ids[1, 3].item() is True
    assert ib.num_tokens_no_spec[1].item() == 4
    assert rbln_model_runner.requests["req_1"].output_token_ids == [201]


def test_get_prompt_logprobs_dict_chunked_and_final(
    rbln_model_runner, dist_init, monkeypatch
):
    """Prompt logprobs should accumulate across chunked prefill steps, return
    only on the final relevant step, and clean in-progress state."""
    req_id = "req_0"
    prompt_token_ids = [10, 11, 12, 13]
    num_prompt_logprobs = 2

    rbln_model_runner._update_states(
        _schedule_new_request(
            req_id,
            prompt_token_ids=[prompt_token_ids],
            sampling_params=[SamplingParams(prompt_logprobs=num_prompt_logprobs)],
        )
    )

    req_state = rbln_model_runner.requests[req_id]
    hidden_size = rbln_model_runner.model_config.get_hidden_size()

    class FakeModel:
        def compute_logits(self, hidden_states):
            return torch.zeros(
                (hidden_states.shape[0], 32),
                dtype=torch.float32,
                device=hidden_states.device,
            )

    class FakeSampler:
        def compute_logprobs(self, logits):
            return logits

        def gather_logprobs(self, logprobs, num_logprobs, tgt_token_ids):
            tgt = tgt_token_ids.to(torch.int32)
            token_ids = torch.stack([tgt, tgt + 100, tgt + 200], dim=1)
            logprobs = torch.zeros_like(token_ids, dtype=torch.float32)
            ranks = torch.arange(
                1,
                tgt.numel() + 1,
                dtype=torch.int32,
                device=tgt.device,
            )
            return token_ids, logprobs, ranks, None

    monkeypatch.setattr(rbln_model_runner, "model", FakeModel(), raising=False)
    monkeypatch.setattr(rbln_model_runner, "sampler", FakeSampler())

    rbln_model_runner.query_start_loc[0] = 0

    # First chunk: create and keep in-progress prompt logprobs.
    req_state.num_computed_tokens = 0
    hidden_states = torch.empty(
        (2, hidden_size),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    out = rbln_model_runner._get_prompt_logprobs_dict(
        hidden_states=hidden_states,
        num_scheduled_tokens={req_id: 2},
    )

    assert out == {}
    assert req_state.in_progress_prompt_logprobs_cpu is not None
    assert req_state.in_progress_prompt_logprobs_cpu.logprob_token_ids[:2].tolist() == [
        [11, 111, 211],
        [12, 112, 212],
    ]

    # Final step: return accumulated logprobs and clear request-local in-progress state.
    req_state.num_computed_tokens = 2
    hidden_states = torch.empty(
        (1, hidden_size),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    out = rbln_model_runner._get_prompt_logprobs_dict(
        hidden_states=hidden_states,
        num_scheduled_tokens={req_id: 2},
    )

    assert list(out) == [req_id]
    final_logprobs = out[req_id]
    assert final_logprobs is not None
    assert final_logprobs.logprob_token_ids.tolist() == [
        [11, 111, 211],
        [12, 112, 212],
        [13, 113, 213],
    ]

    assert req_id not in rbln_model_runner.num_prompt_logprobs
    assert req_state.in_progress_prompt_logprobs_cpu is None


def test_get_nans_in_logits_when_enabled(rbln_model_runner, dist_init, monkeypatch):
    """When VLLM_COMPUTE_NANS_IN_LOGITS is enabled, bookkeeping should report
    per-request NaN counts without affecting sampled token caching."""
    req_ids = ("req_0", "req_1")

    rbln_model_runner._update_states(_schedule_new_request(*req_ids))

    ib = rbln_model_runner.input_batch
    ib.num_tokens_no_spec[:2] = [3, 3]
    rbln_model_runner.discard_request_mask[:2] = False

    monkeypatch.setattr(
        rbln_model_runner_module.envs,
        "VLLM_COMPUTE_NANS_IN_LOGITS",
        True,
    )

    # _get_nans_in_logits currently calls .numpy(), so keep logits on CPU.
    logits = torch.tensor(
        [
            [0.0, float("nan"), 1.0, float("nan")],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    sampler_output = _sampler_output(
        [[101], [202]],
        device=rbln_model_runner.device,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(req_ids),
            resumed_req_ids=set(),
            new_token_ids=[[], []],
            all_token_ids={},
            new_block_ids=[None, None],
            num_computed_tokens=[3, 3],
            num_output_tokens=[0, 0],
        ),
        num_scheduled_tokens={"req_0": 1, "req_1": 1},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    hidden_states = torch.empty(
        (2, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    (
        num_nans_in_logits,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
    ) = rbln_model_runner._bookkeeping_sync(
        scheduler_output=scheduler_output,
        sampler_output=sampler_output,
        logits=logits,
        hidden_states=hidden_states,
        num_scheduled_tokens=2,
    )

    assert num_nans_in_logits == {
        "req_0": 2,
        "req_1": 0,
    }
    assert logprobs_lists is None
    assert valid_sampled_token_ids == [[101], [202]]
    assert prompt_logprobs_dict == {}
    assert req_ids_output_copy == ["req_0", "req_1"]
    assert req_id_to_index_output_copy == {"req_0": 0, "req_1": 1}

    # NaN diagnostics should not affect normal token caching.
    assert ib.token_ids_cpu[0, 3].item() == 101
    assert ib.token_ids_cpu[1, 3].item() == 202
    assert rbln_model_runner.requests["req_0"].output_token_ids == [101]
    assert rbln_model_runner.requests["req_1"].output_token_ids == [202]


# ============================================================================
# Tests for Execute Model State Flow
# ============================================================================
def test_execute_model_empty_scheduler_output_returns_empty_output(
    rbln_model_runner, dist_init, monkeypatch
):
    """A scheduler output with zero scheduled tokens still updates state, then
    returns EMPTY_MODEL_RUNNER_OUTPUT without running the model."""
    req_id = "req_0"

    # Seed an existing request so the zero-token scheduler output can still
    # exercise _update_states() through the finished_req_ids path.
    rbln_model_runner._update_states(_schedule_new_request(req_id))
    assert _is_req_added(rbln_model_runner, req_id)
    assert _is_req_scheduled(rbln_model_runner, req_id)

    monkeypatch.setattr(
        rbln_model_runner,
        "model_executable",
        _unexpected_call("model_executable must not be called"),
        raising=False,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    output = rbln_model_runner.execute_model(scheduler_output)

    assert output is rbln_model_runner_module.EMPTY_MODEL_RUNNER_OUTPUT
    assert rbln_model_runner.execute_model_state is None

    # _update_states() still ran before the early return.
    assert not _is_req_added(rbln_model_runner, req_id)
    assert not _is_req_scheduled(rbln_model_runner, req_id)


def test_execute_model_requires_sample_tokens_before_next_execute(rbln_model_runner):
    """execute_model must reject a second call while execute_model_state is still
    pending, enforcing the execute_model -> sample_tokens lifecycle."""
    pending_state = object()
    rbln_model_runner.execute_model_state = pending_state

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    with pytest.raises(
        RuntimeError,
        match=r"sample_tokens\(\) must be called",
    ):
        rbln_model_runner.execute_model(scheduler_output)

    # The guard fails before touching the pending state.
    assert rbln_model_runner.execute_model_state is pending_state


def test_sample_tokens_clears_execute_model_state_and_returns_output(
    rbln_model_runner, monkeypatch
):
    """sample_tokens should consume execute_model_state, run sampling and
    bookkeeping exactly once, and return a populated ModelRunnerOutput."""
    req_id = "req_0"

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits = torch.randn(
        (1, 16),
        dtype=torch.float32,
        device=rbln_model_runner.device,
    )
    hidden_states = torch.randn(
        (1, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    _set_execute_model_state(
        rbln_model_runner,
        scheduler_output=scheduler_output,
        logits=logits,
        hidden_states=hidden_states,
    )

    sampler_output = _sampler_output([[101]], device=rbln_model_runner.device)

    calls = {}

    def fake_sample(sample_logits, spec_decode_metadata):
        calls["sample_logits"] = sample_logits
        calls["sample_spec_decode_metadata"] = spec_decode_metadata
        return sampler_output

    def fake_bookkeeping_sync(
        bookkeeping_scheduler_output,
        bookkeeping_sampler_output,
        bookkeeping_logits,
        bookkeeping_hidden_states,
        num_scheduled_tokens,
    ):
        calls["bookkeeping_scheduler_output"] = bookkeeping_scheduler_output
        calls["bookkeeping_sampler_output"] = bookkeeping_sampler_output
        calls["bookkeeping_logits"] = bookkeeping_logits
        calls["bookkeeping_hidden_states"] = bookkeeping_hidden_states
        calls["num_scheduled_tokens"] = num_scheduled_tokens

        return _bookkeeping_return(
            num_nans_in_logits={"req_0": 0},
            logprobs_lists="logprobs-lists",
            valid_sampled_token_ids=[[101]],
            prompt_logprobs_dict={"req_0": None},
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
        )

    monkeypatch.setattr(rbln_model_runner, "_sample", fake_sample)
    monkeypatch.setattr(
        rbln_model_runner,
        "_bookkeeping_sync",
        fake_bookkeeping_sync,
    )

    output = rbln_model_runner.sample_tokens(grammar_output=None)

    assert rbln_model_runner.execute_model_state is None

    assert calls["sample_logits"] is logits
    assert calls["sample_spec_decode_metadata"] is None

    assert calls["bookkeeping_scheduler_output"] is scheduler_output
    assert calls["bookkeeping_sampler_output"] is sampler_output
    assert calls["bookkeeping_logits"] is logits
    assert calls["bookkeeping_hidden_states"] is hidden_states
    assert calls["num_scheduled_tokens"] == 1

    assert output.req_ids == [req_id]
    assert output.req_id_to_index == {req_id: 0}
    assert output.sampled_token_ids == [[101]]
    assert output.logprobs == "logprobs-lists"
    assert output.prompt_logprobs_dict == {"req_0": None}
    assert output.num_nans_in_logits == {"req_0": 0}


def test_execute_model_prefill_passes_token_indices_and_stores_execute_state(
    rbln_model_runner, dist_init, monkeypatch
):
    """Prefill execution should pass last-token indices and store pending state
    for sample_tokens()."""
    req_id = "req_0"
    scheduler_output = _schedule_new_request(
        req_id,
        prompt_token_ids=[[10, 11, 12]],
        num_computed_tokens=[0],
        num_scheduled_tokens=[3],
    )

    attn_metadata_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "_build_attention_metadata",
        _recording_attention_metadata_builder(attn_metadata_calls),
    )

    hidden_size = rbln_model_runner.model_config.get_hidden_size()
    vocab_size = 16
    hidden_states = torch.randn(
        (1, 1, hidden_size),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    logits = torch.randn(
        (1, vocab_size),
        dtype=torch.float32,
        device=rbln_model_runner.device,
    )

    model_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "model_executable",
        _recording_model_executable(model_calls, hidden_states, logits),
        raising=False,
    )

    output = rbln_model_runner.execute_model(scheduler_output)

    assert output is None

    # Prefill runs the real input preparation path.
    assert rbln_model_runner.is_prefill is True
    assert attn_metadata_calls["num_tokens"] == 3
    assert attn_metadata_calls["num_reqs"] == 1
    assert attn_metadata_calls["max_query_len"] == 3
    assert attn_metadata_calls["use_spec_decode"] is False
    assert attn_metadata_calls["logits_indices"].tolist() == [2]

    # token_indices should point at the last token of the prefill prompt.
    assert model_calls["token_indices"].tolist() == [2]

    # _preprocess pads prefill inputs to max_num_tokens for RBLN compilation.
    assert model_calls["input_ids"].shape == (1, rbln_model_runner.max_num_tokens)
    assert model_calls["positions"].shape == (1, rbln_model_runner.max_num_tokens)
    assert model_calls["input_ids"][0, :3].tolist() == [10, 11, 12]
    assert model_calls["positions"][0, :3].tolist() == [0, 1, 2]
    assert model_calls["intermediate_tensors"] is None
    assert model_calls["inputs_embeds"] is None

    state = rbln_model_runner.execute_model_state
    assert state is not None
    assert state.scheduler_output is scheduler_output
    assert state.logits is logits
    assert state.spec_decode_metadata is None
    assert state.spec_decode_common_attn_metadata is None
    assert state.hidden_states is hidden_states
    assert state.sample_hidden_states is hidden_states
    assert state.aux_hidden_states is None


def test_execute_model_decode_slices_logits_by_logits_indices(
    rbln_model_runner, dist_init, monkeypatch
):
    """Decode execution should store only logits selected by runner-owned
    logits_indices before sampling."""
    req_ids = ("req_0", "req_1")
    rbln_model_runner._update_states(
        _schedule_new_request(
            *req_ids,
            prompt_token_ids=[[10, 11, 12], [20, 21, 22]],
            block_ids=[([0],), ([1],)],
            num_computed_tokens=[3, 3],
            num_scheduled_tokens=[3, 3],
        )
    )

    ib = rbln_model_runner.input_batch
    # Decode reads from num_computed_tokens_cpu. After a prompt of length 3 has
    # been prefetched, the next decode input is the previously sampled token at
    # position 3, not the prompt's last token at position 2.
    ib.token_ids_cpu[0, 3] = 101
    ib.token_ids_cpu[1, 3] = 202
    ib.is_token_ids[0, 3] = True
    ib.is_token_ids[1, 3] = True
    ib.num_computed_tokens_cpu[:2] = [3, 3]
    ib.num_tokens_no_spec[:2] = [4, 4]
    rbln_model_runner.requests["req_0"].output_token_ids = [101]
    rbln_model_runner.requests["req_1"].output_token_ids = [202]
    assert rbln_model_runner.is_prefill is False

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(req_ids),
            resumed_req_ids=set(),
            new_token_ids=[[], []],
            all_token_ids={},
            new_block_ids=[None, None],
            num_computed_tokens=[3, 3],
            num_output_tokens=[1, 1],
        ),
        num_scheduled_tokens={"req_0": 1, "req_1": 1},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    attn_metadata_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "_build_attention_metadata",
        _recording_attention_metadata_builder(attn_metadata_calls),
    )

    hidden_size = rbln_model_runner.model_config.get_hidden_size()
    vocab_size = 8

    # Decode batch is padded by bucketing. Return logits for padded rows so
    # execute_model must select only real rows via logits_indices.
    padded_reqs = rbln_model_runner.bucketing_manager.find_decode_batch_bucket(2)
    hidden_states = torch.randn(
        (padded_reqs, 1, hidden_size),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    logits = torch.arange(
        padded_reqs * vocab_size,
        dtype=torch.float32,
        device=rbln_model_runner.device,
    ).view(padded_reqs, vocab_size)

    model_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "model_executable",
        _recording_model_executable(model_calls, hidden_states, logits),
        raising=False,
    )

    output = rbln_model_runner.execute_model(scheduler_output)

    assert output is None
    assert rbln_model_runner.is_prefill is False

    # Decode path uses one query token per request and does not pass token_indices
    # to wrapped compute_logits.
    assert attn_metadata_calls["num_tokens"] == 2
    assert attn_metadata_calls["num_reqs"] == 2
    assert attn_metadata_calls["max_query_len"] == 1
    assert attn_metadata_calls["use_spec_decode"] is False
    assert attn_metadata_calls["logits_indices"].tolist() == [0, 1]

    assert model_calls["token_indices"] is None
    assert model_calls["input_ids"].shape == (padded_reqs, 1)
    assert model_calls["positions"].shape == (padded_reqs, 1)
    assert model_calls["input_ids"][:2, 0].tolist() == [101, 202]
    assert model_calls["positions"][:2, 0].tolist() == [3, 3]

    state = rbln_model_runner.execute_model_state
    assert state is not None
    assert state.scheduler_output is scheduler_output
    assert state.spec_decode_metadata is None
    assert state.spec_decode_common_attn_metadata is None
    assert state.hidden_states is hidden_states
    assert state.sample_hidden_states is hidden_states
    assert state.aux_hidden_states is None

    # Critical contract: padded logits are sliced by logits_indices before
    # sample_tokens() sees them.
    expected_logits = logits[attn_metadata_calls["logits_indices"]]
    assert torch.equal(state.logits, expected_logits)
    assert state.logits.shape == (2, vocab_size)


def test_execute_model_spec_decode_stores_spec_metadata_and_common_attention_metadata(
    rbln_model_runner, dist_init, monkeypatch
):
    """Speculative decode execution should store both rejection-sampling
    metadata and common attention metadata for sample_tokens()."""
    req_id = "req_0"

    rbln_model_runner._update_states(
        _schedule_new_request(
            req_id,
            prompt_token_ids=[[10, 11, 12]],
            block_ids=[([0],)],
            num_computed_tokens=[3],
            num_scheduled_tokens=[3],
        )
    )

    ib = rbln_model_runner.input_batch

    # Spec decode starts from the already-prefilled decode position. The query
    # contains one real next token plus kept draft tokens.
    ib.token_ids_cpu[0, 3:6] = [101, 102, 103]
    ib.is_token_ids[0, 3:6] = True
    ib.num_computed_tokens_cpu[0] = 3
    ib.num_tokens_no_spec[0] = 4
    rbln_model_runner.requests[req_id].output_token_ids = [101]
    assert rbln_model_runner.is_prefill is False

    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 2)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={req_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={req_id: [102, 103]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    common_attn_metadata = SimpleNamespace(max_seq_len=6)
    attn_metadata_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "_build_attention_metadata",
        _recording_attention_metadata_builder(
            attn_metadata_calls,
            common_attn_metadata,
        ),
    )

    hidden_size = rbln_model_runner.model_config.get_hidden_size()
    vocab_size = 8
    hidden_states = torch.randn(
        (3, 1, hidden_size),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    logits = torch.arange(
        3 * vocab_size,
        dtype=torch.float32,
        device=rbln_model_runner.device,
    ).view(3, vocab_size)

    model_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "model_executable",
        _recording_model_executable(model_calls, hidden_states, logits),
        raising=False,
    )

    output = rbln_model_runner.execute_model(scheduler_output)

    assert output is None
    assert rbln_model_runner.is_prefill is False

    assert attn_metadata_calls["num_tokens"] == 3
    assert attn_metadata_calls["num_reqs"] == 1
    assert attn_metadata_calls["max_query_len"] == 3
    assert attn_metadata_calls["use_spec_decode"] is True

    spec_md = rbln_model_runner.execute_model_state.spec_decode_metadata
    assert spec_md is not None
    assert spec_md.num_draft_tokens == [2]
    assert spec_md.cu_num_draft_tokens.tolist() == [2]
    assert spec_md.cu_num_sampled_tokens.tolist() == [3]
    assert spec_md.logits_indices.tolist() == [0, 1, 2]
    assert spec_md.target_logits_indices.tolist() == [0, 1]
    assert spec_md.bonus_logits_indices.tolist() == [2]
    assert spec_md.draft_token_ids.tolist() == [102, 103]

    assert torch.equal(attn_metadata_calls["logits_indices"], spec_md.logits_indices)

    padded_reqs = rbln_model_runner.bucketing_manager.find_decode_batch_bucket(1)

    assert model_calls["token_indices"] is None
    assert model_calls["input_ids"].shape == (padded_reqs, 3)
    assert model_calls["positions"].shape == (padded_reqs, 3)
    assert model_calls["input_ids"][0, :3].tolist() == [101, 102, 103]
    assert model_calls["positions"][0, :3].tolist() == [3, 4, 5]
    assert model_calls["intermediate_tensors"] is None
    assert model_calls["inputs_embeds"] is None

    state = rbln_model_runner.execute_model_state
    assert state is not None
    assert state.scheduler_output is scheduler_output
    assert state.spec_decode_metadata is spec_md
    assert state.spec_decode_common_attn_metadata is common_attn_metadata
    assert state.hidden_states is hidden_states
    assert state.sample_hidden_states is hidden_states
    assert state.aux_hidden_states is None

    expected_logits = logits[spec_md.logits_indices]
    assert torch.equal(state.logits, expected_logits)
    assert state.logits.shape == (3, vocab_size)


def test_sample_tokens_proposes_non_eagle_drafts_after_bookkeeping(
    rbln_model_runner, monkeypatch
):
    """Non-EAGLE speculative decoding should propose draft tokens from accepted
    tokens after bookkeeping."""
    req_id = "req_0"

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits = torch.randn(
        (1, 8),
        dtype=torch.float32,
        device=rbln_model_runner.device,
    )
    hidden_states = torch.randn(
        (1, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    common_attn_metadata = SimpleNamespace(max_seq_len=4)

    _set_execute_model_state(
        rbln_model_runner,
        scheduler_output=scheduler_output,
        logits=logits,
        hidden_states=hidden_states,
        common_attn_metadata=common_attn_metadata,
    )

    monkeypatch.setattr(
        rbln_model_runner,
        "speculative_config",
        FakeSpecConfig(method="ngram", num_speculative_tokens=3),
    )
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 3)
    monkeypatch.setattr(
        rbln_model_runner, "effective_drafter_max_model_len", 16, raising=False
    )

    sampler_output = _sampler_output([[101]], device=rbln_model_runner.device)

    calls = []
    bookkeeping_valid_sampled_token_ids = [[101]]
    propose_args = {}

    def fake_sample(sample_logits, spec_decode_metadata):
        calls.append("sample")
        assert sample_logits is logits
        assert spec_decode_metadata is None
        return sampler_output

    def fake_bookkeeping_sync(
        bookkeeping_scheduler_output,
        bookkeeping_sampler_output,
        bookkeeping_logits,
        bookkeeping_hidden_states,
        num_scheduled_tokens,
    ):
        calls.append("bookkeeping")
        assert bookkeeping_scheduler_output is scheduler_output
        assert bookkeeping_sampler_output is sampler_output
        assert bookkeeping_logits is logits
        assert bookkeeping_hidden_states is hidden_states
        assert num_scheduled_tokens == 1

        return _bookkeeping_return(
            valid_sampled_token_ids=bookkeeping_valid_sampled_token_ids,
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
        )

    def fake_propose_draft_token_ids(
        got_scheduler_output,
        got_sampled_token_ids,
        got_sampling_metadata,
        got_hidden_states,
        got_sample_hidden_states,
        got_aux_hidden_states,
        got_spec_decode_metadata,
        got_common_attn_metadata,
    ):
        calls.append("propose")
        propose_args["scheduler_output"] = got_scheduler_output
        propose_args["sampled_token_ids"] = got_sampled_token_ids
        propose_args["sampling_metadata"] = got_sampling_metadata
        propose_args["hidden_states"] = got_hidden_states
        propose_args["sample_hidden_states"] = got_sample_hidden_states
        propose_args["aux_hidden_states"] = got_aux_hidden_states
        propose_args["spec_decode_metadata"] = got_spec_decode_metadata
        propose_args["common_attn_metadata"] = got_common_attn_metadata
        return [[201, 202, 203]]

    monkeypatch.setattr(rbln_model_runner, "_sample", fake_sample)
    monkeypatch.setattr(rbln_model_runner, "_bookkeeping_sync", fake_bookkeeping_sync)
    monkeypatch.setattr(
        rbln_model_runner,
        "propose_draft_token_ids",
        fake_propose_draft_token_ids,
    )

    output = rbln_model_runner.sample_tokens(grammar_output=None)

    assert calls == ["sample", "bookkeeping", "propose"]

    # Non-EAGLE draft proposal must use accepted tokens from bookkeeping, not
    # raw sampler_output.sampled_token_ids.
    assert propose_args["scheduler_output"] is scheduler_output
    assert propose_args["sampled_token_ids"] is bookkeeping_valid_sampled_token_ids
    assert (
        propose_args["sampling_metadata"]
        is rbln_model_runner.input_batch.sampling_metadata
    )
    assert propose_args["hidden_states"] is hidden_states
    assert propose_args["sample_hidden_states"] is hidden_states
    assert propose_args["aux_hidden_states"] is None
    assert propose_args["spec_decode_metadata"] is None
    assert propose_args["common_attn_metadata"] is common_attn_metadata

    assert rbln_model_runner._draft_token_ids == [[201, 202, 203]]
    assert rbln_model_runner.execute_model_state is None

    assert output.req_ids == [req_id]
    assert output.req_id_to_index == {req_id: 0}
    assert output.sampled_token_ids == [[101]]
    assert output.logprobs is None
    assert output.prompt_logprobs_dict == {}
    assert output.num_nans_in_logits == {}


def test_sample_tokens_proposes_eagle_drafts_before_bookkeeping(
    rbln_model_runner, monkeypatch
):
    """EAGLE speculative decoding should propose draft tokens from raw sampler
    output before bookkeeping parses accepted tokens."""
    req_id = "req_0"

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits = torch.randn(
        (1, 8),
        dtype=torch.float32,
        device=rbln_model_runner.device,
    )
    hidden_states = torch.randn(
        (1, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    spec_decode_metadata = SimpleNamespace(name="spec-md")
    common_attn_metadata = SimpleNamespace(max_seq_len=4)

    _set_execute_model_state(
        rbln_model_runner,
        scheduler_output=scheduler_output,
        logits=logits,
        spec_decode_metadata=spec_decode_metadata,
        hidden_states=hidden_states,
        common_attn_metadata=common_attn_metadata,
    )

    monkeypatch.setattr(
        rbln_model_runner,
        "speculative_config",
        FakeSpecConfig(method="eagle", num_speculative_tokens=3),
    )
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 3)
    monkeypatch.setattr(
        rbln_model_runner,
        "effective_drafter_max_model_len",
        16,
        raising=False,
    )

    raw_sampled_token_ids = torch.tensor(
        [[101]],
        dtype=torch.int32,
        device=rbln_model_runner.device,
    )
    sampler_output = _sampler_output(raw_sampled_token_ids)

    calls = []
    propose_args = {}

    def fake_sample(sample_logits, sample_spec_decode_metadata):
        calls.append("sample")
        assert sample_logits is logits
        assert sample_spec_decode_metadata is spec_decode_metadata
        return sampler_output

    def fake_propose_draft_token_ids(
        got_scheduler_output,
        got_sampled_token_ids,
        got_sampling_metadata,
        got_hidden_states,
        got_sample_hidden_states,
        got_aux_hidden_states,
        got_spec_decode_metadata,
        got_common_attn_metadata,
    ):
        calls.append("propose")
        propose_args["scheduler_output"] = got_scheduler_output
        propose_args["sampled_token_ids"] = got_sampled_token_ids
        propose_args["sampling_metadata"] = got_sampling_metadata
        propose_args["hidden_states"] = got_hidden_states
        propose_args["sample_hidden_states"] = got_sample_hidden_states
        propose_args["aux_hidden_states"] = got_aux_hidden_states
        propose_args["spec_decode_metadata"] = got_spec_decode_metadata
        propose_args["common_attn_metadata"] = got_common_attn_metadata
        return torch.tensor(
            [[201, 202, 203]],
            dtype=torch.int32,
            device=rbln_model_runner.device,
        )

    def fake_bookkeeping_sync(
        bookkeeping_scheduler_output,
        bookkeeping_sampler_output,
        bookkeeping_logits,
        bookkeeping_hidden_states,
        num_scheduled_tokens,
    ):
        calls.append("bookkeeping")
        assert bookkeeping_scheduler_output is scheduler_output
        assert bookkeeping_sampler_output is sampler_output
        assert bookkeeping_logits is logits
        assert bookkeeping_hidden_states is hidden_states
        assert num_scheduled_tokens == 1

        return _bookkeeping_return(
            valid_sampled_token_ids=[[101]],
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
        )

    monkeypatch.setattr(rbln_model_runner, "_sample", fake_sample)
    monkeypatch.setattr(
        rbln_model_runner,
        "propose_draft_token_ids",
        fake_propose_draft_token_ids,
    )
    monkeypatch.setattr(rbln_model_runner, "_bookkeeping_sync", fake_bookkeeping_sync)

    output = rbln_model_runner.sample_tokens(grammar_output=None)

    assert calls == ["sample", "propose", "bookkeeping"]

    # EAGLE must receive raw sampler output before bookkeeping converts/parses it.
    assert propose_args["scheduler_output"] is scheduler_output
    assert propose_args["sampled_token_ids"] is raw_sampled_token_ids
    assert (
        propose_args["sampling_metadata"]
        is rbln_model_runner.input_batch.sampling_metadata
    )
    assert propose_args["hidden_states"] is hidden_states
    assert propose_args["sample_hidden_states"] is hidden_states
    assert propose_args["aux_hidden_states"] is None
    assert propose_args["spec_decode_metadata"] is spec_decode_metadata
    assert propose_args["common_attn_metadata"] is common_attn_metadata

    assert torch.equal(
        rbln_model_runner._draft_token_ids,
        torch.tensor(
            [[201, 202, 203]], dtype=torch.int32, device=rbln_model_runner.device
        ),
    )
    assert rbln_model_runner.execute_model_state is None

    assert output.req_ids == [req_id]
    assert output.req_id_to_index == {req_id: 0}
    assert output.sampled_token_ids == [[101]]
    assert output.logprobs is None
    assert output.prompt_logprobs_dict == {}
    assert output.num_nans_in_logits == {}


def test_sample_tokens_skips_draft_proposal_when_drafter_context_overflows(
    rbln_model_runner, monkeypatch
):
    """Draft proposal should be skipped when the next speculative step would
    exceed the drafter's effective max model length."""
    req_id = "req_0"

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    logits = torch.randn(
        (1, 8),
        dtype=torch.float32,
        device=rbln_model_runner.device,
    )
    hidden_states = torch.randn(
        (1, rbln_model_runner.model_config.get_hidden_size()),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )

    # max_seq_len(10) + num_spec_tokens(3) > effective_drafter_max_model_len(12)
    # so sample_tokens() must not call propose_draft_token_ids().
    common_attn_metadata = SimpleNamespace(max_seq_len=10)

    _set_execute_model_state(
        rbln_model_runner,
        scheduler_output=scheduler_output,
        logits=logits,
        hidden_states=hidden_states,
        common_attn_metadata=common_attn_metadata,
    )

    monkeypatch.setattr(
        rbln_model_runner,
        "speculative_config",
        FakeSpecConfig(method="ngram", num_speculative_tokens=3),
    )
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 3)
    monkeypatch.setattr(
        rbln_model_runner,
        "effective_drafter_max_model_len",
        12,
        raising=False,
    )

    sampler_output = _sampler_output([[101]], device=rbln_model_runner.device)

    calls = []

    def fake_sample(sample_logits, sample_spec_decode_metadata):
        calls.append("sample")
        assert sample_logits is logits
        assert sample_spec_decode_metadata is None
        return sampler_output

    def fake_bookkeeping_sync(
        bookkeeping_scheduler_output,
        bookkeeping_sampler_output,
        bookkeeping_logits,
        bookkeeping_hidden_states,
        num_scheduled_tokens,
    ):
        calls.append("bookkeeping")
        assert bookkeeping_scheduler_output is scheduler_output
        assert bookkeeping_sampler_output is sampler_output
        assert bookkeeping_logits is logits
        assert bookkeeping_hidden_states is hidden_states
        assert num_scheduled_tokens == 1

        return _bookkeeping_return(
            valid_sampled_token_ids=[[101]],
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
        )

    monkeypatch.setattr(rbln_model_runner, "_sample", fake_sample)
    monkeypatch.setattr(rbln_model_runner, "_bookkeeping_sync", fake_bookkeeping_sync)
    monkeypatch.setattr(
        rbln_model_runner,
        "propose_draft_token_ids",
        _unexpected_call("draft proposal must be skipped on drafter overflow"),
    )

    # Seed a stale value to verify sample_tokens() clears it before evaluating
    # whether this step can propose drafts.
    rbln_model_runner._draft_token_ids = [[999]]

    output = rbln_model_runner.sample_tokens(grammar_output=None)

    assert calls == ["sample", "bookkeeping"]
    assert rbln_model_runner._draft_token_ids is None
    assert rbln_model_runner.execute_model_state is None

    assert output.req_ids == [req_id]
    assert output.req_id_to_index == {req_id: 0}
    assert output.sampled_token_ids == [[101]]
    assert output.logprobs is None
    assert output.prompt_logprobs_dict == {}
    assert output.num_nans_in_logits == {}


def test_take_draft_token_ids_returns_current_req_ids_and_draft_ids(
    rbln_model_runner, dist_init, monkeypatch
):
    """Draft-token handoff should return current request ids with the runner's
    pending draft token ids."""
    # No speculative decoding configured -> no draft handoff.
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 0)
    rbln_model_runner._draft_token_ids = [[201, 202]]
    assert rbln_model_runner.take_draft_token_ids() is None

    # Speculative decoding configured, but no active requests -> no handoff.
    monkeypatch.setattr(rbln_model_runner, "num_spec_tokens", 2)
    rbln_model_runner.input_batch.req_ids.clear()
    assert rbln_model_runner.take_draft_token_ids() is None

    # Active requests with tensor draft ids: tensor should be converted to a
    # Python list because DraftTokenIds is consumed by scheduler-side code.
    rbln_model_runner._update_states(_schedule_new_request("req_0", "req_1"))
    rbln_model_runner._draft_token_ids = torch.tensor(
        [
            [201, 202],
            [301, 302],
        ],
        dtype=torch.int32,
        device=rbln_model_runner.device,
    )

    draft_token_ids = rbln_model_runner.take_draft_token_ids()

    assert draft_token_ids is not None
    assert draft_token_ids.req_ids == ["req_0", "req_1"]
    assert draft_token_ids.draft_token_ids == [[201, 202], [301, 302]]

    # Active requests with list draft ids: list should pass through unchanged.
    list_drafts = [[401, 402], [501, 502]]
    rbln_model_runner._draft_token_ids = list_drafts

    draft_token_ids = rbln_model_runner.take_draft_token_ids()

    assert draft_token_ids is not None
    assert draft_token_ids.req_ids == ["req_0", "req_1"]
    assert draft_token_ids.draft_token_ids is list_drafts


# ============================================================================
# Tests for Batch Padding and InputBatch Reinitialization
# ============================================================================
def test_determine_batch_padding_decode_uses_bucket(rbln_model_runner, monkeypatch):
    """Prefill keeps the unpadded request count; decode uses the bucketing
    manager. With data_parallel_size == 1, token padding metadata stays None."""
    ib = rbln_model_runner.input_batch

    fake_bucketing_manager = FakeBucketingManager(default=4)
    monkeypatch.setattr(
        rbln_model_runner,
        "bucketing_manager",
        fake_bucketing_manager,
    )

    assert rbln_model_runner.parallel_config.data_parallel_size == 1

    # Prefill path:
    # is_prefill == num_computed_tokens < num_tokens_no_spec - 1
    ib.num_computed_tokens_cpu[0] = 0
    ib.num_tokens_no_spec[0] = 8
    assert rbln_model_runner.is_prefill is True

    num_reqs_padded, num_tokens_padded, num_tokens_across_dp = (
        rbln_model_runner._determine_batch_padding(
            num_reqs_unpadded=3,
            num_tokens_unpadded=8,
        )
    )

    assert num_reqs_padded == 3
    assert num_tokens_padded is None
    assert num_tokens_across_dp is None
    assert fake_bucketing_manager.calls == []

    # Decode path:
    # boundary condition for decode: computed == num_tokens_no_spec - 1
    ib.num_computed_tokens_cpu[0] = 7
    ib.num_tokens_no_spec[0] = 8
    assert rbln_model_runner.is_prefill is False

    num_reqs_padded, num_tokens_padded, num_tokens_across_dp = (
        rbln_model_runner._determine_batch_padding(
            num_reqs_unpadded=3,
            num_tokens_unpadded=3,
        )
    )

    assert num_reqs_padded == 4
    assert num_tokens_padded is None
    assert num_tokens_across_dp is None
    assert fake_bucketing_manager.calls == [3]


def test_determine_batch_padding_data_parallel_prefill_uses_max_num_tokens(
    rbln_model_runner, monkeypatch
):
    """Data-parallel prefill should use max_num_tokens padding and return
    cross-rank token counts."""
    ib = rbln_model_runner.input_batch

    # Force prefill.
    ib.num_computed_tokens_cpu[0] = 0
    ib.num_tokens_no_spec[0] = 8
    assert rbln_model_runner.is_prefill is True

    monkeypatch.setattr(
        rbln_model_runner.parallel_config,
        "data_parallel_size",
        2,
    )
    monkeypatch.setattr(
        rbln_model_runner.parallel_config,
        "data_parallel_rank",
        0,
    )

    calls = []

    def fake_num_tokens_and_reqs_across_dp(
        num_tokens,
        num_reqs,
        dp_size,
        dp_rank,
        is_prefill,
    ):
        calls.append(
            {
                "num_tokens": num_tokens,
                "num_reqs": num_reqs,
                "dp_size": dp_size,
                "dp_rank": dp_rank,
                "is_prefill": is_prefill,
            }
        )
        return torch.tensor([64, 96], dtype=torch.int32), None

    monkeypatch.setattr(
        rbln_model_runner_module.RBLNDPMetadata,
        "num_tokens_and_reqs_across_dp",
        staticmethod(fake_num_tokens_and_reqs_across_dp),
    )

    monkeypatch.setattr(
        rbln_model_runner,
        "bucketing_manager",
        FakeBucketingManager(fail_message="decode bucket must not be used for prefill"),
    )

    num_reqs_padded, num_tokens_padded, num_tokens_across_dp = (
        rbln_model_runner._determine_batch_padding(
            num_reqs_unpadded=3,
            num_tokens_unpadded=64,
        )
    )

    assert calls == [
        {
            "num_tokens": 64,
            "num_reqs": 3,
            "dp_size": 2,
            "dp_rank": 0,
            "is_prefill": True,
        }
    ]

    assert num_reqs_padded == 3
    assert num_tokens_padded == rbln_model_runner.max_num_tokens

    assert isinstance(num_tokens_across_dp, torch.Tensor)
    assert num_tokens_across_dp.dtype == torch.int32
    assert num_tokens_across_dp.device.type == "cpu"
    assert num_tokens_across_dp.tolist() == [64, 96]


def test_determine_batch_padding_specialized_moe_decode_uses_decode_bucket(
    rbln_model_runner, monkeypatch
):
    """Specialized MoE decode should bucket by request count and pad by the
    request bucket times the per-request query length."""
    ib = rbln_model_runner.input_batch

    # Force decode.
    ib.num_computed_tokens_cpu[0] = 7
    ib.num_tokens_no_spec[0] = 8
    assert rbln_model_runner.is_prefill is False

    monkeypatch.setattr(
        rbln_model_runner.parallel_config,
        "data_parallel_size",
        2,
    )
    monkeypatch.setattr(
        rbln_model_runner.parallel_config,
        "data_parallel_rank",
        0,
    )
    monkeypatch.setattr(
        rbln_model_runner,
        "specialized_moe_decode",
        True,
        raising=False,
    )

    calls = []

    def fake_num_tokens_and_reqs_across_dp(
        num_tokens,
        num_reqs,
        dp_size,
        dp_rank,
        is_prefill,
    ):
        calls.append(
            {
                "num_tokens": num_tokens,
                "num_reqs": num_reqs,
                "dp_size": dp_size,
                "dp_rank": dp_rank,
                "is_prefill": is_prefill,
            }
        )
        # Simulate this rank having 3 requests with 4 query tokens each, while
        # another rank requires bucket selection for max request count == 5.
        return torch.tensor([12, 20], dtype=torch.int32), torch.tensor(
            [3, 5], dtype=torch.int32
        )

    monkeypatch.setattr(
        rbln_model_runner_module.RBLNDPMetadata,
        "num_tokens_and_reqs_across_dp",
        staticmethod(fake_num_tokens_and_reqs_across_dp),
    )

    fake_bucketing_manager = FakeBucketingManager(buckets={3: 4, 5: 8})
    monkeypatch.setattr(
        rbln_model_runner,
        "bucketing_manager",
        fake_bucketing_manager,
    )

    num_reqs_padded, num_tokens_padded, num_tokens_across_dp = (
        rbln_model_runner._determine_batch_padding(
            num_reqs_unpadded=3,
            num_tokens_unpadded=12,
        )
    )

    assert calls == [
        {
            "num_tokens": 12,
            "num_reqs": 3,
            "dp_size": 2,
            "dp_rank": 0,
            "is_prefill": False,
        }
    ]

    # First call: initial decode padding for this rank's unpadded request count.
    # Second call: specialized MoE decode repads using cross-DP max request count.
    assert fake_bucketing_manager.calls == [3, 5]

    assert num_reqs_padded == 8
    assert num_tokens_padded == 32

    assert isinstance(num_tokens_across_dp, torch.Tensor)
    assert num_tokens_across_dp.dtype == torch.int32
    assert num_tokens_across_dp.device.type == "cpu"
    assert num_tokens_across_dp.tolist() == [12, 20]


def test_may_reinitialize_input_batch_rebuilds_on_kernel_block_size_change(
    rbln_model_runner,
):
    """InputBatch is rebuilt only when block_sizes or kernel_block_sizes differ
    from the initialized values; unchanged values keep the same object."""
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=rbln_model_runner.model_config.get_num_kv_heads(
            rbln_model_runner.parallel_config
        ),
        head_size=rbln_model_runner.model_config.get_head_size(),
        dtype=rbln_model_runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec),
        ],
    )

    original_input_batch = rbln_model_runner.input_batch
    assert rbln_model_runner._init_block_sizes == [BLOCK_SIZE]
    assert rbln_model_runner._init_kernel_block_sizes == [BLOCK_SIZE]

    # Same block/kernel sizes: no rebuild.
    rbln_model_runner.may_reinitialize_input_batch(
        kv_cache_config,
        kernel_block_sizes=[BLOCK_SIZE],
    )
    assert rbln_model_runner.input_batch is original_input_batch
    assert rbln_model_runner._init_block_sizes == [BLOCK_SIZE]
    assert rbln_model_runner._init_kernel_block_sizes == [BLOCK_SIZE]

    # Different kernel block size: rebuild.
    rbln_model_runner.may_reinitialize_input_batch(
        kv_cache_config,
        kernel_block_sizes=[BLOCK_SIZE // 2],
    )

    rebuilt_input_batch = rbln_model_runner.input_batch
    assert rebuilt_input_batch is not original_input_batch
    assert rbln_model_runner._init_block_sizes == [BLOCK_SIZE]
    assert rbln_model_runner._init_kernel_block_sizes == [BLOCK_SIZE // 2]

    # Repeating with the same updated sizes keeps the rebuilt object.
    rbln_model_runner.may_reinitialize_input_batch(
        kv_cache_config,
        kernel_block_sizes=[BLOCK_SIZE // 2],
    )
    assert rbln_model_runner.input_batch is rebuilt_input_batch


# ============================================================================
# Tests for KV Cache Initialization and Metadata
# ============================================================================
def test_reshape_kv_cache_tensors_respects_backend_stride_order(rbln_model_runner):
    """KV cache reshape should derive num_blocks from page_size_bytes, apply the
    backend stride order, and keep base/view metadata consistent."""
    layer_name = "layer.0"
    num_kv_heads = 2
    head_size = 4
    block_size = 16
    kernel_block_size = 8
    num_blocks = 2

    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )

    class FakeBackend:
        calls: list[Any] = []

        @staticmethod
        def get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size_arg,
            num_kv_heads_arg,
            head_size_arg,
            cache_dtype_str,
        ):
            FakeBackend.calls.append(
                (
                    kernel_num_blocks,
                    kernel_block_size_arg,
                    num_kv_heads_arg,
                    head_size_arg,
                    cache_dtype_str,
                )
            )
            # Generic semantic shape before applying backend stride order:
            # [K/V, kernel blocks, heads, block, head_size].
            return (
                2,
                kernel_num_blocks,
                num_kv_heads_arg,
                kernel_block_size_arg,
                head_size_arg,
            )

        @staticmethod
        def get_kv_cache_stride_order():
            # Backend wants kernel blocks as the outermost allocation dimension.
            return (1, 0, 2, 3, 4)

    fake_group = SimpleNamespace(
        kv_cache_spec=kv_cache_spec,
        backend=FakeBackend,
        kv_cache_group_id=0,
        layer_names=[layer_name],
    )

    rbln_model_runner.attn_groups = [[fake_group]]
    rbln_model_runner.runner_only_attn_layers.clear()

    raw_tensor = torch.empty(
        kv_cache_spec.page_size_bytes * num_blocks,
        dtype=torch.int8,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=raw_tensor.numel(), shared_by=[layer_name]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=kv_cache_spec),
        ],
    )

    kv_caches, kv_cache_bases, kv_cache_view_infos = (
        rbln_model_runner._reshape_kv_cache_tensors(
            kv_cache_config=kv_cache_config,
            kv_cache_raw_tensors={layer_name: raw_tensor},
            kernel_block_sizes=[kernel_block_size],
        )
    )

    # num_blocks is derived from raw bytes / page_size_bytes.
    # kernel_num_blocks additionally expands by block_size / kernel_block_size.
    expected_kernel_num_blocks = num_blocks * (block_size // kernel_block_size)
    assert FakeBackend.calls == [
        (
            expected_kernel_num_blocks,
            kernel_block_size,
            num_kv_heads,
            head_size,
            rbln_model_runner.cache_config.cache_dtype,
        )
    ]

    # FakeBackend returned semantic shape:
    #   (2, 4, 2, 8, 4)
    # stride_order=(1,0,2,3,4) means the allocated base shape becomes:
    #   (4, 2, 2, 8, 4)
    expected_base_shape = (
        expected_kernel_num_blocks,
        2,
        num_kv_heads,
        kernel_block_size,
        head_size,
    )
    expected_kv_cache_shape = (
        2,
        expected_kernel_num_blocks,
        num_kv_heads,
        kernel_block_size,
        head_size,
    )

    assert kv_cache_bases[layer_name].shape == expected_base_shape
    assert kv_caches[layer_name].shape == expected_kv_cache_shape

    # The returned cache is a view sharing storage with the backend-native base.
    assert (
        kv_caches[layer_name].untyped_storage().data_ptr()
        == kv_cache_bases[layer_name].untyped_storage().data_ptr()
    )
    assert (
        kv_caches[layer_name].storage_offset()
        == kv_cache_bases[layer_name].storage_offset()
    )
    assert not kv_caches[layer_name].is_contiguous()

    view_info = kv_cache_view_infos[layer_name]
    assert view_info.view_shape == expected_base_shape
    assert view_info.permute_order == (1, 0, 2, 3, 4)


def test_build_attention_metadata_returns_per_layer_metadata_and_attaches_kv_bindings(
    rbln_model_runner, dist_init, monkeypatch
):
    """Attention metadata should be built per attention group, exposed per layer,
    and receive KV cache bindings."""
    req_ids = ("req_0", "req_1")

    rbln_model_runner._update_states(
        _schedule_new_request(
            *req_ids,
            prompt_token_ids=[[10, 11, 12], [20, 21, 22]],
            block_ids=[([0],), ([1],)],
            num_computed_tokens=[3, 3],
            num_scheduled_tokens=[3, 3],
        )
    )

    ib = rbln_model_runner.input_batch
    ib.num_computed_tokens_cpu[:2] = [3, 3]
    ib.num_tokens_no_spec[:2] = [4, 4]

    rbln_model_runner.query_start_loc[:3] = torch.tensor(
        [0, 1, 2],
        dtype=rbln_model_runner.query_start_loc.dtype,
    )
    rbln_model_runner.seq_lens[:2] = torch.tensor(
        [4, 4],
        dtype=rbln_model_runner.seq_lens.dtype,
    )

    # Exercise the real binding attachment path without depending on the real
    # attention metadata class.
    kv_cache_tensor = torch.empty(1)
    rbln_model_runner.kv_caches = [kv_cache_tensor]
    rbln_model_runner.kv_cache_bases = []
    rbln_model_runner.kv_cache_view_infos = []

    class FakeMetadataBuilder(
        rbln_model_runner_module.RBLNFlashAttentionMetadataBuilder
    ):
        def __init__(self, name):
            self.name = name
            self.calls = []

        def build(
            self,
            common_attn_metadata,
            positions,
            batch_pad,
            is_prefill,
        ):
            metadata = SimpleNamespace(builder_name=self.name)
            self.calls.append(
                {
                    "common_attn_metadata": common_attn_metadata,
                    "positions": positions,
                    "batch_pad": batch_pad,
                    "is_prefill": is_prefill,
                    "metadata": metadata,
                }
            )
            return metadata

    class FakeAttentionGroup:
        def __init__(self, layer_names, builder):
            self.layer_names = layer_names
            self._builder = builder

        def get_metadata_builder(self, index):
            assert index == 0
            return self._builder

    builder_a = FakeMetadataBuilder("group_a")
    builder_b = FakeMetadataBuilder("group_b")

    monkeypatch.setattr(
        rbln_model_runner,
        "attn_groups",
        [
            [
                FakeAttentionGroup(["layer.0", "layer.1"], builder_a),
                FakeAttentionGroup(["layer.2"], builder_b),
            ]
        ],
    )

    attn_metadata, spec_decode_common_attn_metadata = (
        rbln_model_runner._build_attention_metadata(
            num_tokens=2,
            num_reqs=2,
            max_query_len=1,
            num_tokens_padded=None,
            num_reqs_padded=4,
            logits_indices=None,
            use_spec_decode=False,
        )
    )

    assert spec_decode_common_attn_metadata is None

    assert set(attn_metadata) == {"layer.0", "layer.1", "layer.2"}

    # Layers in the same attention group share the same metadata object.
    assert attn_metadata["layer.0"] is attn_metadata["layer.1"]
    assert attn_metadata["layer.0"] is builder_a.calls[0]["metadata"]
    assert attn_metadata["layer.2"] is builder_b.calls[0]["metadata"]
    assert attn_metadata["layer.2"] is not attn_metadata["layer.0"]

    assert len(builder_a.calls) == 1
    assert len(builder_b.calls) == 1

    for call in (builder_a.calls[0], builder_b.calls[0]):
        common = call["common_attn_metadata"]

        assert common.query_start_loc.tolist() == [0, 1, 2]
        assert common.query_start_loc_cpu.tolist() == [0, 1, 2]
        assert common.seq_lens.tolist() == [4, 4]
        assert common.num_reqs == 2
        assert common.num_actual_tokens == 2
        assert common.max_query_len == 1
        assert common.max_seq_len == 4
        block_table_tensor = common.block_table_tensor
        assert block_table_tensor.shape[0] == 2
        assert block_table_tensor[:, 0].tolist() == [0, 1]

        assert call["positions"] is rbln_model_runner.positions
        assert call["batch_pad"] == 4
        assert call["is_prefill"] is False

    # _build_attention_metadata should attach KV cache bindings to every
    # returned per-layer metadata object.
    for metadata in attn_metadata.values():
        assert metadata.kv_caches is rbln_model_runner.kv_caches
        assert metadata.kv_cache_view_infos is None


def test_initialize_kv_cache_rejects_sub_block_cache_with_multi_group(
    rbln_model_runner, monkeypatch
):
    """Sub-block prefix caching should reject multi-group KV cache
    configurations before initialization side effects."""
    monkeypatch.setattr(
        rbln_model_runner_module.envs,
        "VLLM_RBLN_SUB_BLOCK_CACHE",
        True,
    )

    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=rbln_model_runner.model_config.get_num_kv_heads(
            rbln_model_runner.parallel_config
        ),
        head_size=rbln_model_runner.model_config.get_head_size(),
        dtype=rbln_model_runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS

    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
            KVCacheTensor(size=tensor_size, shared_by=["layer.1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["layer.1"], kv_cache_spec=attn_spec),
        ],
    )

    monkeypatch.setattr(
        rbln_model_runner,
        "initialize_attn_backend",
        _unexpected_call("initialize_attn_backend must not be called"),
    )

    with pytest.raises(
        NotImplementedError,
        match="Sub-block prefix caching does not support multi-group KV caches",
    ):
        rbln_model_runner.initialize_kv_cache(kv_cache_config)


def test_dummy_run_prefill_uses_last_token_indices_for_wrapped_logits(
    rbln_model_runner, dist_init, monkeypatch
):
    """Prefill warmup should pass last-token indices used by wrapped
    compute_logits."""
    attn_metadata_calls: dict[str, Any] = {}

    monkeypatch.setattr(
        rbln_model_runner,
        "_build_attention_metadata",
        _recording_attention_metadata_builder(attn_metadata_calls),
    )

    class FakePerformanceCtx:
        def profile(self, *args, **kwargs):
            return contextlib.nullcontext()

    monkeypatch.setattr(
        rbln_model_runner,
        "performance_ctx",
        FakePerformanceCtx(),
    )

    hidden_size = rbln_model_runner.model_config.get_hidden_size()
    vocab_size = 8
    model_calls: dict[str, Any] = {}

    num_reqs = 2
    num_tokens_per_req = 4

    hidden_states = torch.zeros(
        (num_reqs, num_tokens_per_req, hidden_size),
        dtype=rbln_model_runner.dtype,
        device=rbln_model_runner.device,
    )
    logits = torch.zeros(
        (num_reqs, vocab_size),
        dtype=torch.float32,
        device=rbln_model_runner.device,
    )

    monkeypatch.setattr(
        rbln_model_runner,
        "model_executable",
        _recording_model_executable(model_calls, hidden_states, logits),
        raising=False,
    )

    rbln_model_runner._dummy_run(
        num_reqs=num_reqs,
        num_tokens_per_req=num_tokens_per_req,
        is_prefill=True,
    )

    assert attn_metadata_calls["num_tokens"] == num_reqs * num_tokens_per_req
    assert attn_metadata_calls["num_reqs"] == num_reqs
    assert attn_metadata_calls["max_query_len"] == num_tokens_per_req
    assert attn_metadata_calls["use_spec_decode"] is False

    assert model_calls["input_ids"].shape == (num_reqs, num_tokens_per_req)
    assert model_calls["positions"].shape == (num_reqs, num_tokens_per_req)
    assert model_calls["intermediate_tensors"] is None
    assert model_calls["inputs_embeds"] is None

    # Last token of each prefill row: [0..3], [4..7] -> [3, 7].
    assert model_calls["token_indices"].tolist() == [3, 7]
