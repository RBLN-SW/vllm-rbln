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

"""Helpers for RBLN v1/core tests: scheduler harnesses (full RBLNScheduler +
VllmConfig) and manager harnesses (bare RBLNKVCacheManager from a hand-built
KVCacheConfig). On-device KV copy is a model-runner concern, left to e2e."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorOutput,
)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.native.vllm_config import local_model_path
from vllm_rbln.v1.core.rbln_kv_cache_manager import RBLNKVCacheManager, SubBlockIndex
from vllm_rbln.v1.core.rbln_scheduler import RBLNAsyncScheduler, RBLNScheduler

EOS_TOKEN_ID = 50256


def _ensure_none_hash() -> None:
    """Seed NONE_HASH once per process: with PYTHONHASHSEED unset init_none_hash()
    draws a fresh random value, so reseeding between two requests would silently
    stop their block hashes from ever matching."""
    from vllm.v1.core import kv_cache_utils

    if getattr(kv_cache_utils, "NONE_HASH", None) is None:
        init_none_hash(sha256)


@dataclass(frozen=True)
class MockKVConfig:
    """How many tokens the fake remote side reports, and whether the load is
    async (which parks the request in WAITING_FOR_REMOTE_KVS)."""

    matched_tokens: int = 0
    is_async: bool = False


class _MockKVConnectorMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.req_ids: list[str] = []


class MockKVConnector(KVConnectorBase_V1):
    """Only what RBLNScheduler.schedule() calls. External tokens are reported
    solely for requests carrying do_remote_prefill, so a test can decide per
    request whether the remote holds its KV."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        self.config = MockKVConfig(
            matched_tokens=extra["matched_tokens"], is_async=extra["is_async"]
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = getattr(request, "kv_transfer_params", None)
        if params and params.get("do_remote_prefill"):
            return self.config.matched_tokens, self.config.is_async
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        meta = _MockKVConnectorMetadata()
        cached = scheduler_output.scheduled_cached_reqs
        meta.req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs] + [
            req_id for req_id in cached.req_ids if req_id in cached.resumed_req_ids
        ]
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


if "MockKVConnector" not in KVConnectorFactory._registry:
    KVConnectorFactory.register_connector(
        "MockKVConnector", __name__, MockKVConnector.__name__
    )


def create_rbln_scheduler(
    *,
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
    num_speculative_tokens: int | None = None,
    pipeline_parallel_size: int = 1,
    sub_block_size: int | None = None,
    policy: str = "fcfs",
    use_kv_connector: MockKVConfig | None = None,
    async_scheduling: bool = False,
    additional_config: dict | None = None,
) -> RBLNScheduler:
    """Build an RBLNScheduler on CPU (ported from upstream tests/v1/core/utils):
    opt-125m config only, num_gpu_blocks set manually, no KV connector."""
    model_config = ModelConfig(
        model=local_model_path(model), trust_remote_code=True, dtype="float16", seed=42
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy=policy,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    speculative_config: SpeculativeConfig | None = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens
        )
    kv_transfer_config = None
    if use_kv_connector is not None:
        kv_transfer_config = KVTransferConfig(
            kv_connector="MockKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "matched_tokens": use_kv_connector.matched_tokens,
                "is_async": use_kv_connector.is_async,
            },
        )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(pipeline_parallel_size=pipeline_parallel_size),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
        additional_config=additional_config or {},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], full_attention_spec(block_size))],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler_cls = RBLNAsyncScheduler if async_scheduling else RBLNScheduler
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        sub_block_size=sub_block_size,
    )


def create_requests(
    num_requests: int,
    *,
    num_tokens: int = 10,
    max_tokens: int = 16,
    block_size: int = 16,
    req_ids: list[str] | None = None,
    same_prompt: bool = False,
    ignore_eos: bool = False,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    min_tokens: int = 0,
    priority: int = 0,
) -> list[Request]:
    """Requests with block hashers wired for prefix caching. same_prompt shares
    tokens to force matching; priority (lower = higher) applies to all."""
    _ensure_none_hash()
    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=ignore_eos,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        prompt_logprobs=prompt_logprobs,
        min_tokens=min_tokens,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    if req_ids is None:
        req_ids = [str(i) for i in range(num_requests)]
    else:
        assert len(req_ids) == num_requests
    requests = []
    for i in range(num_requests):
        prompt_token_ids = [0] * num_tokens if same_prompt else [i] * num_tokens
        requests.append(
            Request(
                request_id=req_ids[i],
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                pooling_params=None,
                mm_features=None,
                block_hasher=block_hasher,
                priority=priority,
            )
        )
    return requests


def make_model_runner_output(
    scheduler_output: Any,
    sampled_token_id: int | None = None,
    *,
    finished_recving: set[str] | None = None,
) -> ModelRunnerOutput:
    """Echoes the scheduled request ids, optionally with one sampled token each.
    ``finished_recving`` is the worker-side "remote KV arrived" signal that
    promotes a request out of WAITING_FOR_REMOTE_KVS on the next step."""
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[
            [sampled_token_id] if sampled_token_id is not None else [] for _ in req_ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=(
            KVConnectorOutput(finished_recving=finished_recving)
            if finished_recving
            else None
        ),
    )


def advance_to_decode(scheduler: RBLNScheduler, request: Request) -> None:
    """Add a request and run one prefill step + update so it enters decode."""
    scheduler.add_request(request)
    sched_out = scheduler.schedule()
    scheduler.update_from_output(sched_out, make_model_runner_output(sched_out, 1))


def full_attention_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def sliding_window_spec(block_size: int, sliding_window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


def make_kv_cache_config(
    specs: Sequence[KVCacheSpec], num_blocks: int = 10
) -> KVCacheConfig:
    """Build a KVCacheConfig with one group per spec."""
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


def make_manager(
    block_size: int,
    sub_block_size: int,
    num_blocks: int,
    *,
    max_model_len: int = 8192,
    enable_kv_cache_events: bool = False,
    log_stats: bool = False,
) -> RBLNKVCacheManager:
    """A single-group (full attention) RBLNKVCacheManager."""
    return RBLNKVCacheManager(
        kv_cache_config=make_kv_cache_config(
            [full_attention_spec(block_size)], num_blocks
        ),
        max_model_len=max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        sub_block_size=sub_block_size,
        hash_fn=sha256,
        enable_kv_cache_events=enable_kv_cache_events,
        log_stats=log_stats,
    )


def make_hybrid_manager(
    block_size: int,
    sub_block_size: int,
    num_blocks: int,
    sliding_window: int,
    *,
    max_model_len: int = 8192,
    enable_kv_cache_events: bool = False,
) -> RBLNKVCacheManager:
    """A two-group manager: full attention + sliding window (same block_size)."""
    return RBLNKVCacheManager(
        kv_cache_config=make_kv_cache_config(
            [
                full_attention_spec(block_size),
                sliding_window_spec(block_size, sliding_window),
            ],
            num_blocks,
        ),
        max_model_len=max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        sub_block_size=sub_block_size,
        hash_fn=sha256,
        enable_kv_cache_events=enable_kv_cache_events,
    )


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    *,
    max_tokens: int = 17,
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
    prompt_logprobs: int | None = None,
    mm_features: Any = None,
) -> Request:
    _ensure_none_hash()
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features,
        sampling_params=SamplingParams(
            max_tokens=max_tokens, prompt_logprobs=prompt_logprobs
        ),
        pooling_params=None,
        cache_salt=cache_salt,
        lora_request=lora_request,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def sub_block_index(manager: RBLNKVCacheManager, group: int = 0) -> SubBlockIndex:
    """The sub-block index of one group (defaults to the first)."""
    return manager._group_infos[group].sub_block_index


def prefill_request(
    manager: RBLNKVCacheManager, request: Request
) -> tuple[Any, int, Any]:
    """Drive the full get_computed_blocks -> allocate_slots flow, then simulate
    execute_model completion. Returns ``(computed_blocks, total_computed_tokens,
    allocated_blocks)``."""
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
    match = manager.get_computed_blocks_sub_block(request, num_computed_tokens)
    sub_extra = match.num_tokens if match else 0
    total_computed = num_computed_tokens + sub_extra
    blocks = manager.allocate_slots(
        request,
        request.num_tokens - total_computed,
        total_computed,
        computed_blocks,
    )
    if blocks is not None and match is not None:
        manager.apply_sub_block_match(match)
    elif match is not None:
        manager.release_sub_block_match(match)
    # Simulate execute_model completion: num_computed_tokens catches up.
    request.num_computed_tokens = request.num_tokens
    manager.do_pending_indexing()
    return computed_blocks, total_computed, blocks


def _drain(sched, *, token=0, max_steps=300, per_step=None):
    """Run schedule()/update_from_output() until nothing is scheduled. ``token``
    is non-EOS; ``per_step(out)`` runs after each schedule() for invariant
    checks. Returns the number of steps."""
    steps = 0
    while sched.requests:
        out = sched.schedule()
        if not out.num_scheduled_tokens:
            break
        if per_step is not None:
            per_step(out)
        sched.update_from_output(out, make_model_runner_output(out, token))
        steps += 1
        assert steps < max_steps, "run did not converge"
    return steps
