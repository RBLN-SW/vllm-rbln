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

from typing import Optional

import torch
from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig,
                         StructuredOutputsConfig, VllmConfig)
from vllm.multimodal.inputs import (MultiModalFeatureSpec,
                                    MultiModalKwargsItem, PlaceholderRange)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_rbln.v1.core.optimum_scheduler import (RBLNOptimumScheduler,
                                                 RBLNSchedulerOutput)

EOS_TOKEN_ID = 50256
_none_hash_initialized = False


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 128,
    num_blocks: int = 8,
    block_size: int = 16,
    max_model_len: Optional[int] = None,
    async_scheduling: bool = False,
    outer_block_size: int = 0,
    enable_prefix_caching: bool = False,
) -> RBLNOptimumScheduler:
    """Create RBLNOptimumscheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch

    Returns:
      {class}`RBLNOptimumscheduler` instance
    """
    if max_model_len is None:
        max_model_len = max_num_batched_tokens

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
    )
    model_config = ModelConfig(
        model=model,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype=torch.float,
        seed=42,
    )

    cache_config = CacheConfig(
        block_size=block_size,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    structured_outputs_config = StructuredOutputsConfig(backend="guidance", )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        structured_outputs_config=structured_outputs_config,
        additional_config={
            "attn_block_size": outer_block_size,
        },
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size, 1, 1, torch.float32, False),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler_cls = RBLNOptimumScheduler
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
    )


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: Optional[list[list[PlaceholderRange]]] = None,
    max_tokens: int = 16,
    stop_token_ids: Optional[list[int]] = None,
    prompt_logprobs: Optional[int] = None,
    same_prompt: bool = False,
    block_size: int = 16,
    sample_json_schema: str = None,
) -> list[Request]:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    requests = []

    if mm_hashes_list is not None:
        # NOTE: allow manual input; some mm items can have the same identifier
        # no. of mm_hashes and mm_positions for each request should be identical
        assert mm_positions is not None, (
            "mm_positions must be provided when mm_hashes_list is provided")
        assert len(mm_hashes_list) == len(mm_positions) == num_requests
        assert [len(h)
                for h in mm_hashes_list] == [len(p) for p in mm_positions]

        # Since same identifier would imply
        # they are identical encoder output
        # Verify mm items with identical identifier
        # are having mm_position.length
        seen_hashes: dict[str, int] = {}

    if sample_json_schema:
        structured_outputs = StructuredOutputsParams(json=sample_json_schema)
        structured_outputs._backend = "guidance"
    else:
        structured_outputs = None
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs,
                                     structured_outputs=structured_outputs)
    for i in range(num_requests):
        mm_features = []

        for j, position in enumerate(
                mm_positions[i] if mm_positions is not None else []):
            if mm_hashes_list is not None:
                identifier = mm_hashes_list[i][j]

                # Verify if position length is identical
                position_length = position.length
                if identifier in seen_hashes:
                    assert seen_hashes[identifier] == position_length, (
                        f"mm_hash '{identifier}' has inconsistent "
                        f"position lengths: "
                        f"previously {seen_hashes[identifier]}, "
                        f"now {position_length} "
                        f"at request {i}, position {j}")
                else:
                    seen_hashes[identifier] = position_length
            else:
                # Unique dummy hash for each mm item
                identifier = f"hash{i}_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy("dummy_m"),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

        prompt_token_ids = ([0] * num_tokens if same_prompt else [i] *
                            num_tokens)
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=mm_features if mm_features else None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def create_model_runner_output(
    scheduler_output: RBLNSchedulerOutput, ) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={
            req_id: i
            for i, req_id in enumerate(req_ids)
        },
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
