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

from typing import TYPE_CHECKING

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def _apply_prefix_caching_block_size(
    vllm_config: VllmConfig, kvcache_block_size: int, prefill_chunk_size: int
) -> None:
    assert prefill_chunk_size is not None, (
        "prefill_chunk_size must be specified in rbln_config.json"
    )
    # If user set prefix_block_size in additional_config, use it.
    # Otherwise, set it to prefill_chunk_size.
    prefix_block_size = vllm_config.additional_config.get("prefix_block_size", None)
    if prefix_block_size is None:
        prefix_block_size = prefill_chunk_size
        logger.debug(
            "Prefix block size is set to %s based on prefill_chunk_size",
            prefix_block_size,
        )
    else:
        if prefix_block_size % prefill_chunk_size != 0:
            raise ValueError(
                "prefix_block_size ({}) is not divisible "
                "by prefill_chunk_size ({}). "
                "Please check the value of prefill_chunk_size "
                "in rbln_config.json".format(prefix_block_size, prefill_chunk_size)
            )
        if prefix_block_size > kvcache_block_size:
            raise ValueError(
                "prefix_block_size ({}) is greater than "
                "kvcache_block_size ({}). "
                "Please check the value of kvcache_block_size "
                "in rbln_config.json".format(prefix_block_size, kvcache_block_size)
            )
        logger.debug(
            "Prefix block size is set to %s based on additional_config",
            prefix_block_size,
        )
    if kvcache_block_size % prefix_block_size != 0:
        raise ValueError(
            "kvcache_block_size ({}) is not divisible "
            "by prefix_block_size ({}). "
            "Please check the value of prefix_block_size in rbln_config.json".format(
                kvcache_block_size, prefix_block_size
            )
        )
    vllm_config.cache_config.block_size = prefix_block_size
    vllm_config.additional_config["attn_block_size"] = kvcache_block_size


def update_block_size(
    vllm_config: VllmConfig, kvcache_block_size: int, prefill_chunk_size: int
) -> None:
    """
    Update the block size in the vllm_config based on the provided kvcache_block_size
    and prefill_chunk_size. For models with prefix caching enabled, the block size
    is set to the prefix block size, which is determined based on the prefill_chunk_size
    and user-provided prefix_block_size.
    """
    vllm_config.cache_config.user_specified_block_size = True
    if vllm_config.cache_config.enable_prefix_caching:
        _apply_prefix_caching_block_size(
            vllm_config, kvcache_block_size, prefill_chunk_size
        )
    else:
        if vllm_config.cache_config.block_size != kvcache_block_size:
            logger.info(
                "Updating model_cache_config.block_size from %s to %s "
                "based on rbln_config.json",
                vllm_config.cache_config.block_size,
                kvcache_block_size,
            )
            vllm_config.cache_config.block_size = kvcache_block_size


def update_max_num_batched_tokens(vllm_config: VllmConfig, max_model_len: int) -> None:
    """
    Update the max_num_batched_tokens in the vLLM configuration based on the model's
    maximum length and architecture.

    For encoder-decoder multimodal models (e.g. Whisper), max_num_batched_tokens
    must be at least max_source_positions so that vllm's MultiModalBudget
    validation passes (it requires max_tokens_per_mm_item <= max_num_batched_tokens
    when chunked MM input is disabled).
    """
    target_max_num_batched_tokens = max_model_len
    hf_config = vllm_config.model_config.hf_config

    if not is_enc_dec_arch(hf_config):
        return

    max_source_positions = getattr(hf_config, "max_source_positions", 0)
    if max_source_positions > target_max_num_batched_tokens:
        target_max_num_batched_tokens = max_source_positions
        logger.info(
            "Encoder-decoder model detected: setting max_num_batched_tokens "
            "to %d (max_source_positions) instead of %d (max_model_len)",
            max_source_positions,
            max_model_len,
        )

    cur = vllm_config.scheduler_config.max_num_batched_tokens
    if cur != target_max_num_batched_tokens:
        logger.info(
            "Updating scheduler_config.max_num_batched_tokens "
            "from %s to %d based on rbln_config.json",
            cur,
            target_max_num_batched_tokens,
        )
        vllm_config.scheduler_config.max_num_batched_tokens = (
            target_max_num_batched_tokens
        )
