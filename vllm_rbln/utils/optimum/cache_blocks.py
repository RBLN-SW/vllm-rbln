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

"""KV-cache block calculation and synchronisation helpers."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def is_full_block_available(num_blocks: int, vllm_config: VllmConfig) -> bool:
    if vllm_config.cache_config.enable_prefix_caching:
        block_size = vllm_config.additional_config["attn_block_size"]

    else:
        block_size = vllm_config.cache_config.block_size

    max_model_len = vllm_config.model_config.max_model_len
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    blocks_per_seq = math.ceil(max_model_len / block_size)
    ideal_total = max_num_seqs * blocks_per_seq
    return num_blocks >= ideal_total


def get_block_ratio(vllm_config: VllmConfig) -> int:
    if vllm_config.cache_config.enable_prefix_caching:
        ob_size = vllm_config.additional_config["attn_block_size"]
        ib_size = vllm_config.cache_config.block_size
        blk_ratio = ob_size // ib_size
    else:
        blk_ratio = 1
    return blk_ratio


def apply_prefix_caching_block_size(
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


def sync_cache_block_size(
    vllm_config: VllmConfig, kvcache_block_size: int, prefill_chunk_size: int
) -> None:
    if vllm_config.cache_config.enable_prefix_caching:
        apply_prefix_caching_block_size(
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


def sync_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> None:
    # num_blocks is determined by rbln_config or overridden by user.
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
        vllm_config.additional_config["num_blocks_override"] = num_blocks

    blk_ratio = get_block_ratio(vllm_config)

    if is_full_block_available(num_blocks, vllm_config):
        adjusted_num_blocks = num_blocks * blk_ratio + 1
    else:
        adjusted_num_blocks = (num_blocks - 1) * blk_ratio + 1

    vllm_config.cache_config.num_gpu_blocks = adjusted_num_blocks

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        vllm_config.cache_config.num_gpu_blocks_override = adjusted_num_blocks
