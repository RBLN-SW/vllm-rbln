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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def resolve_explicit_num_blocks(
    num_gpu_blocks_override: int | None,
    rbln_overrides: dict[str, Any] | None,
) -> int | None:
    """Resolve user block counts before vLLM's block-size conversion."""
    decoder_overrides = rbln_overrides or {}
    prefix = ""
    for name in ("language_model", "text_model"):
        if name in decoder_overrides:
            decoder_overrides = decoder_overrides[name]
            prefix = f"{name}."
            break
    num_blocks = decoder_overrides.get("kvcache_num_blocks")
    if num_blocks == 0:
        num_blocks = None
    if (
        num_gpu_blocks_override is not None
        and num_blocks is not None
        and num_gpu_blocks_override != num_blocks
    ):
        raise ValueError(
            "Conflicting KV cache block counts: "
            f"num_gpu_blocks_override={num_gpu_blocks_override}, "
            f"optimum_overrides.{prefix}kvcache_num_blocks={num_blocks}. "
            "Specify only one or use matching values."
        )
    return (
        num_gpu_blocks_override if num_gpu_blocks_override is not None else num_blocks
    )


# FIXME This will be refactored with RBLNPrefixKVCacheManager in the future
def get_attn_block_size(vllm_config: VllmConfig) -> int:
    if vllm_config.cache_config.enable_prefix_caching:
        rbln_config: OptimumRBLNConfig = vllm_config.additional_config
        assert rbln_config.attn_block_size is not None, (
            "the block-size sync sets attn_block_size when prefix caching is on"
        )
        block_size = rbln_config.attn_block_size
    else:
        block_size = vllm_config.cache_config.block_size
    return block_size


def get_block_ratio(vllm_config: VllmConfig) -> int:
    if vllm_config.cache_config.enable_prefix_caching:
        ob_size = get_attn_block_size(vllm_config)
        ib_size = vllm_config.cache_config.block_size
        blk_ratio = ob_size // ib_size
    else:
        blk_ratio = 1
    return blk_ratio


def is_full_block_available(num_blocks: int, vllm_config: VllmConfig) -> bool:
    block_size = get_attn_block_size(vllm_config)

    max_model_len = vllm_config.model_config.max_model_len
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    blocks_per_seq = math.ceil(max_model_len / block_size)
    ideal_total = max_num_seqs * blocks_per_seq
    return num_blocks >= ideal_total
