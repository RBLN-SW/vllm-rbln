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

"""Derive and apply vLLM execution settings from the selected RBLN config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.block_size import get_block_ratio, is_full_block_available
from vllm_rbln.utils.optimum.configuration.readers import (
    RblnConfigLike,
    get_field,
    language_model_configs,
)
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_encoder_pooling_arch,
    is_multi_modal,
    is_pooling_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNConfig

logger = init_logger(__name__)


def get_runtime_num_blocks(hf_config: Any, rbln_config: RblnConfigLike) -> int | None:
    """Read the compiled block count or map a pooling batch to vLLM blocks."""
    if is_encoder_pooling_arch(hf_config):
        # Encoder pooling graphs have no KV cache. vLLM uses one block per sequence.
        return get_field(rbln_config, "batch_size")
    sources = (
        language_model_configs(rbln_config)
        if is_multi_modal(hf_config)
        else [rbln_config]
    )
    for source in sources:
        num_blocks = get_field(source, "kvcache_num_blocks")
        if num_blocks is not None:
            return num_blocks
    return None


def _apply_prefix_caching_block_size(
    vllm_config: VllmConfig, kvcache_block_size: int, prefill_chunk_size: int
) -> None:
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    sub_block_size = additional_config.sub_block_size
    if sub_block_size is None:
        sub_block_size = prefill_chunk_size
        logger.debug(
            "Sub-block size is set to %s based on prefill_chunk_size",
            sub_block_size,
        )
    else:
        if sub_block_size % prefill_chunk_size != 0:
            raise ValueError(
                "sub_block_size ({}) is not divisible "
                "by prefill_chunk_size ({}). "
                "Please check the value of prefill_chunk_size "
                "in rbln_config.json".format(sub_block_size, prefill_chunk_size)
            )
        if sub_block_size > kvcache_block_size:
            raise ValueError(
                "sub_block_size ({}) is greater than "
                "kvcache_block_size ({}). "
                "Please check the value of kvcache_block_size "
                "in rbln_config.json".format(sub_block_size, kvcache_block_size)
            )
        logger.debug(
            "Sub-block size is set to %s based on additional_config",
            sub_block_size,
        )
    if kvcache_block_size % sub_block_size != 0:
        raise ValueError(
            "kvcache_block_size ({}) is not divisible "
            "by sub_block_size ({}). "
            "Please check the value of sub_block_size in additional_config".format(
                kvcache_block_size, sub_block_size
            )
        )
    vllm_config.cache_config.block_size = sub_block_size
    additional_config.attn_block_size = kvcache_block_size


def update_block_size(
    vllm_config: VllmConfig,
    kvcache_block_size: int,
    prefill_chunk_size: int | None,
    image_prefill_chunk_size: list[int] | None = None,
) -> None:
    """
    Update the block size in the vllm_config based on the provided kvcache_block_size
    and prefill_chunk_size. For models with prefix caching enabled, the block size
    is set to the sub-block size, which is determined based on the prefill_chunk_size
    and user-provided sub_block_size.
    """
    vllm_config.cache_config.user_specified_block_size = True
    if vllm_config.cache_config.enable_prefix_caching:
        assert prefill_chunk_size is not None, (
            "prefill_chunk_size is required for prefix caching"
        )
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


def is_chunked_prefill_arch(hf_config) -> bool:
    """Whether the compiled model supports chunked prefill."""
    return not (is_enc_dec_arch(hf_config) or is_pooling_arch(hf_config))


def update_max_num_batched_tokens(vllm_config: VllmConfig) -> None:
    """Keep the scheduling budget large enough for a full prefill or decode batch.

    Chunking happens inside optimum-rbln, so a step must fit a whole prompt.
    Whisper also needs max_source_positions for vLLM's MultiModalBudget check.
    """
    hf_config = vllm_config.model_config.hf_config
    scheduler_config = vllm_config.scheduler_config
    if scheduler_config.max_num_scheduled_tokens is not None:
        raise ValueError(
            "max_num_scheduled_tokens cannot be set on the optimum model impl. "
            "The scheduling budget is determined automatically. "
            "Leave max_num_scheduled_tokens unset."
        )
    scheduler_config.max_num_batched_tokens = max(
        vllm_config.model_config.max_model_len,
        scheduler_config.max_num_seqs,
        getattr(hf_config, "max_source_positions", 0)
        if is_enc_dec_arch(hf_config)
        else 0,
    )


def store_image_prefill_chunk_size(
    vllm_config: VllmConfig,
    image_prefill_chunk_size: list[int] | None,
) -> None:
    # Image-prefill buckets (gemma4 multi-bucket / gemma3 single) don't fit the
    # scalar prefill_chunk_size, so they stay in additional_config for
    # RBLNKVCacheManager to size the per-image chunk in its block padding.
    if image_prefill_chunk_size is None:
        return
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    additional_config.image_prefill_chunk_size = image_prefill_chunk_size


def validate_user_prefill_chunk_size(
    vllm_config: VllmConfig, prefill_chunk_size: int | None
) -> None:
    """Require an explicit user chunk size to match the selected model config."""
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    user_chunk = additional_config.prefill_chunk_size
    if user_chunk is not None:
        if not is_chunked_prefill_arch(vllm_config.model_config.hf_config):
            raise ValueError(
                "prefill_chunk_size cannot be specified for a model that does "
                "not support chunked prefill. Leave --rbln-prefill-chunk-size unset."
            )
        if user_chunk != prefill_chunk_size:
            raise ValueError(
                f"prefill_chunk_size ({user_chunk}) conflicts with the selected "
                f"prefill_chunk_size ({prefill_chunk_size}) in rbln_config. "
                "Leave --rbln-prefill-chunk-size unset or select a model "
                "with a matching prefill chunk size."
            )
    additional_config.prefill_chunk_size = prefill_chunk_size


def keep_device_overrides(obj: dict) -> dict:
    """Recursively keep only ``device`` entries from nested dict/list.

    A precompiled model accepts only device placement as a load-time override.
    """
    result: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "device":
            result[k] = v
        elif isinstance(v, dict):
            filtered = keep_device_overrides(v)
            if filtered:
                result[k] = filtered
    return result


def update_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> None:
    # Startup sync runs in the main process and again in workers. Keep the
    # block-count conversion idempotent when VllmConfig is deserialized.
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    if additional_config.num_blocks_synced:
        logger.debug(
            "num_blocks already synced to %s, skipping...",
            vllm_config.cache_config.num_gpu_blocks,
        )
        return
    # num_blocks is determined by rbln_config or overridden by user.
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
        # This is kept for optimum based num blocks
        # not considering ob-ib logic for prefix caching
        additional_config.num_blocks_override = num_blocks
    blk_ratio = get_block_ratio(vllm_config)

    if is_full_block_available(num_blocks, vllm_config):
        adjusted_num_blocks = num_blocks * blk_ratio + 1
    else:
        adjusted_num_blocks = (num_blocks - 1) * blk_ratio + 1

    vllm_config.cache_config.num_gpu_blocks = adjusted_num_blocks

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        vllm_config.cache_config.num_gpu_blocks_override = adjusted_num_blocks
    additional_config.num_blocks_synced = True


def update_mamba_block_size(vllm_config: VllmConfig, max_seq_len: int) -> None:
    cache_config = vllm_config.cache_config
    if (
        not cache_config.user_specified_mamba_block_size
        and cache_config.mamba_block_size is not None
    ):
        assert not cache_config.enable_prefix_caching, (
            "mamba_block_size can only be synced to max_model_len while "
            "prefix caching is disabled, but prefix caching is enabled. "
            "RBLN does not support prefix caching for mamba/hybrid models."
        )
        cache_config.mamba_block_size = max_seq_len
