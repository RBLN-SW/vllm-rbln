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

from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_pooling_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .params import RBLNParams
else:
    VllmConfig = None

logger = init_logger(__name__)


def _apply_prefix_caching_block_size(
    vllm_config: VllmConfig, kvcache_block_size: int, prefill_chunk_size: int
) -> None:
    assert prefill_chunk_size is not None, (
        "prefill_chunk_size must be specified in rbln_config.json"
    )
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    sub_block_size = rbln_config.sub_block_size
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
    rbln_config.attn_block_size = kvcache_block_size


def update_block_size(
    vllm_config: VllmConfig,
    kvcache_block_size: int,
    prefill_chunk_size: int,
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
    scheduler_config.max_num_batched_tokens = max(
        scheduler_config.max_num_batched_tokens,
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
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    rbln_config.image_prefill_chunk_size = image_prefill_chunk_size


def apply_user_prefill_chunk_size(
    vllm_config: VllmConfig,
    params: "RBLNParams",
    *,
    precompiled: bool,
    override_prefill_chunk_size: int | None = None,
) -> None:
    """Resolve the compiled chunk size independently of the scheduling budget."""
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    chunk = rbln_config.prefill_chunk_size
    if chunk is not None:
        if not is_chunked_prefill_arch(vllm_config.model_config.hf_config):
            raise ValueError(
                "prefill_chunk_size cannot be specified for a model that does "
                "not support chunked prefill. Omit --rbln-prefill-chunk-size."
            )
        if precompiled and chunk != params.prefill_chunk_size:
            raise ValueError(
                f"prefill_chunk_size ({chunk}) conflicts with the compiled "
                f"prefill_chunk_size ({params.prefill_chunk_size}) in "
                "rbln_config.json. Omit --rbln-prefill-chunk-size or recompile "
                "the model with a matching prefill chunk size."
            )
        if not precompiled:
            if (
                override_prefill_chunk_size is not None
                and override_prefill_chunk_size != chunk
            ):
                raise ValueError(
                    f"Conflicting prefill chunk size: --rbln-prefill-chunk-size "
                    f"({chunk}) != optimum_overrides['prefill_chunk_size'] "
                    f"({override_prefill_chunk_size}). Set only one."
                )
            params.prefill_chunk_size = chunk
    rbln_config.prefill_chunk_size = params.prefill_chunk_size
