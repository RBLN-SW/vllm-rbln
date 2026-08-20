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
    is_pooling_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .params import RBLNParams
else:
    VllmConfig = None

logger = init_logger(__name__)

# additional_config key holding the user's explicit max_num_batched_tokens
# (``None`` if unset), stashed by the platform before vLLM defaults the value.
# In the RBLN optimum path an explicit value is the prefill chunk size.
USER_MAX_NUM_BATCHED_TOKENS_KEY = "user_max_num_batched_tokens"


def update_block_size(
    vllm_config: VllmConfig,
    kvcache_block_size: int,
    prefill_chunk_size: int,
    image_prefill_chunk_size: list[int] | None = None,
) -> None:
    """
    Update the block size in the vllm_config based on the provided
    kvcache_block_size. The vLLM block is the device block: prefix caching
    shares device blocks between requests through the upstream KVCacheManager,
    so there is no separate prefix block size.
    """
    vllm_config.cache_config.user_specified_block_size = True
    if vllm_config.cache_config.enable_prefix_caching:
        if "prefix_block_size" in vllm_config.additional_config:
            raise ValueError(
                "prefix_block_size is no longer supported: prefix caching "
                "shares device blocks at kvcache_block_size granularity."
            )
        assert prefill_chunk_size is not None, (
            "prefill_chunk_size must be specified in rbln_config.json"
        )
        # A prefix-cache hit makes the prefill resume at a block-aligned
        # offset, and the compiled prefill only resumes at chunk boundaries.
        if kvcache_block_size % prefill_chunk_size != 0:
            raise ValueError(
                "kvcache_block_size ({}) is not divisible by "
                "prefill_chunk_size ({}), so a prefix-cache hit cannot "
                "resume at a chunk-aligned offset.".format(
                    kvcache_block_size, prefill_chunk_size
                )
            )
    if vllm_config.cache_config.block_size != kvcache_block_size:
        logger.info(
            "Updating model_cache_config.block_size from %s to %s "
            "based on rbln_config.json",
            vllm_config.cache_config.block_size,
            kvcache_block_size,
        )
        vllm_config.cache_config.block_size = kvcache_block_size


def is_chunked_prefill_arch(hf_config) -> bool:
    """Whether ``max_num_batched_tokens`` is the prefill chunk size for this model.

    True for decoder and (non-enc-dec) multimodal models: they chunk prefill at
    ``max_num_batched_tokens`` granularity. Encoder-decoder and pooling models
    don't chunk prefill, so the value there is a full-prefill budget instead.
    """
    return not (is_enc_dec_arch(hf_config) or is_pooling_arch(hf_config))


def get_user_max_num_batched_tokens(vllm_config: VllmConfig) -> int | None:
    """Return the user's explicit ``max_num_batched_tokens`` (``None`` if unset).

    Stashed by ``RblnPlatform._capture_user_max_num_batched_tokens`` before vLLM
    fills in its default.
    """
    if vllm_config.additional_config is None:
        return None
    return vllm_config.additional_config.get(USER_MAX_NUM_BATCHED_TOKENS_KEY)


def update_max_num_batched_tokens(
    vllm_config: VllmConfig, params: "RBLNParams"
) -> None:
    """Set ``scheduler_config.max_num_batched_tokens`` from the RBLN config.

    Decoder/multimodal: the prefill chunk size. The scheduler chunks prefill at
    this granularity, and it is read back to size gemma3/4 block padding and pin
    the compiled artefact.

    Enc-dec/pooling don't chunk prefill, so it must fit a full prefill plus a full
    batch; Whisper also needs ``max_source_positions`` for vLLM's MultiModalBudget
    check.
    """
    hf_config = vllm_config.model_config.hf_config
    max_model_len = vllm_config.model_config.max_model_len
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    if is_enc_dec_arch(hf_config):
        max_source_positions = getattr(hf_config, "max_source_positions", 0)
        target = max(max_model_len, max_num_seqs, max_source_positions)
    elif is_pooling_arch(hf_config):
        target = max(max_model_len, max_num_seqs)
    else:
        # decoder / multimodal: max_num_batched_tokens carries the compiled
        # prefill chunk size. It does not bound the decode batch — the
        # scheduler's per-step budget (max_num_scheduled_tokens) is decoupled
        # from max_num_batched_tokens — so it may be smaller than max_num_seqs.
        target = params.prefill_chunk_size

    cur = vllm_config.scheduler_config.max_num_batched_tokens
    if cur != target:
        logger.info(
            "Updating scheduler_config.max_num_batched_tokens "
            "from %s to %d based on rbln_config.json",
            cur,
            target,
        )
        vllm_config.scheduler_config.max_num_batched_tokens = target


def store_image_prefill_chunk_size(
    vllm_config: VllmConfig,
    image_prefill_chunk_size: list[int] | None,
) -> None:
    # Image-prefill buckets (gemma4 multi-bucket / gemma3 single) don't fit the
    # scalar max_num_batched_tokens, so they stay in additional_config for
    # RBLNKVCacheManager to size the per-image chunk in its block padding.
    if image_prefill_chunk_size is None:
        return
    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config["image_prefill_chunk_size"] = image_prefill_chunk_size


def apply_user_prefill_chunk_size(
    vllm_config: VllmConfig,
    params: "RBLNParams",
    *,
    precompiled: bool,
    override_prefill_chunk_size: int | None = None,
) -> None:
    """Fold an explicit ``max_num_batched_tokens`` into ``params.prefill_chunk_size``.

    No-op for enc-dec/pooling (they don't chunk prefill) or when the user left
    ``max_num_batched_tokens`` unset; downstream reads only
    ``params.prefill_chunk_size``.

    Precompiled: the binary fixes the chunk size, so a differing value errors.
    Compile: the user's value wins over the rbln_config default, but passing both
    (``override_prefill_chunk_size``) with different values errors.
    """
    if not is_chunked_prefill_arch(vllm_config.model_config.hf_config):
        return
    user_mnbt = get_user_max_num_batched_tokens(vllm_config)
    if user_mnbt is None:
        return
    if precompiled:
        if user_mnbt != params.prefill_chunk_size:
            raise ValueError(
                f"max_num_batched_tokens ({user_mnbt}) conflicts with the "
                f"compiled prefill_chunk_size ({params.prefill_chunk_size}) in "
                "rbln_config.json. Omit max_num_batched_tokens or recompile the "
                "model with a matching prefill chunk size."
            )
    elif (
        override_prefill_chunk_size is not None
        and override_prefill_chunk_size != user_mnbt
    ):
        raise ValueError(
            f"Conflicting prefill chunk size: max_num_batched_tokens "
            f"({user_mnbt}) != rbln_config['prefill_chunk_size'] "
            f"({override_prefill_chunk_size}). Set only one."
        )
    params.prefill_chunk_size = user_mnbt
