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
    vllm_config: VllmConfig,
    kvcache_block_size: int,
    prefill_chunk_size: int,
    image_prefill_chunk_size: list[int] | None = None,
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

    For decoder and (non-enc-dec) multimodal models this is the prefill chunk
    size: the scheduler chunks prefill at this granularity and the same value is
    read back off ``max_num_batched_tokens`` to size RBLNKVCacheManager's
    gemma3/gemma4 block padding and to pin the compiled artefact. vllm-rbln no
    longer carries a separate ``prefill_chunk_size`` in ``additional_config``.

    Encoder-decoder and pooling models don't chunk prefill, so the budget must
    fit a full-length prefill plus a full batch dispatch. Encoder-decoder models
    (e.g. Whisper) additionally need at least ``max_source_positions`` so vllm's
    MultiModalBudget validation passes (it requires ``max_tokens_per_mm_item <=
    max_num_batched_tokens`` when chunked MM input is disabled).
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
        # decoder / multimodal: max_num_batched_tokens is the prefill chunk size.
        # A decode step schedules one token per running sequence, so the budget
        # must cover a full batch; a smaller chunk would throttle decode.
        target = params.prefill_chunk_size
        assert target >= max_num_seqs, (
            f"prefill_chunk_size ({target}) must be >= max_num_seqs "
            f"({max_num_seqs}); a smaller prefill chunk would throttle the "
            "decode batch. Recompile with a larger prefill_chunk_size or lower "
            "max_num_seqs."
        )

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
    # Image-prefill buckets (gemma4 multi-bucket / gemma3 single) can't be folded
    # into the scalar max_num_batched_tokens, so they stay in additional_config.
    # RBLNKVCacheManager reads them to size the per-image chunk in the
    # chunked-prefill block padding. Both load paths funnel through here.
    if image_prefill_chunk_size is None:
        return
    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config["image_prefill_chunk_size"] = image_prefill_chunk_size
