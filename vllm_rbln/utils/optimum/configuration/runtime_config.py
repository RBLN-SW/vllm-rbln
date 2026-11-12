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

"""Prepare, validate and apply optimum runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_rbln.config import OptimumRBLNRuntimeConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.block_size import (
    get_block_ratio,
    is_full_block_available,
)
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_encoder_pooling_arch,
    is_multi_modal,
    is_pooling_arch,
)

from .rbln_config import (
    RblnConfigLike,
    get_field,
    language_model_configs,
    read_kvcache_block_size,
    read_language_model_fields,
    read_num_devices,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def prepare_runtime_config(
    vllm_config: VllmConfig,
    rbln_config: RblnConfigLike,
    *,
    precompiled: bool,
) -> OptimumRBLNRuntimeConfig:
    """Read and validate the selected model settings without changing vLLM."""
    hf_config = vllm_config.model_config.hf_config
    image_prefill_chunk_size = None
    prefill_chunk_size: int | None
    if is_enc_dec_arch(hf_config):
        batch_size = get_field(rbln_config, "batch_size")
        max_seq_len = get_field(rbln_config, "dec_max_seq_len")
        kvcache_block_size = read_kvcache_block_size(rbln_config)
        prefill_chunk_size = None
    elif is_multi_modal(hf_config):
        values = read_language_model_fields(
            rbln_config,
            "batch_size",
            "max_seq_len",
            "prefill_chunk_size",
            "kvcache_block_size",
            "image_prefill_chunk_size",
        )
        batch_size = values.get("batch_size")
        max_seq_len = values.get("max_seq_len")
        kvcache_block_size = values.get("kvcache_block_size")
        prefill_chunk_size = values.get("prefill_chunk_size")
        image_prefill_chunk_size = values.get("image_prefill_chunk_size")
    else:
        batch_size = get_field(rbln_config, "batch_size")
        max_seq_len = get_field(rbln_config, "max_seq_len")
        if is_encoder_pooling_arch(hf_config):
            # Encoder graphs have no KV cache; vLLM uses one block per sequence.
            kvcache_block_size = max_seq_len
            prefill_chunk_size = None
        else:
            kvcache_block_size = read_kvcache_block_size(rbln_config)
            prefill_chunk_size = get_field(rbln_config, "prefill_chunk_size")
    num_blocks = get_runtime_num_blocks(hf_config, rbln_config)
    num_devices = read_num_devices(rbln_config)
    dtype = get_field(rbln_config, "dtype")
    compiled_dtype = dtype if precompiled else None
    if isinstance(compiled_dtype, str):
        compiled_dtype = getattr(torch, compiled_dtype)

    assert batch_size is not None
    assert max_seq_len is not None
    if is_chunked_prefill_arch(hf_config) and prefill_chunk_size is None:
        raise ValueError(
            "prefill_chunk_size must be specified in the selected rbln_config."
        )
    if precompiled:
        assert num_blocks is not None, (
            "num_blocks must be specified in rbln_config.json"
        )
    if kvcache_block_size is None:
        raise ValueError(
            "kvcache_block_size or kvcache_partition_len must be specified "
            "in the selected rbln_config."
        )
    if num_devices is None:
        raise ValueError("num_devices must be specified in the selected rbln_config.")
    if is_chunked_prefill_arch(hf_config):
        assert prefill_chunk_size is not None
        validate_user_prefill_chunk_size(vllm_config, prefill_chunk_size)

    cache_config = vllm_config.cache_config
    additional_config = vllm_config.additional_config
    block_size = kvcache_block_size
    attn_block_size = None
    if cache_config.enable_prefix_caching:
        assert prefill_chunk_size is not None, (
            "prefill_chunk_size is required for prefix caching"
        )
        sub_block_size = additional_config.sub_block_size
        if sub_block_size is None:
            sub_block_size = prefill_chunk_size
        else:
            if sub_block_size % prefill_chunk_size != 0:
                raise ValueError(
                    f"sub_block_size ({sub_block_size}) is not divisible "
                    f"by prefill_chunk_size ({prefill_chunk_size}). "
                    "Please check the value of prefill_chunk_size in rbln_config.json"
                )
            if sub_block_size > kvcache_block_size:
                raise ValueError(
                    f"sub_block_size ({sub_block_size}) is greater than "
                    f"kvcache_block_size ({kvcache_block_size}). "
                    "Please check the value of kvcache_block_size in rbln_config.json"
                )
        if kvcache_block_size % sub_block_size != 0:
            raise ValueError(
                f"kvcache_block_size ({kvcache_block_size}) is not divisible "
                f"by sub_block_size ({sub_block_size}). "
                "Please check the value of sub_block_size in additional_config"
            )
        block_size = sub_block_size
        attn_block_size = kvcache_block_size

    mamba_block_size = cache_config.mamba_block_size
    if (
        precompiled
        and not cache_config.user_specified_mamba_block_size
        and mamba_block_size is not None
    ):
        assert not cache_config.enable_prefix_caching, (
            "mamba_block_size can only be synced to max_model_len while "
            "prefix caching is disabled, but prefix caching is enabled. "
            "RBLN does not support prefix caching for mamba/hybrid models."
        )
        mamba_block_size = max_seq_len
    return OptimumRBLNRuntimeConfig(
        max_num_seqs=batch_size,
        max_model_len=max_seq_len,
        block_size=block_size,
        attn_block_size=attn_block_size,
        mamba_block_size=mamba_block_size,
        prefill_chunk_size=prefill_chunk_size,
        image_prefill_chunk_size=image_prefill_chunk_size,
        model_num_blocks=resolve_runtime_num_blocks(vllm_config, rbln_config),
        num_devices_per_local_rank=num_devices,
        compiled_dtype=compiled_dtype,
        precompiled=precompiled,
    )


def apply_runtime_config(
    vllm_config: VllmConfig,
    runtime_config: OptimumRBLNRuntimeConfig,
) -> None:
    """Apply validated values and convert them to vLLM's execution units."""
    max_num_seqs = runtime_config.max_num_seqs
    max_model_len = runtime_config.max_model_len
    prefill_chunk_size = runtime_config.prefill_chunk_size
    compiled_dtype = runtime_config.compiled_dtype
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    if runtime_config.precompiled:
        additional_config.optimum_overrides = keep_device_overrides(
            additional_config.optimum_overrides
        )

    # Set max_num_seqs in scheduler_config based on rbln_config.json
    if vllm_config.scheduler_config.max_num_seqs != max_num_seqs:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            max_num_seqs,
        )
        vllm_config.scheduler_config.max_num_seqs = max_num_seqs

    # Set max_model_len in model_config based on rbln_config.json
    if vllm_config.model_config.max_model_len != max_model_len:
        logger.info(
            "Updating model_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            max_model_len,
        )
        vllm_config.model_config.max_model_len = max_model_len

    # Set max_num_batched_tokens: the prefill chunk size for decoder/multimodal
    # models, or a full-prefill-plus-batch budget for enc-dec/pooling models.
    update_max_num_batched_tokens(vllm_config, prefill_chunk_size)

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(vllm_config, runtime_config.image_prefill_chunk_size)
    # Set block_size in cache_config based on rbln_config.json.
    update_block_size(
        vllm_config,
        runtime_config.block_size,
        runtime_config.attn_block_size,
    )
    # The compiled model is the source of truth for dtype: vLLM must feed
    # inputs in exactly the dtype the artefact was compiled with.
    if compiled_dtype is not None and vllm_config.model_config.dtype != compiled_dtype:
        logger.info(
            "Updating model_config.dtype from %s to %s based on rbln_config.json",
            vllm_config.model_config.dtype,
            compiled_dtype,
        )
        vllm_config.model_config.dtype = compiled_dtype

    # Set num_blocks in cache_config based on rbln_config.json
    if runtime_config.precompiled:
        vllm_config.cache_config.mamba_block_size = runtime_config.mamba_block_size
        assert runtime_config.model_num_blocks is not None
        update_num_blocks(vllm_config, runtime_config.model_num_blocks)
    # The compiled model fixes num_devices too.
    additional_config.num_devices_per_local_rank = (
        runtime_config.num_devices_per_local_rank
    )


def resolve_runtime_num_blocks(
    vllm_config: VllmConfig, rbln_config: RblnConfigLike
) -> int | None:
    """Select the model block count before conversion to vLLM allocation units."""
    cache_config = vllm_config.cache_config
    additional_config = vllm_config.additional_config
    if cache_config.num_gpu_blocks_override is not None:
        return (
            additional_config.num_blocks_override
            if additional_config.num_blocks_synced
            else cache_config.num_gpu_blocks_override
        )
    return get_runtime_num_blocks(vllm_config.model_config.hf_config, rbln_config)


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


def update_block_size(
    vllm_config: VllmConfig,
    block_size: int,
    attn_block_size: int | None,
) -> None:
    """Apply the prepared vLLM block size and attention block size."""
    vllm_config.cache_config.user_specified_block_size = True
    if attn_block_size is not None:
        vllm_config.cache_config.block_size = block_size
        vllm_config.additional_config.attn_block_size = attn_block_size
        logger.debug("Sub-block size is set to %s for prefix caching", block_size)
    elif vllm_config.cache_config.block_size != block_size:
        logger.info(
            "Updating model_cache_config.block_size from %s to %s "
            "based on rbln_config.json",
            vllm_config.cache_config.block_size,
            block_size,
        )
        vllm_config.cache_config.block_size = block_size


def is_chunked_prefill_arch(hf_config) -> bool:
    """Whether ``max_num_batched_tokens`` is the prefill chunk size for this model.

    True for decoder and (non-enc-dec) multimodal models: they chunk prefill at
    ``max_num_batched_tokens`` granularity. Encoder-decoder and pooling models
    don't chunk prefill, so the value there is a full-prefill budget instead.
    """
    return not (is_enc_dec_arch(hf_config) or is_pooling_arch(hf_config))


def get_user_max_num_batched_tokens(vllm_config: VllmConfig) -> int | None:
    """Return the user's explicit ``max_num_batched_tokens`` (``None`` if unset).

    Stashed by ``optimum_impl._capture_user_max_num_batched_tokens`` before vLLM
    fills in its default.
    """
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    return additional_config.user_max_num_batched_tokens


def update_max_num_batched_tokens(
    vllm_config: VllmConfig, prefill_chunk_size: int | None
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
        assert prefill_chunk_size is not None
        target = prefill_chunk_size

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
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    additional_config.image_prefill_chunk_size = image_prefill_chunk_size


def validate_user_prefill_chunk_size(
    vllm_config: VllmConfig, prefill_chunk_size: int
) -> None:
    """Require an explicit user chunk size to match the selected model config."""
    if not is_chunked_prefill_arch(vllm_config.model_config.hf_config):
        return
    user_chunk = get_user_max_num_batched_tokens(vllm_config)
    if user_chunk is not None and user_chunk != prefill_chunk_size:
        raise ValueError(
            f"max_num_batched_tokens ({user_chunk}) conflicts with the "
            f"compiled prefill_chunk_size ({prefill_chunk_size}) in "
            "rbln_config.json. Omit max_num_batched_tokens or recompile the "
            "model with a matching prefill chunk size."
        )


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
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
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
