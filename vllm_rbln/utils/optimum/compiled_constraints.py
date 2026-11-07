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

"""Read compiled constraints from rbln_config and apply them to vLLM."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.block_size import get_block_ratio, is_full_block_available
from vllm_rbln.utils.optimum.compile_spec import keep_device_overrides
from vllm_rbln.utils.optimum.rbln_config_fields import (
    RblnConfigLike,
    default_prefill_chunk_size,
    get_field,
    language_model_configs,
    resolve_image_prefill_chunk_size,
    resolve_kvcache_block_size,
    resolve_num_devices,
)
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_multi_modal,
    is_pooling_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


@dataclass
class CompiledConstraints:
    """Values the compiled graph fixes and vLLM must follow.

    Read from rbln_config.json on a cache hit, or from the compile spec that
    is about to be compiled on a cache miss.
    """

    num_blocks: int | None = None
    batch_size: int | None = None
    max_seq_len: int | None = None
    kvcache_block_size: int | None = None
    prefill_chunk_size: int = 128
    num_devices: int = 1
    dtype: str | None = None
    # Image-prefill buckets for multimodal models (gemma3: single value;
    # gemma4: descending list of 128-multiples). None for non-multimodal models.
    image_prefill_chunk_size: list[int] | None = None

    @classmethod
    def from_rbln_config(
        cls, hf_config: Any, rbln_config: RblnConfigLike
    ) -> "CompiledConstraints":
        """Parse rbln_config according to the model architecture."""
        if is_enc_dec_arch(hf_config):
            constraints = cls._parse_enc_dec(rbln_config)
        elif is_multi_modal(hf_config):
            constraints = cls._parse_multimodal(rbln_config)
        elif is_pooling_arch(hf_config):
            constraints = cls._parse_pooling(rbln_config)
        else:
            constraints = cls._parse_decoder(rbln_config)

        constraints.num_devices = resolve_num_devices(rbln_config)
        constraints.dtype = get_field(rbln_config, "dtype")
        return constraints

    @classmethod
    def _parse_enc_dec(cls, cfg: RblnConfigLike) -> "CompiledConstraints":
        max_seq_len = get_field(cfg, "dec_max_seq_len")
        batch_size = get_field(cfg, "batch_size")
        num_blocks = get_field(cfg, "kvcache_num_blocks")
        if num_blocks is None and batch_size is not None:
            num_blocks = batch_size
        return cls(
            num_blocks=num_blocks,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_block_size=max_seq_len,
        )

    @classmethod
    def _parse_pooling(cls, cfg: RblnConfigLike) -> "CompiledConstraints":
        max_seq_len = get_field(cfg, "max_seq_len")
        batch_size = get_field(cfg, "batch_size")
        # For pooling models each sequence occupies exactly one block.
        num_blocks = get_field(cfg, "kvcache_num_blocks")
        kvcache_block_size = resolve_kvcache_block_size(cfg, arch="pooling")
        if kvcache_block_size is None and max_seq_len is not None:
            kvcache_block_size = max_seq_len
        if num_blocks is None and batch_size is not None:
            num_blocks = batch_size
        return cls(
            num_blocks=num_blocks,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_block_size=kvcache_block_size,
        )

    @classmethod
    def _parse_decoder(cls, cfg: RblnConfigLike) -> "CompiledConstraints":
        kvcache_block_size = resolve_kvcache_block_size(cfg, arch="decoder")
        default_chunk = default_prefill_chunk_size()
        return cls(
            num_blocks=get_field(cfg, "kvcache_num_blocks"),
            batch_size=get_field(cfg, "batch_size"),
            max_seq_len=get_field(cfg, "max_seq_len"),
            kvcache_block_size=kvcache_block_size,
            prefill_chunk_size=get_field(cfg, "prefill_chunk_size", default_chunk),
        )

    @classmethod
    def _parse_multimodal(cls, cfg: RblnConfigLike) -> "CompiledConstraints":
        sources = language_model_configs(cfg)
        values: dict[str, Any] = {}
        for source in sources:
            for key in (
                "batch_size",
                "max_seq_len",
                "kvcache_num_blocks",
                "prefill_chunk_size",
            ):
                value = get_field(source, key)
                if value is not None:
                    values.setdefault(key, value)
            block_size = resolve_kvcache_block_size(source, arch="multi-modal")
            if block_size is not None:
                values.setdefault("kvcache_block_size", block_size)
            image_size = resolve_image_prefill_chunk_size(source)
            if image_size is not None:
                values.setdefault("image_prefill_chunk_size", image_size)
        kvcache_block_size = values.get("kvcache_block_size")
        batch_size = values.get("batch_size")
        max_seq_len = values.get("max_seq_len")
        num_blocks = values.get("kvcache_num_blocks")
        prefill_chunk_size = values.get("prefill_chunk_size")
        if prefill_chunk_size is None:
            prefill_chunk_size = default_prefill_chunk_size()
        image_prefill_chunk_size = values.get("image_prefill_chunk_size")

        return cls(
            num_blocks=num_blocks,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_block_size=kvcache_block_size,
            prefill_chunk_size=prefill_chunk_size,
            image_prefill_chunk_size=image_prefill_chunk_size,
        )


def _apply_prefix_caching_block_size(
    vllm_config: VllmConfig, kvcache_block_size: int, prefill_chunk_size: int
) -> None:
    assert prefill_chunk_size is not None, (
        "prefill_chunk_size must be specified in rbln_config.json"
    )
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
    vllm_config: VllmConfig, constraints: "CompiledConstraints"
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
        target = constraints.prefill_chunk_size

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
    """Require an explicit user chunk size to match the selected model constraints."""
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


def apply_compiled_constraints(
    vllm_config: VllmConfig,
    constraints: CompiledConstraints,
    *,
    precompiled: bool,
) -> None:
    """Apply compiled constraints to the vLLM config."""
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    assert constraints.batch_size is not None
    assert constraints.max_seq_len is not None
    if precompiled:
        assert constraints.num_blocks is not None, (
            "num_blocks must be specified in rbln_config.json"
        )
        additional_config.optimum_overrides = keep_device_overrides(
            additional_config.optimum_overrides
        )

    if not precompiled and constraints.kvcache_block_size is None:
        constraints.kvcache_block_size = constraints.max_seq_len
    assert constraints.kvcache_block_size is not None

    # Set max_num_seqs in scheduler_config based on rbln_config.json
    if vllm_config.scheduler_config.max_num_seqs != constraints.batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            constraints.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = constraints.batch_size

    # Set max_model_len in model_config based on rbln_config.json
    if vllm_config.model_config.max_model_len != constraints.max_seq_len:
        logger.info(
            "Updating model_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            constraints.max_seq_len,
        )
        vllm_config.model_config.max_model_len = constraints.max_seq_len

    validate_user_prefill_chunk_size(vllm_config, constraints.prefill_chunk_size)
    # Set max_num_batched_tokens: the prefill chunk size for decoder/multimodal
    # models, or a full-prefill-plus-batch budget for enc-dec/pooling models.
    update_max_num_batched_tokens(vllm_config, constraints)

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(vllm_config, constraints.image_prefill_chunk_size)
    # Set block_size in cache_config based on rbln_config.json.
    update_block_size(
        vllm_config,
        constraints.kvcache_block_size,
        constraints.prefill_chunk_size,
        constraints.image_prefill_chunk_size,
    )
    # The compiled model is the source of truth for dtype: vLLM must feed
    # inputs in exactly the dtype the artefact was compiled with.
    if precompiled and constraints.dtype is not None:
        compiled_dtype = getattr(torch, constraints.dtype)
        if vllm_config.model_config.dtype != compiled_dtype:
            logger.info(
                "Updating model_config.dtype from %s to %s based on rbln_config.json",
                vllm_config.model_config.dtype,
                compiled_dtype,
            )
            vllm_config.model_config.dtype = compiled_dtype

    # Set num_blocks in cache_config based on rbln_config.json
    if precompiled:
        update_mamba_block_size(vllm_config, constraints)
        assert constraints.num_blocks is not None
        update_num_blocks(vllm_config, constraints.num_blocks)
    # The compiled model fixes num_devices too.
    additional_config.num_devices_per_local_rank = constraints.num_devices


def update_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> None:
    # This function is called twice during startup: once in the main process
    # when we first read rbln_config.json, and once in each subprocess when
    # the VllmConfig is deserialized. We only want to perform the num_blocks
    # calculation and update once, in the main process, to avoid redundant
    # calculations and potential inconsistencies. We use an additional_config
    # flag to track whether we have already synced num_blocks.
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


def update_mamba_block_size(
    vllm_config: VllmConfig, constraints: "CompiledConstraints"
) -> None:
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
        cache_config.mamba_block_size = constraints.max_seq_len
