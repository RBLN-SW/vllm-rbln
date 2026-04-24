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

"""Top-level vLLM ↔ RBLN config synchronisation entry points."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.cache_blocks import (
    sync_cache_block_size,
    sync_num_blocks,
)
from vllm_rbln.utils.optimum.rbln_params import (
    get_rbln_config,
    get_rbln_params,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
)

logger = init_logger(__name__)


def get_invalid_leaf_keys(dict_rbln_config, prefix=""):
    """Return a list of leaf keys that are not 'device'."""
    target_key = "device"
    invalid_keys = []
    for key, value in dict_rbln_config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            invalid_keys.extend(get_invalid_leaf_keys(value, prefix=full_key))
        else:
            if key != target_key:
                invalid_keys.append(full_key)
    return invalid_keys


def is_qwen3_pooling(
    vllm_config: VllmConfig,
) -> bool:
    _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
    return (
        model_cls_name in ["RBLNQwen3ForCausalLM"]
        and vllm_config.model_config.runner_type == "pooling"
    )


def sync_vllm_from_rbln_config(
    vllm_config: VllmConfig,
    num_blocks: int,
    batch_size: int,
    max_model_len: int,
    kvcache_block_size: int,
    prefill_chunk_size: int,
) -> None:
    if vllm_config.scheduler_config.max_num_seqs != batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = batch_size

    # For encoder-decoder multimodal models (e.g. Whisper), max_num_batched_tokens
    # must be at least max_source_positions so that vllm's MultiModalBudget
    # validation passes (it requires max_tokens_per_mm_item <= max_num_batched_tokens
    # when chunked MM input is disabled).
    target_max_num_batched_tokens = max_model_len
    hf_config = vllm_config.model_config.hf_config
    if is_enc_dec_arch(hf_config):
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

    if vllm_config.model_config.max_model_len != max_model_len:
        logger.info(
            "Updating model_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            max_model_len,
        )
        vllm_config.model_config.max_model_len = max_model_len

    # Set block_size in cache_config based on rbln_config.json
    sync_cache_block_size(vllm_config, kvcache_block_size, prefill_chunk_size)
    # Set num_blocks in cache_config based on rbln_config.json
    sync_num_blocks(vllm_config, num_blocks)


def prepare_vllm_for_compile(vllm_config: VllmConfig) -> None:
    # NOTE:
    # num_blocks is set after compilation,
    # so we only set other parameters here to compile model internally.
    # 1. block_size
    # Get proper block_size if not set by user
    hf_config = vllm_config.model_config.hf_config
    if not vllm_config.cache_config.user_specified_block_size:
        # Set block_size to 4096 for fast compilation
        if is_multi_modal(hf_config) or is_generation_arch(hf_config):
            vllm_config.cache_config.block_size = 4096
        else:
            vllm_config.cache_config.block_size = vllm_config.model_config.max_model_len
    else:
        if is_multi_modal(hf_config) or is_generation_arch(hf_config):
            assert vllm_config.cache_config.block_size >= 4096, (
                "block_size must be at least 4096 for compilation"
            )

    # Set block_size in cache_config to compile model internally.
    sync_cache_block_size(
        vllm_config, vllm_config.cache_config.block_size, prefill_chunk_size=128
    )

    # 2. max_model_len
    # NOTE: Uses the user-defined max_model_len if provided;
    # otherwise, it defaults to the model's native maximum length.
    # Note that using the default value may significantly increase compilation time.
    vllm_config.scheduler_config.max_num_batched_tokens = max(
        vllm_config.model_config.max_model_len,
        vllm_config.scheduler_config.max_num_seqs,
    )

    logger.info(
        "Prepared vLLM config for compilation: %s",
        vllm_config,
    )


def sync_with_rbln_config(vllm_config: VllmConfig) -> None:
    """
    If compiled model with RBLN config is given,
    synchronise vLLM config with RBLN config.
    If no RBLN config is given, validate vLLM config and set necessary parameters
    to default values to compile model internally.
    """
    try:
        rbln_config = get_rbln_config(vllm_config)
    except Exception as e:
        raise RuntimeError("Failed to get RBLN config: %s", e) from e

    additional_rbln_config = vllm_config.additional_config.get("rbln_config", {})
    # If the pre-compiled model exists, rbln_config is not None
    if rbln_config is not None:
        invalid_keys = get_invalid_leaf_keys(additional_rbln_config)
        if invalid_keys:
            raise RuntimeError(
                "For now, we only support 'device' as a configurable key "
                "in rbln_config passed through additional_config "
                "for pre-compiled optimum models. "
                f"Got unsupported keys: {invalid_keys}"
            )

        (
            num_blocks,
            batch_size,
            max_model_len,
            kvcache_block_size,
            prefill_chunk_size,
        ) = get_rbln_params(vllm_config, rbln_config)
        sync_vllm_from_rbln_config(
            vllm_config,
            num_blocks,
            batch_size,
            max_model_len,
            kvcache_block_size,
            prefill_chunk_size,
        )
    else:
        assert len(additional_rbln_config) == 0, (
            "For now, we don't support passing rbln_config "
            "through additional_config for compilation yet."
        )
        prepare_vllm_for_compile(vllm_config)
