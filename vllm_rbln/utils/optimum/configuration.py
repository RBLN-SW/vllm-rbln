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

import hashlib
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.cache_blocks import (
    sync_cache_block_size,
    sync_num_blocks,
)
from vllm_rbln.utils.optimum.configuration_helper import (
    keep_only_device_keys,
    strip_runtime_only_keys,
)
from vllm_rbln.utils.optimum.rbln_params import (
    RBLNParams,
    get_rbln_config,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)

logger = init_logger(__name__)


def get_attn_block_size(vllm_config: VllmConfig) -> int:
    if vllm_config.cache_config.enable_prefix_caching:
        block_size = vllm_config.additional_config["attn_block_size"]
    else:
        block_size = vllm_config.cache_config.block_size
    return block_size


def generate_model_path_name(
    vllm_config: VllmConfig,
) -> str:
    # Just depends on user-provided parameters
    model_name = str(vllm_config.model_config.model)
    batch_size = vllm_config.scheduler_config.max_num_seqs
    block_size = vllm_config.cache_config.block_size
    max_model_len = vllm_config.model_config.max_model_len
    tp_size = envs.VLLM_RBLN_TP_SIZE
    additional_config = vllm_config.additional_config.get("rbln_config", None)

    # FIXME: To avoid cache collisions, the cache key should also include
    # the versions of the compiler and optimum-rbln.
    config_dict = {
        "model_name": model_name,
        "batch_size": batch_size,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "tp_size": tp_size,
    }
    if additional_config:
        config_dict["rbln_config"] = strip_runtime_only_keys(additional_config)

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


def is_qwen3_pooling(
    vllm_config: VllmConfig,
) -> bool:
    _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
    return (
        model_cls_name in ["RBLNQwen3ForCausalLM"]
        and vllm_config.model_config.runner_type == "pooling"
    )


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


def sync_vllm_from_rbln_config(
    vllm_config: VllmConfig,
    params: RBLNParams,
) -> None:
    assert params.num_blocks is not None, (
        "num_blocks must be specified in rbln_config.json"
    )
    assert params.batch_size is not None, (
        "batch_size must be specified in rbln_config.json"
    )
    assert params.max_seq_len is not None, (
        "max_seq_len must be specified in rbln_config.json"
    )
    assert params.kvcache_block_size is not None, (
        "kvcache_block_size must be specified in rbln_config.json"
    )

    if vllm_config.scheduler_config.max_num_seqs != params.batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = params.batch_size

    update_max_num_batched_tokens(vllm_config, params.max_seq_len)

    if vllm_config.model_config.max_model_len != params.max_seq_len:
        logger.info(
            "Updating model_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            params.max_seq_len,
        )
        vllm_config.model_config.max_model_len = params.max_seq_len

    # Set block_size in cache_config based on rbln_config.json
    sync_cache_block_size(
        vllm_config, params.kvcache_block_size, params.prefill_chunk_size
    )
    # Set num_blocks in cache_config based on rbln_config.json
    sync_num_blocks(vllm_config, params.num_blocks)
    envs.VLLM_RBLN_TP_SIZE = params.tensor_parallel_size


def prepare_vllm_for_compile(vllm_config: VllmConfig) -> None:
    hf_config = vllm_config.model_config.hf_config
    rbln_config = vllm_config.additional_config.get("rbln_config", {})
    # Extract block size from rbln_config
    params = RBLNParams.from_rbln_config(vllm_config, rbln_config)
    max_num_seqs = params.batch_size
    max_model_len = params.max_seq_len
    kvcache_block_size = params.kvcache_block_size
    if max_num_seqs is not None:
        logger.info(
            "Setting max_num_seqs to %d based on rbln_config in additional_config",
            max_num_seqs,
        )
        vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    if max_model_len is not None:
        logger.info(
            "Setting max_model_len to %d based on rbln_config in additional_config",
            max_model_len,
        )
        vllm_config.model_config.max_model_len = max_model_len
    if kvcache_block_size is not None:
        logger.info(
            "Setting block_size to %d based on rbln_config in additional_config",
            kvcache_block_size,
        )
        vllm_config.cache_config.block_size = kvcache_block_size
        vllm_config.cache_config.user_specified_block_size = kvcache_block_size

    if not vllm_config.cache_config.user_specified_block_size:
        # Set block_size to max_model_len
        # for decoder-only, multimodal, and pooling models
        vllm_config.cache_config.block_size = vllm_config.model_config.max_model_len
    else:
        if is_multi_modal(hf_config) or is_generation_arch(hf_config):
            assert vllm_config.cache_config.block_size >= 4096, (
                "block_size must be at least 4096 for compilation."
            )
        if is_pooling_arch(hf_config):
            assert (
                vllm_config.cache_config.block_size
                == vllm_config.model_config.max_model_len
            ), "For pooling models, block_size must be equal to max_model_len."

    # NOTE:
    # num_blocks is set after compilation,
    # so we only set other parameters here to compile model internally.
    # 1. block_size
    # Get proper block_size if not set by user
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

    update_max_num_batched_tokens(
        vllm_config, vllm_config.scheduler_config.max_num_batched_tokens
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

    if rbln_config is None:
        cached_model_path = os.path.join(
            envs.VLLM_CACHE_ROOT,
            "compiled_models",
            generate_model_path_name(vllm_config=vllm_config),
        )
        if os.path.exists(os.path.join(cached_model_path, "rbln_config.json")):
            logger.info("Found cached compiled model at %s", cached_model_path)
            vllm_config.model_config.model = cached_model_path
            rbln_config = get_rbln_config(vllm_config)
        else:
            vllm_config.additional_config["cached_model_path"] = cached_model_path

    # If the pre-compiled model exists, rbln_config is not None
    if rbln_config is not None:
        # NOTE: We can set the device to run submodules
        # Set only device setting using rbln_config
        vllm_config.additional_config["rbln_config"] = keep_only_device_keys(
            additional_rbln_config
        )
        params = RBLNParams.from_rbln_config(vllm_config, rbln_config)
        sync_vllm_from_rbln_config(vllm_config, params)
    else:
        prepare_vllm_for_compile(vllm_config)
