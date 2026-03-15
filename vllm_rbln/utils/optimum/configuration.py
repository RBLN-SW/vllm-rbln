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

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_multi_modal,
    is_pooling_arch,
)

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


def get_rbln_params(
    vllm_config: VllmConfig, rbln_config: dict
) -> tuple[int, int, int, int, int]:
    kvcache_block_size = None
    prefill_chunk_size = 128
    batch_size = None
    max_seq_len = None

    if is_enc_dec_arch(vllm_config.model_config.hf_config):
        max_seq_len = rbln_config.get("dec_max_seq_len")
        kvcache_block_size = max_seq_len
        batch_size = rbln_config.get("batch_size")
        num_blocks = rbln_config.get("kvcache_num_blocks")
    elif is_multi_modal(vllm_config.model_config.hf_config):
        # Get configurations from main module (e.g. Qwen2.5-VL, Whisper)
        kvcache_block_size = rbln_config.get("kvcache_block_size")
        batch_size = rbln_config.get("batch_size")
        max_seq_len = rbln_config.get("max_seq_len")
        num_blocks = rbln_config.get("kvcache_num_blocks")
        if max_seq_len is None:  # Whisper FIXME to be moved to enc-dec
            max_seq_len = rbln_config.get("dec_max_seq_len")
        # Get configurations from submodule
        if kvcache_block_size is None:
            submodules = ["language_model", "text_model"]
            for submodule in submodules:
                if submodule in rbln_config:
                    kvcache_block_size = rbln_config[submodule].get(
                        "kvcache_block_size", None
                    )
                    batch_size = rbln_config[submodule].get("batch_size", None)
                    max_seq_len = rbln_config[submodule].get("max_seq_len", None)
                    num_blocks = rbln_config[submodule].get("kvcache_num_blocks", None)
                    if kvcache_block_size is not None:
                        break

    elif is_pooling_arch(vllm_config.model_config.hf_config):
        max_seq_len = rbln_config.get("max_seq_len")
        kvcache_block_size = max_seq_len
        batch_size = rbln_config.get("batch_size")
        num_blocks = rbln_config.get("kvcache_num_blocks")
        if num_blocks is None:
            num_blocks = batch_size  # for pooling models, each sequence is one block
    else:
        # decoder
        kvcache_block_size = rbln_config.get("kvcache_block_size")
        prefill_chunk_size = rbln_config.get("prefill_chunk_size", 128)
        batch_size = rbln_config.get("batch_size")
        max_seq_len = rbln_config.get("max_seq_len")
        num_blocks = rbln_config.get("kvcache_num_blocks")

    assert num_blocks is not None, "num_blocks must be specified in rbln_config.json"

    assert kvcache_block_size is not None, (
        "kvcache_block_size must be specified in rbln_config.json"
    )
    assert batch_size is not None, "batch_size must be specified in rbln_config.json"
    assert max_seq_len is not None, "max_seq_len must be specified in rbln_config.json"
    # NOTE:
    # prefill_chunk_size is only used for decoder-only models
    # with prefix caching
    return num_blocks, batch_size, max_seq_len, kvcache_block_size, prefill_chunk_size


def set_block_size_for_prefix_caching(
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
                "prefix_block_size (%s) is not divisible "
                "by prefill_chunk_size (%s)."
                "Please check the value of prefill_chunk_size "
                "in rbln_config.json"
            )
        if prefix_block_size > kvcache_block_size:
            raise ValueError(
                "prefix_block_size (%s) is greater than "
                "kvcache_block_size (%s)."
                "Please check the value of kvcache_block_size "
                "in rbln_config.json"
            )
        logger.debug(
            "Prefix block size is set to %s based on additional_config",
            prefix_block_size,
        )
    if kvcache_block_size % prefix_block_size != 0:
        raise ValueError(
            "kvcache_block_size (%s) is not divisible "
            "by prefix_block_size (%s)."
            "Please check the value of prefix_block_size in rbln_config.json"
        )
    vllm_config.cache_config.block_size = prefix_block_size
    vllm_config.additional_config["attn_block_size"] = kvcache_block_size


def update_vllm_config_with_rbln_params(
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

    if vllm_config.scheduler_config.max_num_batched_tokens != (max_model_len):
        logger.info(
            "Updating scheduler_config.max_num_batched_tokens from %s to "
            "%d based on rbln_config.json",
            vllm_config.scheduler_config.max_num_batched_tokens,
            max_model_len,
        )
        vllm_config.scheduler_config.max_num_batched_tokens = max_model_len

    if vllm_config.model_config.max_model_len != max_model_len:
        logger.info(
            "Updating model_config.max_model_len and "
            "scheduler_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            max_model_len,
        )
        vllm_config.model_config.max_model_len = max_model_len
        vllm_config.scheduler_config.max_model_len = max_model_len

    if vllm_config.cache_config.enable_prefix_caching:
        set_block_size_for_prefix_caching(
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


def is_qwen3_pooling(
    vllm_config: VllmConfig,
) -> bool:
    _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
    return (
        model_cls_name in ["RBLNQwen3ForCausalLM"]
        and vllm_config.model_config.runner_type == "pooling"
    )


def get_rbln_config(vllm_config: VllmConfig) -> dict | None:
    rbln_config_path = Path(
        os.path.join(vllm_config.model_config.model, "rbln_config.json")
    )
    if not rbln_config_path.exists():  # for pytest
        logger.warning(
            "rbln_config.json not found in model directory: %s. "
            "Using `block_size` from vllm_config.cache_config instead.",
            rbln_config_path,
        )
        return None
    with open(rbln_config_path, encoding="utf-8") as f:
        rbln_config = json.load(f)
    return rbln_config


def sync_with_rbln_config(vllm_config: VllmConfig) -> None:
    try:
        rbln_config = get_rbln_config(vllm_config)
    except Exception as e:
        raise RuntimeError("Failed to get RBLN config: %s", e) from e

    if rbln_config is not None:
        (
            num_blocks,
            batch_size,
            max_model_len,
            kvcache_block_size,
            prefill_chunk_size,
        ) = get_rbln_params(vllm_config, rbln_config)
        update_vllm_config_with_rbln_params(
            vllm_config,
            num_blocks,
            batch_size,
            max_model_len,
            kvcache_block_size,
            prefill_chunk_size,
        )
