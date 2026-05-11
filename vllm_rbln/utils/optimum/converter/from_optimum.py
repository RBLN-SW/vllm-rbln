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

from typing import TYPE_CHECKING, Any

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.block_size import (
    get_block_ratio,
    is_full_block_available,
)

from .common import update_block_size, update_max_num_batched_tokens
from .params import RBLNParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def _keep_only_device_keys(obj: dict) -> dict:
    """Recursively keep only ``devices`` entries from nested dict/list."""
    result: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "devices":
            result[k] = v
        elif isinstance(v, dict):
            filtered = _keep_only_device_keys(v)
            if filtered:
                result[k] = filtered
    return result


def sync_from_optimum(
    vllm_config: VllmConfig,
    compiled_rbln_config: dict,
) -> None:
    """
    optimum -> vLLM config synchronization
    """

    params = RBLNParams.from_rbln_config(vllm_config, compiled_rbln_config)

    # The compiled artefact is the source of truth. Strip the user's
    # additional_config down to device-only keys so submodule placement
    # can still be overridden, but no other parameter sneaks in.
    vllm_config.additional_config["rbln_config"] = _keep_only_device_keys(
        vllm_config.additional_config.get("rbln_config", {})
    )

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

    # Set max_num_seqs in scheduler_config based on rbln_config.json
    if vllm_config.scheduler_config.max_num_seqs != params.batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = params.batch_size

    # In case of encoder-decoder models,
    # update max_num_seqs in encoder_scheduler_config as well
    vllm_config.scheduler_config.max_num_batched_tokens = max(
        vllm_config.model_config.max_model_len,
        vllm_config.scheduler_config.max_num_seqs,
    )
    update_max_num_batched_tokens(vllm_config, params.max_seq_len)

    # Set max_model_len in model_config based on rbln_config.json
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
    update_block_size(vllm_config, params.kvcache_block_size, params.prefill_chunk_size)
    # Set num_blocks in cache_config based on rbln_config.json
    update_num_blocks(vllm_config, params.num_blocks)
    # Sync tensor_parallel_size in envs with optimum pre-compiled model
    envs.VLLM_RBLN_TP_SIZE = params.tensor_parallel_size


def update_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> None:
    # This function is called twice during startup: once in the main process
    # when we first read rbln_config.json, and once in each subprocess when
    # the VllmConfig is deserialized. We only want to perform the num_blocks
    # calculation and update once, in the main process, to avoid redundant
    # calculations and potential inconsistencies. We use an additional_config
    # flag to track whether we have already synced num_blocks.
    if vllm_config.additional_config.get("num_blocks_synced", False):
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
        vllm_config.additional_config["num_blocks_override"] = num_blocks
    blk_ratio = get_block_ratio(vllm_config)

    if is_full_block_available(num_blocks, vllm_config):
        adjusted_num_blocks = num_blocks * blk_ratio + 1
    else:
        adjusted_num_blocks = (num_blocks - 1) * blk_ratio + 1

    vllm_config.cache_config.num_gpu_blocks = adjusted_num_blocks

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        vllm_config.cache_config.num_gpu_blocks_override = adjusted_num_blocks
    vllm_config.additional_config["num_blocks_synced"] = True
