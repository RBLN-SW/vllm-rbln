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
    is_generation_arch,
    is_multi_modal,
)

logger = init_logger(__name__)


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
    if vllm_config.cache_config.block_size is None:
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
    vllm_config.scheduler_config.max_num_batched_tokens = (
        vllm_config.model_config.max_model_len
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

    if rbln_config is not None:
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
        prepare_vllm_for_compile(vllm_config)
