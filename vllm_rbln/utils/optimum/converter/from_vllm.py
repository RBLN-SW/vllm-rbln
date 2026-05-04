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

from .common import update_block_size, update_max_num_batched_tokens
from .params import RBLNParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def sync_from_vllm(vllm_config: VllmConfig) -> None:
    """
    vllm_config.additional_config["rbln_config"] -> optimum
    1. Parse RBLNParams from vllm_config.additional_config["rbln_config"].
    2. Update vllm_config based on the parsed RBLNParams
    to ensure consistency between vLLM and RBLN configurations.
    3. Validate the updated block size
    """
    rbln_overrides = vllm_config.additional_config.get("rbln_config", {})
    params = RBLNParams.from_rbln_config(vllm_config, rbln_overrides)

    if params.batch_size is not None:
        logger.info(
            "Setting max_num_seqs to %d based on rbln_config in additional_config",
            params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = params.batch_size
    if params.max_seq_len is not None:
        logger.info(
            "Setting max_model_len to %d based on rbln_config in additional_config",
            params.max_seq_len,
        )
        vllm_config.model_config.max_model_len = params.max_seq_len
    if params.kvcache_block_size is not None:
        logger.info(
            "Setting block_size to %d based on rbln_config in additional_config",
            params.kvcache_block_size,
        )
        vllm_config.cache_config.block_size = params.kvcache_block_size
        vllm_config.cache_config.user_specified_block_size = True

    if not vllm_config.cache_config.user_specified_block_size:
        raise ValueError(
            "`block_size` is required to run optimum-rbln models in vLLM RBLN.\n"
            "Set it via one of:\n"
            "  1) vLLM's `block_size` argument "
            "(e.g. `LLM(block_size=...)` or `--block-size`), or\n"
            "  2) `kvcache_block_size` under "
            "`additional_config={'rbln_config': {...}}`.\n"
        )

    update_block_size(
        vllm_config,
        vllm_config.cache_config.block_size,
        prefill_chunk_size=params.prefill_chunk_size,
    )

    # max_num_batched_tokens must fit both a full-length prefill and a full
    # batch dispatch; update_max_num_batched_tokens layers the enc-dec
    # max_source_positions constraint on top.
    vllm_config.scheduler_config.max_num_batched_tokens = max(
        vllm_config.model_config.max_model_len,
        vllm_config.scheduler_config.max_num_seqs,
    )
    update_max_num_batched_tokens(
        vllm_config, vllm_config.scheduler_config.max_num_batched_tokens
    )
