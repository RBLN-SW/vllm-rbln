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

"""Apply validated runtime settings to vLLM."""

from typing import TYPE_CHECKING

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.configuration.execution import (
    keep_device_overrides,
    store_image_prefill_chunk_size,
    update_block_size,
    update_mamba_block_size,
    update_max_num_batched_tokens,
    update_num_blocks,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNConfig, OptimumRBLNRuntimeConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def apply_runtime_config(
    vllm_config: VllmConfig,
    runtime_config: "OptimumRBLNRuntimeConfig",
) -> None:
    """Apply validated values and convert them to vLLM's execution units."""
    batch_size = runtime_config.batch_size
    max_seq_len = runtime_config.max_seq_len
    prefill_chunk_size = runtime_config.prefill_chunk_size
    compiled_dtype = runtime_config.dtype
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    cache_config = vllm_config.cache_config
    utilization = runtime_config.gpu_memory_utilization
    if cache_config.gpu_memory_utilization != utilization:
        logger.info(
            "Setting gpu_memory_utilization from %s to %s because "
            "an explicit KV cache block count=%d is set.",
            cache_config.gpu_memory_utilization,
            utilization,
            runtime_config.explicit_num_blocks,
        )
        cache_config.gpu_memory_utilization = utilization
    if runtime_config.precompiled:
        additional_config.optimum_overrides = keep_device_overrides(
            additional_config.optimum_overrides
        )

    # Set max_num_seqs in scheduler_config based on rbln_config.json
    if vllm_config.scheduler_config.max_num_seqs != batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = batch_size

    # Set max_model_len in model_config based on rbln_config.json
    if vllm_config.model_config.max_model_len != max_seq_len:
        logger.info(
            "Updating model_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            max_seq_len,
        )
        vllm_config.model_config.max_model_len = max_seq_len

    # Set max_num_batched_tokens: the prefill chunk size for decoder/multimodal
    # models, or a full-prefill-plus-batch budget for enc-dec/pooling models.
    update_max_num_batched_tokens(vllm_config, prefill_chunk_size)

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(vllm_config, runtime_config.image_prefill_chunk_size)
    # Set block_size in cache_config based on rbln_config.json.
    update_block_size(
        vllm_config,
        runtime_config.kvcache_block_size,
        prefill_chunk_size,
        runtime_config.image_prefill_chunk_size,
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
        update_mamba_block_size(vllm_config, max_seq_len)
        assert runtime_config.num_blocks is not None
        update_num_blocks(vllm_config, runtime_config.num_blocks)
    # The compiled model fixes num_devices too.
    additional_config.num_devices_per_local_rank = runtime_config.num_devices
