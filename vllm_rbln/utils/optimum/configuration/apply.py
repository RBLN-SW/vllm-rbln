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

"""Read, validate and apply the selected model settings to vLLM."""

from typing import TYPE_CHECKING

import torch

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.configuration.readers import (
    RblnConfigLike,
    get_field,
    read_kvcache_block_size,
    read_language_model_fields,
    read_num_devices,
)
from vllm_rbln.utils.optimum.configuration.runtime import (
    get_runtime_num_blocks,
    is_chunked_prefill_arch,
    keep_device_overrides,
    store_image_prefill_chunk_size,
    update_block_size,
    update_mamba_block_size,
    update_max_num_batched_tokens,
    update_num_blocks,
    validate_user_prefill_chunk_size,
)
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_encoder_pooling_arch,
    is_multi_modal,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def apply_to_vllm_config(
    vllm_config: VllmConfig,
    rbln_config: RblnConfigLike,
    *,
    precompiled: bool,
) -> None:
    """Read and validate the selected settings, then apply them to vllm_config."""
    # 1. Read and validate rbln_config without changing vllm_config.
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
            # The encoder's full sequence is one vLLM block, not a KV-cache block.
            kvcache_block_size = max_seq_len
            prefill_chunk_size = None
        else:
            kvcache_block_size = read_kvcache_block_size(rbln_config)
            prefill_chunk_size = get_field(rbln_config, "prefill_chunk_size")
    num_blocks = get_runtime_num_blocks(hf_config, rbln_config)
    num_devices = read_num_devices(rbln_config)
    dtype = get_field(rbln_config, "dtype")
    compiled_dtype = (
        getattr(torch, dtype) if precompiled and dtype is not None else None
    )

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

    # 2. Apply the selected values and convert them to vLLM's execution units.
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    if precompiled:
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
    store_image_prefill_chunk_size(vllm_config, image_prefill_chunk_size)
    # Set block_size in cache_config based on rbln_config.json.
    update_block_size(
        vllm_config,
        kvcache_block_size,
        prefill_chunk_size,
        image_prefill_chunk_size,
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
    if precompiled:
        update_mamba_block_size(vllm_config, max_seq_len)
        assert num_blocks is not None
        update_num_blocks(vllm_config, num_blocks)
    # The compiled model fixes num_devices too.
    additional_config.num_devices_per_local_rank = num_devices
