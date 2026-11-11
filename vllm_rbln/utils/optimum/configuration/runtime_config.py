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

"""Prepare validated vLLM execution settings from the selected model config."""

from typing import TYPE_CHECKING

import torch

from vllm_rbln.config import OptimumRBLNRuntimeConfig
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_encoder_pooling_arch,
    is_multi_modal,
)

from .compile_config import resolve_explicit_num_blocks
from .execution import (
    get_runtime_num_blocks,
    is_chunked_prefill_arch,
    validate_user_prefill_chunk_size,
)
from .readers import (
    RblnConfigLike,
    get_field,
    read_kvcache_block_size,
    read_language_model_fields,
    read_num_devices,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


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

    explicit_num_blocks = resolve_explicit_num_blocks(vllm_config)
    return OptimumRBLNRuntimeConfig(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        kvcache_block_size=kvcache_block_size,
        prefill_chunk_size=prefill_chunk_size,
        image_prefill_chunk_size=image_prefill_chunk_size,
        num_blocks=num_blocks,
        num_devices=num_devices,
        dtype=compiled_dtype,
        gpu_memory_utilization=(
            1.0
            if explicit_num_blocks is not None
            else vllm_config.cache_config.gpu_memory_utilization
        ),
        explicit_num_blocks=explicit_num_blocks,
        precompiled=precompiled,
    )
