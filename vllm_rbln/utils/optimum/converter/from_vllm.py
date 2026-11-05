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

from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_pooling_arch,
)

from .common import (
    apply_user_prefill_chunk_size,
    store_image_prefill_chunk_size,
    update_block_size,
    update_max_num_batched_tokens,
)
from .params import RBLNParams, _resolve_num_devices

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def sync_from_vllm(vllm_config: VllmConfig) -> RBLNParams:
    """Resolve compile inputs and synchronize block sizes."""
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    rbln_overrides = rbln_config.optimum_overrides
    params = RBLNParams.from_rbln_config(vllm_config, rbln_overrides)

    if params.dtype is not None:
        raise ValueError(
            "`dtype` cannot be set in `additional_config`'s `optimum_overrides`. "
            "optimum-rbln takes the compile dtype as a load argument, not as "
            "an rbln_config field. Use vLLM's `dtype` argument instead "
            "(e.g. `LLM(dtype=...)` or `--dtype`)."
        )

    if params.batch_size is not None:
        logger.info(
            "Setting max_num_seqs to %d based on additional_config.optimum_overrides",
            params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = params.batch_size
    if params.max_seq_len is not None:
        logger.info(
            "Setting max_model_len to %d based on additional_config.optimum_overrides",
            params.max_seq_len,
        )
        vllm_config.model_config.max_model_len = params.max_seq_len
    params.num_devices = _resolve_num_devices(
        rbln_overrides, default=rbln_config.num_devices_per_local_rank
    )
    rbln_config.num_devices_per_local_rank = params.num_devices

    if params.kvcache_block_size is not None:
        logger.info(
            "Setting block_size to %d based on additional_config.optimum_overrides",
            params.kvcache_block_size,
        )
        vllm_config.cache_config.block_size = params.kvcache_block_size
        vllm_config.cache_config.user_specified_block_size = True

    # Enc-dec and pooling models usually
    # don't use paged KV cache (except for Qwen3 models),
    # so block_size has no real effect
    # — default it to max_seq_len so users aren't forced
    # to specify it explicitly.
    if not vllm_config.cache_config.user_specified_block_size:
        hf_config = vllm_config.model_config.hf_config
        if is_enc_dec_arch(hf_config) or is_pooling_arch(hf_config):
            vllm_config.cache_config.block_size = vllm_config.model_config.max_model_len
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

    if params.kvcache_block_size is None:
        params.kvcache_block_size = (
            rbln_config.attn_block_size
            if vllm_config.cache_config.enable_prefix_caching
            and rbln_config.attn_block_size is not None
            else vllm_config.cache_config.block_size
        )
    assert params.kvcache_block_size is not None

    apply_user_prefill_chunk_size(
        vllm_config,
        params,
        precompiled=False,
        override_prefill_chunk_size=rbln_overrides.get("prefill_chunk_size"),
    )

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(vllm_config, params.image_prefill_chunk_size)
    update_block_size(
        vllm_config,
        params.kvcache_block_size,
        prefill_chunk_size=params.prefill_chunk_size,
        image_prefill_chunk_size=params.image_prefill_chunk_size,
    )

    update_max_num_batched_tokens(vllm_config, params)

    params.batch_size = vllm_config.scheduler_config.max_num_seqs
    params.max_seq_len = vllm_config.model_config.max_model_len
    return params
