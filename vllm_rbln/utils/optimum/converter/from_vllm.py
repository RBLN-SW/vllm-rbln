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
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_pooling_arch,
)

from .common import (
    get_user_max_num_batched_tokens,
    is_chunked_prefill_arch,
    store_image_prefill_chunk_size,
    update_block_size,
    update_max_num_batched_tokens,
)
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
    # In the compile path an explicit max_num_batched_tokens is the user's
    # prefill chunk size. Fold it into params so block sizing, the compile pin,
    # and max_num_batched_tokens all agree on one value. Enc-dec/pooling models
    # don't chunk prefill, so the value there is a full-prefill budget, not a
    # chunk size — leave params untouched for them.
    user_mnbt = get_user_max_num_batched_tokens(vllm_config)
    if user_mnbt is not None and is_chunked_prefill_arch(
        vllm_config.model_config.hf_config
    ):
        override_pcs = rbln_overrides.get("prefill_chunk_size")
        if override_pcs is not None and override_pcs != user_mnbt:
            raise ValueError(
                f"Conflicting prefill chunk size: max_num_batched_tokens "
                f"({user_mnbt}) != rbln_config['prefill_chunk_size'] "
                f"({override_pcs}). Set only one."
            )
        params.prefill_chunk_size = user_mnbt

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(vllm_config, params.image_prefill_chunk_size)
    update_block_size(
        vllm_config,
        vllm_config.cache_config.block_size,
        prefill_chunk_size=params.prefill_chunk_size,
        image_prefill_chunk_size=params.image_prefill_chunk_size,
    )

    # Set max_num_batched_tokens: the prefill chunk size for decoder/multimodal
    # models, or a full-prefill-plus-batch budget for enc-dec/pooling models.
    print(
        "@@@ before prefill_chunk_size",
        vllm_config.scheduler_config.max_num_batched_tokens,
    )
    update_max_num_batched_tokens(vllm_config, params)
    print(
        "@@@ after prefill_chunk_size",
        vllm_config.scheduler_config.max_num_batched_tokens,
    )
