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

from copy import deepcopy
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


def sync_from_vllm(vllm_config: VllmConfig) -> None:
    """Resolve the compile spec and synchronize runtime settings from it."""
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    if rbln_config.compile_spec is None:
        from vllm_rbln.platform import RblnPlatform
        from vllm_rbln.utils.optimum.compilation import RBLNCompileSpec

        # 1. Resolve user overrides and defaults into compile inputs.
        rbln_overrides = rbln_config.optimum_overrides
        input_params = RBLNParams.from_rbln_config(vllm_config, rbln_overrides)

        if input_params.dtype is not None:
            raise ValueError(
                "`dtype` cannot be set in `additional_config`'s `optimum_overrides`. "
                "optimum-rbln takes the compile dtype as a load argument, not as "
                "an rbln_config field. Use vLLM's `dtype` argument instead "
                "(e.g. `LLM(dtype=...)` or `--dtype`)."
            )

        if input_params.batch_size is None:
            input_params.batch_size = vllm_config.scheduler_config.max_num_seqs
        if input_params.max_seq_len is None:
            input_params.max_seq_len = vllm_config.model_config.max_model_len
        input_params.num_devices = _resolve_num_devices(
            rbln_overrides, default=rbln_config.num_devices_per_local_rank
        )

        if input_params.kvcache_block_size is None:
            # Enc-dec and pooling models usually
            # don't use paged KV cache (except for Qwen3 models),
            # so block_size has no real effect
            # — default it to max_seq_len so users aren't forced
            # to specify it explicitly.
            hf_config = vllm_config.model_config.hf_config
            if not vllm_config.cache_config.user_specified_block_size:
                if is_enc_dec_arch(hf_config) or is_pooling_arch(hf_config):
                    input_params.kvcache_block_size = input_params.max_seq_len
                else:
                    raise ValueError(
                        "`block_size` is required to run optimum-rbln models "
                        "in vLLM RBLN.\n"
                        "Set it via one of:\n"
                        "  1) vLLM's `block_size` argument "
                        "(e.g. `LLM(block_size=...)` or `--block-size`), or\n"
                        "  2) `kvcache_block_size` under "
                        "`additional_config={'rbln_config': {...}}`.\n"
                    )
            elif (
                vllm_config.cache_config.enable_prefix_caching
                and rbln_config.attn_block_size is not None
            ):
                input_params.kvcache_block_size = rbln_config.attn_block_size
            else:
                input_params.kvcache_block_size = vllm_config.cache_config.block_size
        assert input_params.kvcache_block_size is not None

        apply_user_prefill_chunk_size(
            vllm_config,
            input_params,
            precompiled=False,
            override_prefill_chunk_size=rbln_overrides.get("prefill_chunk_size"),
        )
        # 2. Finalize model-specific constraints and snapshot the compile spec.
        spec = RBLNCompileSpec.for_architecture(
            vllm_config.model_config.hf_config,
            batch_size=input_params.batch_size,
            block_size=input_params.kvcache_block_size,
            max_model_len=input_params.max_seq_len,
            num_devices=input_params.num_devices,
            prefill_chunk_size=input_params.prefill_chunk_size,
            memory_budget=vllm_config.cache_config.gpu_memory_utilization,
            rbln_overrides=rbln_overrides,
        )
        rbln_config.compile_spec = {
            "model": str(vllm_config.model_config.model),
            "dtype": vllm_config.model_config.dtype,
            "model_cls": spec.model_cls.__name__,
            "rbln_config": deepcopy(spec.rbln_config),
            "hf_config": vllm_config.model_config.hf_config.to_dict(),
            "npu_name": RblnPlatform.get_device_name().strip().upper(),
        }

    # 3. Read resolved parameters after applying model-specific constraints.
    resolved_params = RBLNParams.from_rbln_config(
        vllm_config, rbln_config.compile_spec["rbln_config"]
    )
    assert resolved_params.batch_size is not None
    assert resolved_params.max_seq_len is not None
    if resolved_params.kvcache_block_size is None:
        resolved_params.kvcache_block_size = resolved_params.max_seq_len
    apply_user_prefill_chunk_size(vllm_config, resolved_params, precompiled=True)

    # 4. Synchronize vLLM settings with the resolved parameters.
    if vllm_config.scheduler_config.max_num_seqs != resolved_params.batch_size:
        logger.info(
            "Setting max_num_seqs to %d based on the resolved compile config",
            resolved_params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = resolved_params.batch_size
    if vllm_config.model_config.max_model_len != resolved_params.max_seq_len:
        logger.info(
            "Setting max_model_len to %d based on the resolved compile config",
            resolved_params.max_seq_len,
        )
        vllm_config.model_config.max_model_len = resolved_params.max_seq_len
    rbln_config.num_devices_per_local_rank = resolved_params.num_devices

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(
        vllm_config, resolved_params.image_prefill_chunk_size
    )
    update_block_size(
        vllm_config,
        resolved_params.kvcache_block_size,
        prefill_chunk_size=resolved_params.prefill_chunk_size,
        image_prefill_chunk_size=resolved_params.image_prefill_chunk_size,
    )

    update_max_num_batched_tokens(vllm_config, resolved_params)
