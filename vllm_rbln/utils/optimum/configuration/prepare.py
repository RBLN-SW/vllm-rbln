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

"""Prepare owned compilation inputs from vLLM configuration.

Cache lookup and compilation share these inputs to stay consistent.
"""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from vllm_rbln import envs
from vllm_rbln.config import OptimumRBLNCompileInputs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.configuration.inputs import (
    default_prefill_chunk_size,
    find_override_conflicts,
    get_compile_num_devices,
    get_user_block_size,
    merge_overrides,
    require_compile_block_size,
    set_submodule_num_devices,
    validate_prefill_chunk_override,
)
from vllm_rbln.utils.optimum.configuration.multimodal import (
    MULTIMODAL_CONFIG_BUILDERS,
)
from vllm_rbln.utils.optimum.configuration.readers import (
    language_model_configs,
    read_kvcache_block_size,
    read_language_model_fields,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_decoder_pooling_arch,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def prepare_compile_inputs(vllm_config: VllmConfig) -> OptimumRBLNCompileInputs:
    """Resolve user settings into owned inputs for cache lookup and compilation."""
    from vllm_rbln.platform import RblnPlatform

    model_config = vllm_config.model_config
    additional_config = vllm_config.additional_config

    if additional_config.prefill_chunk_size is not None and (
        is_enc_dec_arch(model_config.hf_config)
        or is_pooling_arch(model_config.hf_config)
    ):
        raise ValueError(
            "prefill_chunk_size cannot be specified for a model that does "
            "not support chunked prefill. Leave --rbln-prefill-chunk-size unset."
        )

    optimum_overrides = deepcopy(additional_config.optimum_overrides)
    if optimum_overrides.get("dtype") is not None:
        raise ValueError(
            "`dtype` cannot be set in `additional_config`'s `optimum_overrides`. "
            "optimum-rbln takes the compile dtype as a load argument, not as "
            "an rbln_config field. Use vLLM's `dtype` argument instead "
            "(e.g. `LLM(dtype=...)` or `--dtype`)."
        )
    set_submodule_num_devices(optimum_overrides)

    hf_config = model_config.hf_config
    if is_generation_arch(hf_config):
        model_cls, rbln_config = _for_decoder(vllm_config, optimum_overrides)
    elif is_pooling_arch(hf_config):
        model_cls, rbln_config = _for_pooling(vllm_config, optimum_overrides)
    elif is_multi_modal(hf_config):
        model_cls, rbln_config = _for_multimodal(vllm_config, optimum_overrides)
    elif is_enc_dec_arch(hf_config):
        model_cls, rbln_config = _for_enc_dec(vllm_config, optimum_overrides)
    else:
        architectures = getattr(hf_config, "architectures", [])
        raise NotImplementedError(
            f"Compilation is not implemented for architecture {architectures[0]}"
        )

    # optimum_overrides must not overwrite fields vllm derives from its own
    # config (batch_size, max_seq_len, memory_budget, ...); a silent
    # mismatch would compile a model that disagrees with the runtime. Adding
    # a new key not set by the builder is allowed.
    if optimum_overrides:
        conflicts = find_override_conflicts(rbln_config, optimum_overrides)
        if conflicts:
            raise ValueError(
                "optimum_overrides conflict with vllm-derived compile config: "
                + "; ".join(conflicts)
            )
        merge_overrides(rbln_config, optimum_overrides)

    return OptimumRBLNCompileInputs(
        model=str(model_config.model),
        dtype=model_config.dtype,
        model_cls=model_cls.__name__,
        rbln_config=rbln_config,
        hf_config=deepcopy(hf_config),
        npu_name=RblnPlatform.get_device_name().strip().upper(),
    )


def _for_decoder(
    vllm_config: VllmConfig,
    optimum_overrides: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from optimum.rbln import RBLNAutoModelForCausalLM

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    additional_config = vllm_config.additional_config

    batch_size = optimum_overrides.get("batch_size")
    if batch_size is None:
        batch_size = vllm_config.scheduler_config.max_num_seqs
    max_model_len = optimum_overrides.get("max_seq_len")
    if max_model_len is None:
        max_model_len = model_config.max_model_len

    block_size = read_kvcache_block_size(optimum_overrides)
    if block_size is None:
        block_size = get_user_block_size(vllm_config)
    block_size = require_compile_block_size(block_size)

    num_devices = get_compile_num_devices(vllm_config, optimum_overrides)

    prefill_chunk_size = additional_config.prefill_chunk_size
    override_prefill_chunk_size = optimum_overrides.get("prefill_chunk_size")
    validate_prefill_chunk_override(prefill_chunk_size, override_prefill_chunk_size)
    if prefill_chunk_size is None:
        prefill_chunk_size = override_prefill_chunk_size
    if prefill_chunk_size is None:
        prefill_chunk_size = default_prefill_chunk_size()

    rbln_config: dict[str, Any] = {
        "num_devices": num_devices,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
        "kvcache_block_size": block_size,
        "memory_budget": cache_config.gpu_memory_utilization,
        "prefill_chunk_size": prefill_chunk_size,
    }
    if block_size != max_model_len:
        rbln_config["kvcache_partition_len"] = block_size
        rbln_config["attn_impl"] = "flash_attn"
    if envs.VLLM_RBLN_USE_NESTED_COMPILE_REGION:
        rbln_config["nested_compile_region"] = True
    return RBLNAutoModelForCausalLM, rbln_config


def _for_pooling(
    vllm_config: VllmConfig,
    optimum_overrides: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import optimum.rbln

    model_config = vllm_config.model_config

    batch_size = optimum_overrides.get("batch_size")
    if batch_size is None:
        batch_size = vllm_config.scheduler_config.max_num_seqs
    max_model_len = optimum_overrides.get("max_seq_len")
    if max_model_len is None:
        max_model_len = model_config.max_model_len

    block_size = read_kvcache_block_size(optimum_overrides)
    if block_size is None:
        block_size = optimum_overrides.get("max_seq_len")
    if block_size is None:
        block_size = get_user_block_size(vllm_config)
    if block_size is None:
        block_size = max_model_len

    num_devices = get_compile_num_devices(vllm_config, optimum_overrides)

    _, model_cls_name = get_rbln_model_info(model_config.hf_config)
    model_cls = getattr(optimum.rbln, model_cls_name)
    assert model_cls is not None

    rbln_config: dict[str, Any] = {
        "num_devices": num_devices,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
    }
    # FIXME: We need a more generalized logic to specify block sizes
    # as the number of supported models continues to grow.
    if is_decoder_pooling_arch(model_config.hf_config):
        rbln_config["kvcache_block_size"] = block_size
        if block_size != max_model_len:
            rbln_config["kvcache_partition_len"] = block_size
            rbln_config["attn_impl"] = "flash_attn"
    return model_cls, rbln_config


def _for_multimodal(
    vllm_config: VllmConfig,
    optimum_overrides: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import optimum.rbln

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    additional_config = vllm_config.additional_config

    override_values = read_language_model_fields(
        optimum_overrides,
        "batch_size",
        "max_seq_len",
        "prefill_chunk_size",
        "kvcache_block_size",
    )

    batch_size = override_values.get(
        "batch_size", vllm_config.scheduler_config.max_num_seqs
    )
    max_model_len = override_values.get("max_seq_len", model_config.max_model_len)

    block_size = override_values.get("kvcache_block_size")
    if block_size is None:
        block_size = get_user_block_size(vllm_config)
    block_size = require_compile_block_size(block_size)

    num_devices = get_compile_num_devices(vllm_config, optimum_overrides)

    prefill_chunk_size = additional_config.prefill_chunk_size
    override_prefill_chunk_size = override_values.get("prefill_chunk_size")
    validate_prefill_chunk_override(prefill_chunk_size, override_prefill_chunk_size)
    if prefill_chunk_size is None:
        prefill_chunk_size = override_prefill_chunk_size
    model_name, model_cls_name = get_rbln_model_info(model_config.hf_config)
    if model_name == "paligemma":
        if prefill_chunk_size is not None and prefill_chunk_size != max_model_len:
            logger.warning(
                "PaliGemma requires full-prompt prefill. Adjusting "
                "prefill_chunk_size from %s to max_model_len (%s).",
                prefill_chunk_size,
                max_model_len,
            )
        prefill_chunk_size = max_model_len
        additional_config.prefill_chunk_size = prefill_chunk_size
    if prefill_chunk_size is None:
        prefill_chunk_size = default_prefill_chunk_size()

    compile_fn = MULTIMODAL_CONFIG_BUILDERS.get(model_name)
    if compile_fn is None:
        raise ValueError(
            f"Unknown multimodal model alias: {model_name}. "
            f"Supported aliases: {sorted(MULTIMODAL_CONFIG_BUILDERS.keys())}"
        )
    model_cls = getattr(optimum.rbln, model_cls_name)
    assert model_cls is not None
    # Pass the resolved prefill_chunk_size so each compile_fn pins it on the
    # compiled model, keeping it in sync with the KV-cache block padding.
    rbln_config = compile_fn(
        batch_size,
        max_model_len,
        block_size,
        num_devices,
        cache_config.gpu_memory_utilization,
        prefill_chunk_size,
    )
    decoder_config = language_model_configs(rbln_config)[0]
    decoder_config["kvcache_block_size"] = block_size
    if envs.VLLM_RBLN_USE_NESTED_COMPILE_REGION:
        decoder_config["nested_compile_region"] = True
    return model_cls, rbln_config


def _for_enc_dec(
    vllm_config: VllmConfig,
    optimum_overrides: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from optimum.rbln import RBLNAutoModelForSpeechSeq2Seq

    model_config = vllm_config.model_config
    config = model_config.hf_config

    batch_size = optimum_overrides.get("batch_size")
    if batch_size is None:
        batch_size = vllm_config.scheduler_config.max_num_seqs
    max_model_len = optimum_overrides.get("dec_max_seq_len")
    if max_model_len is None:
        max_model_len = model_config.max_model_len

    block_size = optimum_overrides.get("dec_max_seq_len")
    if block_size is None:
        block_size = get_user_block_size(vllm_config)
    if block_size is None:
        block_size = max_model_len

    num_devices = get_compile_num_devices(vllm_config, optimum_overrides)

    architectures = getattr(config, "architectures", [])
    assert architectures[0] == "WhisperForConditionalGeneration"
    # Whisper does not support varying block_size or max_model_len.
    assert block_size == max_model_len, (
        "block_size must be equal to max_model_len for Whisper models."
    )
    assert max_model_len == config.max_target_positions, (
        f"max_model_len ({max_model_len}) must match the Whisper model's "
        f"max_target_positions ({config.max_target_positions}) "
        "from the HuggingFace config."
    )
    return RBLNAutoModelForSpeechSeq2Seq, {
        "num_devices": num_devices,
        "batch_size": batch_size,
        "token_timestamps": False,
        "dec_max_seq_len": max_model_len,
        "kvcache_block_size": block_size,
    }
