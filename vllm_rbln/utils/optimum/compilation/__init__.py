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
from typing import TYPE_CHECKING, Any

import optimum.rbln
from optimum.rbln import (
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForSpeechSeq2Seq,
)
from transformers import PretrainedConfig

from vllm_rbln import envs
from vllm_rbln.utils.optimum.params import (
    RBLNCompileSpec,
    _cfg_get,
    _language_model_configs,
    _resolve_default_prefill_chunk_size,
    _resolve_kvcache_block_size,
    _resolve_num_devices,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)

from .multimodal import _COMPILE_MULTIMODAL_FNS

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge ``overrides`` into ``base`` in place.

    Nested dicts are merged key-by-key; non-dict values overwrite. This
    preserves untouched sub-keys when the user only overrides part of a
    nested config (e.g. ``language_model.max_seq_len`` for multimodal).
    """
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value


def _find_conflicts(base: dict, overrides: dict, _prefix: str = "") -> list[str]:
    """Return dotted paths where ``overrides`` would overwrite an existing
    value in ``base`` with a different one.

    Mirrors ``_deep_merge``'s traversal: nested dicts recurse, leaves compare
    by value. A key present only in ``overrides`` is not a conflict (the user
    is adding a new knob, not clobbering a derived field).
    """
    conflicts: list[str] = []
    for key, value in overrides.items():
        if key not in base:
            continue
        path = f"{_prefix}{key}"
        existing = base[key]
        if isinstance(value, dict) and isinstance(existing, dict):
            conflicts.extend(_find_conflicts(existing, value, f"{path}."))
        elif existing != value:
            conflicts.append(f"{path} (compiled={existing!r}, override={value!r})")
    return conflicts


def _sync_submodule_tp_with_device(rbln_config: dict) -> None:
    """Align each submodule's ``tensor_parallel_size`` with its device count.

    A submodule (e.g. ``visual``) is configured on a fixed set of devices via
    its nested ``device`` list, but optimum-rbln requires its
    ``tensor_parallel_size`` to match that device count. The user only supplies
    ``device``, so we derive ``tensor_parallel_size`` from ``len(device)`` here.
    The top-level config is left untouched (its TP is driven by
    ``--rbln-num-devices-per-local-rank``).
    """
    for value in rbln_config.values():
        if not isinstance(value, dict):
            continue
        device = value.get("device")
        if isinstance(device, list) and device:
            value["num_devices"] = len(device)


def build_compile_spec(vllm_config: VllmConfig) -> RBLNCompileSpec:
    """Resolve user inputs into an owned snapshot for cache lookup and compilation."""
    from vllm_rbln.platform import RblnPlatform

    additional_config = vllm_config.additional_config
    optimum_overrides = deepcopy(additional_config.optimum_overrides)
    if optimum_overrides.get("dtype") is not None:
        raise ValueError(
            "`dtype` cannot be set in `additional_config`'s `optimum_overrides`. "
            "optimum-rbln takes the compile dtype as a load argument, not as "
            "an rbln_config field. Use vLLM's `dtype` argument instead "
            "(e.g. `LLM(dtype=...)` or `--dtype`)."
        )
    _sync_submodule_tp_with_device(optimum_overrides)
    model_config = vllm_config.model_config
    config = model_config.hf_config
    enc_dec = is_enc_dec_arch(config)
    pooling = is_pooling_arch(config)
    multimodal = is_multi_modal(config) and not enc_dec
    sources = (
        _language_model_configs(optimum_overrides)
        if multimodal
        else [optimum_overrides]
    )
    compile_inputs: dict[str, Any] = {}
    block_size = None
    for source in sources:
        for key in (
            "batch_size",
            "max_seq_len",
            "dec_max_seq_len",
            "prefill_chunk_size",
        ):
            value = _cfg_get(source, key)
            if value is not None:
                compile_inputs.setdefault(key, value)
        if not enc_dec:
            source_block_size = _resolve_kvcache_block_size(source, arch="compile")
            if block_size is None:
                block_size = source_block_size

    batch_size = compile_inputs.get(
        "batch_size", vllm_config.scheduler_config.max_num_seqs
    )
    length_key = "dec_max_seq_len" if enc_dec else "max_seq_len"
    max_model_len = compile_inputs.get(length_key, model_config.max_model_len)
    if enc_dec or pooling and block_size is None:
        block_size = compile_inputs.get(length_key)
    if block_size is None:
        cache_config = vllm_config.cache_config
        if not cache_config.user_specified_block_size:
            if enc_dec or pooling:
                block_size = max_model_len
            else:
                raise ValueError(
                    "`block_size` is required to run optimum-rbln models "
                    "in vLLM RBLN.\n"
                    "Set it via one of:\n"
                    "  1) vLLM's `block_size` argument "
                    "(e.g. `LLM(block_size=...)` or `--block-size`), or\n"
                    "  2) `kvcache_block_size` under "
                    "`additional_config={'optimum_overrides': {...}}`.\n"
                )
        elif (
            cache_config.enable_prefix_caching
            and additional_config.attn_block_size is not None
        ):
            block_size = additional_config.attn_block_size
        else:
            block_size = cache_config.block_size
    assert block_size is not None

    num_devices = _resolve_num_devices(
        optimum_overrides, default=additional_config.num_devices_per_local_rank
    )
    memory_budget = vllm_config.cache_config.gpu_memory_utilization
    prefill_chunk_size = None
    if not (enc_dec or pooling):
        default_chunk = _resolve_default_prefill_chunk_size()
        prefill_chunk_size = (
            compile_inputs.get("prefill_chunk_size", default_chunk)
            if multimodal
            else optimum_overrides.get("prefill_chunk_size", default_chunk)
        )
        user_chunk = additional_config.user_max_num_batched_tokens
        override_chunk = optimum_overrides.get("prefill_chunk_size")
        if user_chunk is not None:
            if override_chunk is not None and override_chunk != user_chunk:
                raise ValueError(
                    f"Conflicting prefill chunk size: max_num_batched_tokens "
                    f"({user_chunk}) != rbln_config['prefill_chunk_size'] "
                    f"({override_chunk}). Set only one."
                )
            prefill_chunk_size = user_chunk
        if multimodal and get_rbln_model_info(config)[0] == "paligemma":
            prefill_chunk_size = max_model_len
    if is_generation_arch(config):
        model_cls, rbln_config = _for_decoder(
            batch_size,
            block_size,
            max_model_len,
            num_devices,
            memory_budget,
            prefill_chunk_size,
        )
    elif is_pooling_arch(config):
        model_cls, rbln_config = _for_pooling(
            config,
            batch_size,
            block_size,
            max_model_len,
            num_devices,
        )
    elif is_multi_modal(config):
        model_cls, rbln_config = _for_multimodal(
            config,
            batch_size,
            block_size,
            max_model_len,
            num_devices,
            memory_budget,
            prefill_chunk_size,
        )
    elif is_enc_dec_arch(config):
        model_cls, rbln_config = _for_enc_dec(
            config,
            batch_size,
            block_size,
            max_model_len,
            num_devices,
        )
    else:
        architectures = getattr(config, "architectures", [])
        raise NotImplementedError(
            f"Compilation is not implemented for architecture {architectures[0]}"
        )

    if envs.VLLM_RBLN_USE_NESTED_COMPILE_REGION and (
        is_generation_arch(config) or is_multi_modal(config)
    ):
        decoder_config = rbln_config
        for name in ("language_model", "text_model"):
            if name in decoder_config:
                decoder_config = decoder_config[name]
                break
        decoder_config["nested_compile_region"] = True

    # optimum_overrides must not overwrite fields vllm derives from its own
    # config (batch_size, max_seq_len, memory_budget, ...); a silent
    # mismatch would compile a model that disagrees with the runtime. Adding
    # a new key not set by the builder is allowed.
    if optimum_overrides:
        conflicts = _find_conflicts(rbln_config, optimum_overrides)
        if conflicts:
            raise ValueError(
                "optimum_overrides conflict with vllm-derived compile config: "
                + "; ".join(conflicts)
            )
        _deep_merge(rbln_config, optimum_overrides)

    return RBLNCompileSpec(
        model=str(model_config.model),
        dtype=model_config.dtype,
        model_cls=model_cls.__name__,
        rbln_config=rbln_config,
        hf_config=deepcopy(config.to_dict()),
        npu_name=RblnPlatform.get_device_name().strip().upper(),
    )


def _for_decoder(
    batch_size: int,
    block_size: int,
    max_model_len: int,
    num_devices: int,
    memory_budget: float,
    prefill_chunk_size: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    rbln_config: dict[str, Any] = {
        "num_devices": num_devices,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
        "memory_budget": memory_budget,
    }
    if block_size != max_model_len:
        rbln_config["kvcache_partition_len"] = block_size
        rbln_config["attn_impl"] = "flash_attn"
    # Pin the compiled prefill_chunk_size to the resolved value so it stays
    # in sync with the one used for KV-cache block padding.
    if prefill_chunk_size is not None:
        rbln_config["prefill_chunk_size"] = prefill_chunk_size
    return RBLNAutoModelForCausalLM, rbln_config


def _for_pooling(
    config: PretrainedConfig,
    batch_size: int,
    block_size: int,
    max_model_len: int,
    num_devices: int,
) -> tuple[Any, dict[str, Any]]:
    _, model_cls_name = get_rbln_model_info(config)
    model_cls = getattr(optimum.rbln, model_cls_name)
    assert model_cls is not None

    rbln_config: dict[str, Any] = {
        "num_devices": num_devices,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
    }
    # FIXME: We need a more generalized logic to specify block sizes
    # as the number of supported models continues to grow.
    architectures = getattr(config, "architectures", [])
    if architectures[0] == "Qwen3Model" and block_size != max_model_len:
        rbln_config["kvcache_partition_len"] = block_size
        rbln_config["attn_impl"] = "flash_attn"
    return model_cls, rbln_config


def _for_multimodal(
    config: PretrainedConfig,
    batch_size: int,
    block_size: int,
    max_model_len: int,
    num_devices: int,
    memory_budget: float,
    prefill_chunk_size: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    model_name, model_cls_name = get_rbln_model_info(config)
    compile_fn = _COMPILE_MULTIMODAL_FNS.get(model_name)
    if compile_fn is None:
        raise ValueError(
            f"Unknown multimodal model alias: {model_name}. "
            f"Supported aliases: {sorted(_COMPILE_MULTIMODAL_FNS.keys())}"
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
        memory_budget,
        prefill_chunk_size,
    )
    return model_cls, rbln_config


def _for_enc_dec(
    config: PretrainedConfig,
    batch_size: int,
    block_size: int,
    max_model_len: int,
    num_devices: int,
) -> tuple[Any, dict[str, Any]]:
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
    }
