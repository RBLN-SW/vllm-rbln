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

"""Define the compile spec, build it from vLLM inputs and derive its cache key.

The spec is the single snapshot that both the compile-cache key and the
compilation read, so the two cannot disagree.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from vllm_rbln import envs
from vllm_rbln.utils.optimum.multimodal_configs import MULTIMODAL_CONFIG_BUILDERS
from vllm_rbln.utils.optimum.rbln_config_utils import (
    default_prefill_chunk_size,
    get_field,
    language_model_configs,
    resolve_kvcache_block_size,
    resolve_num_devices,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


@dataclass
class RBLNCompileSpec:
    """Serializable snapshot shared by cache lookup, compilation and runtime."""

    model: str
    dtype: Any
    model_cls: str
    rbln_config: dict[str, Any]
    hf_config: dict[str, Any]
    npu_name: str


# Keys that affect only loading the compiled model, not the compiled binary,
# and therefore must be stripped before hashing so the same compiled artifact
# is shared between compile-only and inference invocations.
_LOAD_ONLY_KEYS: frozenset[str] = frozenset(
    {"create_runtimes", "device", "kvcache_num_blocks"}
)


def _strip_load_only_keys(obj: dict) -> dict:
    """Recursively drop :data:`_LOAD_ONLY_KEYS` from nested dict/list."""
    if isinstance(obj, dict):
        return {
            k: _strip_load_only_keys(v)
            for k, v in obj.items()
            if k not in _LOAD_ONLY_KEYS
        }
    if isinstance(obj, list):
        return [_strip_load_only_keys(item) for item in obj]
    return obj


def compile_cache_name(spec: RBLNCompileSpec) -> str:
    """Hash compile-relevant spec fields into a stable artifact directory name."""
    config_dict: dict[str, Any] = {
        field.name: getattr(spec, field.name) for field in fields(spec)
    }
    config_dict["rbln_config"] = _strip_load_only_keys(config_dict["rbln_config"])
    model_name = str(config_dict["model"])

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


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


def _set_submodule_num_devices(rbln_config: dict) -> None:
    """Set each submodule's num_devices from its explicit device list."""
    for value in rbln_config.values():
        if not isinstance(value, dict):
            continue
        device = value.get("device")
        if isinstance(device, list) and device:
            value["num_devices"] = len(device)


def _user_block_size(vllm_config: VllmConfig) -> int | None:
    """Preserve the attention block when prefix caching uses smaller blocks."""
    cache_config = vllm_config.cache_config
    additional_config = vllm_config.additional_config
    if not cache_config.user_specified_block_size:
        return None
    if (
        cache_config.enable_prefix_caching
        and additional_config.attn_block_size is not None
    ):
        return additional_config.attn_block_size
    return cache_config.block_size


def _require_block_size(block_size: int | None) -> int:
    if block_size is None:
        raise ValueError(
            "`block_size` is required to run optimum-rbln models "
            "in vLLM RBLN.\n"
            "Set it via one of:\n"
            "  1) vLLM's `block_size` argument "
            "(e.g. `LLM(block_size=...)` or `--block-size`), or\n"
            "  2) `kvcache_block_size` under "
            "`additional_config={'optimum_overrides': {...}}`.\n"
        )
    return block_size


def _resolve_compile_num_devices(
    vllm_config: VllmConfig, optimum_overrides: dict[str, Any]
) -> int:
    num_devices = resolve_num_devices(optimum_overrides)
    if num_devices is None:
        num_devices = vllm_config.additional_config.num_devices_per_local_rank
    return num_devices


def _validate_prefill_chunk_size(
    user_chunk: int | None, override_chunk: int | None
) -> None:
    if (
        user_chunk is not None
        and override_chunk is not None
        and override_chunk != user_chunk
    ):
        raise ValueError(
            f"Conflicting prefill chunk size: max_num_batched_tokens "
            f"({user_chunk}) != rbln_config['prefill_chunk_size'] "
            f"({override_chunk}). Set only one."
        )


def build_compile_spec(vllm_config: VllmConfig) -> RBLNCompileSpec:
    """Resolve user inputs into an owned snapshot for cache lookup and compilation."""
    from vllm_rbln.platform import RblnPlatform

    model_config = vllm_config.model_config
    additional_config = vllm_config.additional_config

    optimum_overrides = deepcopy(additional_config.optimum_overrides)
    if optimum_overrides.get("dtype") is not None:
        raise ValueError(
            "`dtype` cannot be set in `additional_config`'s `optimum_overrides`. "
            "optimum-rbln takes the compile dtype as a load argument, not as "
            "an rbln_config field. Use vLLM's `dtype` argument instead "
            "(e.g. `LLM(dtype=...)` or `--dtype`)."
        )
    _set_submodule_num_devices(optimum_overrides)

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
        hf_config=deepcopy(hf_config.to_dict()),
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

    block_size = resolve_kvcache_block_size(optimum_overrides)
    if block_size is None:
        block_size = _user_block_size(vllm_config)
    block_size = _require_block_size(block_size)

    num_devices = _resolve_compile_num_devices(vllm_config, optimum_overrides)

    prefill_chunk_size = optimum_overrides.get("prefill_chunk_size")
    user_chunk = additional_config.user_max_num_batched_tokens
    _validate_prefill_chunk_size(user_chunk, prefill_chunk_size)
    if user_chunk is not None:
        prefill_chunk_size = user_chunk
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

    block_size = resolve_kvcache_block_size(optimum_overrides)
    if block_size is None:
        block_size = optimum_overrides.get("max_seq_len")
    if block_size is None:
        block_size = _user_block_size(vllm_config)
    if block_size is None:
        block_size = max_model_len

    num_devices = _resolve_compile_num_devices(vllm_config, optimum_overrides)

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
    architectures = getattr(model_config.hf_config, "architectures", [])
    if architectures[0] == "Qwen3Model":
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

    # Resolve each field independently: language_model, text_model, then root.
    override_values: dict[str, Any] = {}
    for source in language_model_configs(optimum_overrides):
        for key in ("batch_size", "max_seq_len", "prefill_chunk_size"):
            value = get_field(source, key)
            if value is not None:
                override_values.setdefault(key, value)
        # Validate both aliases even after a higher-priority size is selected.
        source_block_size = resolve_kvcache_block_size(source)
        if source_block_size is not None:
            override_values.setdefault("kvcache_block_size", source_block_size)

    batch_size = override_values.get(
        "batch_size", vllm_config.scheduler_config.max_num_seqs
    )
    max_model_len = override_values.get("max_seq_len", model_config.max_model_len)

    block_size = override_values.get("kvcache_block_size")
    if block_size is None:
        block_size = _user_block_size(vllm_config)
    block_size = _require_block_size(block_size)

    num_devices = _resolve_compile_num_devices(vllm_config, optimum_overrides)

    prefill_chunk_size = override_values.get("prefill_chunk_size")
    user_chunk = additional_config.user_max_num_batched_tokens
    override_chunk = optimum_overrides.get("prefill_chunk_size")
    _validate_prefill_chunk_size(user_chunk, override_chunk)
    if user_chunk is not None:
        prefill_chunk_size = user_chunk
    model_name, model_cls_name = get_rbln_model_info(model_config.hf_config)
    if model_name == "paligemma":
        prefill_chunk_size = max_model_len
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
        block_size = _user_block_size(vllm_config)
    if block_size is None:
        block_size = max_model_len

    num_devices = _resolve_compile_num_devices(vllm_config, optimum_overrides)

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
