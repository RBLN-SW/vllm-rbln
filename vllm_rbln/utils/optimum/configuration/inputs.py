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

"""Input selection, override validation and compile-cache key generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from .readers import read_num_devices

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNCompileInputs


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


def compile_cache_name(compile_inputs: OptimumRBLNCompileInputs) -> str:
    """Hash compile-relevant input fields into a stable artifact directory name."""
    config_dict: dict[str, Any] = {
        field.name: getattr(compile_inputs, field.name)
        for field in fields(compile_inputs)
    }
    config_dict["rbln_config"] = _strip_load_only_keys(config_dict["rbln_config"])
    config_dict["hf_config"] = compile_inputs.hf_config.to_dict()
    model_name = str(config_dict["model"])

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


def merge_overrides(base: dict, overrides: dict) -> None:
    """Recursively merge ``overrides`` into ``base`` in place.

    Nested dicts are merged key-by-key; non-dict values overwrite. This
    preserves untouched sub-keys when the user only overrides part of a
    nested config (e.g. ``language_model.max_seq_len`` for multimodal).
    """
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merge_overrides(existing, value)
        else:
            base[key] = value


def find_override_conflicts(
    base: dict, overrides: dict, _prefix: str = ""
) -> list[str]:
    """Return dotted paths where ``overrides`` would overwrite an existing
    value in ``base`` with a different one.

    Mirrors ``merge_overrides``'s traversal: nested dicts recurse, leaves compare
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
            conflicts.extend(find_override_conflicts(existing, value, f"{path}."))
        elif existing != value:
            conflicts.append(f"{path} (compiled={existing!r}, override={value!r})")
    return conflicts


def set_submodule_num_devices(rbln_config: dict) -> None:
    """Set each submodule's num_devices from its explicit device list."""
    for value in rbln_config.values():
        if not isinstance(value, dict):
            continue
        device = value.get("device")
        if isinstance(device, list) and device:
            value["num_devices"] = len(device)


def get_user_block_size(vllm_config: VllmConfig) -> int | None:
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


def require_compile_block_size(block_size: int | None) -> int:
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


def get_compile_num_devices(
    vllm_config: VllmConfig, optimum_overrides: dict[str, Any]
) -> int:
    num_devices = read_num_devices(optimum_overrides)
    if num_devices is None:
        num_devices = vllm_config.additional_config.num_devices_per_local_rank
    return num_devices


def validate_prefill_chunk_override(
    user_chunk: int | None, override_chunk: int | None
) -> None:
    if (
        user_chunk is not None
        and override_chunk is not None
        and override_chunk != user_chunk
    ):
        raise ValueError(
            f"Conflicting prefill chunk size: --rbln-prefill-chunk-size "
            f"({user_chunk}) != optimum_overrides['prefill_chunk_size'] "
            f"({override_chunk}). Set only one."
        )


_PREFILL_CHUNK_SIZE_BY_FAMILY = {
    "ca": 128,  # ATOM
    "cr": 512,  # REBEL
}


def default_prefill_chunk_size() -> int:
    from vllm_rbln.platform import RblnPlatform

    device_name = RblnPlatform.get_device_name().lower()
    for family, chunk_size in _PREFILL_CHUNK_SIZE_BY_FAMILY.items():
        if family in device_name:
            return chunk_size
    raise RuntimeError(
        f"Unknown NPU device {device_name!r}; "
        "expected an ATOM (ca) or REBEL (cr) device."
    )
