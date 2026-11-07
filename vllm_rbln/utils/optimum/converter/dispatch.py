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

import hashlib
import json
import os
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from vllm_rbln import envs
from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.params import (
    load_compiled_rbln_config,
    read_runtime_settings,
)
from vllm_rbln.utils.optimum.paths import resolve_compiled_model_path

from .runtime import apply_runtime_settings

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


# Keys that affect only runtime loading, not the compiled binary, and
# therefore must be stripped before hashing so the same compiled artifact
# is shared between compile-only and inference invocations.
_RUNTIME_ONLY_KEYS: frozenset[str] = frozenset(
    {"create_runtimes", "device", "kvcache_num_blocks"}
)


def _strip_runtime_only_keys(obj: dict) -> dict:
    """Recursively drop :data:`_RUNTIME_ONLY_KEYS` from nested dict/list."""
    if isinstance(obj, dict):
        return {
            k: _strip_runtime_only_keys(v)
            for k, v in obj.items()
            if k not in _RUNTIME_ONLY_KEYS
        }
    if isinstance(obj, list):
        return [_strip_runtime_only_keys(item) for item in obj]
    return obj


def _generate_model_path_name(
    vllm_config: VllmConfig,
) -> str:
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    assert additional_config.compile_spec is not None, (
        "compile spec must be resolved first"
    )
    config_dict: dict[str, Any] = {
        field.name: getattr(additional_config.compile_spec, field.name)
        for field in fields(additional_config.compile_spec)
    }
    config_dict["rbln_config"] = _strip_runtime_only_keys(config_dict["rbln_config"])
    model_name = str(config_dict["model"])

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


def _resolve_cached_model_path(vllm_config: VllmConfig) -> str:
    """Compute and store the cache path from the resolved compile spec."""
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    cached_model_path = os.path.join(
        envs.VLLM_CACHE_ROOT,
        "compiled_models",
        _generate_model_path_name(vllm_config=vllm_config),
    )
    additional_config.cached_model_path = cached_model_path
    return cached_model_path


def sync_vllm_and_optimum(vllm_config: VllmConfig) -> None:
    """
    Reconcile vllm_config with the optimum-rbln side.

    If a compiled model exists, it is the source of truth (optimum→vllm).
    Otherwise stage user overrides for the upcoming compile (vllm→optimum).
    """
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    # 1. Load rbln_config.json from the model path or a known cache path.
    artifact_path = resolve_compiled_model_path(
        vllm_config.model_config.model, additional_config.cached_model_path
    )
    rbln_config = (
        load_compiled_rbln_config(artifact_path) if artifact_path is not None else None
    )
    # 2. Resolve the compile spec and look up the corresponding cached model.
    if rbln_config is None:
        if additional_config.compile_spec is None:
            from vllm_rbln.utils.optimum.compilation import build_compile_spec

            additional_config.compile_spec = build_compile_spec(vllm_config)
        spec = additional_config.compile_spec
        artifact_path = _resolve_cached_model_path(vllm_config)
        rbln_config = load_compiled_rbln_config(artifact_path)
    if rbln_config is not None:
        # 3. Use settings from the compiled model's rbln_config.json.
        logger.info("Found compiled model at %s", artifact_path)
        selected_config = rbln_config
        precompiled = True
    else:
        # 3. Use settings from the compile spec for a new compilation.
        selected_config = spec.rbln_config
        precompiled = False
    # 4. Read and apply runtime settings once from the selected configuration.
    runtime_settings = read_runtime_settings(vllm_config, selected_config)
    apply_runtime_settings(vllm_config, runtime_settings, precompiled=precompiled)
