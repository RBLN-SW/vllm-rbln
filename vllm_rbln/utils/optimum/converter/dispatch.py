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
from typing import TYPE_CHECKING

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

from .from_optimum import sync_from_optimum
from .from_vllm import sync_from_vllm
from .params import load_compiled_rbln_config

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
    # Just depends on user-provided parameters
    model_name = str(vllm_config.model_config.model)
    batch_size = vllm_config.scheduler_config.max_num_seqs
    block_size = vllm_config.cache_config.block_size
    max_model_len = vllm_config.model_config.max_model_len
    tp_size = envs.VLLM_RBLN_TP_SIZE
    additional_config = vllm_config.additional_config.get("rbln_config", None)

    # FIXME: To avoid cache collisions, the cache key should also include
    # the versions of the compiler and optimum-rbln.
    config_dict = {
        "model_name": model_name,
        "batch_size": batch_size,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "tp_size": tp_size,
    }
    if additional_config:
        config_dict["rbln_config"] = _strip_runtime_only_keys(additional_config)

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


def _resolve_rbln_config(vllm_config: VllmConfig) -> dict | None:
    """Locate a compiled ``rbln_config.json`` for this vLLM config.

    1. pre-compiled path (user passed an already-compiled directory)
    2. cache hit (hashed path already contains a compiled artifact)
    3. cache miss (compilation still needed; stages `cached_model_path`)
    """
    try:
        compiled_rbln_config = load_compiled_rbln_config(vllm_config)
    except Exception as e:
        raise RuntimeError("Failed to get RBLN config: %s", e) from e
    if compiled_rbln_config is not None:
        return compiled_rbln_config

    cached_model_path = os.path.join(
        envs.VLLM_CACHE_ROOT,
        "compiled_models",
        _generate_model_path_name(vllm_config=vllm_config),
    )
    vllm_config.additional_config["cached_model_path"] = cached_model_path
    if os.path.exists(os.path.join(cached_model_path, "rbln_config.json")):
        logger.info("Found cached compiled model at %s", cached_model_path)
        vllm_config.model_config.model = cached_model_path
        return load_compiled_rbln_config(vllm_config)
    return None


def sync_vllm_and_optimum(vllm_config: VllmConfig) -> None:
    """
    Reconcile vllm_config with the optimum-rbln side.

    If a compiled model exists, it is the source of truth (optimum→vllm).
    Otherwise stage user overrides for the upcoming compile (vllm→optimum).
    """
    compiled_rbln_config = _resolve_rbln_config(vllm_config)
    if compiled_rbln_config is None:
        sync_from_vllm(vllm_config)
    else:
        sync_from_optimum(vllm_config, compiled_rbln_config)
