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

from vllm_rbln import envs
from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.paths import is_compiled_dir

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
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    assert rbln_config.compile_spec is not None, "compile spec must be resolved first"
    config_dict = dict(rbln_config.compile_spec)
    config_dict["rbln_config"] = _strip_runtime_only_keys(config_dict["rbln_config"])
    model_name = str(config_dict["model"])

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


def _resolve_cached_rbln_config(vllm_config: VllmConfig) -> dict | None:
    """Locate a cached artifact after synchronizing the compile inputs."""
    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    cached_model_path = os.path.join(
        envs.VLLM_CACHE_ROOT,
        "compiled_models",
        _generate_model_path_name(vllm_config=vllm_config),
    )
    rbln_config.cached_model_path = cached_model_path
    if is_compiled_dir(cached_model_path):
        logger.info("Found cached compiled model at %s", cached_model_path)
        return load_compiled_rbln_config(vllm_config)
    return None


def sync_vllm_and_optimum(vllm_config: VllmConfig) -> None:
    """
    Reconcile vllm_config with the optimum-rbln side.

    If a compiled model exists, it is the source of truth (optimum→vllm).
    Otherwise stage user overrides for the upcoming compile (vllm→optimum).
    """
    # 1. Synchronize from existing compiled metadata when available.
    compiled_config = load_compiled_rbln_config(vllm_config)
    if compiled_config is not None:
        sync_from_optimum(vllm_config, compiled_config)
        return

    # 2. Resolve the compile spec and synchronize runtime settings from it.
    sync_from_vllm(vllm_config)
    # 3. Look up the finalized spec's cache and synchronize from its metadata.
    cached_config = _resolve_cached_rbln_config(vllm_config)
    if cached_config is not None:
        sync_from_optimum(vllm_config, cached_config)
