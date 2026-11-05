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
from copy import deepcopy
from typing import TYPE_CHECKING

from vllm_rbln import envs
from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.paths import is_compiled_dir

from .common import (
    apply_user_prefill_chunk_size,
    is_chunked_prefill_arch,
    update_max_num_batched_tokens,
)
from .from_optimum import sync_from_optimum
from .from_vllm import sync_from_vllm
from .params import RBLNParams, load_compiled_rbln_config

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
    config_dict = _strip_runtime_only_keys(rbln_config.compile_spec)
    model_name = str(config_dict["model"])

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

    rbln_config: OptimumRBLNConfig = vllm_config.additional_config
    if rbln_config.compile_spec is None:
        from vllm_rbln.model_executor.models.optimum.compilation import RBLNCompileSpec
        from vllm_rbln.utils.optimum.block_size import get_attn_block_size

        if is_chunked_prefill_arch(vllm_config.model_config.hf_config):
            overrides = rbln_config.optimum_overrides
            params = RBLNParams.from_rbln_config(vllm_config, overrides)
            apply_user_prefill_chunk_size(
                vllm_config,
                params,
                precompiled=False,
                override_prefill_chunk_size=overrides.get("prefill_chunk_size"),
            )
            update_max_num_batched_tokens(vllm_config, params)

        resolved_config = deepcopy(vllm_config)
        sync_from_vllm(resolved_config)
        spec = RBLNCompileSpec.for_architecture(
            resolved_config.model_config.hf_config,
            batch_size=resolved_config.scheduler_config.max_num_seqs,
            block_size=get_attn_block_size(resolved_config),
            max_model_len=resolved_config.model_config.max_model_len,
            num_devices=resolved_config.additional_config.num_devices_per_local_rank,
            prefill_chunk_size=resolved_config.scheduler_config.max_num_batched_tokens,
            memory_budget=resolved_config.cache_config.gpu_memory_utilization,
            rbln_overrides=resolved_config.additional_config.optimum_overrides,
        )
        rbln_config.compile_spec = {
            "model": str(resolved_config.model_config.model),
            "dtype": resolved_config.model_config.dtype,
            "model_cls": spec.model_cls.__name__,
            "rbln_config": deepcopy(spec.rbln_config),
        }

    cached_model_path = os.path.join(
        envs.VLLM_CACHE_ROOT,
        "compiled_models",
        _generate_model_path_name(vllm_config=vllm_config),
    )
    rbln_config.cached_model_path = cached_model_path
    if is_compiled_dir(cached_model_path):
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
