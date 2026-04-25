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
from vllm_rbln.utils.optimum.params import RBLNParams, get_rbln_config

from .dispatch_helper import (
    keep_only_device_keys,
    strip_runtime_only_keys,
)
from .optimum_to_vllm import sync_to_vllm
from .vllm_to_optimum import sync_from_vllm

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def generate_model_path_name(
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
        config_dict["rbln_config"] = strip_runtime_only_keys(additional_config)

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
        rbln_config = get_rbln_config(vllm_config)
    except Exception as e:
        raise RuntimeError("Failed to get RBLN config: %s", e) from e
    if rbln_config is not None:
        return rbln_config

    cached_model_path = os.path.join(
        envs.VLLM_CACHE_ROOT,
        "compiled_models",
        generate_model_path_name(vllm_config=vllm_config),
    )
    if os.path.exists(os.path.join(cached_model_path, "rbln_config.json")):
        logger.info("Found cached compiled model at %s", cached_model_path)
        vllm_config.model_config.model = cached_model_path
        return get_rbln_config(vllm_config)

    vllm_config.additional_config["cached_model_path"] = cached_model_path
    return None


def sync_vllm_and_optimum(vllm_config: VllmConfig) -> None:
    """
    If compiled model with RBLN config is given,
    synchronise vLLM config with RBLN config.
    If no RBLN config is given, validate vLLM config and set necessary parameters
    to default values to compile model internally.
    """
    rbln_config = _resolve_rbln_config(vllm_config)
    if rbln_config is None:
        sync_from_vllm(vllm_config)
        return

    # Pre-compiled (or cache-hit): the compiled artefact is the source of
    # truth. Strip the user's additional_config down to device-only keys so
    # submodule placement can still be overridden.
    vllm_config.additional_config["rbln_config"] = keep_only_device_keys(
        vllm_config.additional_config.get("rbln_config", {})
    )
    params = RBLNParams.from_rbln_config(vllm_config, rbln_config)
    sync_to_vllm(vllm_config, params)
