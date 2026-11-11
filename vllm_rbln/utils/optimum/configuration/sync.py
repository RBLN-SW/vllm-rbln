# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare optimum compile config and synchronize the selected model settings."""

from typing import TYPE_CHECKING

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.path import (
    get_compile_cache_path,
    resolve_compiled_model_path,
)

from .apply import apply_runtime_config
from .compile_config import compile_cache_name, prepare_compile_config
from .readers import load_compiled_rbln_config
from .runtime_config import prepare_runtime_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def sync_vllm_and_optimum(vllm_config: VllmConfig) -> None:
    """Prepare compile config, select model settings and apply them to vLLM.

    Existing artifacts determine the vLLM config updates. On a cache miss, the
    resolved compile config supplies them and is retained for compilation.
    """
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    # 1. Load rbln_config.json from the model path or a known cache path.
    artifact_path = resolve_compiled_model_path(
        vllm_config.model_config.model, additional_config.artifact_path
    )
    rbln_config = (
        load_compiled_rbln_config(artifact_path) if artifact_path is not None else None
    )
    # 2. Prepare compile config and look up the corresponding cached model.
    if rbln_config is None:
        if additional_config.compile_config is None:
            additional_config.compile_config = prepare_compile_config(vllm_config)
        compile_config = additional_config.compile_config
        artifact_path = get_compile_cache_path(
            compile_cache_name(compile_config), envs.VLLM_CACHE_ROOT
        )
        rbln_config = load_compiled_rbln_config(artifact_path)
    if rbln_config is not None:
        # 3. Use settings from the compiled model's rbln_config.json.
        logger.info("Found compiled model at %s", artifact_path)
        selected_config = rbln_config
        precompiled = True
    else:
        # 3. Use settings from the compile config for a new compilation.
        logger.warning(
            "The compiled model was not found in the cache. "
            "If you already have a compiled model, set `model` to the "
            "directory containing its rbln_config.json to reuse it.",
        )
        selected_config = compile_config.rbln_config
        precompiled = False
    # 4. Prepare and apply the selected configuration to vLLM once.
    runtime_config = prepare_runtime_config(
        vllm_config, selected_config, precompiled=precompiled
    )
    apply_runtime_config(vllm_config, runtime_config)
    additional_config.artifact_path = artifact_path
    additional_config.precompiled = precompiled
