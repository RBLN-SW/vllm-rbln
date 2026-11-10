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

"""Prepare optimum compile inputs and synchronize the selected model settings."""

from typing import TYPE_CHECKING

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.path import (
    get_compile_cache_path,
    resolve_compiled_model_path,
)

from .apply import apply_to_vllm_config
from .inputs import compile_cache_name, resolve_explicit_num_blocks
from .prepare import prepare_compile_inputs
from .readers import load_compiled_rbln_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import OptimumRBLNConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def _apply_explicit_block_memory_policy(vllm_config: VllmConfig) -> None:
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    cache_config = vllm_config.cache_config
    num_blocks = resolve_explicit_num_blocks(
        cache_config.num_gpu_blocks_override, additional_config.optimum_overrides
    )
    if num_blocks is not None and cache_config.gpu_memory_utilization != 1.0:
        logger.info(
            "Setting gpu_memory_utilization from %s to 1.0 because "
            "an explicit KV cache block count=%d is set.",
            cache_config.gpu_memory_utilization,
            num_blocks,
        )
        cache_config.gpu_memory_utilization = 1.0


def sync_vllm_and_optimum(vllm_config: VllmConfig) -> None:
    """Prepare compile inputs, select model settings and apply them to vLLM.

    Existing artifacts determine the vLLM config updates. On a cache miss, the
    resolved compile inputs supply them and are retained for compilation.
    """
    # 1. Apply the explicit-block memory policy before cache lookup.
    _apply_explicit_block_memory_policy(vllm_config)
    additional_config: OptimumRBLNConfig = vllm_config.additional_config
    # 2. Load rbln_config.json from the model path or a known cache path.
    artifact_path = resolve_compiled_model_path(
        vllm_config.model_config.model, additional_config.artifact_path
    )
    rbln_config = (
        load_compiled_rbln_config(artifact_path) if artifact_path is not None else None
    )
    # 3. Prepare compile inputs and look up the corresponding cached model.
    if rbln_config is None:
        if additional_config.compile_inputs is None:
            additional_config.compile_inputs = prepare_compile_inputs(vllm_config)
        compile_inputs = additional_config.compile_inputs
        artifact_path = get_compile_cache_path(
            compile_cache_name(compile_inputs), envs.VLLM_CACHE_ROOT
        )
        rbln_config = load_compiled_rbln_config(artifact_path)
    if rbln_config is not None:
        # 4. Use settings from the compiled model's rbln_config.json.
        logger.info("Found compiled model at %s", artifact_path)
        selected_config = rbln_config
        precompiled = True
    else:
        # 4. Use settings from the compile inputs for a new compilation.
        logger.warning(
            "The compiled model was not found in the cache. "
            "If you already have a compiled model, set `model` to the "
            "directory containing its rbln_config.json to reuse it.",
        )
        selected_config = compile_inputs.rbln_config
        precompiled = False
    # 5. Apply the selected configuration to vLLM once.
    apply_to_vllm_config(vllm_config, selected_config, precompiled=precompiled)
    additional_config.artifact_path = artifact_path
    additional_config.precompiled = precompiled
