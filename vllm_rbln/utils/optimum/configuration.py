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

"""Top-level vLLM ↔ RBLN config synchronisation entry points."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
)

logger = init_logger(__name__)


def is_qwen3_pooling(
    vllm_config: VllmConfig,
) -> bool:
    _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
    return (
        model_cls_name in ["RBLNQwen3ForCausalLM"]
        and vllm_config.model_config.runner_type == "pooling"
    )
