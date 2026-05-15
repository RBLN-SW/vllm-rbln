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

"""
vLLM-aware predicates over RBLN model identity and runtime intent.

Kept out of :mod:`registry` so that registry stays free of ``vllm`` imports
(it is on the early import path and pulling ``vllm.config`` from there
triggers circular imports).
"""

from typing import TYPE_CHECKING

from vllm_rbln.utils.optimum.registry import get_rbln_model_info

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def is_qwen3_pooling(model_config: "ModelConfig") -> bool:
    """Return True if the model is the Qwen3 backbone used as a pooler."""
    _, model_cls_name = get_rbln_model_info(model_config)
    return (
        model_cls_name == "RBLNQwen3ForCausalLM"
        and model_config.runner_type == "pooling"
    )
