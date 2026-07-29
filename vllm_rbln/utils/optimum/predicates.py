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

from collections.abc import Sized
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


def is_qwen3_reranker(model_config: "ModelConfig") -> bool:
    """Return True for the original Qwen3-Reranker loaded for ``score()``.

    Keyed on ``classifier_from_token`` rather than the architecture name, so the
    predicate still holds after the model runner has remapped the HF arch to
    ``Qwen3Model`` for optimum-rbln. A checkpoint already converted offline to
    sequence classification carries a real ``score`` layer and no
    ``classifier_from_token``, so it does not match here.
    """
    if model_config.runner_type != "pooling":
        return False

    hf_config = model_config.hf_config
    text_config = hf_config.get_text_config()
    if not getattr(hf_config, "is_original_qwen3_reranker", False):
        return False

    tokens: Sized | None = getattr(
        hf_config,
        "classifier_from_token",
        getattr(text_config, "classifier_from_token", None),
    )
    if tokens is None:
        return False
    return len(tokens) == 2


def forces_fp32_dtype(model_config: "ModelConfig") -> bool:
    import optimum.rbln

    _, model_cls_name = get_rbln_model_info(model_config)
    model_cls = getattr(optimum.rbln, model_cls_name)
    return not getattr(model_cls, "_supports_non_fp32", False)
