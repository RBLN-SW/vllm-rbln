# SPDX-License-Identifier: Apache-2.0
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

from collections.abc import Callable
from typing import Any

from optimum.rbln import RBLNAutoModelForImageTextToText, RBLNAutoModelForVision2Seq

from .blip2 import get_param_blip2
from .gemma3 import get_param_gemma3
from .idefics3 import get_param_idefics3
from .llava import get_param_llava, get_param_llava_next
from .paligemma import get_param_paligemma
from .qwen import get_param_qwen2_5_vl, get_param_qwen2_vl


def get_multimodal_cls(architecture: str) -> type[Any]:
    if architecture == "Gemma3ForConditionalGeneration":
        return RBLNAutoModelForImageTextToText
    else:
        return RBLNAutoModelForVision2Seq


_COMPILE_FNS: dict[str, Callable[[int, int, int, int], dict]] = {
    "blip2": get_param_blip2,
    "idefics3": get_param_idefics3,
    "llava": get_param_llava,
    "llava_next": get_param_llava_next,
    "paligemma": get_param_paligemma,
    "gemma3": get_param_gemma3,
    "qwen2_vl": get_param_qwen2_vl,
    "qwen2_5_vl": get_param_qwen2_5_vl,
}


def compile_multimodal(
    model_name: str,
    architecture: str,
    model_alias: str,
    batch_size: int,
    max_model_len: int,
    block_size: int,
    tp_size: int,
) -> dict:
    model_cls = get_multimodal_cls(architecture)
    compile_fn = _COMPILE_FNS.get(model_alias)
    if compile_fn is None:
        raise ValueError(
            f"Unknown multimodal model alias: {model_alias}. "
            f"Supported aliases: {sorted(_COMPILE_FNS.keys())}"
        )
    param = compile_fn(batch_size, max_model_len, block_size, tp_size)
    model = model_cls.from_pretrained(model_name, export=True, rbln_config=param)
    return model
