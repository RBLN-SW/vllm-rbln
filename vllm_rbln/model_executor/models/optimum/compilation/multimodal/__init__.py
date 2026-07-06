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

from .blip2 import get_param_blip2
from .exaone4_5 import get_param_exaone4_5
from .gemma3 import get_param_gemma3, get_param_gemma4
from .idefics3 import get_param_idefics3
from .llava import get_param_llava, get_param_llava_next
from .paligemma import get_param_paligemma
from .qwen import (
    get_param_qwen2_5_vl,
    get_param_qwen2_vl,
    get_param_qwen3_vl,
    get_param_qwen3_vl_moe,
)

_COMPILE_MULTIMODAL_FNS: dict[str, Callable[[int, int, int, int, int | None], dict]] = {
    "blip2": get_param_blip2,
    "exaone4_5": get_param_exaone4_5,
    "idefics3": get_param_idefics3,
    "llava": get_param_llava,
    "llava_next": get_param_llava_next,
    "paligemma": get_param_paligemma,
    "gemma3": get_param_gemma3,
    "gemma4": get_param_gemma4,
    "qwen2_vl": get_param_qwen2_vl,
    "qwen2_5_vl": get_param_qwen2_5_vl,
    "qwen3_vl": get_param_qwen3_vl,
    "qwen3_vl_moe": get_param_qwen3_vl_moe,
}
