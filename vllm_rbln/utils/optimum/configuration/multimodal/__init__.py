# SPDX-License-Identifier: Apache-2.0
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

from collections.abc import Callable

from .blip2 import build_blip2_config
from .exaone import build_exaone4_5_config
from .gemma import build_gemma3_config, build_gemma4_config
from .idefics3 import build_idefics3_config
from .llama4 import build_llama4_config
from .llava import build_llava_config, build_llava_next_config
from .paligemma import build_paligemma_config
from .qwen import (
    build_qwen2_5_vl_config,
    build_qwen2_vl_config,
    build_qwen3_5_config,
    build_qwen3_asr_config,
    build_qwen3_vl_config,
    build_qwen3_vl_moe_config,
)

MULTIMODAL_CONFIG_BUILDERS: dict[
    str, Callable[[int, int, int, int, float, int | None], dict]
] = {
    "blip2": build_blip2_config,
    "exaone4_5": build_exaone4_5_config,
    "idefics3": build_idefics3_config,
    "llama4": build_llama4_config,
    "llava": build_llava_config,
    "llava_next": build_llava_next_config,
    "paligemma": build_paligemma_config,
    "gemma3": build_gemma3_config,
    "gemma4": build_gemma4_config,
    "qwen2_vl": build_qwen2_vl_config,
    "qwen2_5_vl": build_qwen2_5_vl_config,
    "qwen3_vl": build_qwen3_vl_config,
    "qwen3_vl_moe": build_qwen3_vl_moe_config,
    "qwen3_5": build_qwen3_5_config,
    "qwen3_asr": build_qwen3_asr_config,
}
