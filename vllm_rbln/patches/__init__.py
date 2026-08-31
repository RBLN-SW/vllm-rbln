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

# NOTE(RBLN): Runtime monkey-patches applied when the RBLN plugin is loaded.

from vllm_rbln.patches.registry import (
    PatchDescriptor,
    RegistrationDescriptor,
    add_registration,
    apply_registered_patches,
    apply_registrations,
    register_patch,
)

# ruff: noqa: F401
from . import (
    attention,
    axk2,
    deepseek_mtp,
    deepseek_v2,
    distributed_utils,
    dynamic_kv,
    fp8_moe_method,
    gpt_oss,
    gpt_oss_mxfp4_config,
    llama_eagle3,
    metrics,
    minimax_m2,
    mla,
    modelopt_mixed_config,
    models_utils,
    multi_connector,
    oot,
    profiler,
    qwen2_moe,
    qwen3_moe,
    rotary_embedding,
    speculative_config,
)

__all__ = (
    "PatchDescriptor",
    "RegistrationDescriptor",
    "add_registration",
    "apply_registered_patches",
    "apply_registrations",
    "register_patch",
)
