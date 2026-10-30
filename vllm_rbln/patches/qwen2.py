# Copyright 2026 Rebellions Inc. All rights reserved.
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

import torch

from vllm_rbln import envs
from vllm_rbln.patches import register_patch

if envs.VLLM_RBLN_USE_NESTED_COMPILE_REGION:
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

    register_patch(
        target="vllm.model_executor.models.qwen2.Qwen2DecoderLayer.forward",
        owner_module=__name__,
        reason=(
            "Upstream Qwen2 exposes no decoder compile-region hook. Preserve "
            "decoder bodies for RBLN loop-compilation bring-up using "
            "invoke_subgraph boundaries."
        ),
    )(torch.compiler.nested_compile_region(Qwen2DecoderLayer.forward))
