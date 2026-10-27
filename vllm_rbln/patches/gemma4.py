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

from vllm.config import VllmConfig
from vllm.model_executor.models.config import Gemma4Config

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.registry import add_registration

upstream_verify_and_update_config = Gemma4Config.verify_and_update_config


@register_patch(
    target="vllm.model_executor.models.config.Gemma4Config.verify_and_update_config",
    reason=(
        "Upstream picks a CUDA attention backend for Gemma4's mixed head sizes "
        "(256 sliding, 512 full) and falls back to TRITON_ATTN when FA4 is "
        "missing, which RBLN refuses. RBLN's one backend serves both sizes, and "
        "the hook runs inside VllmConfig.__post_init__, before any platform hook "
        "could undo the choice."
    ),
)
def keep_the_rbln_attention_backend(vllm_config: VllmConfig) -> None:
    """Leave the attention backend to the RBLN platform."""


@add_registration(
    reason=(
        "Upstream's Gemma4ForConditionalGeneration merges image embeddings "
        "outside forward() and runs its vision tower through HF's dynamic "
        "attention dispatch. The RBLN subclass merges them in the compiled "
        "graph and compiles the tower at one patch count."
    )
)
def register_gemma4() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Gemma4ForConditionalGeneration",
        "vllm_rbln.model_executor.models.gemma4:RBLNGemma4ForConditionalGeneration",
    )
