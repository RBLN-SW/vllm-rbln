# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm_rbln.patches.registry import add_registration

_MODULE = "vllm_rbln.model_executor.models.glm5_next"


@add_registration(
    reason=(
        "Upstream's GLM-5.3 model binds FLA / TileLang / DeepGEMM kernels and the hybrid "
        "mamba state manager, so the RBLN text model (KDA, kpool DSA and mHC custom ops) "
        "replaces both of its architectures."
    )
)
def register_glm5_next() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Glm5NextForCausalLM", f"{_MODULE}:RBLNGlm5NextForCausalLM"
    )
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        f"{_MODULE}:RBLNGlm5NextForConditionalGeneration",
    )
    # The MTP drafter vLLM's speculative config selects for a glm5_next target.
    ModelRegistry.register_model("Glm5NextMTPModel", f"{_MODULE}:RBLNGlm5NextMTP")
