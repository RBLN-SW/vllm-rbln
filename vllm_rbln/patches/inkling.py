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

from vllm_rbln.patches.registry import add_registration

_MODULE = "vllm_rbln.model_executor.models.inkling"


@add_registration(
    reason=(
        "Upstream's Inkling binds CUDA / Triton / FA4 kernels"
        "attention kernel takes its per-head relative-position bias, so the RBLN"
        "need to replace"
    )
)
def register_inkling() -> None:
    from vllm.model_executor.models import ModelRegistry

    for arch in ("InklingForCausalLM", "InklingForConditionalGeneration"):
        ModelRegistry.register_model(arch, f"{_MODULE}:RBLNInklingForCausalLM")
