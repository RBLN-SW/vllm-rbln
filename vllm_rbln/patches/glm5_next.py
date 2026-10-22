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

from vllm_rbln.patches.registry import add_registration, register_patch

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


def _glm5_layer_index(name: str) -> int:
    # model.layers.<layer>.self_attn.attn.<slot>: the layer is the integer after "layers."
    return int(name.split("layers.", 1)[1].split(".", 1)[0])


@register_patch(
    target="vllm.v1.core.kv_cache_utils._pp_balanced_mamba_group_count",
    reason=(
        "Upstream reads a GLM-5.3 cache's layer with extract_layer_index, which asserts "
        "one integer per name; the RBLN caches carry a slot after the layer "
        "(model.layers.<layer>.self_attn.attn.<slot>), so pipeline parallel fails here."
    ),
)
def _pp_balanced_mamba_group_count(
    vllm_config, mamba_layer_names: list[str], mla_layer_names: list[str]
) -> int | None:
    from vllm.distributed.utils import get_pp_indices
    from vllm.utils.math_utils import cdiv

    num_groups = cdiv(len(mamba_layer_names), len(mla_layer_names))
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if pp_size == 1:
        return num_groups
    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    mamba_indices = [_glm5_layer_index(name) for name in mamba_layer_names]
    mla_indices = [_glm5_layer_index(name) for name in mla_layer_names]
    for rank in range(pp_size):
        start, end = get_pp_indices(total_layers, rank, pp_size)
        num_mamba = sum(start <= index < end for index in mamba_indices)
        num_mla = sum(start <= index < end for index in mla_indices)
        if not num_mamba:
            continue
        if not num_mla:
            return None
        num_groups = max(num_groups, cdiv(num_mamba, num_mla))
    return num_groups
