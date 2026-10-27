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

import functools

from vllm_rbln.patches.registry import add_registration, register_patch

_MODULE = "vllm_rbln.model_executor.models.glm5_next"


@add_registration(
    reason=(
        "Upstream's GLM-5.3 model binds FLA / TileLang / DeepGEMM kernels and the "
        "hybrid mamba state manager, so the RBLN model replaces its architectures."
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


@add_registration(
    reason=(
        "The RBLN GLM-5.3 model is not IsHybrid (its KDA states are MambaSpec head "
        "groups on MLA latent blocks), so vLLM skips the Mamba cache setup a hybrid "
        "model gets. Mapping its architectures to MambaModelConfig gives prefix "
        "caching the 'align' mode and a Mamba block of block_size."
    )
)
def register_glm5_next_mamba_config() -> None:
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

    for arch in ("Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"):
        MODELS_CONFIG_MAP[arch] = MambaModelConfig


@register_patch(
    target="vllm.v1.core.kv_cache_utils._pp_balanced_mamba_group_count",
    reason=(
        "Upstream reads each GLM-5.3 cache's layer with extract_layer_index, which "
        "asserts one integer a name; the RBLN caches carry a slot after the layer "
        "(model.layers.<layer>.self_attn.attn.<slot>), so pipeline parallel fails "
        "there. The wrapper drops the slot and calls upstream."
    ),
    build=True,
)
def _build_pp_balanced_mamba_group_count():
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils._pp_balanced_mamba_group_count

    @functools.wraps(original)
    def _pp_balanced_mamba_group_count(
        vllm_config, mamba_layer_names: list[str], mla_layer_names: list[str]
    ) -> int | None:
        return original(
            vllm_config,
            [name.rsplit(".", 1)[0] for name in mamba_layer_names],
            [name.rsplit(".", 1)[0] for name in mla_layer_names],
        )

    return _pp_balanced_mamba_group_count
