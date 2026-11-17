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


@register_patch(
    target="vllm.v1.core.kv_cache_utils._get_kv_cache_groups_glm5_next",
    reason=(
        "Upstream exposes no group-placement hook. Its KDA groups make "
        "whole-pool graph inputs cross attention periods inconsistently. "
        "Groups aligned with attention phases while retaining scheduler's group count."
    ),
    build=True,
)
def _build_glm5_next_phase_groups():
    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_glm5_next
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    @functools.wraps(_get_kv_cache_groups_glm5_next)
    def phase_groups(vllm_config, kv_cache_spec):
        groups = _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec)
        if groups is None:
            return groups
        config = vllm_config.model_config.hf_text_config
        if config.model_type not in ("glm5_next", "glm5_next_text"):
            return groups
        if vllm_config.kv_transfer_config is not None:
            return groups

        layer_types = config.layer_types
        period = next(
            p
            for p in range(1, len(layer_types) + 1)
            if all(kind == layer_types[i % p] for i, kind in enumerate(layer_types))
        )
        states = sorted(
            (
                extract_layer_index(name.rsplit(".", 1)[0]),
                int(name.rsplit(".", 1)[1]),
                name,
            )
            for name, spec in kv_cache_spec.items()
            if isinstance(spec, MambaSpec)
        )
        phases = sorted({(layer % period, shard) for layer, shard, _ in states})
        state_groups = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)]

        if len(phases) > len(state_groups):
            return groups

        mla_layers = [
            extract_layer_index(name.rsplit(".", 1)[0])
            for name, spec in kv_cache_spec.items()
            if type(spec) is MLAAttentionSpec and spec.tokens_per_state == 1
        ]
        phase_ids = {phase: i for i, phase in enumerate(phases)}
        state_layers = {name: layer for layer, _, name in states}
        names: list[list[str]] = [[] for _ in state_groups]
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        for rank in range(pp_size):
            start, end = get_pp_indices(total_layers, rank, pp_size)
            num_target_layers = min(end, total_layers) - start
            if rank == pp_size - 1:
                # Draft attention belongs to the last stage and also owns slots.
                end = max(end, max(mla_layers) + 1, states[-1][0] + 1)
            # Carrying a stage's device input peels one period; rolling needs
            # two more. Avoid enlarging aux pools on stages too short to roll.
            if pp_size > 1 and num_target_layers < 3 * period:
                for selected, group in zip(names, state_groups):
                    selected.extend(
                        name
                        for name in group.layer_names
                        if start <= state_layers[name] < end
                    )
                continue
            capacity = sum(start <= layer < end for layer in mla_layers)
            local: list[list[str]] = [[] for _ in state_groups]
            overflow = []
            for layer, shard, name in states:
                if start <= layer < end:
                    group_id = phase_ids[layer % period, shard]
                    if len(local[group_id]) < capacity:
                        local[group_id].append(name)
                    else:
                        overflow.append(name)

            extra_groups = len(state_groups) - len(phases)
            assert not overflow or extra_groups > 0
            for i, name in enumerate(overflow):
                local[len(phases) + i % extra_groups].append(name)
            assert all(len(group) <= capacity for group in local)
            for selected, stage_names in zip(names, local):
                selected.extend(stage_names)

        for group, selected in zip(state_groups, names):
            group.layer_names = selected
        return groups

    return phase_groups
