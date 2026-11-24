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

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.utils import get_pp_indices
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_rbln import envs
from vllm_rbln.v1.worker.utils import kv_cache_extents


@pytest.mark.parametrize(
    "pp, pool_start, same_phase_layers", [(4, 11, (12, 16)), (8, 10, (10, 14))]
)
@pytest.mark.parametrize("shards", [1, 2, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_short_pp_stages_align_kda_groups_and_cache_pools(
    monkeypatch, pp, pool_start, same_phase_layers, shards, mtp
):
    monkeypatch.setattr(envs, "VLLM_RBLN_NUM_HIDDEN_LAYERS", 0)
    layer_types = ["dsa" if i % 4 == 3 else "kda" for i in range(45)]
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="glm5_next_text", layer_types=layer_types
            ),
            get_total_num_hidden_layers=lambda: 45,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=128, prefix_cache_retention_interval=None
        ),
    )
    specs = {}
    for layer in range(45 + int(mtp)):
        prefix = f"model.layers.{layer}.self_attn.attn."
        if layer % 4 == 3 or layer == 45:
            for slot, head_size, tokens_per_state in ((0, 512, 1), (1, 128, 4)):
                specs[prefix + str(slot)] = MLAAttentionSpec(
                    block_size=1024,
                    num_kv_heads=1,
                    head_size=head_size,
                    dtype=torch.uint8,
                    tokens_per_state=tokens_per_state,
                )
        else:
            for shard in range(shards):
                specs[prefix + str(shard)] = MambaSpec(
                    block_size=16384,
                    shapes=((1024, 512),),
                    dtypes=(torch.uint8,),
                )

    groups = kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs)
    assert groups is not None
    group_ids = {
        name: i for i, group in enumerate(groups) for name in group.layer_names
    }
    assert sorted(name for group in groups for name in group.layer_names) == sorted(
        specs
    )
    for shard in range(shards):
        first = f"model.layers.{same_phase_layers[0]}.self_attn.attn.{shard}"
        second = f"model.layers.{same_phase_layers[1]}.self_attn.attn.{shard}"
        assert group_ids[first] == group_ids[second]

    for rank in range(pp):
        start, end = get_pp_indices(45, rank, pp)
        if rank == pp - 1:
            end += int(mtp)
        worker = {n: s for n, s in specs.items() if start <= int(n.split(".")[2]) < end}
        projected = kv_cache_utils._project_kv_cache_groups_to_worker(groups, worker)
        capacity = sum(
            isinstance(spec, MLAAttentionSpec) and spec.tokens_per_state == 1
            for spec in worker.values()
        )
        assert all(
            len(group.layer_names) <= capacity
            for group in projected
            if isinstance(group.kv_cache_spec, MambaSpec)
        )
        cache = kv_cache_utils.get_kv_cache_config_from_groups(
            config, projected, 1 << 30
        )
        extents = kv_cache_extents(cache)
        assert set(extents) == set(worker)
        if start <= pool_start < end:
            extent = extents[f"model.layers.{pool_start}.self_attn.attn.0"]
            for layer in range(pool_start, pool_start + 4):
                for shard in range(1 if layer % 4 == 3 else shards):
                    assert (
                        extents[f"model.layers.{layer}.self_attn.attn.{shard}"]
                        == extent
                    )
