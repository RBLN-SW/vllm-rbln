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


@pytest.fixture(autouse=True)
def _full_cache_layout(monkeypatch):
    # The synthetic cache specs already include every layer, independently of
    # the suite's model-construction limit.
    monkeypatch.setattr(envs, "VLLM_RBLN_NUM_HIDDEN_LAYERS", 0)


def _config_and_specs(*, pp=1, shards=1, layers=45, mtp=False, mode="none"):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="glm5_next_text",
                layer_types=[
                    "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
                    for i in range(layers)
                ],
            ),
            get_total_num_hidden_layers=lambda: layers,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=128, prefix_cache_retention_interval=None
        ),
    )
    specs = {}
    for i in range(layers + int(mtp)):
        prefix = f"model.layers.{i}.self_attn.attn."
        if i % 4 == 3 or i == layers:
            specs[prefix + "0"] = MLAAttentionSpec(
                block_size=1024,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float8_e4m3fn,
            )
            specs[prefix + "1"] = MLAAttentionSpec(
                block_size=1024,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.float8_e4m3fn,
                tokens_per_state=4,
            )
        else:
            for shard in range(shards):
                specs[prefix + str(shard)] = MambaSpec(
                    block_size=16384 if mode == "none" else 1024,
                    shapes=((1024, 512),),
                    dtypes=(torch.uint8,),
                    mamba_cache_mode=mode,
                )
    return config, specs


def test_groups_align_the_three_kda_phases_and_keep_the_tail_separate():
    config, specs = _config_and_specs()
    groups = kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs)
    assert groups is not None
    assert [group.layer_names for group in groups[1:]] == [
        [f"model.layers.{i}.self_attn.attn.0" for i in range(phase, 44, 4)]
        for phase in range(3)
    ] + [["model.layers.44.self_attn.attn.0"]]

    cache = kv_cache_utils.get_kv_cache_config_from_groups(config, groups, 1 << 30)
    extents = kv_cache_extents(cache)
    for start in range(0, 44, 4):
        assert (
            len(
                {
                    extents[f"model.layers.{i}.self_attn.attn.0"]
                    for i in range(start, start + 4)
                }
            )
            == 1
        )
    assert cache.num_blocks == 128
    assert len(set(extents.values())) == 22


@pytest.mark.parametrize("pp", [1, 2, 4])
@pytest.mark.parametrize("shards", [1, 2, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_phase_groups_fit_every_pipeline_stage_without_losing_states(pp, shards, mtp):
    config, specs = _config_and_specs(pp=pp, shards=shards, mtp=mtp)
    config.cache_config.num_gpu_blocks_override = None
    groups = kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs)
    assert groups is not None
    original = kv_cache_utils._get_kv_cache_groups_glm5_next.__wrapped__(config, specs)
    assert [g.kv_cache_spec for g in groups] == [g.kv_cache_spec for g in original]
    if mtp and pp == shards == 1:
        # The draft DSA supplies a twelfth slot, so upstream already has the
        # three phase groups and needs no rearrangement.
        assert groups == original
    names = [name for group in groups for name in group.layer_names]
    assert sorted(names) == sorted(specs)
    for rank in range(pp):
        start, end = get_pp_indices(45, rank, pp)
        if rank == pp - 1:
            end += int(mtp)
        worker = {n: s for n, s in specs.items() if start <= int(n.split(".")[2]) < end}
        projected = kv_cache_utils._project_kv_cache_groups_to_worker(groups, worker)
        original_projected = kv_cache_utils._project_kv_cache_groups_to_worker(
            original, worker
        )
        for phase in range(3):
            for shard in range(shards):
                group = projected[1 + phase * shards + shard]
                assert all(
                    int(name.split(".")[2]) % 4 == phase
                    and name.endswith(f".attn.{shard}")
                    for name in group.layer_names
                )
        slots = sum(
            isinstance(s, MLAAttentionSpec) and s.tokens_per_state == 1
            for s in worker.values()
        )
        assert all(len(g.layer_names) <= slots for g in projected[1:])
        cache = kv_cache_utils.get_kv_cache_config_from_groups(
            config, projected, 1 << 30
        )
        before = kv_cache_utils.get_kv_cache_config_from_groups(
            config, original_projected, 1 << 30
        )
        assert cache.num_blocks == before.num_blocks
        assert set(kv_cache_extents(cache).values()) == set(
            kv_cache_extents(before).values()
        )
        tensors = {
            name: tensor for tensor in cache.kv_cache_tensors for name in tensor.layers
        }
        assert set(tensors) == set(worker)
        intervals = []
        for group_id, group in enumerate(projected):
            for name in group.layer_names:
                tensor = tensors[name]
                offset = tensor.offset + (group_id + 1) * tensor.block_stride
                intervals.append((offset, offset + tensor.block_stride))
        intervals.sort()
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("pp", [1, 4])
def test_regrouping_preserves_allocations_and_prefix_cache_hits(mode, pp):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.request import Request

    config, specs = _config_and_specs(pp=pp, mode=mode)
    changed = kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs)
    original = kv_cache_utils._get_kv_cache_groups_glm5_next.__wrapped__(config, specs)
    assert [g.layer_names for g in changed] != [g.layer_names for g in original]
    register_all_kvcache_specs(config)
    kv_cache_utils.init_none_hash(sha256)
    hasher = kv_cache_utils.get_request_block_hasher(1024, sha256)
    traces = []
    for groups in (original, changed):
        caches = []
        for rank in range(pp):
            start, end = get_pp_indices(45, rank, pp)
            worker = {
                n: s for n, s in specs.items() if start <= int(n.split(".")[2]) < end
            }
            projected = kv_cache_utils._project_kv_cache_groups_to_worker(
                groups, worker
            )
            caches.append(
                kv_cache_utils.get_kv_cache_config_from_groups(
                    config, projected, 1 << 30
                )
            )
        cache = kv_cache_utils.generate_scheduler_kv_cache_config(caches)
        manager = KVCacheManager(
            cache,
            max_model_len=16384,
            scheduler_block_size=16384 if mode == "none" else 1024,
            hash_block_size=1024,
            enable_caching=mode == "align",
        )
        trace, active = [], []
        for i, length in enumerate([2048, 2064, 8192, 8208, 15360, 15376, 22, 38]):
            manager.coordinator.new_step_starts()
            request = Request(
                str(i),
                [i // 2 + 10] * length,
                SamplingParams(max_tokens=32),
                None,
                block_hasher=hasher,
            )
            computed, hit, boundary = manager.get_computed_blocks(request)
            allocated = manager.allocate_slots(
                request,
                length - hit,
                num_new_computed_tokens=hit,
                new_computed_blocks=computed,
            )
            assert allocated is not None
            request.num_computed_tokens = length
            manager.cache_blocks(request, length)
            trace.append(
                (
                    hit,
                    boundary,
                    allocated.get_block_ids(),
                    manager.block_pool.get_num_free_blocks(),
                )
            )
            active.append(request)
            if i % 2 == 1:
                for finished in reversed(active):
                    manager.free(finished)
                active.clear()
        traces.append(trace)
    assert traces[0] == traces[1]
    if mode == "align":
        assert any(row[0] > 0 for row in traces[1])


@pytest.mark.parametrize("layers", [5, 8, 12, 44, 45, 48, 49])
def test_short_and_complete_stacks_keep_every_cache(layers):
    config, specs = _config_and_specs(layers=layers)
    groups = kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs)
    cache = kv_cache_utils.get_kv_cache_config_from_groups(config, groups, 1 << 30)
    assert sorted(n for t in cache.kv_cache_tensors for n in t.layers) == sorted(specs)
    assert len(groups) == (5 if layers % 4 else 4)


def test_nonperiodic_and_other_models_keep_upstream_grouping():
    for other_model in (False, True):
        config, specs = _config_and_specs()
        if other_model:
            config.model_config.hf_text_config.model_type = "other"
        else:
            config.model_config.hf_text_config.layer_types[10] = "other_attention"
        expected = kv_cache_utils._get_kv_cache_groups_glm5_next.__wrapped__(
            config, specs
        )
        assert kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs) == expected


@pytest.mark.parametrize("pp", [1, 2, 4])
def test_kv_connectors_keep_the_existing_layer_to_group_mapping(pp):
    config, specs = _config_and_specs(pp=pp)
    config.kv_transfer_config = SimpleNamespace(kv_connector="RblnNixlConnector")
    expected = kv_cache_utils._get_kv_cache_groups_glm5_next.__wrapped__(config, specs)
    assert kv_cache_utils._get_kv_cache_groups_glm5_next(config, specs) == expected


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
