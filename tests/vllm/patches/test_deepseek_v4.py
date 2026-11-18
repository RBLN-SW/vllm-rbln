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
from vllm.v1.core import kv_cache_utils
from vllm.v1.engine import core as engine_core
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_rbln.patches.deepseek_v4 import deepseek_v4_kv_cache_groups
from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec


def test_config_keeps_upstream_quantization_without_mamba_setup():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    hf_config = SimpleNamespace(
        model_type="deepseek_v4", quantization_config={"quant_method": "fp8"}
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
        kv_transfer_config=None,
    )
    adapter = MODELS_CONFIG_MAP["DeepseekV4ForCausalLM"]
    adapter.verify_and_update_model_config(config.model_config)
    adapter.verify_and_update_config(config)
    assert hf_config.quantization_config["quant_method"] == "deepseek_v4_fp8"


def test_config_rejects_unsupported_ring_transfer():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            block_size=2048,
        ),
        speculative_config=None,
        kv_transfer_config=object(),
    )
    with pytest.raises(ValueError, match="KV transfer"):
        MODELS_CONFIG_MAP["DeepseekV4ForCausalLM"].verify_and_update_config(config)


def test_prefix_caching_requires_ring_checkpoints():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="DeepseekV4ForCausalLM"),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            block_size=2048,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    with pytest.raises(ValueError, match="ring-state checkpoint"):
        MODELS_CONFIG_MAP["DeepseekV4ForCausalLM"].verify_and_update_config(config)


@pytest.mark.parametrize("drafts", [4, 8])
def test_speculation_rejects_unsupported_verification_geometry(monkeypatch, drafts):
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        DeepseekV4ForCausalLMConfig,
    )

    monkeypatch.setattr(
        DeepseekV4ForCausalLMConfig,
        "verify_and_update_config",
        classmethod(lambda cls, cfg: None),
    )

    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts),
    )
    with pytest.raises(ValueError, match="at most 3"):
        MODELS_CONFIG_MAP["DeepseekV4ForCausalLM"].verify_and_update_config(config)


@pytest.fixture
def cache_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v4"),
            max_model_len=64,
            original_max_model_len=64,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, decode_context_parallel_size=1
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        num_prefill_lookahead_tokens=0,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None, prefix_cache_retention_interval=None
        ),
    )


@pytest.fixture
def cache_specs():
    return {
        "csa": MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=32, dtype=torch.uint8
        ),
        "hca": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.uint8,
            tokens_per_state=4,
        ),
        "swa": RBLNDeepseekV4StateSpec(
            num_state_slots=3,
            block_size=8,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.uint8,
        ),
        "compressor": RBLNDeepseekV4StateSpec(
            num_state_slots=3,
            block_size=4,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.float32,
        ),
    }


@pytest.mark.parametrize("num_blocks", [6, 13])
def test_fixed_rings_are_reserved_before_paged_sizing(
    cache_config, cache_specs, num_blocks
):
    from vllm_rbln.patches.dynamic_kv import check_enough_kv_cache_blocks_after_resize
    from vllm_rbln.v1.worker.dynamic_kv_sizer import kv_cache_config_at

    groups = deepseek_v4_kv_cache_groups(cache_config, cache_specs)
    page = cache_specs["csa"].page_size_bytes + cache_specs["hca"].page_size_bytes
    fixed = 3 * (
        cache_specs["swa"].page_size_bytes + cache_specs["compressor"].page_size_bytes
    )
    memory = fixed + num_blocks * page
    cfg = engine_core.get_kv_cache_configs(cache_config, [cache_specs], [memory])[0]
    assert kv_cache_utils._get_kv_cache_bytes_per_block(groups) == page
    assert cfg.num_blocks == num_blocks
    check_enough_kv_cache_blocks_after_resize(cache_config, cfg)
    with pytest.raises(ValueError, match="needs 6"):
        check_enough_kv_cache_blocks_after_resize(
            cache_config, kv_cache_config_at(cfg, 5)
        )
    assert sum(t.layer_stride for t in cfg.kv_cache_tensors) == memory
    assert kv_cache_utils.get_max_concurrency_for_kv_cache_config(
        cache_config, cfg
    ) == min((num_blocks - 1) / 5, 2)


@pytest.mark.parametrize("override", [None, 6])
def test_pp_reconciliation_keeps_fixed_rings(cache_config, cache_specs, override):
    cache_config.cache_config.num_gpu_blocks_override = override
    workers = [
        {"csa": cache_specs["csa"], "swa": cache_specs["swa"]},
        {"hca": cache_specs["hca"]},
        {"compressor": cache_specs["compressor"]},
    ]
    memories = [
        3 * cache_specs["swa"].page_size_bytes
        + 10 * cache_specs["csa"].page_size_bytes,
        6 * cache_specs["hca"].page_size_bytes,
        3 * cache_specs["compressor"].page_size_bytes,
    ]
    configs = engine_core.get_kv_cache_configs(cache_config, workers, memories)
    for cfg, memory in zip(configs, memories, strict=True):
        assert cfg.num_blocks == 6
        assert sum(t.layer_stride for t in cfg.kv_cache_tensors) <= memory
        for tensor in cfg.kv_cache_tensors:
            spec = cache_specs[tensor.layers[0]]
            slots = (
                spec.num_state_slots if isinstance(spec, RBLNDeepseekV4StateSpec) else 6
            )
            assert tensor.layer_stride == slots * spec.page_size_bytes


def test_auto_fit_accounts_for_fixed_rings_and_null_block(cache_config, cache_specs):
    cache_config.model_config.original_max_model_len = -1
    page = cache_specs["csa"].page_size_bytes + cache_specs["hca"].page_size_bytes
    fixed = 3 * (
        cache_specs["swa"].page_size_bytes + cache_specs["compressor"].page_size_bytes
    )
    cfg = engine_core.get_kv_cache_configs(
        cache_config, [cache_specs], [fixed + 5 * page]
    )[0]
    assert cache_config.model_config.max_model_len == 48
    assert cfg.num_blocks == 5


@pytest.mark.parametrize("deferred", [False, True])
def test_scheduler_admission_and_free_use_independent_ring_pools(
    cache_config, cache_specs, deferred
):
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from tests.vllm.v1.core.utils import make_request

    cache_config.cache_config.num_gpu_blocks_override = 64
    cfg = engine_core.get_kv_cache_configs(cache_config, [cache_specs], [1 << 20])[0]
    cfg = kv_cache_utils.generate_scheduler_kv_cache_config([cfg])
    manager = KVCacheManager(cfg, 256, 64, 16, enable_caching=False)
    requests = [make_request(str(i), list(range(32)), 16) for i in range(3)]
    a, b, c = requests
    first = manager.allocate_slots(a, 16)
    second = manager.allocate_slots(b, 16)
    assert first is not None and second is not None
    groups = manager.coordinator.single_type_managers
    rings = [g for g in groups if isinstance(g.kv_cache_spec, CircularBufferSpec)]
    assert len(rings) == 2
    for ring in rings:
        assert ring.block_pool is not manager.block_pool
        assert len(ring.req_to_blocks[a.request_id]) == 1
        assert ring.block_pool.get_num_free_blocks() == 0
    assert manager.block_pool.get_num_free_blocks() > 0
    assert manager.allocate_slots(c, 16) is None

    a.num_computed_tokens = 16
    assert manager.allocate_slots(a, 16) is not None
    for ring in rings:
        assert len(ring.req_to_blocks[a.request_id]) == 1
    if deferred:
        pending = manager.pop_blocks_for_free(a)
        assert manager.allocate_slots(c, 16) is None
        manager.block_pool.free_blocks(reversed(pending))
    else:
        manager.free(a)
    assert manager.allocate_slots(c, 16) is not None
    manager.free(b)
    manager.free(c)
    assert manager.block_pool.get_num_free_blocks() == 63
    assert all(ring.block_pool.get_num_free_blocks() == 2 for ring in rings)
    # Preemption returns the old slot before the request restarts its prefill.
    a.num_computed_tokens = 0
    assert manager.allocate_slots(a, 16) is not None
    manager.free(a)
    assert all(not ring.req_to_blocks for ring in rings)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_non_v4_states_keep_upstream_block_accounting(dtype):
    spec = MambaSpec(block_size=16, shapes=((8, 32),), dtypes=(dtype,))
    groups = [
        KVCacheGroupSpec(["layer.0", "layer.1"], spec),
        KVCacheGroupSpec(["layer.2"], spec),
    ]
    assert (
        kv_cache_utils._get_kv_cache_bytes_per_block(groups) == 2 * spec.page_size_bytes
    )


def test_glm_states_do_not_add_to_the_shared_attention_pool():
    specs = {
        "mla": MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=32, dtype=torch.uint8
        ),
        "indexer": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.uint8,
            tokens_per_state=4,
        ),
    }
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    groups = [
        KVCacheGroupSpec(list(specs), uniform),
        KVCacheGroupSpec(
            ["kda"],
            MambaSpec(block_size=16, shapes=((16, 32),), dtypes=(torch.uint8,)),
        ),
    ]
    assert kv_cache_utils._get_kv_cache_bytes_per_block(groups) == sum(
        spec.page_size_bytes for spec in specs.values()
    )


def test_ring_allocation_is_bounded_by_running_requests():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v4")
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=64, prefix_cache_retention_interval=None
        ),
    )
    ring = RBLNDeepseekV4StateSpec(
        num_state_slots=3,
        block_size=256,
        num_kv_heads=1,
        head_size=768,
        dtype=torch.uint8,
    )
    groups = deepseek_v4_kv_cache_groups(
        config,
        {
            "layers.0.attn": MLAAttentionSpec(
                block_size=2048,
                num_kv_heads=1,
                head_size=768,
                tokens_per_state=4,
                dtype=torch.uint8,
            ),
            "layers.0.ring": ring,
        },
    )
    cfg = kv_cache_utils.get_kv_cache_config_from_groups(config, groups, 1 << 30)
    state = next(t for t in cfg.kv_cache_tensors if t.layers == ["layers.0.ring"])
    assert cfg.num_blocks == 64
    assert state.layer_stride == 3 * ring.page_size_bytes

    from vllm_rbln.v1.worker.dynamic_kv_sizer import kv_cache_config_at

    resized = kv_cache_config_at(cfg, 128)
    state_after = next(t for t in resized.kv_cache_tensors if t.layers == state.layers)
    assert state_after.layer_stride == state.layer_stride
    assert (
        resized.kv_cache_tensors[0].layer_stride
        == 2 * cfg.kv_cache_tensors[0].layer_stride
    )


def test_state_pool_reuses_finished_requests_without_consuming_attention_ids():
    from vllm.v1.core.block_pool import BlockPool

    from vllm_rbln.v1.kv_cache import RBLNDeepseekV4StateManager

    attention = BlockPool(64, False, 16)
    state = RBLNDeepseekV4StateManager(
        RBLNDeepseekV4StateSpec(
            num_state_slots=3,
            block_size=256,
            num_kv_heads=1,
            head_size=768,
            dtype=torch.uint8,
        ),
        block_pool=attention,
        enable_caching=False,
        kv_cache_group_id=1,
        scheduler_block_size=1024,
    )
    a = state.allocate_new_blocks("a", 100, 100)[0]
    b = state.allocate_new_blocks("b", 100, 100)[0]
    assert 0 < a.block_id < 3 and 0 < b.block_id < 3
    assert a.block_id != b.block_id
    assert attention.get_num_free_blocks() == 63
    assert state.get_num_blocks_to_allocate("c", 100, [], 0, 0, 100) > 63
    state.free("a")
    assert state.get_num_blocks_to_allocate("c", 100, [], 0, 0, 100) == 0
    c = state.allocate_new_blocks("c", 100, 100)[0]
    assert c is a
    assert state.req_to_blocks["b"] == [b]
    assert attention.get_num_free_blocks() == 63

    pending = state.pop_blocks_for_free("b")
    assert pending == [b]
    assert state.block_pool.get_num_free_blocks() == 0
    attention.free_blocks(reversed(pending))
    assert state.block_pool.get_num_free_blocks() == 1
    assert attention.get_num_free_blocks() == 63
