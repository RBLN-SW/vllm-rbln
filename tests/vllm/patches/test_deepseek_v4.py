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
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_rbln.patches.deepseek_v4 import deepseek_v4_kv_cache_groups
from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec


def test_prefix_caching_requires_ring_checkpoints():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="DeepseekV4ForCausalLM"),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="none",
            mamba_block_size=None,
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


@pytest.mark.parametrize("layers", [2, 3])
@pytest.mark.parametrize("num_blocks", [2, 7])
def test_pp_replan_preserves_reconciled_block_count(layers, num_blocks):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v4")
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None, prefix_cache_retention_interval=None
        ),
    )
    specs = {}
    for layer in range(layers):
        specs[f"layers.{layer}.attn"] = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=32, dtype=torch.uint8
        )
        specs[f"layers.{layer}.ring"] = RBLNDeepseekV4StateSpec(
            num_state_slots=3, block_size=16, shapes=((8, 32),), dtypes=(torch.uint8,)
        )
    groups = deepseek_v4_kv_cache_groups(config, specs)
    # PP reconciliation re-plans each larger stage with the smallest stage's count.
    memory = num_blocks * kv_cache_utils._get_kv_cache_bytes_per_block(groups)
    if num_blocks < 3:
        with pytest.raises(ValueError, match="fixed state slots"):
            kv_cache_utils.get_kv_cache_config_from_groups(config, groups, memory)
        return
    replanned = kv_cache_utils.get_kv_cache_config_from_groups(config, groups, memory)
    assert replanned.num_blocks == num_blocks
    assert sum(t.layer_stride for t in replanned.kv_cache_tensors) <= memory


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
        block_size=1048576,
        shapes=((256, 768),),
        dtypes=(torch.uint8,),
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
            block_size=1024,
            shapes=((256, 768),),
            dtypes=(torch.uint8,),
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
