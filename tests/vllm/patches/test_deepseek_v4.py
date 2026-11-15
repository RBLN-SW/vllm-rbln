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


@pytest.mark.parametrize("layers", [2, 3])
def test_pp_replan_preserves_reconciled_block_count(layers):
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
            block_size=16, shapes=((8, 32),), dtypes=(torch.uint8,)
        )
    groups = deepseek_v4_kv_cache_groups(config, specs)
    # PP reconciliation re-plans each larger stage with the smallest stage's count.
    memory = 7 * kv_cache_utils._get_kv_cache_bytes_per_block(groups)
    replanned = kv_cache_utils.get_kv_cache_config_from_groups(config, groups, memory)
    assert replanned.num_blocks == 7
    assert sum(t.layer_stride for t in replanned.kv_cache_tensors) == memory


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
