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

"""The RBLN GatedDeltaNet layer's state spec, its out-of-tree registration and
the architectures it refuses."""

from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.config import RBLNConfig
from vllm_rbln.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    RBLNQwenGatedDeltaNetAttention,
)
from vllm_rbln.model_executor.models.qwen3_5 import RBLNQwen3_5ForConditionalGeneration


class TestStateSpec:
    """State specs describe the physical KV page used by the runner."""

    @staticmethod
    def _layer():
        layer = object.__new__(RBLNQwenGatedDeltaNetAttention)
        layer.num_kv_heads = 4
        layer.kv_head_size = 256
        layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
        layer.cache_config = SimpleNamespace(
            block_size=1024,
            mamba_block_size=8192,
            mamba_page_size_padded=None,
            mamba_cache_mode="none",
            use_kda_recoverssm=False,
        )
        return layer

    def test_spec_uses_full_attention_page(self):
        layer = self._layer()
        config = SimpleNamespace(
            cache_config=layer.cache_config,
            num_speculative_tokens=0,
            additional_config=RBLNConfig(),
        )
        spec = layer.get_kv_cache_spec(config)
        assert spec.shapes == ((2, 4, 1, 1024, 256),)
        assert spec.real_page_size_bytes == spec.page_size_bytes

    def test_model_state_sizing_uses_valid_shape(self):
        layer = self._layer()
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                hf_text_config=SimpleNamespace(
                    linear_num_value_heads=48,
                    linear_key_head_dim=128,
                    linear_value_head_dim=128,
                ),
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            cache_config=layer.cache_config,
            num_speculative_tokens=0,
            additional_config=RBLNConfig(),
        )
        model = RBLNQwen3_5ForConditionalGeneration
        assert model.get_mamba_state_shape_from_config(vllm_config) == ((48, 128, 128),)
        assert model.get_mamba_state_dtype_from_config(vllm_config) == (
            self._layer().get_state_dtype()
        )


def test_upstreams_layer_builds_the_rbln_one():
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    assert isinstance(layer, RBLNQwenGatedDeltaNetAttention)


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3NextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        # Upstream's own classes: no RBLN class regroups their weights.
        "Qwen3_5ForCausalLM",
        "InternS2MobiusForConditionalGeneration",
    ],
)
def test_another_architectures_layer_is_refused(architecture):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architecture=architecture)
    )
    with pytest.raises(NotImplementedError, match="dense Qwen3.5"):
        RBLNQwenGatedDeltaNetAttention(None, vllm_config)
