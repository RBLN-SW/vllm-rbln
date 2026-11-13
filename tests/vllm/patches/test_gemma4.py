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

"""Gemma4's attention backend choice on RBLN.

Config objects only -- no checkpoint, no device.
"""

from types import SimpleNamespace

from vllm.model_executor.models.config import Gemma4Config


class _Layers(list):
    """`model_arch_config` as the hook reads it: indexable per layer."""

    @property
    def total_num_hidden_layers(self) -> int:
        return len(self)


def _vllm_config() -> SimpleNamespace:
    """One sliding (256) and one full (512) layer, Gemma4's head sizes."""
    layers = _Layers([SimpleNamespace(head_size=256), SimpleNamespace(head_size=512)])
    model_config = SimpleNamespace(
        model_arch_config=layers,
        hf_text_config=SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"]
        ),
    )
    return SimpleNamespace(
        model_config=model_config,
        attention_config=SimpleNamespace(backend=None, flash_attn_version=None),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def test_the_backend_is_left_to_the_platform():
    config = _vllm_config()
    Gemma4Config.verify_and_update_config(config)
    assert config.attention_config.backend is None


def test_the_moe_block_routes_the_runner_on_the_residual():
    import torch

    from vllm_rbln.patches.gemma4 import patched_gemma4_decoder_layer_forward

    def same(x):
        return x

    seen = {}

    def experts(*, hidden_states, router):
        seen["logits"] = router(hidden_states)
        return hidden_states

    layer = SimpleNamespace(
        input_layernorm=same,
        self_attn=lambda positions, hidden_states: hidden_states,
        post_attention_layernorm=same,
        pre_feedforward_layernorm=same,
        mlp=same,
        enable_moe_block=True,
        post_feedforward_layernorm_1=same,
        pre_feedforward_layernorm_2=lambda x: x * 3,
        router=lambda x: x + 100,
        experts=experts,
        post_feedforward_layernorm_2=same,
        post_feedforward_layernorm=same,
        per_layer_input_gate=None,
        layer_scalar=torch.ones(1),
    )
    hidden_states = torch.randn(1, 2, 4)

    patched_gemma4_decoder_layer_forward(layer, torch.arange(2), hidden_states, None)

    # The residual after attention is 2x; the experts' input (6x) is ignored.
    torch.testing.assert_close(seen["logits"], hidden_states * 2 + 100)


def test_the_split_mlp_matches_the_merged_one():
    import torch
    from torch import nn
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.gemma4 import Gemma4MLP

    from vllm_rbln.model_executor.models.gemma4 import split_gate_up_weights
    from vllm_rbln.patches.gemma4 import split_gate_up_mlp_forward

    torch.manual_seed(0)
    mlp = Gemma4MLP.__new__(Gemma4MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = nn.Module()
    mlp.gate_up_proj.weight = nn.Parameter(torch.randn(2 * 8, 4))
    mlp.gate_up_proj.quant_method = UnquantizedLinearMethod()
    mlp.act_fn = SimpleNamespace(approximate="tanh")
    mlp.down_proj = lambda x: (x.sum(-1, keepdim=True), None)
    x = torch.randn(3, 4)
    gate, up = (x @ mlp.gate_up_proj.weight.T).chunk(2, dim=-1)
    merged = (nn.functional.gelu(gate, approximate="tanh") * up).sum(-1, keepdim=True)

    split_gate_up_weights(mlp)

    torch.testing.assert_close(split_gate_up_mlp_forward(mlp, x), merged)
    assert not hasattr(mlp.gate_up_proj, "weight")
