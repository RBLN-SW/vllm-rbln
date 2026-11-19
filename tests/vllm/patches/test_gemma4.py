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

"""Gemma4 on RBLN: the attention backend choice, the decoder layer forward
(MoE routing, the dense prefill residual on the token-cut ops) and the split MLP.

Config objects and stand-in layers only -- no checkpoint, no device.
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

    assert split_gate_up_weights(mlp) == {"gate_weight", "up_weight"}

    torch.testing.assert_close(split_gate_up_mlp_forward(mlp, x), merged)
    assert not hasattr(mlp.gate_up_proj, "weight")


def _rms_norm(x, weight, eps):
    """vLLM's RMSNorm math: fp32 statistics, cast back before the weight."""
    rows = x.float()
    normed = rows * (rows.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (normed.to(weight.dtype) * weight).to(x.dtype)


def _dense_layer(hidden=64):
    """A dense Gemma4DecoderLayer's residual path around stand-in blocks, each
    norm with its own eps."""
    import torch

    layer = SimpleNamespace(enable_moe_block=False, per_layer_input_gate=None)
    for i, name in enumerate(
        (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        )
    ):
        eps = (i + 1) * 1e-6
        weight = 1.0 + 0.1 * torch.randn(hidden)
        setattr(layer, name, _Norm(weight, eps))
    attn = torch.randn(hidden, hidden) / hidden**0.5
    mlp = torch.randn(hidden, hidden) / hidden**0.5
    layer.self_attn = lambda positions, hidden_states: hidden_states @ attn
    layer.mlp = lambda x: x @ mlp
    layer.layer_scalar = torch.tensor([0.75])
    return layer


class _Norm:
    def __init__(self, weight, eps):
        self.weight = weight
        self.variance_epsilon = eps

    def __call__(self, x):
        return _rms_norm(x, self.weight, self.variance_epsilon)


def _token_cut_ops(monkeypatch):
    """The two ops as their CPU math, recording each call."""
    import torch

    calls = []

    def read(residual, weight, eps):
        calls.append("read")
        return _rms_norm(residual, weight, eps)

    def write(residual, y, weight, eps):
        calls.append("write")
        return _rms_norm(y, weight, eps) + residual

    ops = torch.ops.rbln_custom_ops
    monkeypatch.setattr(ops, "residual_read", read, raising=False)
    monkeypatch.setattr(ops, "gemma4_dense_residual_write", write, raising=False)
    return calls


def _run(layer, hidden_states, prefill, monkeypatch, **kwargs):
    import torch

    from vllm_rbln.patches import gemma4

    monkeypatch.setattr(gemma4, "is_prefill_step", lambda: prefill)
    positions = torch.arange(hidden_states.shape[-2])
    out, _ = gemma4.patched_gemma4_decoder_layer_forward(
        layer, positions, hidden_states, None, **kwargs
    )
    return out


def test_the_dense_prefill_residual_runs_on_the_token_cut_ops(monkeypatch):
    import torch

    torch.manual_seed(0)
    calls = _token_cut_ops(monkeypatch)
    layer = _dense_layer()
    hidden_states = torch.randn(1, 8, 64)

    cut = _run(layer, hidden_states, True, monkeypatch)
    assert calls == ["read", "write", "read", "write"]
    plain = _run(layer, hidden_states, False, monkeypatch)
    torch.testing.assert_close(cut, plain)


def test_decode_keeps_the_plain_layer(monkeypatch):
    import torch

    calls = _token_cut_ops(monkeypatch)

    _run(_dense_layer(), torch.randn(4, 1, 64), False, monkeypatch)

    assert calls == []


def test_a_per_layer_input_keeps_the_plain_layer(monkeypatch):
    import torch

    calls = _token_cut_ops(monkeypatch)
    layer = _dense_layer()
    layer.per_layer_input_gate = lambda x: x[..., :8]
    layer.per_layer_projection = lambda x: x.repeat(1, 1, 8)
    layer.post_per_layer_input_norm = lambda x: x

    _run(
        layer,
        torch.randn(1, 8, 64),
        True,
        monkeypatch,
        per_layer_input=torch.randn(1, 8, 8),
    )

    assert calls == []
