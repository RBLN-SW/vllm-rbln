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

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.model_executor.layers.mamba.mamba_mixer2 import RBLNMambaMixer2
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionBackend
from vllm_rbln.v1.attention.backends.mamba2_attn import (
    RBLNMamba2AttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import (
    KVCacheViewInfo,
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)

HIDDEN, HEADS, GROUPS, HEAD_DIM, STATE, KERNEL = 64, 8, 4, 64, 64, 4
INNER, GROUP_CHANNELS = HEADS * HEAD_DIM, GROUPS * STATE
CONV_DIM = INNER + 2 * GROUP_CHANNELS
EPS = 1e-5
PREFIX = "model.layers.0.mixer"
# The state cache: the request's block, with a neighbour on each side.
NUM_BLOCKS, BLOCK = 4, 2


class _Proj(torch.nn.Linear):
    """vLLM's linear layers return (output, bias)."""

    def forward(self, x):
        return super().forward(x), None


def _gated_norm(weight):
    """Mixer2RMSNormGated at TP 1: RMSNorm over each group of x * silu(gate)."""

    def norm(x, gate):
        y = x.float() * F.silu(gate.float())
        y = y.view(*y.shape[:-1], GROUPS, -1)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
        return weight * y.reshape(*x.shape).to(x.dtype)

    return norm


def _layer(ring_x, ring_bc):
    torch.manual_seed(0)
    layer = SimpleNamespace(
        prefix=PREFIX,
        tp_size=1,
        local_groups=GROUPS,
        local_heads=HEADS,
        tped_intermediate_size=INNER,
        tped_conv_size=CONV_DIM,
        tped_dt_size=HEADS,
        split_hidden_states_B_C_fn=lambda xbc: xbc.split(
            [INNER, GROUP_CHANNELS, GROUP_CHANNELS], dim=-1
        ),
        groups_ssm_state_size=GROUP_CHANNELS,
        ssm_state_size=STATE,
        head_dim=HEAD_DIM,
        conv_kernel_size=KERNEL,
        num_spec=0,
        conv_ring_x=SimpleNamespace(inputs=lambda: ring_x),
        conv_ring_bc=SimpleNamespace(inputs=lambda: ring_bc),
    )
    layer.in_proj = _Proj(HIDDEN, 2 * INNER + 2 * GROUP_CHANNELS + HEADS, bias=False)
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(CONV_DIM, 1, KERNEL) * 0.3, bias=torch.randn(CONV_DIM) * 0.1
    )
    layer.A = -torch.rand(HEADS) * 4 - 0.5
    layer.dt_bias = torch.rand(HEADS) - 2.0
    layer.D = torch.randn(HEADS)
    layer.norm_weight = torch.rand(INNER) + 0.5
    layer.norm = _gated_norm(layer.norm_weight)
    layer.out_proj = _Proj(INNER, HIDDEN, bias=False)
    return layer


def _reference(layer, hidden, conv_history, ssm_state):
    """Upstream MambaMixer2 over the valid tokens, one token at a time:
    conv(+bias) and SiLU over x | B | C, dt = softplus(dt + dt_bias),
    S <- exp(dt A) S + dt x B^T, y = S C + D x, the gated group norm."""
    seq_len = hidden.shape[1]
    projected = layer.in_proj(hidden)[0][0]
    gate, xbc, dt = projected.split([INNER, CONV_DIM, HEADS], dim=-1)
    if conv_history is None:
        conv_history = torch.zeros(CONV_DIM, KERNEL - 1)
    window = torch.cat([conv_history, xbc.T], dim=-1)
    conv = F.conv1d(
        window[None], layer.conv1d.weight, layer.conv1d.bias, groups=CONV_DIM
    )[0]
    x, b, c = F.silu(conv).T.split([INNER, GROUP_CHANNELS, GROUP_CHANNELS], dim=-1)
    x = x.view(seq_len, HEADS, HEAD_DIM)
    b = b.view(seq_len, GROUPS, STATE).repeat_interleave(HEADS // GROUPS, dim=1)
    c = c.view(seq_len, GROUPS, STATE).repeat_interleave(HEADS // GROUPS, dim=1)
    dt = F.softplus(dt + layer.dt_bias)
    state = torch.zeros(HEADS, HEAD_DIM, STATE) if ssm_state is None else ssm_state
    ys = []
    for t in range(seq_len):
        decay = torch.exp(dt[t] * layer.A).view(HEADS, 1, 1)
        update = (dt[t].view(HEADS, 1, 1) * x[t].unsqueeze(-1)) * b[t].unsqueeze(1)
        state = state * decay + update
        ys.append((state * c[t].unsqueeze(1)).sum(-1) + layer.D.view(HEADS, 1) * x[t])
    y = torch.stack(ys).reshape(1, seq_len, INNER)
    output = layer.out_proj(layer.norm(y, gate[None]))[0]
    return output, window[:, -(KERNEL - 1) :], state


def _common(query_len: int, seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=query_len,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table_tensor=torch.tensor([[BLOCK]], dtype=torch.int32),
    )


def _run(layer, vllm_config, builder, state, hidden, common, positions, prefill, pad):
    metadata = builder.build(common, positions, pad, prefill)
    infos = [KVCacheViewInfo(base_index=0, view_shape=tuple(state.shape))]
    attach_kv_cache_bindings(
        metadata, [state], [state], infos, kv_cache_positions={PREFIX: 0}
    )
    layer.conv_ssm_forward = MethodType(RBLNMambaMixer2.conv_ssm_forward, layer)
    with set_forward_context(
        {PREFIX: metadata},
        vllm_config,
        **build_kv_cache_forward_context_kwargs([state]),
    ):
        return RBLNMambaMixer2.forward(layer, hidden)


@pytest.mark.parametrize(
    "steps",
    [
        # (valid tokens, the prefill window they are padded to) a step
        [(70, 128), (58, 128)],
        [(200, 256), (58, 128)],
    ],
)
def test_padded_prefill_then_decode_tracks_reference(steps):
    vllm_config = make_vllm_config(
        model="JackFram/llama-68m",
        max_model_len=2048,
        max_num_batched_tokens=max(window for _, window in steps),
        max_num_seqs=2,
        dtype="bfloat16",
    )
    dtype = torch.bfloat16
    with set_current_vllm_config(vllm_config):
        kv_block = RBLNFlashAttentionBackend.get_kv_cache_shape(1, 256, 1, 128)[1:]
    state = torch.zeros(NUM_BLOCKS, *kv_block, dtype=dtype)
    before = state.clone()
    ring_x = torch.zeros(NUM_BLOCKS, KERNEL - 1, 1, INNER // 64, 64, dtype=dtype)
    ring_bc = torch.zeros(
        NUM_BLOCKS, KERNEL - 1, 2, GROUP_CHANNELS // 64, 64, dtype=dtype
    )
    layer = _layer(ring_x, ring_bc)
    # The layer as the model loads it: every weight in the activation dtype but
    # A, which upstream keeps in float32.
    device_layer = SimpleNamespace(**vars(layer))
    device_layer.in_proj = copy.deepcopy(layer.in_proj).to(dtype)
    device_layer.out_proj = copy.deepcopy(layer.out_proj).to(dtype)
    device_layer.conv1d = SimpleNamespace(
        weight=layer.conv1d.weight.to(dtype), bias=layer.conv1d.bias.to(dtype)
    )
    device_layer.D = layer.D.to(dtype)
    device_layer.dt_bias = layer.dt_bias.to(dtype)
    device_layer.norm = _gated_norm(layer.norm_weight.to(dtype))
    vllm_config.compilation_config.static_forward_context[PREFIX] = device_layer
    spec = MambaSpec(
        block_size=vllm_config.model_config.max_model_len,
        shapes=(kv_block,),
        dtypes=(dtype,),
    )
    with set_current_vllm_config(vllm_config):
        builder = RBLNMamba2AttentionMetadataBuilder(
            spec, [PREFIX], vllm_config, torch.device("cpu")
        )
    torch.manual_seed(1)
    total = sum(valid for valid, _ in steps)
    tokens = torch.randn(1, total + 1, HIDDEN).to(dtype).float()

    start, history, ref_state = 0, None, None
    for valid, window in steps:
        hidden = torch.zeros(1, window, HIDDEN, dtype=dtype)
        hidden[:, :valid] = tokens[:, start : start + valid].to(dtype)
        out = _run(
            device_layer,
            vllm_config,
            builder,
            state,
            hidden,
            _common(valid, start + valid),
            torch.arange(start, start + window),
            True,
            1,
        )
        ref, history, ref_state = _reference(
            layer, tokens[:, start : start + valid], history, ref_state
        )
        torch.testing.assert_close(out[:, :valid].float(), ref, atol=3e-2, rtol=3e-2)
        start += valid

    # Decode: one token on row 0 of a padded batch of two.
    hidden = torch.zeros(2, 1, HIDDEN, dtype=dtype)
    hidden[0] = tokens[:, total].to(dtype)
    out = _run(
        device_layer,
        vllm_config,
        builder,
        state,
        hidden,
        _common(1, total + 1),
        torch.tensor([total]),
        False,
        2,
    )
    ref, _, _ = _reference(layer, tokens[:, total:], history, ref_state)
    torch.testing.assert_close(out[:1].float(), ref, atol=3e-2, rtol=3e-2)
    # The padded row took the null block, so the request's neighbours are
    # untouched.
    torch.testing.assert_close(
        state[[BLOCK - 1, BLOCK + 1]], before[[BLOCK - 1, BLOCK + 1]]
    )
    for ring in (ring_x, ring_bc):
        assert not ring[BLOCK - 1].any() and not ring[BLOCK + 1].any()
