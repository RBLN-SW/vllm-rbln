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

"""The RBLN GatedDeltaNet forward tracks the HF reference across a padded
prefill chunk, a continued chunk and a decode step, on the request's state
block. The custom ops run their CPU bodies here, so no NPU is needed."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from rebel.ops.torch_custom_ops.kda import state_view
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.patches.qwen3_next import patched_gdn_forward
from vllm_rbln.v1.attention.backends.gdn import RBLNGDNAttentionMetadataBuilder

HIDDEN, HEADS_K, HEADS_V, HEAD_DIM, KERNEL = 64, 2, 4, 128, 4
KEY_DIM, VALUE_DIM = HEADS_K * HEAD_DIM, HEADS_V * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CHUNK = 128
PREFIX = "model.layers.0.linear_attn"
# The request's state block (the MambaSpec group's block table entry).
BLOCK = 2


class _Proj(torch.nn.Linear):
    """vLLM's linear layers return (output, bias)."""

    def forward(self, x):
        return super().forward(x), None


def _layer():
    torch.manual_seed(0)
    layer = SimpleNamespace(
        prefix=PREFIX,
        tp_size=1,
        num_k_heads=HEADS_K,
        num_v_heads=HEADS_V,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        conv_kernel_size=KERNEL,
        conv_dim=CONV_DIM,
    )
    layer.in_proj_qkvz = _Proj(HIDDEN, CONV_DIM + VALUE_DIM, bias=False)
    layer.in_proj_ba = _Proj(HIDDEN, 2 * HEADS_V, bias=False)
    layer.conv1d = SimpleNamespace(weight=torch.randn(CONV_DIM, 1, KERNEL) * 0.2)
    layer.A_log = torch.log(torch.rand(HEADS_V) * 8 + 0.5)
    layer.dt_bias = torch.rand(HEADS_V)
    layer.norm = Qwen3_5RMSNormGated(HEAD_DIM, eps=1e-6)
    layer.out_proj = _Proj(VALUE_DIM, HIDDEN, bias=False)
    return layer


def _reference(layer, hidden, conv_history, recurrent_state):
    """HF Qwen3_5GatedDeltaNet.forward over the valid tokens only."""
    seq_len = hidden.shape[1]
    mixed_qkv, z = layer.in_proj_qkvz(hidden)[0].split([CONV_DIM, VALUE_DIM], dim=-1)
    b, a = layer.in_proj_ba(hidden)[0].chunk(2, dim=-1)
    x = mixed_qkv.transpose(1, 2)
    if conv_history is None:
        conv_history = torch.zeros(1, CONV_DIM, KERNEL - 1)
    x = torch.cat([conv_history, x], dim=-1)
    conv = F.silu(F.conv1d(x, layer.conv1d.weight, groups=CONV_DIM)).transpose(1, 2)
    query, key, value = conv.split([KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    ratio = HEADS_V // HEADS_K
    query = query.reshape(1, seq_len, HEADS_K, HEAD_DIM).repeat_interleave(ratio, dim=2)
    key = key.reshape(1, seq_len, HEADS_K, HEAD_DIM).repeat_interleave(ratio, dim=2)
    value = value.reshape(1, seq_len, HEADS_V, HEAD_DIM)
    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
    rule = (
        torch_recurrent_gated_delta_rule
        if seq_len == 1
        else torch_chunk_gated_delta_rule
    )
    core, state = rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    core = layer.norm(core.reshape(-1, HEAD_DIM), z.reshape(-1, HEAD_DIM))
    output = layer.out_proj(core.reshape(1, seq_len, -1))[0]
    return output, x[:, :, -(KERNEL - 1) :], state


def _common(query_len: int, seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table_tensor=torch.tensor([[BLOCK]], dtype=torch.int32),
    )


@pytest.fixture
def vllm_config():
    return make_vllm_config(
        max_num_batched_tokens=CHUNK, max_num_seqs=2, dtype="float32"
    )


def _run(layer, vllm_config, builder, hidden, common, positions, is_prefill, batch_pad):
    metadata = builder.build(common, positions, batch_pad, is_prefill, [BLOCK])
    with set_forward_context({PREFIX: metadata}, vllm_config):
        return patched_gdn_forward(layer, hidden)


def _ssm_state(builder) -> torch.Tensor:
    # R1's bytes of the block's row -> HF [1, heads_v, dk, dv]
    row = builder.rows.row_of[BLOCK]
    state = builder.states[PREFIX].ssm_state
    return state_view(state, HEADS_V, torch.float32)[row : row + 1]


def test_padded_prefill_then_decode_tracks_hf(vllm_config):
    layer = _layer()
    vllm_config.compilation_config.static_forward_context[PREFIX] = layer
    builder = RBLNGDNAttentionMetadataBuilder(
        None, [PREFIX], vllm_config, torch.device("cpu")
    )
    torch.manual_seed(1)
    tokens = torch.randn(1, 70 + 58 + 1, HIDDEN)

    # Chunk 1: 70 valid tokens in a 128-token prefill window, no carried state.
    hidden = torch.zeros(1, CHUNK, HIDDEN)
    hidden[:, :70] = tokens[:, :70]
    out = _run(
        layer,
        vllm_config,
        builder,
        hidden,
        _common(70, 70),
        torch.arange(CHUNK),
        True,
        1,
    )
    ref, history, state = _reference(layer, tokens[:, :70], None, None)
    torch.testing.assert_close(out[:, :70], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(builder), state, atol=2e-3, rtol=2e-2)

    # Chunk 2: 58 more tokens, carrying the conv history and the recurrent state.
    hidden = torch.zeros(1, CHUNK, HIDDEN)
    hidden[:, :58] = tokens[:, 70:128]
    out = _run(
        layer,
        vllm_config,
        builder,
        hidden,
        _common(58, 128),
        torch.arange(70, 70 + CHUNK),
        True,
        1,
    )
    ref, history, state = _reference(layer, tokens[:, 70:128], history, state)
    torch.testing.assert_close(out[:, :58], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(builder), state, atol=2e-3, rtol=2e-2)

    # Decode: one token on row 0 of a padded batch of two.
    hidden = torch.zeros(2, 1, HIDDEN)
    hidden[0] = tokens[:, 128]
    out = _run(
        layer,
        vllm_config,
        builder,
        hidden,
        _common(1, 129),
        torch.tensor([128]),
        False,
        2,
    )
    ref, _, state = _reference(layer, tokens[:, 128:129], history, state)
    torch.testing.assert_close(out[:1], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(builder), state, atol=2e-3, rtol=2e-2)
    # The padded row took its scratch row, so no other live row was touched.
    rows = builder.states[PREFIX].ssm_state
    assert builder.rows.row_of == {BLOCK: 0}
    assert rows[0].any() and not rows[1:].any()


def test_forward_without_metadata_is_an_error(vllm_config):
    layer = _layer()
    with (
        set_forward_context({}, vllm_config),
        pytest.raises(RuntimeError, match="no RBLN GDN metadata"),
    ):
        patched_gdn_forward(layer, torch.zeros(1, CHUNK, HIDDEN))


class TestProjectQkvGate:
    """q, k, v and the gate all keep the leading dims of `qkv`, so the gated
    product never broadcasts [tokens, hidden] against [batch, 1, hidden]."""

    HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8

    def _layer(self, gate: bool):
        from vllm_rbln.patches.qwen3_next import patched_qwen3_next_project_qkv_gate

        layer = SimpleNamespace(
            attn_output_gate=gate,
            num_heads=self.HEADS,
            num_kv_heads=self.KV_HEADS,
            head_dim=self.HEAD_DIM,
            q_size=self.HEADS * self.HEAD_DIM,
            kv_size=self.KV_HEADS * self.HEAD_DIM,
            q_norm=lambda x: x * 2.0,
            k_norm=lambda x: x * 3.0,
            rotary_emb=lambda positions, q, k: (q + 1.0, k - 1.0),
        )
        return layer, patched_qwen3_next_project_qkv_gate

    @pytest.mark.parametrize("gate", [True, False])
    def test_leading_dims_are_kept(self, gate):
        layer, project = self._layer(gate)
        width = layer.q_size * (2 if gate else 1) + 2 * layer.kv_size
        torch.manual_seed(0)
        qkv = torch.randn(2, 3, width)
        positions = torch.zeros(2, 3, dtype=torch.long)
        q, k, v, g = project(layer, qkv, positions)
        assert q.shape == (2, 3, layer.q_size)
        assert k.shape == (2, 3, layer.kv_size)
        assert v.shape == (2, 3, layer.kv_size)
        assert (g is None) is not gate
        if gate:
            assert g.shape == (2, 3, layer.q_size)
        # The same math as the flattened [tokens, ...] form upstream computes.
        q_flat, k_flat, v_flat, g_flat = project(
            layer, qkv.reshape(6, width), positions.reshape(6)
        )
        assert torch.equal(q.reshape(6, -1), q_flat)
        assert torch.equal(k.reshape(6, -1), k_flat)
        assert torch.equal(v.reshape(6, -1), v_flat)
        if gate:
            assert torch.equal(g.reshape(6, -1), g_flat)

    def test_gate_splits_per_head(self):
        # q_gate interleaves [q | gate] per head; the chunk must follow that layout.
        layer, project = self._layer(True)
        qkv = torch.zeros(1, 1, layer.q_size * 2 + 2 * layer.kv_size)
        per_head = qkv[..., : layer.q_size * 2].view(
            1, 1, self.HEADS, 2 * self.HEAD_DIM
        )
        per_head[..., : self.HEAD_DIM] = 1.0  # q half of every head
        per_head[..., self.HEAD_DIM :] = 5.0  # gate half
        q, _, _, g = project(layer, qkv, torch.zeros(1, 1, dtype=torch.long))
        assert torch.all(q == 1.0 * 2.0 + 1.0)  # q_norm then rotary
        assert torch.all(g == 5.0)
