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

"""The RBLN GatedDeltaNet layer declares its state as the full-attention KV
block it fills, and its forward tracks the HF reference across
a padded prefill chunk, a continued chunk and a decode step on that block. The
custom ops run their CPU bodies here, so no NPU is needed."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from rebel.ops.torch_custom_ops.gdn import state_blocks, unblock_state
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.model_executor.models.qwen3_5 import RBLNQwen3_5ForConditionalGeneration
from vllm_rbln.patches.qwen3_next import RBLNQwenGatedDeltaNetAttention
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionMetadataBuilder,
    gdn_state_shape,
    grouped_qkv_rows,
)

HIDDEN, HEADS_K, HEADS_V, HEAD_DIM, KERNEL = 64, 2, 4, 128, 4
KEY_DIM, VALUE_DIM = HEADS_K * HEAD_DIM, HEADS_V * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CHUNK = 128
PREFIX = "model.layers.0.linear_attn"
# The state cache: the request's block, with a neighbour on each side.
NUM_BLOCKS, BLOCK = 4, 2
# A full-attention KV block of the layer's HEADS_V * HEAD_DIM * HEAD_DIM state
# elements: 1 KV head, 256 tokens, head size 128.
KV_BLOCK = gdn_state_shape(1, 256, 128)


class _Proj(torch.nn.Linear):
    """vLLM's linear layers return (output, bias)."""

    def forward(self, x):
        return super().forward(x), None


def _layer():
    torch.manual_seed(0)
    layer = SimpleNamespace(
        prefix=PREFIX,
        layer_index=0,
        tp_size=1,
        num_k_heads=HEADS_K,
        num_v_heads=HEADS_V,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        conv_kernel_size=KERNEL,
        conv_dim=CONV_DIM,
        num_drafts=0,
        drafts=None,
    )
    layer.in_proj_qkvz = _Proj(HIDDEN, CONV_DIM + VALUE_DIM, bias=False)
    layer.in_proj_ba = _Proj(HIDDEN, 2 * HEADS_V, bias=False)
    layer.conv1d = SimpleNamespace(weight=torch.randn(CONV_DIM, 1, KERNEL) * 0.2)
    layer.A_log = torch.log(torch.rand(HEADS_V) * 8 + 0.5)
    layer.dt_bias = torch.rand(HEADS_V)
    layer.norm = Qwen3_5RMSNormGated(HEAD_DIM, eps=1e-6)
    layer.out_proj = _Proj(VALUE_DIM, HIDDEN, bias=False)
    return layer


def _grouped(layer, ring):
    """The layer as RBLNQwen3_5ForConditionalGeneration.load_weights leaves it:
    the q | k | v rows of in_proj_qkvz and conv1d regrouped by key head, with
    its conv ring beside it."""
    rows = grouped_qkv_rows(HEADS_K, HEADS_V // HEADS_K, HEAD_DIM)
    grouped = SimpleNamespace(**vars(layer))
    grouped.in_proj_qkvz = _Proj(HIDDEN, CONV_DIM + VALUE_DIM, bias=False)
    grouped.in_proj_qkvz.weight.data = layer.in_proj_qkvz.weight.data.clone()
    grouped.in_proj_qkvz.weight.data[: rows.numel()] = layer.in_proj_qkvz.weight[rows]
    grouped.conv1d = SimpleNamespace(weight=layer.conv1d.weight.clone())
    grouped.conv1d.weight[: rows.numel()] = layer.conv1d.weight[rows]
    grouped.conv_ring = SimpleNamespace(inputs=lambda: ring)
    return grouped


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
        num_actual_tokens=query_len,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table_tensor=torch.tensor([[BLOCK]], dtype=torch.int32),
    )


@pytest.fixture
def vllm_config():
    return make_vllm_config(
        max_num_batched_tokens=CHUNK, max_num_seqs=2, dtype="float32"
    )


def _run(
    layer,
    vllm_config,
    builder,
    state,
    hidden,
    common,
    positions,
    is_prefill,
    pad,
    **window,
):
    metadata = builder.build(common, positions, pad, is_prefill, **window)
    # What the runner attaches to every step's metadata (attach_kv_cache_bindings).
    metadata.kv_caches = [state]
    metadata.kv_cache_view_infos = None
    with set_forward_context({PREFIX: metadata}, vllm_config):
        return RBLNQwenGatedDeltaNetAttention.forward(layer, hidden)


def _ssm_state(state: torch.Tensor) -> torch.Tensor:
    # The block's bytes as the blocked state they hold -> HF [1, heads_v, dk, dv]
    blocked = (
        state[BLOCK]
        .float()
        .reshape(1, HEADS_K, HEADS_V // HEADS_K, *state_blocks(HEAD_DIM))
    )
    return unblock_state(blocked).reshape(1, HEADS_V, HEAD_DIM, HEAD_DIM)


def test_padded_prefill_then_decode_tracks_hf(vllm_config):
    layer = _layer()
    # The state cache and the ring beside it, as the runner allocates them.
    state = torch.zeros(NUM_BLOCKS, *KV_BLOCK)
    ring = torch.zeros(NUM_BLOCKS, KERNEL - 1, 1, CONV_DIM // 64, 64)
    device_layer = _grouped(layer, ring)
    vllm_config.compilation_config.static_forward_context[PREFIX] = device_layer
    spec = MambaSpec(
        block_size=vllm_config.model_config.max_model_len,
        shapes=(KV_BLOCK,),
        dtypes=(torch.float32,),
    )
    with set_current_vllm_config(vllm_config):
        builder = RBLNGDNAttentionMetadataBuilder(
            spec, [PREFIX], vllm_config, torch.device("cpu")
        )
    torch.manual_seed(1)
    tokens = torch.randn(1, 70 + 58 + 1, HIDDEN)

    # Chunk 1: 70 valid tokens in a 128-token prefill window, no carried state.
    hidden = torch.zeros(1, CHUNK, HIDDEN)
    hidden[:, :70] = tokens[:, :70]
    out = _run(
        device_layer,
        vllm_config,
        builder,
        state,
        hidden,
        _common(70, 70),
        torch.arange(CHUNK),
        True,
        1,
    )
    ref, history, ref_state = _reference(layer, tokens[:, :70], None, None)
    torch.testing.assert_close(out[:, :70], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(state), ref_state, atol=2e-3, rtol=2e-2)

    # Chunk 2: 58 more tokens, carrying the conv history and the recurrent state.
    hidden = torch.zeros(1, CHUNK, HIDDEN)
    hidden[:, :58] = tokens[:, 70:128]
    out = _run(
        device_layer,
        vllm_config,
        builder,
        state,
        hidden,
        _common(58, 128),
        torch.arange(70, 70 + CHUNK),
        True,
        1,
    )
    ref, history, ref_state = _reference(layer, tokens[:, 70:128], history, ref_state)
    torch.testing.assert_close(out[:, :58], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(state), ref_state, atol=2e-3, rtol=2e-2)

    # Decode: one token on row 0 of a padded batch of two.
    hidden = torch.zeros(2, 1, HIDDEN)
    hidden[0] = tokens[:, 128]
    out = _run(
        device_layer,
        vllm_config,
        builder,
        state,
        hidden,
        _common(1, 129),
        torch.tensor([128]),
        False,
        2,
    )
    ref, _, ref_state = _reference(layer, tokens[:, 128:129], history, ref_state)
    torch.testing.assert_close(out[:1], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(state), ref_state, atol=2e-3, rtol=2e-2)
    # The padded row took the null block, so the request's neighbours are
    # untouched.
    assert not state[BLOCK - 1].any() and not state[BLOCK + 1].any()
    assert not ring[BLOCK - 1].any() and not ring[BLOCK + 1].any()


def _decoded(vllm_config, num_drafts: int, windows):
    """A 70-token prefill, then `windows` of (tokens, position, valid tokens,
    drafts of the window before that were accepted) on a bfloat16 layer, the
    16-bit activations the verify kernel takes. Returns each window's outputs
    and the state it left."""
    layer = _layer()
    # A slow decay, so a token missing from the state shows.
    layer.A_log = torch.full((HEADS_V,), -3.0)
    dtype = torch.bfloat16
    state = torch.zeros(NUM_BLOCKS, *KV_BLOCK, dtype=dtype)
    ring = torch.zeros(
        NUM_BLOCKS, KERNEL - 1 + num_drafts, 1, CONV_DIM // 64, 64, dtype=dtype
    )
    device_layer = _grouped(layer, ring)
    for name in ("in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
        getattr(device_layer, name).to(dtype)
    device_layer.num_drafts = num_drafts
    if num_drafts:
        ratio = HEADS_V // HEADS_K
        drafts = torch.zeros(
            NUM_BLOCKS, HEADS_K, 2 * ratio + 1, num_drafts, HEAD_DIM, dtype=dtype
        )
        device_layer.drafts = SimpleNamespace(inputs=lambda: drafts)
    vllm_config.compilation_config.static_forward_context[PREFIX] = device_layer
    spec = MambaSpec(
        block_size=vllm_config.model_config.max_model_len,
        shapes=(KV_BLOCK,),
        dtypes=(dtype,),
    )
    with set_current_vllm_config(vllm_config):
        builder = RBLNGDNAttentionMetadataBuilder(
            spec, [PREFIX], vllm_config, torch.device("cpu")
        )
    torch.manual_seed(1)
    hidden = torch.zeros(1, CHUNK, HIDDEN, dtype=dtype)
    hidden[:, :70] = torch.randn(1, 70, HIDDEN)
    _run(
        device_layer,
        vllm_config,
        builder,
        state,
        hidden,
        _common(70, 70),
        torch.arange(CHUNK),
        True,
        1,
    )
    outputs, states = [], []
    for tokens, position, valid, accepted in windows:
        width = tokens.shape[0]
        hidden = torch.zeros(2, width, HIDDEN, dtype=dtype)
        hidden[0] = tokens
        window = {}
        if num_drafts:
            window = {
                "back_pad": torch.tensor([width - valid]),
                "num_accepted": torch.tensor([accepted]),
            }
        out = _run(
            device_layer,
            vllm_config,
            builder,
            state,
            hidden,
            _common(width, position + valid),
            torch.arange(position, position + width),
            False,
            2,
            **window,
        )
        outputs.append(out[0].float())
        states.append(_ssm_state(state))
    return outputs, states


def test_verify_steps_stand_for_the_decode_steps_of_their_accepted_tokens(
    vllm_config,
):
    """An MTP step takes a confirmed token and two drafts, keeps the state
    after the confirmed token only and replays the accepted drafts on the next
    step, so a rejected draft leaves no trace in the state or the conv ring."""
    torch.manual_seed(2)
    x = torch.randn(5, HIDDEN)
    wrong = torch.randn(2, HIDDEN)
    pad = torch.zeros(HIDDEN)
    plain, plain_states = _decoded(
        vllm_config, 0, [(x[i : i + 1], 70 + i, 1, 0) for i in range(5)]
    )
    verified, verified_states = _decoded(
        vllm_config,
        2,
        [
            # The first decode step carries no draft.
            (torch.stack([x[0], pad, pad]), 70, 1, 0),
            # x[2] is accepted, the draft after it rejected.
            (torch.stack([x[1], x[2], wrong[0]]), 71, 3, 0),
            (torch.stack([x[3], x[4], wrong[1]]), 73, 3, 1),
        ],
    )
    close = {"atol": 1e-2, "rtol": 0}
    torch.testing.assert_close(verified[0][:1], plain[0], **close)
    torch.testing.assert_close(verified[1][:2], torch.cat(plain[1:3]), **close)
    torch.testing.assert_close(verified[2][:2], torch.cat(plain[3:5]), **close)
    # The state after x[0], x[1] and x[3]: each window's confirmed token.
    for window, step in enumerate((0, 1, 3)):
        torch.testing.assert_close(verified_states[window], plain_states[step], **close)


def test_forward_without_metadata_is_an_error(vllm_config):
    layer = _layer()
    with (
        set_forward_context({}, vllm_config),
        pytest.raises(RuntimeError, match="no RBLN GDN metadata"),
    ):
        RBLNQwenGatedDeltaNetAttention.forward(layer, torch.zeros(1, CHUNK, HIDDEN))


class TestStateSpec:
    """The layer declares its state as a full-attention KV block, and the model
    declares the same state for the platform's page arithmetic."""

    @staticmethod
    def _layer():
        # The attention layers keep 4 KV heads of 256, 256 tokens a block.
        layer = object.__new__(RBLNQwenGatedDeltaNetAttention)
        layer.num_kv_heads = 4
        layer.kv_head_size = 256
        layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
        layer.cache_config = SimpleNamespace(block_size=256)
        return layer

    def test_the_layer_declares_one_ssm_state_a_block(self):
        layer = self._layer()
        assert layer.get_state_shape() == ((2, 4, 1, 256, 256),)
        # The activation dtype: the kernels take no other.
        assert layer.get_state_dtype() == (torch.bfloat16,)

    def test_the_state_takes_no_block_a_draft(self):
        layer = self._layer()
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                mamba_block_size=8192,
                mamba_page_size_padded=None,
                mamba_cache_mode="none",
                use_kda_recoverssm=False,
            ),
            num_speculative_tokens=3,
        )
        # Upstream keeps a state block a draft; the RBLN verify kernel keeps the
        # drafts in the layer's draft buffer instead.
        assert layer.get_kv_cache_spec(vllm_config).num_speculative_blocks == 0

    def test_a_cache_dtype_option_does_not_move_the_state_dtype(self):
        layer = self._layer()
        # What upstream's Qwen3.5 hook takes from the HF config's mamba_ssm_dtype.
        layer.cache_config.mamba_ssm_cache_dtype = "float32"
        assert layer.get_state_dtype() == (torch.bfloat16,)

    def test_the_model_declares_the_layers_state(self):
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                get_num_kv_heads=lambda parallel_config: 4,
                get_head_size=lambda: 256,
            ),
            parallel_config=None,
            cache_config=SimpleNamespace(block_size=256),
        )
        model = RBLNQwen3_5ForConditionalGeneration
        assert model.get_mamba_state_shape_from_config(vllm_config) == (
            self._layer().get_state_shape()
        )
        assert model.get_mamba_state_dtype_from_config(vllm_config) == (
            self._layer().get_state_dtype()
        )


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
