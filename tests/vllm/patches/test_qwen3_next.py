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

"""The RBLN GatedDeltaNet layer's physical state spec and KV cache.
Its forward tracks the HF reference across
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
from vllm_rbln.config import RBLNConfig
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.model_executor.models.qwen3_5 import RBLNQwen3_5ForConditionalGeneration
from vllm_rbln.patches.qwen3_next import RBLNQwenGatedDeltaNetAttention
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionBackend
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionMetadataBuilder,
    grouped_qkv_rows,
)
from vllm_rbln.v1.attention.kv_cache_bindings import (
    KVCacheViewInfo,
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)

HIDDEN, HEADS_K, HEADS_V, HEAD_DIM, KERNEL = 64, 2, 4, 128, 4
KEY_DIM, VALUE_DIM = HEADS_K * HEAD_DIM, HEADS_V * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CHUNK = 128
PREFIX = "model.layers.0.linear_attn"
# The state cache: the request's block, with a neighbour on each side.
NUM_BLOCKS, BLOCK = 4, 2


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
    shared_binding,
):
    metadata = builder.build(common, positions, pad, is_prefill)
    bases = [state] if shared_binding else []
    infos = (
        [KVCacheViewInfo(base_index=0, view_shape=tuple(state.shape))]
        if shared_binding
        else []
    )
    attach_kv_cache_bindings(metadata, [state], bases, infos)
    with set_forward_context(
        {PREFIX: metadata}, vllm_config, **build_kv_cache_forward_context_kwargs(bases)
    ):
        return RBLNQwenGatedDeltaNetAttention.forward(layer, hidden)


def _ssm_state(state: torch.Tensor) -> torch.Tensor:
    # The block's bytes as the blocked state they hold -> HF [1, heads_v, dk, dv]
    blocked = (
        state[BLOCK]
        .flatten()[: HEADS_V * HEAD_DIM * HEAD_DIM]
        .float()
        .reshape(1, HEADS_K, HEADS_V // HEADS_K, *state_blocks(HEAD_DIM))
    )
    return unblock_state(blocked).reshape(1, HEADS_V, HEAD_DIM, HEAD_DIM)


@pytest.mark.parametrize("block_size", [256, 1024])
@pytest.mark.parametrize("shared_binding", [False, True])
@pytest.mark.parametrize(
    "chunk_sizes, steps",
    [
        # (valid tokens, the prefill window they are padded to) a step
        ([128], [(70, 128), (58, 128)]),
        # The long step takes the long window, the rest of the prompt the short.
        ([128, 256], [(200, 256), (58, 128)]),
    ],
)
def test_padded_prefill_then_decode_tracks_hf(
    block_size, shared_binding, chunk_sizes, steps
):
    vllm_config = make_vllm_config(
        model="JackFram/llama-68m",
        max_model_len=2048,
        max_num_batched_tokens=max(chunk_sizes),
        max_num_seqs=2,
        dtype="float32",
        additional_config={"prefill_chunk_size": chunk_sizes},
    )
    layer = _layer()
    # The state cache and the ring beside it, as the runner allocates them.
    with set_current_vllm_config(vllm_config):
        kv_block = RBLNFlashAttentionBackend.get_kv_cache_shape(1, block_size, 1, 128)[
            1:
        ]
    state = torch.zeros(NUM_BLOCKS, *kv_block)
    state_elements = HEADS_V * HEAD_DIM * HEAD_DIM
    state.flatten(1)[:, state_elements:] = 13
    before = state.clone()
    ring = torch.zeros(NUM_BLOCKS, KERNEL - 1, 1, CONV_DIM // 64, 64)
    device_layer = _grouped(layer, ring)
    vllm_config.compilation_config.static_forward_context[PREFIX] = device_layer
    spec = MambaSpec(
        block_size=vllm_config.model_config.max_model_len,
        shapes=(kv_block,),
        dtypes=(torch.float32,),
        page_size_padded=state[0].numel() * state.element_size(),
    )
    with set_current_vllm_config(vllm_config):
        builder = RBLNGDNAttentionMetadataBuilder(
            spec, [PREFIX], vllm_config, torch.device("cpu")
        )
    torch.manual_seed(1)
    total = sum(valid for valid, _ in steps)
    tokens = torch.randn(1, total + 1, HIDDEN)

    # Each prefill step pads its valid tokens to a window and carries the conv
    # history and the recurrent state of the steps before it.
    start, history, ref_state = 0, None, None
    for valid, window in steps:
        hidden = torch.zeros(1, window, HIDDEN)
        hidden[:, :valid] = tokens[:, start : start + valid]
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
            shared_binding,
        )
        ref, history, ref_state = _reference(
            layer, tokens[:, start : start + valid], history, ref_state
        )
        torch.testing.assert_close(out[:, :valid], ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(_ssm_state(state), ref_state, atol=2e-3, rtol=2e-2)
        start += valid

    # Decode: one token on row 0 of a padded batch of two.
    hidden = torch.zeros(2, 1, HIDDEN)
    hidden[0] = tokens[:, total]
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
        shared_binding,
    )
    ref, _, ref_state = _reference(layer, tokens[:, total:], history, ref_state)
    torch.testing.assert_close(out[:1], ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(_ssm_state(state), ref_state, atol=2e-3, rtol=2e-2)
    # The padded row took the null block, so the request's neighbours are
    # untouched.
    torch.testing.assert_close(
        state[[BLOCK - 1, BLOCK + 1]], before[[BLOCK - 1, BLOCK + 1]]
    )
    torch.testing.assert_close(
        state.flatten(1)[:, state_elements:], before.flatten(1)[:, state_elements:]
    )
    assert not ring[BLOCK - 1].any() and not ring[BLOCK + 1].any()


def test_forward_without_metadata_is_an_error(vllm_config):
    layer = _layer()
    with (
        set_forward_context({}, vllm_config),
        pytest.raises(RuntimeError, match="no RBLN GDN metadata"),
    ):
        RBLNQwenGatedDeltaNetAttention.forward(layer, torch.zeros(1, CHUNK, HIDDEN))


class TestStateSpec:
    """State specs describe the physical KV page used by the runner."""

    @staticmethod
    def _layer():
        layer = object.__new__(RBLNQwenGatedDeltaNetAttention)
        layer.num_kv_heads = 4
        layer.kv_head_size = 256
        layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
        layer.num_v_heads = 48
        layer.tp_size = 1
        layer.head_k_dim = layer.head_v_dim = 128
        layer.cache_config = SimpleNamespace(
            block_size=1024,
            mamba_block_size=8192,
            mamba_page_size_padded=None,
            mamba_cache_mode="none",
            use_kda_recoverssm=False,
        )
        return layer

    @pytest.mark.parametrize(
        "value_heads, block_size", [(32, 256), (32, 1024), (48, 384), (48, 1024)]
    )
    @pytest.mark.parametrize("explicit_padding", [False, True])
    def test_spec_uses_full_attention_page(
        self, value_heads, block_size, explicit_padding
    ):
        layer = self._layer()
        layer.num_v_heads = value_heads
        layer.cache_config.block_size = block_size
        if explicit_padding:
            layer.cache_config.mamba_page_size_padded = 2 * 4 * block_size * 256 * 2
        config = SimpleNamespace(
            cache_config=layer.cache_config,
            num_speculative_tokens=0,
            additional_config=RBLNConfig(),
        )
        spec = layer.get_kv_cache_spec(config)
        assert spec.shapes == ((2, 4, 1, block_size, 256),)
        assert spec.real_page_size_bytes == spec.page_size_bytes
        assert spec.page_size_bytes == 2 * 4 * block_size * 256 * 2
        assert spec.page_size_padded == layer.cache_config.mamba_page_size_padded
        assert spec.block_size == config.cache_config.mamba_block_size

    def test_layer_cache_uses_full_attention_layout(self):
        layer = self._layer()
        with set_current_vllm_config(SimpleNamespace(additional_config=RBLNConfig())):
            assert layer.get_state_shape() == ((2, 4, 1, 1024, 256),)
        # The activation dtype: the kernels take no other.
        assert layer.get_state_dtype() == (torch.bfloat16,)

    def test_triton_cache_layout_is_rejected(self):
        layer = self._layer()
        config = SimpleNamespace(additional_config=RBLNConfig(use_custom_kernel=True))
        with (
            set_current_vllm_config(config),
            pytest.raises(AssertionError, match="GDN requires KV blocks shaped"),
        ):
            layer.get_state_shape()

    def test_a_cache_dtype_option_does_not_move_the_state_dtype(self):
        layer = self._layer()
        # What upstream's Qwen3.5 hook takes from the HF config's mamba_ssm_dtype.
        layer.cache_config.mamba_ssm_cache_dtype = "float32"
        assert layer.get_state_dtype() == (torch.bfloat16,)

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
