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
a padded prefill chunk, a continued chunk and a decode step on that block, and
its MTP verify steps track one-token decode steps. The custom ops run their CPU
bodies here, so no NPU is needed."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.config import RBLNConfig
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    RBLNQwenGatedDeltaNetAttention,
)
from vllm_rbln.model_executor.models.qwen3_5 import RBLNQwen3_5ForConditionalGeneration
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionBackend
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import (
    KVCacheViewInfo,
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.worker.utils import get_kv_cache_names

HIDDEN, HEADS_K, HEADS_V, HEAD_DIM, KERNEL = 64, 4, 4, 128, 4
KEY_DIM, VALUE_DIM = HEADS_K * HEAD_DIM, HEADS_V * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
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
        num_spec=0,
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


def _grouped(layer, ring, monkeypatch):
    """The layer as RBLNQwen3_5ForConditionalGeneration.load_weights leaves it:
    the q | k | v rows of in_proj_qkvz and conv1d regrouped by key head, with
    its conv ring beside it."""
    grouped = SimpleNamespace(**vars(layer))
    grouped.in_proj_qkvz = _Proj(HIDDEN, CONV_DIM + VALUE_DIM, bias=False)
    grouped.in_proj_qkvz.weight.data = layer.in_proj_qkvz.weight.data.clone()
    grouped.conv1d = SimpleNamespace(weight=layer.conv1d.weight.clone())
    grouped.conv_ring = SimpleNamespace(inputs=lambda: ring)
    # The model's own regrouping, over a checkpoint its parent has loaded. The
    # second decoder layer is a full-attention one, which it passes over.
    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration, "load_weights", lambda self, weights: set()
    )
    model = RBLNQwen3_5ForConditionalGeneration.__new__(
        RBLNQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(linear_attn=grouped), SimpleNamespace()]
        )
    )
    model.load_weights([])
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
    **window,
):
    metadata = builder.build(common, positions, pad, is_prefill, **window)
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


@pytest.mark.parametrize("block_size", [256, 1024])
@pytest.mark.parametrize("shared_binding", [False, True])
@pytest.mark.parametrize(
    "steps",
    [
        # (valid tokens, the prefill window they are padded to) a step
        [(70, 128), (58, 128)],
        # A long step in the long window, the rest of the prompt in the short.
        [(200, 256), (58, 128)],
    ],
)
def test_padded_prefill_then_decode_tracks_hf(
    monkeypatch, block_size, shared_binding, steps
):
    vllm_config = make_vllm_config(
        model="JackFram/llama-68m",
        max_model_len=2048,
        max_num_batched_tokens=max(window for _, window in steps),
        max_num_seqs=2,
        dtype="float32",
    )
    layer = _layer()
    with set_current_vllm_config(vllm_config):
        kv_block = RBLNFlashAttentionBackend.get_kv_cache_shape(1, block_size, 1, 128)[
            1:
        ]
    state = torch.zeros(NUM_BLOCKS, *kv_block)
    before = state.clone()
    ring = torch.zeros(NUM_BLOCKS, KERNEL - 1, 1, CONV_DIM // 64, 64)
    device_layer = _grouped(layer, ring, monkeypatch)
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
    ref, _, _ = _reference(layer, tokens[:, total:], history, ref_state)
    torch.testing.assert_close(out[:1], ref, atol=2e-3, rtol=2e-2)
    # The padded row took the null block, so the request's neighbours are
    # untouched.
    torch.testing.assert_close(
        state[[BLOCK - 1, BLOCK + 1]], before[[BLOCK - 1, BLOCK + 1]]
    )
    assert not ring[BLOCK - 1].any() and not ring[BLOCK + 1].any()


@pytest.mark.use_device
def test_mtp_adds_a_verify_only_draft_buffer(monkeypatch):
    # The layer as the model builds it; upstream's constructor puts parameters
    # on the device.
    import vllm.distributed.parallel_state as parallel_state

    config = make_vllm_config(block_size=256, dtype="bfloat16")
    text = config.model_config.hf_text_config
    text.linear_num_key_heads = HEADS_K
    text.linear_num_value_heads = HEADS_V
    text.linear_key_head_dim = text.linear_value_head_dim = HEAD_DIM
    text.linear_conv_kernel_dim = KERNEL
    config.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    monkeypatch.setattr(
        config.model_config, "_architecture", "Qwen3_5ForConditionalGeneration"
    )
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    with set_current_vllm_config(config):
        RBLNQwenGatedDeltaNetAttention(text, config, PREFIX)
    specs = {
        name.removeprefix(PREFIX): module.get_aux_cache_spec(config)
        for name, module in config.compilation_config.static_forward_context.items()
        if hasattr(module, "get_aux_cache_spec")
    }
    ring, drafts = specs[".conv_ring"], specs[".drafts"]
    # The ring holds the two drafts beside the taps.
    assert ring.block_shape == (KERNEL - 1 + 2, 1, CONV_DIM // 64, 64)
    assert (ring.prefix_cacheable, ring.verify_only) == (True, False)
    # Only a verify step reads the drafts, and a prefix resuming at a block
    # boundary replays none.
    assert drafts.block_shape == (HEADS_K, 2 * (HEADS_V // HEADS_K) + 1, 2, HEAD_DIM)
    assert (drafts.prefix_cacheable, drafts.verify_only) == (False, True)


def _decoded(vllm_config, num_drafts: int, windows, monkeypatch, shared_binding):
    """A 70-token prefill, then `windows` of (tokens, position, valid tokens,
    drafts of the window before that were accepted) on a bfloat16 layer, the
    16-bit activations the verify kernel takes. Returns each window's outputs
    and the state it left."""
    layer = _layer()
    # A slow decay, so a token missing from the state shows.
    layer.A_log = torch.full((HEADS_V,), -3.0)
    dtype = torch.bfloat16
    with set_current_vllm_config(vllm_config):
        kv_block = RBLNFlashAttentionBackend.get_kv_cache_shape(1, 256, 1, 128)[1:]
    state = torch.zeros(NUM_BLOCKS, *kv_block, dtype=dtype)
    ring = torch.zeros(
        NUM_BLOCKS, KERNEL - 1 + num_drafts, 1, CONV_DIM // 64, 64, dtype=dtype
    )
    device_layer = _grouped(layer, ring, monkeypatch)
    for name in ("in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
        getattr(device_layer, name).to(dtype)
    device_layer.conv1d.weight = device_layer.conv1d.weight.to(dtype)
    device_layer.num_spec = num_drafts
    if num_drafts:
        ratio = HEADS_V // HEADS_K
        drafts = torch.zeros(
            NUM_BLOCKS, HEADS_K, 2 * ratio + 1, num_drafts, HEAD_DIM, dtype=dtype
        )
        device_layer.drafts = SimpleNamespace(inputs=lambda: drafts)
    vllm_config.compilation_config.static_forward_context[PREFIX] = device_layer
    # The builder holds a decode step to one token and the drafts.
    vllm_config.speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_drafts) if num_drafts else None
    )
    spec = MambaSpec(
        block_size=vllm_config.model_config.max_model_len,
        shapes=(kv_block,),
        dtypes=(dtype,),
    )
    with set_current_vllm_config(vllm_config):
        builder = RBLNGDNAttentionMetadataBuilder(
            spec, [PREFIX], vllm_config, torch.device("cpu")
        )
    torch.manual_seed(1)
    hidden = torch.zeros(1, 128, HIDDEN, dtype=dtype)
    hidden[:, :70] = torch.randn(1, 70, HIDDEN)
    _run(
        device_layer,
        vllm_config,
        builder,
        state,
        hidden,
        _common(70, 70),
        torch.arange(128),
        True,
        1,
        shared_binding,
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
            shared_binding,
            **window,
        )
        outputs.append(out[0].float())
        states.append(state[BLOCK].float())
    return outputs, states


@pytest.mark.parametrize("shared_binding", [False, True])
def test_verify_steps_stand_for_the_decode_steps_of_their_accepted_tokens(
    monkeypatch, shared_binding
):
    """An MTP step takes a confirmed token and two drafts, keeps the state
    after the confirmed token only and replays the accepted drafts on the next
    step, so a rejected draft leaves no trace in the state or the conv ring."""
    vllm_config = make_vllm_config(
        model="JackFram/llama-68m",
        max_model_len=2048,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        dtype="bfloat16",
    )
    torch.manual_seed(2)
    x = torch.randn(8, HIDDEN)
    wrong = torch.randn(2, HIDDEN)
    pad = torch.zeros(HIDDEN)
    plain, plain_states = _decoded(
        vllm_config,
        0,
        [(x[i : i + 1], 70 + i, 1, 0) for i in range(8)],
        monkeypatch,
        shared_binding,
    )
    verified, verified_states = _decoded(
        vllm_config,
        2,
        [
            # The first decode step carries no draft.
            (torch.stack([x[0], pad, pad]), 70, 1, 0),
            # x[2] is accepted, the draft after it rejected.
            (torch.stack([x[1], x[2], wrong[0]]), 71, 3, 0),
            # Both of this window's drafts are accepted: the next one replays
            # the whole draft buffer.
            (torch.stack([x[3], x[4], x[5]]), 73, 3, 1),
            (torch.stack([x[6], x[7], wrong[1]]), 76, 3, 2),
        ],
        monkeypatch,
        shared_binding,
    )
    close = {"atol": 1e-2, "rtol": 0}
    torch.testing.assert_close(verified[0][:1], plain[0], **close)
    torch.testing.assert_close(verified[1][:2], torch.cat(plain[1:3]), **close)
    torch.testing.assert_close(verified[2], torch.cat(plain[3:6]), **close)
    torch.testing.assert_close(verified[3][:2], torch.cat(plain[6:8]), **close)
    # The state after x[0], x[1], x[3] and x[6]: each window's confirmed token.
    for window, step in enumerate((0, 1, 3, 6)):
        torch.testing.assert_close(verified_states[window], plain_states[step], **close)


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
            additional_config=RBLNConfig(),
        )
        # Upstream keeps a state block a draft; the RBLN verify kernel keeps the
        # drafts in the layer's draft buffer instead.
        assert layer.get_kv_cache_spec(vllm_config).num_speculative_blocks == 0

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


def test_gdn_aux_caches_sort_after_their_state(monkeypatch):
    def init_upstream(layer, _text_config, config, prefix):
        torch.nn.Module.__init__(layer)
        layer.head_k_dim = layer.head_v_dim = HEAD_DIM
        layer.num_k_heads, layer.num_v_heads = HEADS_K, HEADS_V
        layer.conv_dim, layer.conv_kernel_size = CONV_DIM, KERNEL
        layer.tp_size, layer.num_spec = 1, 2
        config.compilation_config.static_forward_context[prefix] = layer

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "__init__", init_upstream)
    config = SimpleNamespace(
        compilation_config=CompilationConfig(),
        model_config=SimpleNamespace(
            architecture="Qwen3_5ForConditionalGeneration",
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda _: HEADS_K,
            get_head_size=lambda: HEAD_DIM,
        ),
        parallel_config=None,
    )
    context = config.compilation_config.static_forward_context
    with set_current_vllm_config(config):
        RBLNQwenGatedDeltaNetAttention(None, config, PREFIX)

    assert get_kv_cache_names(context) == [
        PREFIX,
        f"{PREFIX}.conv_ring",
        f"{PREFIX}.drafts",
    ]


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
