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
from unittest.mock import patch

import pytest

from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNModelOptNvFp4Config,
)
from vllm_rbln.model_executor.models import inkling
from vllm_rbln.model_executor.models.inkling import RBLNInklingForCausalLM

MAPPER = RBLNInklingForCausalLM.hf_to_vllm_mapper
EXPERTS = "model.llm.layers.3.mlp.experts."
ROUTED = "model.layers.3.mlp.experts.routed_experts."


@pytest.mark.parametrize(
    ("checkpoint", "param"),
    [
        (EXPERTS + "w13_weight", ROUTED + "w13_weight"),
        (EXPERTS + "w13_weight.scale", ROUTED + "w13_weight_scale"),
        (EXPERTS + "w2_weight.scale", ROUTED + "w2_weight_scale"),
        (EXPERTS + "w13_weight.scale2", ROUTED + "w13_weight_scale_2"),
        (EXPERTS + "w2_weight.scale2", ROUTED + "w2_weight_scale_2"),
        (EXPERTS + "w13_weight.input_amax", ROUTED + "w13_input_scale"),
        (EXPERTS + "w2_weight.input_amax", ROUTED + "w2_input_scale"),
        (EXPERTS + "w13_weight.original_shape", None),
    ],
)
def test_nvfp4_expert_tensors_map_onto_routed_expert_params(checkpoint, param):
    assert MAPPER._map_name(checkpoint) == param


def test_modelopt_exclusions_reach_the_rbln_modules():
    # Entries as Inkling-Small-NVFP4's hf_quant_config.json writes them.
    config = RBLNModelOptNvFp4Config(
        "NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=[
            "model.llm.unembed",
            "model.llm.layers.0.attn",
            "model.llm.layers.0.mlp.w13_dn",
            "model.llm.layers.0.mlp.w2_md",
            "model.llm.layers.2.mlp.experts",
            "model.llm.layers.3.attn",
            "model.llm.layers.3.mlp.gate",
            "model.llm.layers.3.mlp.shared_experts",
        ],
    )
    config.apply_vllm_mapper(MAPPER)

    for prefix in (
        "lm_head",
        "model.layers.0.attn.q_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.2.mlp.experts",
        "model.layers.3.attn.r_proj",
        "model.layers.3.mlp.gate",
        "model.layers.3.mlp.shared_experts.1.gate_up_proj",
    ):
        assert config.is_layer_excluded(prefix), prefix
    assert not config.is_layer_excluded("model.layers.3.mlp.experts")


def test_speculative_decoding_is_refused():
    # The runner replays accepted drafts only for MambaSpec groups, which
    # Inkling's rings are not, so nothing downstream would catch it.
    config = SimpleNamespace(
        use_sconv=True,
        use_embed_norm=True,
        q_bias=False,
        o_bias=False,
        final_logit_softcapping=None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        speculative_config=SimpleNamespace(method="ngram"),
    )
    with (
        patch.object(
            inkling, "get_pp_group", return_value=SimpleNamespace(world_size=1)
        ),
        pytest.raises(NotImplementedError, match="speculative decoding"),
    ):
        RBLNInklingForCausalLM(vllm_config=vllm_config)


def _attention_layers(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config

    from tests.vllm.vllm_config import make_vllm_config

    config = make_vllm_config(block_size=128)
    text = SimpleNamespace(
        hidden_size=128,
        rms_norm_eps=1e-6,
        sconv_kernel_size=4,
        d_rel=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        sliding_window_size=64,
        rel_extent=128,
        log_scaling_n_floor=128,
        log_scaling_alpha=0.1,
    )
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    with set_current_vllm_config(config):
        for i, is_local in enumerate((True, False)):
            inkling.RBLNInklingAttention(text, is_local, None, f"model.layers.{i}.attn")
    return config


def test_local_layers_keep_only_their_window(monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    config = _attention_layers(monkeypatch)
    context = config.compilation_config.static_forward_context
    local = context["model.layers.0.attn"].get_kv_cache_spec(config)
    full = context["model.layers.1.attn"].get_kv_cache_spec(config)
    assert type(local) is SlidingWindowSpec
    assert local.sliding_window == 64
    assert type(full) is FullAttentionSpec


def test_stream_conv_adds_the_conv_rounded_once(monkeypatch):
    import torch
    from vllm.config import set_current_vllm_config

    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.v1.attention import kv_cache_bindings
    from vllm_rbln.v1.attention.backends.inkling import RBLNInklingAttentionMetadata

    tokens, channels = 16, 128
    with set_current_vllm_config(make_vllm_config(block_size=128)):
        ring = inkling.RBLNInklingConvRing("ring", "owner", 2, channels, 4)
    metadata = RBLNInklingAttentionMetadata.__new__(RBLNInklingAttentionMetadata)
    index = torch.tensor([1], dtype=torch.int32)
    metadata.kv_caches = [torch.zeros(2, *ring.block_shape, dtype=torch.bfloat16)]
    metadata.kv_cache_positions = {"ring": 0}
    metadata.kv_cache_view_infos = None
    metadata.conv_load_blocks = metadata.conv_store_blocks = index
    metadata.conv_start_positions = torch.tensor([0], dtype=torch.int32)
    metadata.conv_query_lens = torch.tensor([tokens], dtype=torch.int32)
    metadata.has_initial_state = torch.tensor([0], dtype=torch.int32)
    context = SimpleNamespace(attn_metadata={"owner": metadata}, additional_kwargs={})
    for module in (inkling, kv_cache_bindings):
        monkeypatch.setattr(module, "get_forward_context", lambda: context)

    generator = torch.Generator().manual_seed(0)
    xs = [
        torch.randn(1, tokens, channels, generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]
    taps = [
        torch.randn(channels, 4, generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]
    # Both streams side by side, from an empty ring: x + causal conv, rounded once.
    seq = torch.cat([torch.zeros(3, 2 * channels), torch.cat(xs, -1)[0].float()]).T
    weight = torch.cat(taps).float()[:, None]
    conv = torch.nn.functional.conv1d(seq[None], weight, groups=2 * channels)
    conv = conv[0].T.to(torch.bfloat16).float().split(channels, dim=-1)
    for x, c, got in zip(xs, conv, ring.convolve(xs, taps)):
        assert torch.equal(got, (x.float() + c).to(x.dtype))
