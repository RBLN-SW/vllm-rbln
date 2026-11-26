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


@pytest.mark.parametrize(
    "chunks,scheduled,threshold,indexer_dtype,error",
    [
        ([512], None, 130, "fp8", None),
        ([512], 130, 0, "fp8", None),
        ([512], None, 0, "fp8", None),
        ([512], None, 128, "fp8", None),
        ([64], None, 0, "fp8", None),
        ([64, 512], None, 64, "fp8", None),
        ([512], None, 0, "auto", None),
        ([512], None, 0, "mxfp4", "indexer.*fp8"),
    ],
)
def test_model_validates_compression_and_indexer_constraints(
    monkeypatch, chunks, scheduled, threshold, indexer_dtype, error
):
    from vllm.config import AttentionConfig

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4

    monkeypatch.setattr(
        dsv4,
        "RBLNDeepseekV4Model",
        lambda **kwargs: SimpleNamespace(make_empty_intermediate_tensors=None),
    )
    monkeypatch.setattr(
        dsv4, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(dsv4, "LogitsProcessor", lambda vocab_size: torch.nn.Identity())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(compress_ratios=[0, 4, 128], vocab_size=16)
        ),
        quant_config=None,
        attention_config=AttentionConfig(indexer_kv_dtype=indexer_dtype),
        additional_config=SimpleNamespace(prefill_chunk_size=chunks),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512,
            max_num_scheduled_tokens=scheduled,
            long_prefill_token_threshold=threshold,
        ),
    )
    if error is None:
        dsv4.RBLNDeepseekV4ForCausalLM(vllm_config=config)
    else:
        with pytest.raises(ValueError, match=error):
            dsv4.RBLNDeepseekV4ForCausalLM(vllm_config=config)


@pytest.mark.parametrize("window", [64, 128, 256])
def test_attention_rejects_unsupported_window_before_allocating_caches(
    monkeypatch, window
):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from tests.vllm.vllm_config import make_vllm_config

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        dsv4, "build_deepseek_v4_rope", lambda *args, **kwargs: torch.nn.Identity()
    )
    hf_config = SimpleNamespace(
        num_attention_heads=2,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=1,
        sliding_window=window,
        rms_norm_eps=1e-6,
        compress_ratios=[0],
        num_hidden_layers=1,
        hidden_size=16,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config, max_model_len=2048),
        quant_config=None,
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    context = make_vllm_config()
    with set_current_vllm_config(context):
        if window == 128:
            attention = dsv4.RBLNDeepseekV4Attention(config, 0, "model.layers.0.attn")
            assert attention.swa_cache.block_size == window
        else:
            with pytest.raises(ValueError, match="sliding_window=128"):
                dsv4.RBLNDeepseekV4Attention(config, 0, "model.layers.0.attn")
            assert not context.compilation_config.static_forward_context


@pytest.mark.parametrize("chunked", [False, True])
def test_compressor_preserves_state_across_prefill_chunks(monkeypatch, chunked):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from tests.vllm.vllm_config import make_vllm_config

    class NoRope(torch.nn.Module):
        def forward(self, positions, x):
            return x, None

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    with set_current_vllm_config(make_vllm_config()):
        compressor = dsv4.RBLNDeepseekV4Compressor(
            SimpleNamespace(hidden_size=4, rms_norm_eps=1e-6),
            4,
            64,
            dsv4.SLOT_STATE,
            NoRope(),
            "model.layers.0.attn.compressor",
        )
    rng = torch.Generator().manual_seed(17)
    with torch.no_grad():
        compressor.fused_wkv_wgate.weight.copy_(torch.randn(256, 4, generator=rng))
        compressor.ape.copy_(torch.randn(4, 128, generator=rng))
    x = torch.randn(1, 4, 4, generator=rng).bfloat16()
    kv, score = torch.nn.functional.linear(
        x.float(), compressor.fused_wkv_wgate.weight
    ).chunk(2, dim=-1)
    pooled = (kv[..., 64:] * (score + compressor.ape)[..., 64:].softmax(1)).sum(
        1, keepdim=True
    )
    expected = compressor.norm(pooled.to(x.dtype))
    state = torch.empty(2, 64, 256, dtype=compressor.state_cache.dtype)
    slot = torch.ones(1, 1, dtype=torch.int32)
    query_len = torch.tensor([[3 if chunked else 4]], dtype=torch.int32)
    monkeypatch.setattr(
        dsv4.RBLNDeepseekV4WindowCache,
        "inputs",
        lambda self: (state, slot, query_len, None),
    )
    if chunked:
        compressor(x[:, :3], torch.tensor([[0]], dtype=torch.int32))
        query_len.fill_(1)
        actual = compressor(x[:, 3:], torch.tensor([[3]], dtype=torch.int32))
    else:
        actual = compressor(x, torch.tensor([[0]], dtype=torch.int32))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mtp_masks_only_position_zero_embeddings():
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4MTPLayer

    class Projection(torch.nn.Module):
        def forward(self, x):
            return x, None

    layer = RBLNDeepseekV4MTPLayer.__new__(RBLNDeepseekV4MTPLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_mult = 4
    layer.enorm = layer.hnorm = torch.nn.Identity()
    layer.e_proj = layer.h_proj = Projection()
    layer.forward = lambda positions, streams, ids: streams
    positions = torch.tensor([[0, 1, 2], [5, 0, 7]])
    embeds = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
    original = embeds.clone()
    previous = torch.ones(2, 3, 16)

    actual = layer.forward_mtp(positions, embeds, previous, torch.ones_like(positions))

    masked = torch.where(positions.unsqueeze(-1) == 0, 0, embeds)
    expected = previous + masked.repeat(1, 1, 4)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeds, original)


@pytest.mark.parametrize("query_len", [3, 4])
def test_compressor_padding_does_not_index_past_the_rope_table(monkeypatch, query_len):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from tests.vllm.vllm_config import make_vllm_config

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    config = DeepseekV4Config(
        hidden_size=4,
        head_dim=64,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        compress_rope_theta=160000,
        rope_parameters={"rope_type": "default"},
    )
    with set_current_vllm_config(make_vllm_config()):
        rope = build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=64,
            compress_ratio=4,
        )
        compressor = dsv4.RBLNDeepseekV4Compressor(
            config, 4, 64, dsv4.SLOT_STATE, rope, "model.layers.0.attn.compressor"
        )
    rng = torch.Generator().manual_seed(18)
    with torch.no_grad():
        compressor.fused_wkv_wgate.weight.copy_(torch.randn(256, 4, generator=rng))
        compressor.ape.copy_(torch.randn(4, 128, generator=rng))
    monkeypatch.setattr(
        dsv4.RBLNDeepseekV4WindowCache,
        "inputs",
        lambda self: (
            torch.zeros(2, 64, 256),
            torch.ones(1, 2, dtype=torch.int32),
            torch.tensor([[query_len]], dtype=torch.int32),
            None,
        ),
    )
    x = torch.randn(1, 16, 4, generator=rng).bfloat16()
    start = torch.tensor([[60]], dtype=torch.int32)
    expected = compressor(x[:, :query_len], start)
    actual = compressor(x, start)
    torch.testing.assert_close(actual[:, :1], expected, atol=0, rtol=0)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("layout", ["headless", "token_major", "head_major"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_matches_upstream_with_activation_dtype_coefficients(
    monkeypatch, nested, ratio, inverse, layout, dtype
):
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        RBLNDeepseekV4ScalingRotaryEmbedding,
    )

    scaling = dict(
        rope_type="yarn",
        factor=4.0,
        original_max_position_embeddings=64,
        beta_fast=32,
        beta_slow=1,
    )
    parameters = (
        {"main": {"rope_type": "default"}, "compress": scaling} if nested else scaling
    )
    config = DeepseekV4Config(
        head_dim=128,
        qk_rope_head_dim=64,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_parameters=parameters,
    )
    with set_current_vllm_config(make_vllm_config()):
        rope = build_deepseek_v4_rope(
            config,
            head_dim=128,
            rope_head_dim=64,
            max_position_embeddings=256,
            compress_ratio=ratio,
        )
    assert isinstance(rope, RBLNDeepseekV4ScalingRotaryEmbedding)
    positions = torch.tensor([[0, 13, 65], [127, 129, 255]])
    shape = (2, 3, 128) if layout == "headless" else (2, 3, 2, 128)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(9)).to(dtype)
    if layout == "head_major":
        actual, _ = rope.forward_oot(
            positions, x.transpose(1, 2), inverse=inverse, head_major=True
        )
        actual = actual.transpose(1, 2)
    else:
        actual, _ = rope.forward_oot(positions, x, inverse=inverse)
    assert rope.cos_sin_cache.dtype == torch.float32
    monkeypatch.setattr(rope, "cos_sin_cache", rope.cos_sin_cache.to(dtype))
    expected, _ = rope.forward_native(
        positions.flatten(), x.reshape(6, -1, 128), inverse=inverse
    )
    torch.testing.assert_close(actual, expected.reshape_as(x), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp4_experts_preserve_scales_clamp_and_routing(dtype):
    from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import (
        RBLNDeepseekV4Mxfp4MoEMethod,
    )

    method = object.__new__(RBLNDeepseekV4Mxfp4MoEMethod)
    method.swiglu_limit = 0.75
    # Two equal FP4 nibbles per byte: 0x22 = (1, 1), 0x44 = (2, 2),
    # 0x11 = (0.5, 0.5). Scales produce dense weights 0.5, 0.25 and 0.125.
    layer = SimpleNamespace(
        w1_weight=torch.full((1, 32, 16), 0x22, dtype=torch.uint8),
        w3_weight=torch.full((1, 32, 16), 0x44, dtype=torch.uint8),
        w2_weight=torch.full((1, 32, 16), 0x11, dtype=torch.uint8),
        w1_weight_scale=torch.full((1, 32, 1), 0.5, dtype=torch.bfloat16),
        w3_weight_scale=torch.full((1, 32, 1), 0.125, dtype=torch.bfloat16),
        w2_weight_scale=torch.full((1, 32, 1), 0.25, dtype=torch.bfloat16),
        expert_map=None,
    )
    x = torch.tensor([-0.25, -0.03125, 0.0625, 0.25], dtype=dtype)
    x = x.view(2, 2, 1).expand(2, 2, 32).contiguous()
    routing = torch.tensor([[0.25, 0.5, 1.0, 0.0]], dtype=dtype)
    gate = torch.nn.functional.linear(x, torch.full((32, 32), 0.5, dtype=dtype))
    up = torch.nn.functional.linear(x, torch.full((32, 32), 0.25, dtype=dtype))
    gate, up = gate.clamp(max=0.75), up.clamp(-0.75, 0.75)
    expected = torch.nn.functional.linear(
        gate * gate.sigmoid() * up, torch.full((32, 32), 0.125, dtype=dtype)
    ) * routing.view(2, 2, 1)

    actual = method.apply(layer, x, routing)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_exponent_scales_preserve_the_e8m0_value(dtype):
    from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import e8m0_to_bf16

    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    actual = e8m0_to_bf16(raw.view(dtype))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize(
    "block_size,ratio,expected",
    [(2048, 4, 2048), (2048, 128, 8192), (8192, 128, 8192), (16384, 128, 16384)],
)
def test_paged_block_size_follows_the_cache_shard_contract(block_size, ratio, expected):
    from vllm_rbln.model_executor.models.deepseek_v4 import _paged_block_size

    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size))
    assert _paged_block_size(config, ratio) == expected


def test_cache_inputs_use_names_in_the_rank_local_cache_order():
    from vllm.config import set_current_vllm_config

    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.forward_context import set_forward_context
    from vllm_rbln.model_executor.models.deepseek_v4 import (
        RBLNDeepseekV4AuxCache,
        RBLNDeepseekV4PagedCache,
        RBLNDeepseekV4WindowCache,
    )
    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionMetadata,
    )

    config = make_vllm_config()
    prefix = "model.layers.12.self_attn.attn"
    with set_current_vllm_config(config):
        window = RBLNDeepseekV4WindowCache(f"{prefix}.0", 128, 768, torch.uint8, 128)
        paged = RBLNDeepseekV4PagedCache(f"{prefix}.2", 4, 128, torch.float8_e4m3fn)
        scale = RBLNDeepseekV4AuxCache(f"{prefix}.3", paged, torch.bfloat16)
    caches = [torch.zeros(2, 64), torch.zeros(2, 64, 128), torch.zeros(2, 128, 768)]
    metadata = RBLNFlashAttentionMetadata(
        seq_lens=torch.tensor([[0]]),
        block_tables=torch.tensor([[1]]),
        is_prefill=True,
        cache_seq_lens=torch.tensor([[0]]),
        cache_offsets=torch.tensor([[4]]),
        kv_caches=caches,
        kv_cache_positions={scale.prefix: 0, paged.prefix: 1, window.prefix: 2},
    )
    with set_forward_context({window.prefix: metadata, paged.prefix: metadata}, config):
        assert window.inputs()[0] is caches[2]
        assert paged.inputs()[0] is caches[1]
        assert scale.inputs() is caches[0]


@pytest.mark.parametrize("segment,block_size", [(128, 128), (8, 64)])
@pytest.mark.parametrize("retention", [128, 512, 2048])
def test_window_pool_reserves_periodic_prefixes(segment, block_size, retention):
    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4WindowCache

    window = RBLNDeepseekV4WindowCache.__new__(RBLNDeepseekV4WindowCache)
    torch.nn.Module.__init__(window)
    window.segment, window.block_size = segment, block_size
    window.width, window.dtype = 768, torch.uint8
    config = make_vllm_config(max_model_len=17408, max_num_batched_tokens=512)
    config.scheduler_config.max_num_seqs = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    baseline = window.get_kv_cache_spec(config).num_state_slots
    config.cache_config.prefix_cache_retention_interval = retention
    spec = window.get_kv_cache_spec(config)
    pages = {
        token // block_size
        for boundary in range(
            retention, config.model_config.max_model_len + 1, retention
        )
        for token in range(max(0, boundary - (2 * segment - 1)), boundary)
    }
    assert spec.num_state_slots >= baseline + len(pages) * 16


@pytest.mark.parametrize("is_prefill", [False, True])
def test_window_metadata_preserves_absolute_positions(is_prefill):
    from tests.vllm.v1.attention.utils import make_builder, make_common_attn_metadata
    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4WindowCache

    window = RBLNDeepseekV4WindowCache.__new__(RBLNDeepseekV4WindowCache)
    torch.nn.Module.__init__(window)
    window.segment, window.block_size, window.width, window.dtype = (
        128,
        128,
        768,
        torch.uint8,
    )
    window.prefix = "model.layers.0.self_attn.attn.0"
    config = make_vllm_config(max_model_len=4096, max_num_batched_tokens=16)
    spec = window.get_kv_cache_spec(config)
    builder = make_builder(
        config, spec=spec, builder_cls=window.get_attn_backend().get_builder_cls()
    )
    count = 1 if is_prefill else 2
    metadata = builder.build(
        make_common_attn_metadata(
            num_reqs=count,
            query_start_loc=torch.tensor([0, 4, 8][: count + 1]),
            seq_lens=torch.tensor([130, 259][:count]),
            block_table_tensor=torch.tensor([[2], [1]][:count]),
        ),
        torch.arange(8),
        batch_pad=1 if is_prefill else 3,
        is_prefill=is_prefill,
        back_pad=torch.tensor([1, 3][:count]),
    )
    expected_slots = [2] if is_prefill else [2, 1, 0]
    expected_start = [127] if is_prefill else [127, 258, 0]
    expected_end = [130] if is_prefill else [130, 259, 0]
    assert metadata.block_tables.flatten().tolist() == expected_slots
    assert metadata.cache_seq_lens.flatten().tolist() == expected_start
    assert metadata.cache_offsets.flatten().tolist() == expected_end


@pytest.mark.parametrize("target_layers", [6, 43])
def test_mtp_compression_uses_checkpoint_tail(target_layers):
    from vllm_rbln.model_executor.models.deepseek_v4 import _compress_ratio

    config = SimpleNamespace(
        compress_ratios=[4, 128] * 22 + [0],
        num_hidden_layers=target_layers,
        num_nextn_predict_layers=1,
    )
    assert _compress_ratio(config, target_layers) == 0


@pytest.mark.parametrize("stages", [1, 2, 3])
@pytest.mark.parametrize("tokens", [1, 3])
def test_target_preserves_streams_for_drafting(monkeypatch, stages, tokens):
    from vllm.sequence import IntermediateTensors

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4Model

    class StampStreams(torch.nn.Module):
        def forward(self, positions, streams, input_ids):
            offsets = torch.arange(4).reshape(1, 4, 1, 1)
            return streams + offsets

    model = RBLNDeepseekV4Model.__new__(RBLNDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 4
    model.hc_head_fn = torch.zeros(4, 8)
    model.hc_head_base = torch.zeros(4)
    model.hc_head_scale = torch.ones(1)
    model.norm_eps, model.hc_eps = 1e-5, 1e-6
    model.norm = torch.nn.Identity()
    model.layers = torch.nn.ModuleList([StampStreams() for _ in range(6)])
    embeddings = torch.arange(4 * tokens, dtype=torch.float32).reshape(2, tokens, 2)
    intermediate = None
    for rank in range(stages):
        group = SimpleNamespace(
            is_first_rank=rank == 0, is_last_rank=rank == stages - 1
        )
        monkeypatch.setattr(dsv4, "get_pp_group", lambda group=group: group)
        model.start_layer, model.end_layer = (
            rank * 6 // stages,
            (rank + 1) * 6 // stages,
        )
        result = model(
            None, None, intermediate, inputs_embeds=embeddings if rank == 0 else None
        )
        if rank < stages - 1:
            assert isinstance(result, IntermediateTensors)
            assert result["streams"].shape == (2, tokens, 8)
            intermediate = result
    expected = torch.cat([embeddings + 6 * i for i in range(4)], dim=-1)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("stages", [1, 2])
def test_mtp_loads_own_embedding_on_last_pp_stage(monkeypatch, stages):
    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4

    group = SimpleNamespace(world_size=stages, is_last_rank=True)
    monkeypatch.setattr(dsv4, "get_pp_group", lambda: group)
    model = dsv4.RBLNDeepseekV4MTP.__new__(dsv4.RBLNDeepseekV4MTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=43)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(4, 2)
    model.model.embed_tokens.weight.data.zero_()
    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    model.load_weights(iter([("embed.weight", weight)]))
    expected = weight if stages > 1 else torch.zeros_like(weight)
    torch.testing.assert_close(model.model.embed_tokens.weight, expected)


@pytest.mark.parametrize("sampled_only", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_head_relocation_preserves_reduction(monkeypatch, sampled_only, prefill, dtype):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.model_executor.models.deepseek_v4 import _hc_head_hidden

    reads = []
    residual_read = torch.ops.rbln_custom_ops.residual_read

    def read(x, weight, eps):
        reads.append(tuple(x.shape))
        return residual_read(x, weight, eps)

    monkeypatch.setattr(dsv4, "is_prefill_step", lambda: prefill)
    monkeypatch.setattr(torch.ops.rbln_custom_ops, "residual_read", read)
    rng = torch.Generator().manual_seed(7)
    streams = torch.randn(2, 4, 3, 8, generator=rng).to(dtype)
    fn = torch.randn(4, 32, generator=rng)
    base = torch.randn(4, generator=rng)
    scale = torch.tensor([0.25])
    with set_current_vllm_config(make_vllm_config()):
        norm = RMSNorm(8, eps=1e-5, dtype=dtype)
    norm.weight.data = (1.0 + 0.1 * torch.randn(8, generator=rng)).to(dtype)
    flat = streams.transpose(1, 2).reshape(2, 3, 32)
    normalized = flat.float() * torch.rsqrt(
        flat.float().square().mean(-1, keepdim=True) + 1e-5
    )
    gates = torch.sigmoid(normalized @ fn.T * scale + base) + 1e-6
    expected = norm((streams * gates.transpose(1, 2).unsqueeze(-1)).sum(1).to(dtype))
    if sampled_only:
        flat = flat[:, -1]
        expected = expected[:, -1]
    actual = _hc_head_hidden(flat, 4, fn, scale, base, 1e-5, 1e-6, norm)
    assert reads == ([] if prefill else [(2, 1 if sampled_only else 3, 8)])
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)


class _Norm:
    def __init__(self, hidden):
        self.weight = 1.0 + 0.1 * torch.randn(hidden, dtype=torch.bfloat16)

    def __call__(self, x):
        rows = x.float()
        normed = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (normed.to(self.weight.dtype) * self.weight).to(x.dtype)


def _mhc_layer(cls, hidden=64):
    class Attention(torch.nn.Module):
        swa_cache = SimpleNamespace(prefix="attn.swa_cache")

        def forward(self, positions, h):
            return torch.sin(h)

    torch.manual_seed(0)
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.hc_mult, layer.norm_eps, layer.hc_eps = 4, 1e-5, 1e-6
    layer.hc_iters, layer.hc_post_mult = 20, 2.0
    mix = (2 + layer.hc_mult) * layer.hc_mult
    for kind in ("attn", "ffn"):
        setattr(layer, f"hc_{kind}_fn", 0.05 * torch.randn(mix, layer.hc_mult * hidden))
        setattr(layer, f"hc_{kind}_base", 0.1 * torch.randn(mix))
        setattr(layer, f"hc_{kind}_scale_vec", 0.5 + torch.rand(mix))
    layer.attn_norm = _Norm(hidden)
    layer.ffn_norm = _Norm(hidden)
    layer.attn = Attention()
    layer.ffn = lambda h, input_ids: torch.tanh(h)
    return layer


def _mhc_kernels(monkeypatch):
    ops = torch.ops.rbln_custom_ops
    calls = []
    n = 4

    def mix(streams, mix_weight, scale, base, eps):
        calls.append("mix")
        batch, _, tokens, hidden = streams.shape
        flat = streams.transpose(1, 2).reshape(batch, tokens, n * hidden).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        logits = torch.nn.functional.linear(flat, mix_weight) * scale + base
        return logits.to(streams.dtype)

    def read(streams, logits, eps):
        calls.append("read")
        return ops.hc_stream_reduce(streams, logits[..., :n].contiguous(), eps)

    def write(x, streams, logits, post_mult, iters, eps):
        calls.append("write")
        comb = ops.sinkhorn_normalize(
            logits[..., 2 * n :].contiguous(), iters=iters, eps=eps
        )
        out = ops.hc_residual_add(
            x, streams, logits[..., n : 2 * n].contiguous(), comb, post_mult=post_mult
        )
        return out.to(torch.bfloat16).to(streams.dtype)

    for name, kernel in (("mhc_mix", mix), ("mhc_read", read), ("mhc_write", write)):
        monkeypatch.setattr(ops, name, kernel, raising=False)
    return calls


@pytest.fixture
def mhc_metadata():
    from vllm.forward_context import ForwardContext, override_forward_context

    metadata = SimpleNamespace(is_prefill=True)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"attn.swa_cache": metadata},
        slot_mapping={},
    )
    with override_forward_context(context):
        yield metadata


@pytest.mark.parametrize("prefill", [True, False])
def test_every_step_runs_the_mhc_on_the_cut_kernels(monkeypatch, mhc_metadata, prefill):
    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4

    calls = _mhc_kernels(monkeypatch)
    layer = _mhc_layer(dsv4.RBLNDeepseekV4DecoderLayer)
    batch, tokens = (1, 8) if prefill else (8, 1)
    mhc_metadata.is_prefill = prefill
    streams = torch.randn(batch, 4, tokens, 64).to(torch.bfloat16)
    positions = torch.arange(tokens).repeat(batch, 1)
    input_ids = torch.zeros(batch, tokens, dtype=torch.long)

    out = layer(positions, streams, input_ids)

    assert calls == ["mix", "read", "write"] * 2
    assert out.shape == streams.shape and out.dtype == streams.dtype


@pytest.mark.parametrize(
    "first_rank,last_rank", [(True, True), (True, False), (False, False), (False, True)]
)
def test_the_model_returns_the_streams_in_the_input_dtype_after_a_cut_prefill(
    monkeypatch,
    mhc_metadata,
    first_rank,
    last_rank,
):
    from vllm.sequence import IntermediateTensors

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4

    monkeypatch.setattr(
        dsv4,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=first_rank, is_last_rank=last_rank),
    )
    model = dsv4.RBLNDeepseekV4Model.__new__(dsv4.RBLNDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 4
    model.layers = torch.nn.ModuleList([_mhc_layer(dsv4.RBLNDeepseekV4DecoderLayer)])
    model.start_layer, model.end_layer = 0, 1
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    embeds = torch.randn(1, 8, 64).to(torch.bfloat16)
    intermediate = None
    if not first_rank:
        intermediate = IntermediateTensors({"streams": embeds.repeat(1, 1, 4)})

    out = model(input_ids, torch.arange(8).unsqueeze(0), intermediate, embeds)
    if not last_rank:
        out = out["streams"]

    assert out.dtype == torch.bfloat16


def test_the_drafter_returns_the_streams_in_the_input_dtype_after_a_cut_prefill(
    monkeypatch,
    mhc_metadata,
):
    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4

    class Projection(torch.nn.Module):
        def forward(self, x):
            return x, None

    _mhc_kernels(monkeypatch)
    layer = _mhc_layer(dsv4.RBLNDeepseekV4MTPLayer)
    layer.enorm = layer.hnorm = torch.nn.Identity()
    layer.e_proj = layer.h_proj = Projection()
    positions = torch.arange(1, 9).unsqueeze(0)
    embeds = torch.randn(1, 8, 64).to(torch.bfloat16)
    previous = torch.randn(1, 8, 256).to(torch.bfloat16)

    out = layer.forward_mtp(positions, embeds, previous, torch.zeros_like(positions))

    assert out.dtype == torch.bfloat16
