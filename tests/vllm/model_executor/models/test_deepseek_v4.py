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
        ([512], None, 130, "fp8", "prefill.*compression boundary"),
        ([512], 130, 0, "fp8", "prefill.*compression boundary"),
        ([130], None, 0, "fp8", "prefill.*compression boundary"),
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


@pytest.mark.parametrize("chunked", [False, True])
def test_compressor_preserves_state_across_prefill_chunks(monkeypatch, chunked):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from tests.vllm.vllm_config import make_vllm_config

    class NoRope(torch.nn.Module):
        def forward(self, x, positions):
            return x

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
    state = torch.empty(1, 8, 256, dtype=compressor.state_cache.dtype)
    slot = torch.zeros(1, 1, dtype=torch.int32)
    query_len = torch.tensor([[3 if chunked else 4]], dtype=torch.int32)
    monkeypatch.setattr(dsv4, "_ring_inputs", lambda cache: (state, slot, query_len))
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
        compressor = dsv4.RBLNDeepseekV4Compressor(
            config,
            4,
            64,
            dsv4.SLOT_STATE,
            dsv4.RBLNDeepseekV4Rope(config, 4, 64),
            "model.layers.0.attn.compressor",
        )
    rng = torch.Generator().manual_seed(18)
    with torch.no_grad():
        compressor.fused_wkv_wgate.weight.copy_(torch.randn(256, 4, generator=rng))
        compressor.ape.copy_(torch.randn(4, 128, generator=rng))
    monkeypatch.setattr(
        dsv4,
        "_ring_inputs",
        lambda cache: (
            torch.zeros(1, 8, 256),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([[query_len]], dtype=torch.int32),
        ),
    )
    x = torch.randn(1, 16, 4, generator=rng).bfloat16()
    start = torch.tensor([[60]], dtype=torch.int32)
    expected = compressor(x[:, :query_len], start)
    actual = compressor(x, start)
    torch.testing.assert_close(actual[:, :1], expected, atol=0, rtol=0)


def test_rope_sharing_is_local_to_each_model(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    import vllm_rbln.model_executor.models.deepseek_v4 as dsv4
    from tests.vllm.vllm_config import make_vllm_config

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        dsv4, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False)
    )
    monkeypatch.setattr(dsv4, "RBLNDeepseekV4MoE", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(
        dsv4,
        "make_layers",
        lambda count, factory, prefix: (
            0,
            count,
            torch.nn.ModuleList(factory(f"{prefix}.{i}") for i in range(count)),
        ),
    )
    hf_config = DeepseekV4Config(
        num_hidden_layers=2,
        num_attention_heads=1,
        hidden_size=4,
        head_dim=64,
        qk_rope_head_dim=64,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=1,
        sliding_window=128,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 0],
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config, max_model_len=64),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        quant_config=None,
    )
    with set_current_vllm_config(make_vllm_config()):
        cpu_model = dsv4.RBLNDeepseekV4Model(vllm_config=config, prefix="cpu")
        with torch.device("meta"):
            meta_model = dsv4.RBLNDeepseekV4Model(vllm_config=config, prefix="meta")
    cpu_rope = cpu_model.layers[0].attn.rope
    meta_rope = meta_model.layers[0].attn.rope
    assert cpu_rope is cpu_model.layers[1].attn.rope
    assert meta_rope is meta_model.layers[1].attn.rope
    assert cpu_rope.cos_cache.device.type == "cpu"
    assert meta_rope.cos_cache.device.type == "meta"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_matches_upstream(nested, ratio, inverse):
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4Rope

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
        upstream = build_deepseek_v4_rope(
            config,
            head_dim=128,
            rope_head_dim=64,
            max_position_embeddings=256,
            compress_ratio=ratio,
        )
        rope = RBLNDeepseekV4Rope(config, ratio, 256)
    positions = torch.tensor([[0, 13, 65], [127, 129, 255]])
    x = torch.randn(2, 3, 2, 128, generator=torch.Generator().manual_seed(9)).bfloat16()
    expected, _ = upstream.forward_native(
        positions.flatten(),
        x.reshape(6, 2, 128),
        inverse=inverse,
    )
    torch.testing.assert_close(
        rope(x, positions, inverse), expected.reshape_as(x), atol=0, rtol=0
    )


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


@pytest.mark.parametrize("history", [8, 128])
@pytest.mark.parametrize("drafts", [0, 1, 3, 8])
def test_ring_capacity_preserves_history_across_rejected_drafts(history, drafts):
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4RingCache

    ring = RBLNDeepseekV4RingCache.__new__(RBLNDeepseekV4RingCache)
    torch.nn.Module.__init__(ring)
    ring.window, ring.width, ring.dtype = history, 768, torch.uint8
    config = SimpleNamespace(
        cache_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts)
        if drafts
        else None,
    )
    rows = ring.get_kv_cache_spec(config).block_size
    tokens = list(range(rows))
    start = rows
    for p in range(start, start + drafts + 1):
        tokens[p % rows] = p
    next_query = start + 1
    for p in range(next_query - history + 1, next_query):
        assert tokens[p % rows] == p
    if not drafts:
        assert rows == history


@pytest.mark.parametrize("is_prefill", [False, True])
def test_ring_metadata_keeps_one_slot_past_ring_boundaries(is_prefill):
    from tests.vllm.v1.attention.utils import make_builder, make_common_attn_metadata
    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4RingCache

    ring = RBLNDeepseekV4RingCache.__new__(RBLNDeepseekV4RingCache)
    torch.nn.Module.__init__(ring)
    ring.window, ring.width, ring.dtype = 128, 768, torch.uint8
    config = make_vllm_config(max_model_len=4096, max_num_batched_tokens=16)
    spec = ring.get_kv_cache_spec(config)
    builder = make_builder(
        config, spec=spec, builder_cls=ring.get_attn_backend().get_builder_cls()
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
def test_head_relocation_preserves_reduction(sampled_only):
    from vllm_rbln.model_executor.models.deepseek_v4 import _hc_head_hidden

    rng = torch.Generator().manual_seed(7)
    streams = torch.randn(2, 4, 3, 8, generator=rng)
    fn = torch.randn(4, 32, generator=rng)
    base = torch.randn(4, generator=rng)
    scale = torch.tensor([0.25])
    norm = torch.nn.RMSNorm(8, eps=1e-5)
    flat = streams.transpose(1, 2).reshape(2, 3, 32)
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    gates = torch.sigmoid(normalized @ fn.T * scale + base) + 1e-6
    expected = norm((streams * gates.transpose(1, 2).unsqueeze(-1)).sum(1))
    if sampled_only:
        flat = flat[:, -1]
        expected = expected[:, -1]
    actual = _hc_head_hidden(flat, 4, fn, scale, base, 1e-5, 1e-6, norm)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "tokens,cut", [(1, False), (4, False), (256, True), (512, True), (260, False)]
)
def test_mhc_sublayer_token_cut(tokens, cut):
    from vllm_rbln.model_executor.models.deepseek_v4 import RBLNDeepseekV4DecoderLayer

    class Sublayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fn", torch.randn(24, 4 * 128) * 0.02)
            self.register_buffer("scale", torch.linspace(0.5, 1.5, 24))
            self.register_buffer("base", torch.randn(24) * 0.1)
            self.norm = torch.nn.RMSNorm(128, eps=1e-5, dtype=torch.bfloat16)
            self.config = SimpleNamespace(norm_eps=1e-5, hc_eps=1e-6, hc_iters=20)

        def forward(self, streams):
            return RBLNDeepseekV4DecoderLayer._sublayer(
                self.config,
                streams,
                self.fn,
                self.scale,
                self.base,
                self.norm,
                lambda x, factor: x * factor,
                0.5,
            )

    torch.manual_seed(7)
    module = Sublayer().eval()
    streams = torch.randn(1, 4, tokens, 128, dtype=torch.bfloat16)
    graph = torch.export.export(module, (streams,)).graph
    targets = {str(node.target) for node in graph.nodes if node.op == "call_function"}
    for name in ("mhc_mix", "mhc_read", "mhc_write"):
        assert (f"rbln_custom_ops.{name}.default" in targets) == cut

    flat = streams.transpose(1, 2).reshape(1, tokens, -1).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    logits = torch.nn.functional.linear(flat, module.fn) * module.scale + module.base
    pre = torch.sigmoid(logits[..., :4]) + 1e-6
    x = (streams.float() * pre.transpose(1, 2).unsqueeze(-1)).sum(1)
    y = (module.norm(x.to(streams.dtype)) * 0.5).float()
    comb = torch.softmax(logits[..., 8:].reshape(1, tokens, 4, 4), dim=-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    expected = torch.matmul(comb.transpose(-1, -2), streams.float().transpose(1, 2))
    expected += (2 * torch.sigmoid(logits[..., 4:8])).unsqueeze(-1) * y.unsqueeze(-2)
    got = module(streams)
    assert got.dtype == streams.dtype
    torch.testing.assert_close(
        got, expected.transpose(1, 2).to(streams.dtype), atol=2e-2, rtol=2e-2
    )
