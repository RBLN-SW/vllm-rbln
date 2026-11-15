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
