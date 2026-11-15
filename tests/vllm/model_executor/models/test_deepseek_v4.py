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
