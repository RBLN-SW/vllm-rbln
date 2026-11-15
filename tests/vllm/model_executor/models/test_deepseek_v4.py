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


def test_target_preserves_streams_for_drafting():
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
    model.layers = torch.nn.ModuleList([StampStreams()])
    model.start_layer, model.end_layer = 0, 1
    embeddings = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    result = model(None, None, None, inputs_embeds=embeddings)
    expected = torch.cat([embeddings + i for i in range(4)], dim=-1)
    torch.testing.assert_close(result, expected)


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
