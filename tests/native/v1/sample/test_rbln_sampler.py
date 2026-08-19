# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from vllm_rbln.v1.sample import rbln_sampler
from vllm_rbln.v1.sample.rbln_sampler import RBLNTopKTopPSampler


def test_topk_topp_returns_its_own_alternating_buffers(monkeypatch):
    """The sampling graph's own output must not leave the sampler.

    Async scheduling keeps the returned tensor past the step boundary; keeping the
    graph's output there stalls that graph's next launch.
    """
    graph_outputs: list[torch.Tensor] = []

    def fake_compile_sampler(op, compile_context):
        def run(logits, temperature, k, p):
            out = torch.full((logits.shape[0],), len(graph_outputs), dtype=torch.int32)
            graph_outputs.append(out)
            return out

        return run

    monkeypatch.setattr(rbln_sampler, "compile_sampler", fake_compile_sampler)
    sampler = RBLNTopKTopPSampler()

    logits = torch.zeros(4, 8)
    temperature = torch.ones(4)
    returned = []
    for _ in range(4):
        got = sampler(logits, {}, temperature, None, None)[0]
        # Checked here, not after the loop: two slots means call N+2 overwrites it.
        assert got.data_ptr() != graph_outputs[-1].data_ptr()
        assert torch.equal(got, graph_outputs[-1])
        returned.append(got)

    slots = {t.data_ptr() for t in returned}
    assert len(slots) == 2
    assert returned[0].data_ptr() == returned[2].data_ptr()
    assert returned[1].data_ptr() == returned[3].data_ptr()
