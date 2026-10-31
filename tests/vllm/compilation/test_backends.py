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

"""Tests for vllm_rbln.compilation.backends: the conformance surface (it wraps
rebel's torch.compile backend, so a drift in that dependency must surface here).
The log/format helpers and the actual NPU compile are not unit-tested."""

import weakref

import torch

import vllm_rbln.compilation.backends as backends
from vllm_rbln.compilation import rbln_backend


class TestBackendConformance:
    def test_compile_records_real_inputs_without_wrapping_runtime(self, monkeypatch):
        cache = torch.zeros(4, 8)
        torch._dynamo.mark_dynamic(cache, 0)
        runtimes = []

        def compile_graph(graph, inputs, **kwargs):
            runtime = graph.forward
            runtimes.append(runtime)
            return runtime

        monkeypatch.setattr(backends, "rbln_torch_compile_backend", compile_graph)
        bindings: dict[int, tuple[weakref.ReferenceType[torch.Tensor], ...]] = {}
        token = backends.program_input_bindings.set(bindings)
        try:

            def forward(x, state):
                return x + state.sum(0)

            compiled = torch.compile(forward, backend=rbln_backend, fullgraph=True)
            assert torch.equal(compiled(torch.ones(8), cache), torch.ones(8))
        finally:
            backends.program_input_bindings.reset(token)
        assert len(runtimes) == 1
        refs = bindings[id(runtimes[0])]
        assert len(refs) == 2
        assert any(ref() is cache for ref in refs)
        del cache
        assert all(ref() is None for ref in refs)

    def test_rebel_backend_importable_and_callable(self):
        # backends.py imports this at module load; the drift alarm if rebel
        # moves or renames its torch.compile backend.
        from rebel.core.torch_compile import rbln_backend as rebel_backend

        assert callable(rebel_backend)

    def test_rbln_backend_exposed_and_callable(self):
        # The package re-exports backends.rbln_backend as the Dynamo backend.
        assert callable(rbln_backend)
        assert backends.rbln_backend is rbln_backend
