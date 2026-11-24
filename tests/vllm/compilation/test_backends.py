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
from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.compilation.backends as backends
from vllm_rbln.compilation import rbln_backend
from vllm_rbln.v1.attention.kv_cache_bindings import kv_cache_tensor_key


class TestBackendConformance:
    def test_compile_records_input_keys_without_retaining_tensors(self, monkeypatch):
        cache = torch.zeros(4, 8)
        cache_ref = weakref.ref(cache)
        torch._dynamo.mark_dynamic(cache, 0)
        runtimes = []

        def compile_graph(graph, inputs, **kwargs):
            runtime = graph.forward
            runtimes.append(runtime)
            return runtime

        monkeypatch.setattr(backends, "rbln_torch_compile_backend", compile_graph)
        bindings: dict[int, tuple[tuple, ...]] = {}
        token = backends.program_input_bindings.set(bindings)
        try:

            def forward(x, state):
                return x + state.sum(0)

            compiled = torch.compile(forward, backend=rbln_backend, fullgraph=True)
            x = torch.ones(8)
            assert torch.equal(compiled(x, cache), x)
        finally:
            backends.program_input_bindings.reset(token)
        assert len(runtimes) == 1
        keys = bindings[id(runtimes[0])]
        assert len(keys) == 2
        assert set(keys) == {kv_cache_tensor_key(cache), kv_cache_tensor_key(x)}
        del cache
        assert cache_ref() is None

    @pytest.mark.parametrize("capture", [False, True])
    def test_backend_without_program_preserves_callable(self, monkeypatch, capture):
        graph = torch.fx.GraphModule({}, torch.fx.Graph())
        runtime = graph.forward

        def compile_graph(graph, inputs, **kwargs):
            return runtime

        monkeypatch.setattr(backends, "rbln_torch_compile_backend", compile_graph)
        if not capture:
            monkeypatch.setattr(torch, "rbln", SimpleNamespace())
        bindings: dict[int, tuple[tuple, ...]] = {}
        token = backends.program_input_bindings.set(bindings if capture else None)
        try:
            assert rbln_backend(graph, []) is runtime
        finally:
            backends.program_input_bindings.reset(token)
        assert bindings == ({id(runtime): ()} if capture else {})

    def test_rebel_backend_importable_and_callable(self):
        # backends.py imports this at module load; the drift alarm if rebel
        # moves or renames its torch.compile backend.
        from rebel.core.torch_compile import rbln_backend as rebel_backend

        assert callable(rebel_backend)

    def test_rbln_backend_exposed_and_callable(self):
        # The package re-exports backends.rbln_backend as the Dynamo backend.
        assert callable(rbln_backend)
        assert backends.rbln_backend is rbln_backend
