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

# compile()'s option-building contract (kwargs + env -> torch.compile options)
# and the RBLN dynamo settings. The real compile needs an NPU and lives in the
# model-compile tests.

import pytest
import torch

import vllm_rbln.compilation.compiler as compiler
from vllm_rbln.compilation import compile
from vllm_rbln.compilation.dispatch import Dispatcher


@pytest.fixture(autouse=True)
def _dynamo_isolation():
    # compile() and _ensure_torch_dynamo_configured mutate the global dynamo
    # config; snapshot and restore so tests don't leak into each other.
    cfg = torch._dynamo.config
    saved = (cfg.cache_size_limit, compiler._DYNAMO_CONFIGURED)
    yield
    cfg.cache_size_limit, compiler._DYNAMO_CONFIGURED = saved


@pytest.fixture
def captured_compile(monkeypatch):
    # Stub torch.compile to capture the args compile() passes. (torch.compile is
    # lazy anyway; stubbing keeps the test free of any tracing.)
    calls: dict[str, object] = {}

    def fake_compile(target, *, backend, dynamic, fullgraph, options):
        calls.update(
            target=target,
            backend=backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
            options=options,
        )
        return "COMPILED"

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


class TestCompileOptions:
    def test_returns_torch_compile_result(self, captured_compile):
        # compile() returns whatever torch.compile returns.
        assert compile(object()) == "COMPILED"

    def test_skips_unset_values(self, captured_compile, monkeypatch):
        # Unset options never reach torch.compile.
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", True)
        monkeypatch.setattr(compiler.envs, "VLLM_RBLN_COMPILE_ONLY", False)
        compile(object())
        assert captured_compile["options"] == {}

    def test_num_devices_is_the_devices_option(self, captured_compile):
        compile(object(), num_devices=4)
        assert captured_compile["options"]["devices"] == 4

    def test_guard_filter_fn_is_forwarded(self, captured_compile):
        def keep_all(guards):
            return [True] * len(guards)

        compile(object(), guard_filter_fn=keep_all)
        assert captured_compile["options"]["guard_filter_fn"] is keep_all

    def test_compile_only_is_the_mode_when_env_set(self, captured_compile, monkeypatch):
        monkeypatch.setattr(compiler.envs, "VLLM_RBLN_COMPILE_ONLY", True)
        compile(object())
        assert captured_compile["options"]["mode"] == "compile_only"

    def test_cache_dir_defaults_under_cache_root(self, captured_compile, monkeypatch):
        # Default cache_dir is VLLM_CACHE_ROOT/rbln.
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", False)
        monkeypatch.setattr(compiler.envs, "VLLM_CACHE_ROOT", "/tmp/cacheroot")
        compile(object())
        assert captured_compile["options"]["cache_dir"] == "/tmp/cacheroot/rbln"

    def test_cache_dir_explicit_honored(self, captured_compile, monkeypatch):
        # An explicit cache_dir overrides the default.
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", False)
        compile(object(), cache_dir="/my/dir")
        assert captured_compile["options"]["cache_dir"] == "/my/dir"

    def test_cache_dir_skipped_when_disabled(self, captured_compile, monkeypatch):
        # VLLM_DISABLE_COMPILE_CACHE drops cache_dir entirely.
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", True)
        compile(object(), cache_dir="/my/dir")
        assert "cache_dir" not in captured_compile["options"]

    def test_cache_dir_skipped_when_use_cache_false(
        self, captured_compile, monkeypatch
    ):
        # use_cache=False drops cache_dir even when caching is enabled.
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", False)
        compile(object(), use_cache=False)
        assert "cache_dir" not in captured_compile["options"]

    def test_direct_dispatch_wraps_the_compiled_callable(self, captured_compile):
        def target(x):
            return x

        assert isinstance(
            compile(target, fullgraph=True, use_direct_dispatch=True), Dispatcher
        )

    def test_direct_dispatch_requires_fullgraph(self, captured_compile):
        # Whatever Dynamo leaves outside the one graph is unreachable from a clone.
        with pytest.raises(ValueError, match="fullgraph"):
            compile(object(), use_direct_dispatch=True)

    def test_forwards_backend_dynamic_fullgraph(self, captured_compile):
        # backend / dynamic / fullgraph pass straight through to torch.compile.
        backend = object()
        compile(object(), backend=backend, dynamic=True, fullgraph=True)
        assert captured_compile["backend"] is backend
        assert captured_compile["dynamic"] is True
        assert captured_compile["fullgraph"] is True


class TestEnsureTorchDynamoConfigured:
    def test_sets_rbln_flags(self):
        # Applies the RBLN dynamo settings (a larger cache size limit).
        compiler._DYNAMO_CONFIGURED = False
        torch._dynamo.config.cache_size_limit = 8
        compiler._ensure_torch_dynamo_configured()
        assert torch._dynamo.config.cache_size_limit == 64

    def test_idempotent(self):
        # After the first call the guard makes further calls no-ops.
        compiler._DYNAMO_CONFIGURED = False
        compiler._ensure_torch_dynamo_configured()
        assert compiler._DYNAMO_CONFIGURED is True
        torch._dynamo.config.cache_size_limit = 8
        compiler._ensure_torch_dynamo_configured()
        assert torch._dynamo.config.cache_size_limit == 8
