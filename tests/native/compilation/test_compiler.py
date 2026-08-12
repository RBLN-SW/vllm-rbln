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

# compile()'s option-building contract (kwargs + env -> torch.compile options),
# the RBLN dynamo settings, and conformance against rebel. The real compile needs
# an NPU and lives in the model-compile tests.

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.compilation.compiler as compiler
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
    create_compile_context,
)


@pytest.fixture(autouse=True)
def _dynamo_isolation():
    # compile() and _ensure_torch_dynamo_configured mutate the global dynamo
    # config; snapshot and restore so tests don't leak into each other.
    cfg = torch._dynamo.config
    saved = (
        cfg.inline_inbuilt_nn_modules,
        cfg.cache_size_limit,
        compiler._DYNAMO_CONFIGURED,
    )
    yield
    cfg.inline_inbuilt_nn_modules, cfg.cache_size_limit = saved[0], saved[1]
    compiler._DYNAMO_CONFIGURED = saved[2]


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

    def test_skips_none_and_empty_values(self, captured_compile):
        # Unset (None / "") options never reach torch.compile.
        compile(object())
        opts = captured_compile["options"]
        for key in (
            "num_devices",
            "model_trace_method",
            "process_group_dict",
            "guard_filter_fn",
            "_runtime_holder",
            "compile_context",
            "global_device_id",
            "use_global_ctx",
        ):
            assert key not in opts

    def test_sets_provided_values(self, captured_compile):
        # Provided values are forwarded under their option keys.
        compile(object(), num_devices=4, model_trace_method="trace")
        opts = captured_compile["options"]
        assert opts["num_devices"] == 4
        assert opts["model_trace_method"] == "trace"

    def test_forwards_static_output_option(self, captured_compile):
        compile(object(), use_static_output=True)
        assert captured_compile["options"]["use_static_output"] is True

    def test_runtime_holder_uses_underscore_key(self, captured_compile):
        # runtime_holder maps to the "_runtime_holder" option key.
        holder: list = []
        compile(object(), runtime_holder=holder)
        assert captured_compile["options"]["_runtime_holder"] is holder
        assert "runtime_holder" not in captured_compile["options"]

    def test_mode_str_becomes_list(self, captured_compile):
        # A str mode is wrapped into a single-element list.
        compile(object(), mode="foo")
        assert captured_compile["options"]["mode"] == ["foo"]

    def test_mode_empty_skipped(self, captured_compile):
        # An empty mode is not set.
        compile(object(), mode="")
        assert "mode" not in captured_compile["options"]

    def test_compile_only_appended_when_env_set(self, captured_compile, monkeypatch):
        # VLLM_RBLN_COMPILE_ONLY appends "compile_only" to a str mode.
        monkeypatch.setattr(compiler.envs, "VLLM_RBLN_COMPILE_ONLY", True)
        compile(object(), mode="foo")
        assert captured_compile["options"]["mode"] == ["foo", "compile_only"]

    # TODO(RBLN): compile_only is appended only when mode is a str, so a
    # list-valued mode silently drops it. Fix the source, then drop this xfail.
    @pytest.mark.xfail(
        strict=True,
        reason="compile_only not appended when mode is passed as a list",
    )
    def test_compile_only_appended_for_list_mode(self, captured_compile, monkeypatch):
        monkeypatch.setattr(compiler.envs, "VLLM_RBLN_COMPILE_ONLY", True)
        compile(object(), mode=["foo"])
        assert captured_compile["options"]["mode"] == ["foo", "compile_only"]

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

    def test_caching_goes_through_the_mega_cache_only(
        self, captured_compile, monkeypatch
    ):
        # The bundle is the only compile cache: with this set the backend neither
        # reads nor writes per-graph <cache_dir>/<hash>.rbln files.
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", False)
        compile(object())
        assert captured_compile["options"]["mega_cache_only"] is True

    @pytest.mark.parametrize(
        ("disabled", "use_cache"), [(True, True), (False, False)], ids=["env", "kwarg"]
    )
    def test_mega_cache_only_follows_cache_dir(
        self, captured_compile, monkeypatch, disabled, use_cache
    ):
        monkeypatch.setattr(compiler.envs, "VLLM_DISABLE_COMPILE_CACHE", disabled)
        compile(object(), use_cache=use_cache)
        assert "mega_cache_only" not in captured_compile["options"]

    def test_forwards_backend_dynamic_fullgraph(self, captured_compile):
        # backend / dynamic / fullgraph pass straight through to torch.compile.
        backend = object()
        compile(object(), backend=backend, dynamic=True, fullgraph=True)
        assert captured_compile["backend"] is backend
        assert captured_compile["dynamic"] is True
        assert captured_compile["fullgraph"] is True


class TestEnsureTorchDynamoConfigured:
    def test_sets_rbln_flags(self):
        # Applies the RBLN dynamo settings (nn.Module params must not become
        # graph inputs; larger cache size limit).
        compiler._DYNAMO_CONFIGURED = False
        torch._dynamo.config.inline_inbuilt_nn_modules = True
        torch._dynamo.config.cache_size_limit = 8
        compiler._ensure_torch_dynamo_configured()
        assert torch._dynamo.config.inline_inbuilt_nn_modules is False
        assert torch._dynamo.config.cache_size_limit == 64

    def test_idempotent(self):
        # After the first call the guard makes further calls no-ops.
        compiler._DYNAMO_CONFIGURED = False
        compiler._ensure_torch_dynamo_configured()
        assert compiler._DYNAMO_CONFIGURED is True
        torch._dynamo.config.inline_inbuilt_nn_modules = True
        compiler._ensure_torch_dynamo_configured()
        assert torch._dynamo.config.inline_inbuilt_nn_modules is True


class TestBuildProcessGroupDict:
    def test_maps_device_and_cpu_names_for_all_groups(self, monkeypatch):
        # Both the device- and cpu-group names of tp/pp/dp, each mapped to that
        # group's ranks; the RBLN backend wires collectives from this.
        def grp(dev_name, cpu_name, ranks):
            return SimpleNamespace(
                device_group=SimpleNamespace(group_name=dev_name),
                cpu_group=SimpleNamespace(group_name=cpu_name),
                ranks=ranks,
            )

        monkeypatch.setattr(
            compiler, "get_tp_group", lambda: grp("tp_d", "tp_c", [0, 1])
        )
        monkeypatch.setattr(compiler, "get_pp_group", lambda: grp("pp_d", "pp_c", [0]))
        monkeypatch.setattr(compiler, "get_dp_group", lambda: grp("dp_d", "dp_c", [2]))
        assert build_process_group_dict() == {
            "tp_d": [0, 1],
            "tp_c": [0, 1],
            "pp_d": [0],
            "pp_c": [0],
            "dp_d": [2],
            "dp_c": [2],
        }


class TestCompilerConformance:
    def test_create_compile_context_forwards_args(self, monkeypatch):
        # create_compile_context forwards its two flags to rebel's CompileContext.
        captured = {}

        class FakeCtx:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(compiler, "CompileContext", FakeCtx)
        create_compile_context(use_weight_sharing=True, use_global_ctx=True)
        assert captured == {"use_weight_sharing": True, "use_global_ctx": True}

    def test_rebel_compile_context_signature(self):
        # Drift alarm against the rebel dependency: the params
        # create_compile_context forwards must still exist on CompileContext.
        from rebel import CompileContext

        params = inspect.signature(CompileContext).parameters
        assert "use_weight_sharing" in params
        assert "use_global_ctx" in params
