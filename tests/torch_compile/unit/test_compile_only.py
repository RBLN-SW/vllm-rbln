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

"""Tests for VLLM_RBLN_COMPILE_ONLY (compile + cache on an NPU-less host).

The flag lets a CPU-only build worker compile every graph and write the
``.rbln`` artifact to the cache (the rbln backend builds the runtime on a
dummy device), so a real NPU host can later reuse it via cache-hit. These
unit tests cover the vllm-rbln side: the env var, the get_device_name guard
for ``rebel.get_npu_name() == None``, the cache-disabled conflict check, and
the ``compile_only`` option injection at the compile sites.
"""

from types import SimpleNamespace

import pytest
import rebel

from vllm_rbln import envs
from vllm_rbln.platform import RblnPlatform


# --------------------------------------------------------------------------
# Env var
# --------------------------------------------------------------------------
def test_compile_only_defaults_false(monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_COMPILE_ONLY", raising=False)
    assert envs.VLLM_RBLN_COMPILE_ONLY is False


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("True", True), ("0", False), ("false", False)],
)
def test_compile_only_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_RBLN_COMPILE_ONLY", value)
    assert envs.VLLM_RBLN_COMPILE_ONLY is expected


# --------------------------------------------------------------------------
# get_device_name: handle rebel.get_npu_name() == None (CPU-only host)
# --------------------------------------------------------------------------
def test_get_device_name_returns_npu_name(monkeypatch):
    monkeypatch.setattr(rebel, "get_npu_name", lambda device_id=0: "RBLN-CA25")
    assert RblnPlatform.get_device_name() == "RBLN-CA25"


def test_get_device_name_raises_when_npu_none(monkeypatch):
    # On a host without an NPU mounted rebel.get_npu_name() returns None; we
    # should surface an actionable error pointing at RBLN_TARGET_SOC rather
    # than a bare AssertionError.
    monkeypatch.setattr(rebel, "get_npu_name", lambda device_id=0: None)
    with pytest.raises(RuntimeError, match="RBLN_TARGET_SOC"):
        RblnPlatform.get_device_name()


# --------------------------------------------------------------------------
# validate_and_setup_prerequisite: compile-only needs the cache enabled
# --------------------------------------------------------------------------
def _stub_config(dp=1, tp=1, pp=1, ep=False, chunked_prefill=True):
    # Single-rank config so use_model_parallel is False and the method does
    # not touch the distributed env (RBLN_CTX_STANDALONE, etc.).
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked_prefill),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
        ),
    )


def test_validate_rejects_compile_only_with_cache_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_COMPILE_ONLY", "1")
    monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    with pytest.raises(ValueError, match="compile cache"):
        RblnPlatform.validate_and_setup_prerequisite(_stub_config())


def test_validate_rejects_compile_only_without_vllm_model(monkeypatch):
    # compile-only only applies to the vLLM-native model path; on the
    # optimum-rbln path (VLLM_RBLN_USE_VLLM_MODEL unset) it must error rather
    # than silently do nothing.
    monkeypatch.setenv("VLLM_RBLN_COMPILE_ONLY", "1")
    monkeypatch.delenv("VLLM_RBLN_USE_VLLM_MODEL", raising=False)
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    with pytest.raises(ValueError, match="VLLM_RBLN_USE_VLLM_MODEL"):
        RblnPlatform.validate_and_setup_prerequisite(_stub_config())


def test_validate_allows_compile_only_with_cache_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_COMPILE_ONLY", "1")
    monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    # Must not raise.
    RblnPlatform.validate_and_setup_prerequisite(_stub_config())


def test_validate_noop_when_compile_only_unset(monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_COMPILE_ONLY", raising=False)
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    # Disabling the cache alone is fine when not compiling-only.
    RblnPlatform.validate_and_setup_prerequisite(_stub_config())


# --------------------------------------------------------------------------
# compile_only is injected into the torch.compile options at the spec-decode
# compile sites (eagle / medusa share the identical inline block as the main
# model runner's _compile_model).
# --------------------------------------------------------------------------
def _capture_compile_options(monkeypatch, **compile_kwargs):
    from vllm_rbln import compilation

    captured = {}

    def fake_compile(target, **kwargs):
        captured["options"] = kwargs.get("options")
        return target

    monkeypatch.setattr("torch.compile", fake_compile)
    compilation.compile(lambda x: x, mode="strict", **compile_kwargs)
    return captured["options"]


def test_compile_only_injected_into_options(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_COMPILE_ONLY", "1")
    monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)

    options = _capture_compile_options(monkeypatch)
    assert options["mode"] == ["strict", "compile_only"]
    # compile-only requires a cache_dir to write the artifact to.
    assert options.get("cache_dir")


def test_compile_only_absent_when_flag_unset(monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_COMPILE_ONLY", raising=False)
    monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)

    options = _capture_compile_options(monkeypatch)
    assert options["mode"] == ["strict"]
