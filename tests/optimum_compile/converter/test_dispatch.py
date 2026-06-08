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


from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from vllm_rbln.utils.optimum.converter import dispatch
from vllm_rbln.utils.optimum.converter.dispatch import (
    _generate_model_path_name,
    _toolchain_versions,
)


def _make_vllm_config(
    model="meta-llama/Llama-3-8B",
    max_num_seqs=4,
    block_size=128,
    max_model_len=8192,
    rbln_config=None,
):
    """Build a minimal stand-in for ``VllmConfig``.

    Only the attributes touched by ``_generate_model_path_name`` are
    populated.
    """
    additional_config = {}
    if rbln_config is not None:
        additional_config["rbln_config"] = rbln_config
    return SimpleNamespace(
        model_config=SimpleNamespace(model=model, max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(block_size=block_size),
        additional_config=additional_config,
    )


class TestToolchainVersions:
    def test_resolves_both_packages(self, monkeypatch):
        monkeypatch.setattr(
            dispatch,
            "version",
            lambda pkg: {
                "rebel-compiler": "1.2.3",
                "optimum-rbln": "4.5.6",
            }[pkg],
        )
        assert _toolchain_versions() == {
            "rebel-compiler": "1.2.3",
            "optimum-rbln": "4.5.6",
        }

    def test_missing_package_falls_back_to_unknown(self, monkeypatch):
        def fake_version(pkg):
            if pkg == "optimum-rbln":
                raise PackageNotFoundError(pkg)
            return "1.2.3"

        monkeypatch.setattr(dispatch, "version", fake_version)
        versions = _toolchain_versions()
        assert versions["rebel-compiler"] == "1.2.3"
        assert versions["optimum-rbln"] == "unknown"

    def test_keys_cover_all_toolchain_packages(self, monkeypatch):
        monkeypatch.setattr(dispatch, "version", lambda pkg: "0.0.0")
        versions = _toolchain_versions()
        assert set(versions) == set(dispatch._TOOLCHAIN_PACKAGES)


class TestGenerateModelPathName:
    def test_deterministic_for_same_inputs(self, monkeypatch):
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)
        monkeypatch.setattr(
            dispatch, "_toolchain_versions", lambda: {"rebel-compiler": "1.0"}
        )
        first = _generate_model_path_name(_make_vllm_config())
        second = _generate_model_path_name(_make_vllm_config())
        assert first == second

    def test_toolchain_version_change_changes_hash(self, monkeypatch):
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)

        monkeypatch.setattr(
            dispatch, "_toolchain_versions", lambda: {"rebel-compiler": "1.0"}
        )
        old = _generate_model_path_name(_make_vllm_config())

        monkeypatch.setattr(
            dispatch, "_toolchain_versions", lambda: {"rebel-compiler": "2.0"}
        )
        new = _generate_model_path_name(_make_vllm_config())

        assert old != new

    def test_user_param_change_changes_hash(self, monkeypatch):
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)
        monkeypatch.setattr(dispatch, "_toolchain_versions", lambda: {})
        base = _generate_model_path_name(_make_vllm_config(max_num_seqs=4))
        bumped = _generate_model_path_name(_make_vllm_config(max_num_seqs=8))
        assert base != bumped

    def test_runtime_only_keys_do_not_affect_hash(self, monkeypatch):
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)
        monkeypatch.setattr(dispatch, "_toolchain_versions", lambda: {})
        without = _generate_model_path_name(
            _make_vllm_config(rbln_config={"tensor_parallel_size": 4})
        )
        # Adding only runtime-only keys must not change the compiled-artifact
        # cache key.
        with_runtime = _generate_model_path_name(
            _make_vllm_config(
                rbln_config={
                    "tensor_parallel_size": 4,
                    "create_runtimes": True,
                    "device": [0, 1],
                    "kvcache_num_blocks": 1024,
                }
            )
        )
        assert without == with_runtime
