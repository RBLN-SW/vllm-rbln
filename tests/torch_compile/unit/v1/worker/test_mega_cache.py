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
"""Unit tests for mega-cache bundle keying (config signature + path)."""

from types import SimpleNamespace

import pytest

from vllm_rbln.v1.worker import mega_cache


class TestRebelMajorMinor:
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("0.11.0", "0.11"),
            ("0.11.1", "0.11"),
            ("0.11.9", "0.11"),
            ("0.12.0", "0.12"),
            ("0.10.5.dev224+gb3c3db0d", "0.10"),
            ("1.0.0", "1.0"),
            ("", "unknown"),
            ("garbage", "unknown"),
        ],
    )
    def test_parse(self, version, expected):
        assert mega_cache._rebel_major_minor(version) == expected

    def test_patch_bump_shares_minor(self):
        assert mega_cache._rebel_major_minor("0.11.0") == mega_cache._rebel_major_minor(
            "0.11.7"
        )

    def test_minor_bump_differs(self):
        assert mega_cache._rebel_major_minor("0.11.0") != mega_cache._rebel_major_minor(
            "0.12.0"
        )


class TestBundlePath:
    def test_includes_signature_and_rank(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "3")
        path = mega_cache.bundle_path("meta/Llama-3", "abc123def456")
        assert "abc123def456" in path
        assert "rank3" in path
        assert path.endswith("mega_cache.bin")

    def test_signature_separates_files(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        a = mega_cache.bundle_path("m", "sig_aaaa")
        b = mega_cache.bundle_path("m", "sig_bbbb")
        assert a != b


class TestConfigSignature:
    def _cfg(self, h="cfghash"):
        return SimpleNamespace(compute_hash=lambda: h)

    def test_deterministic(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        import vllm_rbln.rbln_envs as rbln_envs

        monkeypatch.setattr(
            rbln_envs, "environment_variables", {"VLLM_RBLN_X": lambda: "1"}
        )
        s1 = mega_cache.config_signature(self._cfg())
        s2 = mega_cache.config_signature(self._cfg())
        assert s1 == s2

    def test_rbln_env_change_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        import vllm_rbln.rbln_envs as rbln_envs

        monkeypatch.setattr(
            rbln_envs, "environment_variables", {"VLLM_RBLN_X": lambda: "1"}
        )
        s1 = mega_cache.config_signature(self._cfg())
        monkeypatch.setattr(
            rbln_envs, "environment_variables", {"VLLM_RBLN_X": lambda: "2"}
        )
        s2 = mega_cache.config_signature(self._cfg())
        assert s1 != s2

    def test_vllm_config_change_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        import vllm_rbln.rbln_envs as rbln_envs

        monkeypatch.setattr(rbln_envs, "environment_variables", {})
        s1 = mega_cache.config_signature(self._cfg("h1"))
        s2 = mega_cache.config_signature(self._cfg("h2"))
        assert s1 != s2

    def test_rebel_patch_stable_minor_invalidates(self, monkeypatch):
        import vllm_rbln.rbln_envs as rbln_envs

        monkeypatch.setattr(rbln_envs, "environment_variables", {})
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        same_minor = mega_cache.config_signature(self._cfg())
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.12")
        next_minor = mega_cache.config_signature(self._cfg())
        assert same_minor != next_minor

    def test_env_resolution_error_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        import vllm_rbln.rbln_envs as rbln_envs

        def _boom():
            raise RuntimeError("no hardware")

        monkeypatch.setattr(rbln_envs, "environment_variables", {"VLLM_RBLN_X": _boom})
        # must not propagate; the failing var is folded in as "<err>"
        assert isinstance(mega_cache.config_signature(self._cfg()), str)


class TestConfigSignatureRealRegistry:
    """Exercise the real 269-entry env registry (no monkeypatch of the dict)."""

    def _cfg(self):
        return SimpleNamespace(compute_hash=lambda: "fixed")

    def test_decode_bucket_strategy_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "exponential")
        exp = mega_cache.config_signature(self._cfg())
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "linear")
        lin = mega_cache.config_signature(self._cfg())
        assert exp != lin

    def test_same_strategy_same_signature(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "linear")
        a = mega_cache.config_signature(self._cfg())
        b = mega_cache.config_signature(self._cfg())
        assert a == b

    def test_manual_buckets_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "manual")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "1,2,4")
        s1 = mega_cache.config_signature(self._cfg())
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "1,2,4,8")
        s2 = mega_cache.config_signature(self._cfg())
        assert s1 != s2
