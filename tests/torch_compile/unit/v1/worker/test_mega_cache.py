# Copyright 2026 Rebellions Inc. All rights reserved.

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

import dataclasses
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


class TestStableComputeHash:
    """Per-launch open ports in ParallelConfig must not reach the signature."""

    @dataclasses.dataclass
    class _PC:
        _coord_store_port: int = 0
        tensor_parallel_size: int = 1

        def compute_hash(self):
            import hashlib

            raw = f"{self._coord_store_port}|{self.tensor_parallel_size}"
            return hashlib.sha1(raw.encode()).hexdigest()

    class _Cfg:
        def __init__(self, pc):
            self.parallel_config = pc

        def compute_hash(self):
            return self.parallel_config.compute_hash()

    def test_volatile_ports_ignored(self):
        h1 = mega_cache._stable_compute_hash(
            self._Cfg(self._PC(_coord_store_port=38735))
        )
        h2 = mega_cache._stable_compute_hash(
            self._Cfg(self._PC(_coord_store_port=41200))
        )
        assert h1 == h2

    def test_non_port_field_still_matters(self):
        h1 = mega_cache._stable_compute_hash(
            self._Cfg(self._PC(tensor_parallel_size=1))
        )
        h2 = mega_cache._stable_compute_hash(
            self._Cfg(self._PC(tensor_parallel_size=4))
        )
        assert h1 != h2

    def test_fields_restored(self):
        pc = self._PC(_coord_store_port=38735)
        mega_cache._stable_compute_hash(self._Cfg(pc))
        assert pc._coord_store_port == 38735


class TestConfigSignature:
    """Composition of the three parts (env factors mocked out)."""

    def _cfg(self, h="cfghash"):
        return SimpleNamespace(compute_hash=lambda: h)

    def test_deterministic(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "envhash")
        s1 = mega_cache.config_signature(self._cfg())
        s2 = mega_cache.config_signature(self._cfg())
        assert s1 == s2

    def test_vllm_config_change_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "envhash")
        s1 = mega_cache.config_signature(self._cfg("h1"))
        s2 = mega_cache.config_signature(self._cfg("h2"))
        assert s1 != s2

    def test_env_factors_change_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "envA")
        s1 = mega_cache.config_signature(self._cfg())
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "envB")
        s2 = mega_cache.config_signature(self._cfg())
        assert s1 != s2

    def test_rebel_patch_stable_minor_invalidates(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "envhash")
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        same_minor = mega_cache.config_signature(self._cfg())
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.12")
        next_minor = mega_cache.config_signature(self._cfg())
        assert same_minor != next_minor


class TestCompileEnvFactors:
    """_compile_env_factors keys on an rbln allowlist, not vLLM's ~240 factors."""

    def test_deterministic(self):
        assert mega_cache._compile_env_factors() == mega_cache._compile_env_factors()

    def test_allowlist_names_all_resolve(self):
        # An allowlist entry that no longer names a real var reads as None and
        # silently stops keying the bundle, so pin the names here.
        import vllm_rbln.envs as rbln_envs

        unresolved = [
            name
            for name in sorted(mega_cache._RBLN_COMPILE_ENV)
            if getattr(rbln_envs, name, None) is None
        ]
        assert not unresolved

    def test_rank_invariant(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        r0 = mega_cache._compile_env_factors()
        monkeypatch.setenv("LOCAL_RANK", "3")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        r3 = mega_cache._compile_env_factors()
        assert r0 == r3

    def test_dp_rank_invariant(self, monkeypatch):
        # DP replicas compile identical graphs → one bundle signature.
        monkeypatch.setenv("VLLM_DP_RANK", "0")
        monkeypatch.setenv("VLLM_DP_RANK_LOCAL", "0")
        d0 = mega_cache._compile_env_factors()
        monkeypatch.setenv("VLLM_DP_RANK", "7")
        monkeypatch.setenv("VLLM_DP_RANK_LOCAL", "7")
        d7 = mega_cache._compile_env_factors()
        assert d0 == d7

    def test_compile_relevant_env_invalidates(self, monkeypatch):
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "exponential")
        a = mega_cache._compile_env_factors()
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "linear")
        b = mega_cache._compile_env_factors()
        assert a != b

    def test_strict_mode_invalidates(self, monkeypatch):
        # A compile option rebel's per-blob key cannot see, so the bundle
        # signature is the only thing separating the two artifacts.
        monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "0")
        off = mega_cache._compile_env_factors()
        monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
        on = mega_cache._compile_env_factors()
        assert off != on

    def test_sampler_flag_invariant(self, monkeypatch):
        # Sampler graphs are compiled with use_cache=False, so they never enter
        # the bundle — toggling the flag must not discard a valid model bundle.
        monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
        on = mega_cache._compile_env_factors()
        monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")
        off = mega_cache._compile_env_factors()
        assert on == off

    def test_host_invariant(self, monkeypatch):
        # A bundle compiled on one host/user must hit on the serving host.
        monkeypatch.setenv("HOME", "/home/someone")
        monkeypatch.setenv("VLLM_CONFIG_ROOT", "/home/someone/.config/vllm")
        monkeypatch.setenv("VLLM_XLA_CACHE_PATH", "/home/someone/.cache/xla")
        a = mega_cache._compile_env_factors()
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("VLLM_CONFIG_ROOT", "/root/.config/vllm")
        monkeypatch.setenv("VLLM_XLA_CACHE_PATH", "/root/.cache/xla")
        assert mega_cache._compile_env_factors() == a

    @pytest.mark.parametrize(
        "name,value",
        [
            ("VLLM_NIXL_SIDE_CHANNEL_PORT", "5999"),
            ("VLLM_MOONCAKE_BOOTSTRAP_PORT", "9999"),
            ("VLLM_SYSTEM_START_DATE", "2026-08-05"),
            ("VLLM_API_KEY", "sk-secret"),
            ("VLLM_RPC_TIMEOUT", "20000"),
            ("VLLM_RAY_BUNDLE_INDICES", "0,1"),
            ("VLLM_USAGE_SOURCE", "staging"),
            ("VLLM_TARGET_DEVICE", "rocm"),
            ("VLLM_ROCM_USE_AITER", "1"),
            ("VLLM_USE_DEEP_GEMM", "0"),
            ("VLLM_TPU_MOST_MODEL_LEN", "4096"),
            ("VLLM_XPU_ENABLE_XPU_GRAPH", "1"),
            ("VLLM_GC_DEBUG", "1"),
            ("VLLM_TRACE_FUNCTION", "1"),
        ],
    )
    def test_unrelated_env_invariant(self, monkeypatch, name, value):
        # Deployment details and other backends' flags cannot reach an rbln
        # graph; keying on vLLM's compile_factors() let each discard the bundle.
        before = mega_cache._compile_env_factors()
        monkeypatch.setenv(name, value)
        assert mega_cache._compile_env_factors() == before

    def test_rbln_runtime_env_invariant(self, monkeypatch):
        for name in (
            "VLLM_RBLN_ENABLE_WARM_UP",
            "VLLM_RBLN_METRICS",
            "VLLM_RBLN_NUMA",
            "VLLM_RBLN_AUTO_PORT",
            "VLLM_RBLN_SUB_BLOCK_CACHE",
            "VLLM_RBLN_SORT_BATCH",
            "VLLM_RBLN_DISABLE_OFFLOAD",
            "VLLM_RBLN_NIXL_SWA_VIEW_OPT",
            # must stay out so a CPU-compiled bundle hits on the NPU host
            "VLLM_RBLN_COMPILE_ONLY",
        ):
            before = mega_cache._compile_env_factors()
            monkeypatch.setenv(name, "1")
            assert mega_cache._compile_env_factors() == before, name
            monkeypatch.delenv(name)


class TestConfigSignatureRealEnv:
    """End-to-end config_signature over the real compile env."""

    def _cfg(self):
        return SimpleNamespace(compute_hash=lambda: "fixed")

    def test_rank_invariant(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("LOCAL_RANK", "0")
        s0 = mega_cache.config_signature(self._cfg())
        monkeypatch.setenv("LOCAL_RANK", "7")
        s7 = mega_cache.config_signature(self._cfg())
        assert s0 == s7

    def test_decode_bucket_strategy_invalidates(self, monkeypatch):
        import vllm_rbln.envs  # noqa: F401

        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "exponential")
        exp = mega_cache.config_signature(self._cfg())
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "linear")
        lin = mega_cache.config_signature(self._cfg())
        assert exp != lin

    def test_same_strategy_same_signature(self, monkeypatch):
        import vllm_rbln.envs  # noqa: F401

        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "linear")
        a = mega_cache.config_signature(self._cfg())
        b = mega_cache.config_signature(self._cfg())
        assert a == b

    def test_manual_buckets_invalidates(self, monkeypatch):
        import vllm_rbln.envs  # noqa: F401

        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "manual")
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "1,2,4")
        s1 = mega_cache.config_signature(self._cfg())
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "1,2,4,8")
        s2 = mega_cache.config_signature(self._cfg())
        assert s1 != s2
