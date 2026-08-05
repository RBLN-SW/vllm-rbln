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
import os
import re
from types import SimpleNamespace

import pytest

from vllm_rbln.v1.worker import mega_cache


@pytest.fixture(autouse=True)
def _unshadow_rbln_envs():
    # monkeypatch.setattr(envs, NAME, ...) plants NAME statically on undo, which
    # permanently bypasses the __getattr__ that reads os.environ (test_medusa.py).
    import vllm_rbln.envs as rbln_envs

    for name in mega_cache._RBLN_COMPILE_ENV:
        rbln_envs.__dict__.pop(name, None)
    yield


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
        # A renamed entry reads as None and silently stops keying the bundle.
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
        # A compile option rebel's per-blob key cannot see.
        monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "0")
        off = mega_cache._compile_env_factors()
        monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
        on = mega_cache._compile_env_factors()
        assert off != on

    def test_sampler_flag_invariant(self, monkeypatch):
        # Sampler graphs compile with use_cache=False, so they never enter it.
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
        # Cannot reach an rbln graph; compile_factors() let each discard it.
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


class TestBundlePathEdges:
    def test_respects_cache_root(self, monkeypatch):
        import vllm.envs as vllm_envs

        monkeypatch.setattr(vllm_envs, "VLLM_CACHE_ROOT", "/tmp/some-cache-root")
        path = mega_cache.bundle_path("m", "sig")
        assert path.startswith("/tmp/some-cache-root/rbln/")

    def test_cache_root_helper_matches_prefix(self, monkeypatch):
        import vllm.envs as vllm_envs

        monkeypatch.setattr(vllm_envs, "VLLM_CACHE_ROOT", "/tmp/other-root")
        assert mega_cache.cache_root() == "/tmp/other-root/rbln"
        assert mega_cache.bundle_path("m", "s").startswith(mega_cache.cache_root())

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "/abs/path/to/local model",
            "../../escape",
            "weird:name*with?chars",
            "한글모델",
        ],
        ids=["hf-repo", "local-space", "traversal", "shell-chars", "non-ascii"],
    )
    def test_model_component_is_a_single_safe_segment(self, monkeypatch, model):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        path = mega_cache.bundle_path(model, "sig")
        # <root>/rbln/<model-seg>/<sig>/rank0/mega_cache.bin
        model_seg = path.split(os.sep)[-4]
        assert model_seg not in (".", "..")
        assert re.fullmatch(r"[A-Za-z0-9._-]+", model_seg), model_seg
        root = os.path.join(mega_cache.cache_root(), "")
        assert os.path.abspath(path).startswith(os.path.abspath(root))

    def test_same_sanitized_name_still_separates(self, monkeypatch):
        # Both sanitize to the same safe name; the sha1 suffix must keep them apart.
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        a = mega_cache.bundle_path("org/model", "sig")
        b = mega_cache.bundle_path("org:model", "sig")
        assert a != b

    @pytest.mark.parametrize("model", ["", "///", "***"])
    def test_degenerate_model_names(self, monkeypatch, model):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        path = mega_cache.bundle_path(model, "sig")
        assert path.endswith(os.path.join("sig", "rank0", "mega_cache.bin"))

    def test_default_rank_is_zero(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert "rank0" in mega_cache.bundle_path("m", "sig")

    def test_ranks_do_not_share_a_file(self, monkeypatch):
        paths = set()
        for rank in ("0", "1", "7"):
            monkeypatch.setenv("LOCAL_RANK", rank)
            paths.add(mega_cache.bundle_path("m", "sig"))
        assert len(paths) == 3


@pytest.fixture
def bundle_env(tmp_path, monkeypatch):
    """Redirect the bundle under tmp_path and stub the torch/rebel boundaries.

    Returns a namespace whose `saved`/`loaded` record what crossed each boundary,
    so the tests assert on the persisted bundle rather than on real compilation.
    """
    import sys

    import vllm.envs as vllm_envs
    from rebel.core import mega_cache as rbln_mega_cache

    monkeypatch.setattr(vllm_envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    state = SimpleNamespace(
        artifact=b"bundle-bytes",
        save_result=None,  # set below; None means "torch returned nothing"
        loaded=[],
        set_dirs=[],
        flushed=0,
        save_raises=None,
    )
    state.save_result = (state.artifact, object())

    def fake_save():
        if state.save_raises is not None:
            raise state.save_raises
        return state.save_result

    monkeypatch.setattr(rbln_mega_cache, "set_dir", state.set_dirs.append)

    def fake_flush():
        state.flushed += 1

    monkeypatch.setattr(rbln_mega_cache, "flush_to_bundle", fake_flush)
    monkeypatch.setattr(
        sys.modules["torch"].compiler, "save_cache_artifacts", fake_save
    )
    monkeypatch.setattr(
        sys.modules["torch"].compiler,
        "load_cache_artifacts",
        state.loaded.append,
    )
    state.path = mega_cache.bundle_path("m", "sig")
    return state


class TestSaveLoadRoundTrip:
    def test_round_trip(self, bundle_env):
        mega_cache.save("m", "sig")
        assert os.path.isfile(bundle_env.path)
        with open(bundle_env.path, "rb") as f:
            assert f.read() == bundle_env.artifact

        mega_cache.load("m", "sig")
        assert bundle_env.loaded == [bundle_env.artifact]

    def test_save_flushes_staged_blobs_first(self, bundle_env):
        # Disk-staged .rbln blobs only enter the bundle via flush_to_bundle().
        mega_cache.save("m", "sig")
        assert bundle_env.flushed == 1

    def test_both_point_rebel_at_cache_root(self, bundle_env):
        mega_cache.save("m", "sig")
        mega_cache.load("m", "sig")
        assert bundle_env.set_dirs == [mega_cache.cache_root()] * 2

    def test_save_leaves_no_tmp_file(self, bundle_env):
        mega_cache.save("m", "sig")
        leftovers = [
            os.path.join(root, name)
            for root, _, files in os.walk(os.path.dirname(bundle_env.path))
            for name in files
            if name.endswith(".tmp")
        ]
        assert not leftovers

    def test_load_miss_is_silent(self, bundle_env):
        mega_cache.load("m", "sig")  # nothing saved yet
        assert bundle_env.loaded == []

    def test_signature_miss_does_not_read_other_bundle(self, bundle_env):
        mega_cache.save("m", "sig")
        mega_cache.load("m", "other-sig")
        assert bundle_env.loaded == []

    def test_rank_miss_does_not_read_other_rank(self, bundle_env, monkeypatch):
        mega_cache.save("m", "sig")
        monkeypatch.setenv("LOCAL_RANK", "1")
        mega_cache.load("m", "sig")
        assert bundle_env.loaded == []

    def test_save_writes_nothing_when_torch_returns_none(self, bundle_env):
        bundle_env.save_result = None
        mega_cache.save("m", "sig")
        assert not os.path.exists(bundle_env.path)

    def test_save_failure_leaves_no_bundle(self, bundle_env):
        bundle_env.save_raises = RuntimeError("boom")
        mega_cache.save("m", "sig")  # must not propagate
        assert not os.path.exists(bundle_env.path)

    def test_save_failure_keeps_previous_bundle(self, bundle_env):
        mega_cache.save("m", "sig")
        bundle_env.save_raises = RuntimeError("boom")
        mega_cache.save("m", "sig")
        with open(bundle_env.path, "rb") as f:
            assert f.read() == bundle_env.artifact

    def test_corrupt_bundle_does_not_raise(self, bundle_env, monkeypatch):
        import sys

        os.makedirs(os.path.dirname(bundle_env.path), exist_ok=True)
        with open(bundle_env.path, "wb") as f:
            f.write(b"garbage")

        def boom(_):
            raise RuntimeError("bad bundle")

        monkeypatch.setattr(sys.modules["torch"].compiler, "load_cache_artifacts", boom)
        mega_cache.load("m", "sig")  # warns, must not propagate

    def test_resave_replaces_in_place(self, bundle_env):
        mega_cache.save("m", "sig")
        bundle_env.artifact = b"second-bundle"
        bundle_env.save_result = (bundle_env.artifact, object())
        mega_cache.save("m", "sig")
        with open(bundle_env.path, "rb") as f:
            assert f.read() == b"second-bundle"

    def test_disabled_compile_cache_is_a_no_op(self, bundle_env, monkeypatch):
        import vllm.envs as vllm_envs

        monkeypatch.setattr(vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", True)
        mega_cache.save("m", "sig")
        assert not os.path.exists(bundle_env.path)
        mega_cache.load("m", "sig")
        assert bundle_env.loaded == []
        assert bundle_env.set_dirs == []


def _real_vllm_config(**overrides):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    return VllmConfig(
        model_config=ModelConfig(
            model=overrides.get("model", "facebook/opt-125m"),
            dtype=overrides.get("dtype", "float16"),
            max_model_len=overrides.get("max_model_len", 2048),
        ),
        cache_config=CacheConfig(
            block_size=overrides.get("block_size", 1024),
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        ),
        scheduler_config=SchedulerConfig.default_factory(),
        parallel_config=ParallelConfig(
            data_parallel_size=overrides.get("data_parallel_size", 2),
            tensor_parallel_size=overrides.get("tensor_parallel_size", 1),
        ),
    )


class TestConfigSignatureRealVllmConfig:
    """Against a real VllmConfig, not a SimpleNamespace stand-in."""

    def test_launch_stable(self):
        # Two independently built identical configs must key one bundle.
        assert mega_cache.config_signature(
            _real_vllm_config()
        ) == mega_cache.config_signature(_real_vllm_config())

    def test_real_coord_store_port_ignored(self):
        # The field that made the signature move every launch.
        cfg = _real_vllm_config()
        object.__setattr__(cfg.parallel_config, "_coord_store_port", 38735)
        a = mega_cache.config_signature(cfg)
        object.__setattr__(cfg.parallel_config, "_coord_store_port", 41200)
        assert mega_cache.config_signature(cfg) == a

    def test_every_real_port_field_is_swept(self):
        cfg = _real_vllm_config()
        port_fields = [
            f.name
            for f in dataclasses.fields(cfg.parallel_config)
            if "port" in f.name.lower()
        ]
        # Guard against the sweep silently becoming a no-op upstream.
        assert port_fields
        base = mega_cache.config_signature(cfg)
        for name in port_fields:
            original = getattr(cfg.parallel_config, name)
            probe = [50001, 50002] if isinstance(original, list) else 50000
            object.__setattr__(cfg.parallel_config, name, probe)
            assert mega_cache.config_signature(cfg) == base, name
            object.__setattr__(cfg.parallel_config, name, original)

    def test_real_port_fields_restored(self):
        cfg = _real_vllm_config()
        object.__setattr__(cfg.parallel_config, "_coord_store_port", 38735)
        before = {
            f.name: getattr(cfg.parallel_config, f.name)
            for f in dataclasses.fields(cfg.parallel_config)
            if "port" in f.name.lower()
        }
        mega_cache.config_signature(cfg)
        after = {name: getattr(cfg.parallel_config, name) for name in before}
        assert after == before

    @pytest.mark.parametrize(
        "overrides",
        [
            {"block_size": 512},
            {"tensor_parallel_size": 4},
            {"max_model_len": 1024},
            {"dtype": "bfloat16"},
        ],
        ids=["block_size", "tp", "max_model_len", "dtype"],
    )
    def test_graph_relevant_config_invalidates(self, overrides):
        base = mega_cache.config_signature(_real_vllm_config())
        assert mega_cache.config_signature(_real_vllm_config(**overrides)) != base

    def test_signature_shape(self):
        sig = mega_cache.config_signature(_real_vllm_config())
        assert re.fullmatch(r"[0-9a-f]{16}", sig)

    def test_signature_is_a_single_path_segment(self):
        # It becomes a directory name in bundle_path().
        sig = mega_cache.config_signature(_real_vllm_config())
        assert os.sep not in sig
