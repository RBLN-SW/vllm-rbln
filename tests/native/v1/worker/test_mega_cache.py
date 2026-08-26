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

# How a torch.compiler mega-cache bundle is keyed (config signature + path) and
# how it is persisted. The torch and rebel boundaries are stubbed, so nothing
# here compiles; what is asserted is the bundle that would reach disk.

from __future__ import annotations

import contextlib
import dataclasses
import errno
import inspect
import logging
import os
import re
from types import SimpleNamespace

import pytest
import torch

from tests.native.vllm_config import make_vllm_config
from vllm_rbln.v1.worker import mega_cache, rbln_model_runner

MODEL = "meta-llama/Llama-3"
SIG = "sig"


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


def _stub_config(digest: str = "cfghash"):
    return SimpleNamespace(compute_hash=lambda: digest)


class TestRebelVersion:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("0.11.0", "0.11"),
            ("0.11.9", "0.11"),  # a patch bump keeps the bundle
            ("0.12.0", "0.12"),
            ("0.10.5.dev224+gb3c3db0d", "0.10"),
            ("", "unknown"),
            ("garbage", "unknown"),
        ],
    )
    def test_major_minor(self, version, expected):
        assert mega_cache._rebel_major_minor(version) == expected


class TestSignatureComposition:
    """Each of the three parts must reach the digest; pinned so only the one
    under test moves."""

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "envhash")

    def test_vllm_config_hash_invalidates(self):
        assert mega_cache.config_signature(
            _stub_config("h1")
        ) != mega_cache.config_signature(_stub_config("h2"))

    def test_env_factors_invalidate(self, monkeypatch):
        before = mega_cache.config_signature(_stub_config())
        monkeypatch.setattr(mega_cache, "_compile_env_factors", lambda: "other")
        assert mega_cache.config_signature(_stub_config()) != before

    def test_rebel_minor_bump_invalidates(self, monkeypatch):
        before = mega_cache.config_signature(_stub_config())
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.12")
        assert mega_cache.config_signature(_stub_config()) != before


# Variables the built graph depends on; one per resolved type, since what has to
# survive is the round trip through normalize_value()/hash_factors().
GRAPH_ENV = [
    ("VLLM_RBLN_NUM_HIDDEN_LAYERS", "0", "4"),  # int
    ("VLLM_RBLN_USE_W8A8", "0", "1"),  # bool
    ("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "exponential", "linear"),  # str
    ("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "1,2,4", "1,2,4,8"),  # list
]

# Variables that must not move it. Each value differs from that variable's
# default, so the row cannot pass by probing with the default itself.
RUNTIME_ENV = [
    # Sampler graphs compile with use_cache=False, so they never enter a bundle.
    ("VLLM_RBLN_SAMPLER", "0"),
    # Must stay out, or a bundle compiled on a CPU host misses on the NPU host.
    ("VLLM_RBLN_COMPILE_ONLY", "1"),
    ("VLLM_RBLN_ENABLE_WARM_UP", "0"),
    ("VLLM_RBLN_NUMA", "0"),
]

# Noise the bundle has to survive: vLLM's own compile_factors() walks ~240
# variables and would discard the bundle over any of them.
UNRELATED_ENV = [
    ("VLLM_NIXL_SIDE_CHANNEL_PORT", "5999"),
    ("VLLM_API_KEY", "sk-secret"),
    ("HOME", "/home/someone-else"),
    ("VLLM_TARGET_DEVICE", "rocm"),
]


class TestSignatureEnv:
    """What the rbln compile-env partition does and does not key on."""

    @pytest.fixture(autouse=True)
    def _pin_rebel(self, monkeypatch):
        monkeypatch.setattr(mega_cache, "_rebel_major_minor", lambda: "0.11")

    def _sig(self) -> str:
        return mega_cache.config_signature(_stub_config())

    @pytest.mark.parametrize(("name", "before", "after"), GRAPH_ENV)
    def test_graph_env_invalidates(self, monkeypatch, name, before, after):
        monkeypatch.setenv(name, before)
        first = self._sig()
        monkeypatch.setenv(name, after)
        assert self._sig() != first

    @pytest.mark.parametrize(("name", "value"), RUNTIME_ENV)
    def test_runtime_env_invariant(self, monkeypatch, name, value):
        from vllm_rbln import envs as rbln_envs

        monkeypatch.delenv(name, raising=False)
        default = getattr(rbln_envs, name)
        before = self._sig()
        monkeypatch.setenv(name, value)
        assert getattr(rbln_envs, name) != default, "probe does not flip the variable"
        assert self._sig() == before

    @pytest.mark.parametrize(("name", "value"), UNRELATED_ENV)
    def test_unrelated_env_invariant(self, monkeypatch, name, value):
        before = self._sig()
        monkeypatch.setenv(name, value)
        assert self._sig() == before

    @pytest.mark.parametrize(
        "names",
        [("LOCAL_RANK",), ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL")],
        ids=["local-rank", "dp-rank"],
    )
    def test_rank_invariant(self, monkeypatch, names):
        # All ranks compile the same graphs; the rank subdir isolates the files.
        for name in names:
            monkeypatch.setenv(name, "0")
        rank0 = self._sig()
        for name in names:
            monkeypatch.setenv(name, "7")
        assert self._sig() == rank0


class TestSignatureVllmConfig:
    """Against a real engine-built VllmConfig, not a stand-in."""

    def test_launch_stable(self):
        assert mega_cache.config_signature(
            make_vllm_config()
        ) == mega_cache.config_signature(make_vllm_config())

    @pytest.mark.parametrize(
        "overrides",
        [
            {"block_size": 512},
            {"max_model_len": 1024},
            {"tensor_parallel_size": 4},
            {"dtype": "float32"},
        ],
        ids=["block_size", "max_model_len", "tp", "dtype"],
    )
    def test_graph_relevant_config_invalidates(self, overrides):
        base = mega_cache.config_signature(make_vllm_config())
        assert mega_cache.config_signature(make_vllm_config(**overrides)) != base

    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_num_seqs": 8},
            {"num_gpu_blocks_override": 64},
            {"gpu_memory_utilization": 0.5},
        ],
        ids=["max_num_seqs", "num_gpu_blocks_override", "gpu_memory_utilization"],
    )
    def test_warmup_graph_set_config_invalidates(self, overrides):
        # compute_hash() drops all three, but each moves a warm-up graph shape,
        # and a partly-hitting bundle costs a duplicate weight set on device.
        base = mega_cache.config_signature(make_vllm_config())
        assert mega_cache.config_signature(make_vllm_config(**overrides)) != base

    def test_speculative_tokens_invalidate(self):
        # num_spec_tokens sets the decode query_len the warm-up compiles, and
        # SpeculativeConfig.compute_hash() keys only on the eagle3 factors.
        spec = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
        }
        base = mega_cache.config_signature(make_vllm_config(speculative_config=spec))
        other = make_vllm_config(
            speculative_config={**spec, "num_speculative_tokens": 5}
        )
        assert mega_cache.config_signature(other) != base

    def test_npu_name_invalidates(self, monkeypatch):
        # The per-graph hash stamps meta=npu:...; the bundle file must split too.
        import rebel

        monkeypatch.setattr(rebel, "get_npu_name", lambda device_id=0: None)
        monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25")
        atom = mega_cache.config_signature(make_vllm_config())
        monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CR13")
        assert mega_cache.config_signature(make_vllm_config()) != atom

    def test_every_factor_is_a_real_field(self):
        # A getattr default would drop an axis from the key on an upstream rename.
        config = make_vllm_config()
        assert hasattr(config.scheduler_config, "max_num_seqs")
        for name in ("num_gpu_blocks_override", "gpu_memory_utilization"):
            assert hasattr(config.cache_config, name), name

        spec = make_vllm_config(
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 2,
            }
        ).speculative_config
        for name in ("num_speculative_tokens", "method", "draft_tensor_parallel_size"):
            assert hasattr(spec, name), name

    def test_every_port_field_is_swept(self):
        # Ports are auto-queried per launch; one reaching the hash moves the
        # signature every restart. The empty guard catches an upstream rename
        # turning the sweep into a no-op.
        config = make_vllm_config()
        fields = [
            f.name
            for f in dataclasses.fields(config.parallel_config)
            if "port" in f.name.lower()
        ]
        assert fields
        base = mega_cache.config_signature(config)
        for name in fields:
            original = getattr(config.parallel_config, name)
            probe = [50001, 50002] if isinstance(original, list) else 50000
            object.__setattr__(config.parallel_config, name, probe)
            assert mega_cache.config_signature(config) == base, name
            object.__setattr__(config.parallel_config, name, original)

    def test_port_fields_restored(self):
        config = make_vllm_config()
        object.__setattr__(config.parallel_config, "_coord_store_port", 38735)
        mega_cache.config_signature(config)
        assert config.parallel_config._coord_store_port == 38735

    def test_signature_shape(self):
        # It becomes one directory name in bundle_path().
        sig = mega_cache.config_signature(make_vllm_config())
        assert re.fullmatch(r"[0-9a-f]{16}", sig)


class TestBundlePath:
    @pytest.fixture(autouse=True)
    def _cache_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
        monkeypatch.delenv("LOCAL_RANK", raising=False)

    def test_layout(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "3")
        path = mega_cache.bundle_path(MODEL, "abc123def456")
        assert path.startswith(mega_cache.cache_root() + os.sep)
        assert path.endswith(os.path.join("abc123def456", "rank3", "mega_cache.bin"))

    def test_signature_separates_bundles(self):
        assert mega_cache.bundle_path(MODEL, "aaaa") != mega_cache.bundle_path(
            MODEL, "bbbb"
        )

    def test_ranks_do_not_share_a_bundle(self, monkeypatch):
        paths = {mega_cache.bundle_path(MODEL, SIG)}  # LOCAL_RANK unset -> rank0
        for rank in ("1", "7"):
            monkeypatch.setenv("LOCAL_RANK", rank)
            paths.add(mega_cache.bundle_path(MODEL, SIG))
        assert len(paths) == 3
        assert any("rank0" in path for path in paths)

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "/abs/path/to/local model",
            "../../escape",
            "weird:name*with?chars",
            "한글모델",
            "***",
        ],
        ids=[
            "hf-repo",
            "local-space",
            "traversal",
            "shell-chars",
            "non-ascii",
            "all-unsafe",
        ],
    )
    def test_model_is_one_safe_segment(self, model):
        path = mega_cache.bundle_path(model, SIG)
        # <root>/rbln/<model-seg>/<sig>/rank0/mega_cache.bin
        model_seg = path.split(os.sep)[-4]
        assert re.fullmatch(r"[A-Za-z0-9._-]+", model_seg), model_seg
        assert os.path.abspath(path).startswith(
            os.path.abspath(mega_cache.cache_root()) + os.sep
        )

    def test_same_sanitized_name_still_separates(self):
        # Both sanitize to "org_model"; the hash suffix keeps them apart.
        assert mega_cache.bundle_path("org/model", SIG) != mega_cache.bundle_path(
            "org:model", SIG
        )


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Redirect the bundle under tmp_path and stub the torch/rebel boundaries.

    `loaded` and `set_dirs` record what crossed each boundary, so the tests
    assert on the persisted bundle rather than on real compilation.
    """
    from rebel.core import mega_cache as rbln_mega_cache

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    # The native suite disables the compile cache session-wide; this is the one
    # module that has to run with it on.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    state = SimpleNamespace(
        artifact=b"bundle-bytes",
        save_result=None,
        save_raises=None,
        loaded=[],
        set_dirs=[],
        steps=[],  # boundary calls in the order they happened
    )
    state.save_result = (state.artifact, object())
    state.path = mega_cache.bundle_path(MODEL, SIG)

    def fake_save():
        state.steps.append("serialize")
        if state.save_raises is not None:
            raise state.save_raises
        return state.save_result

    def fake_load(data):
        state.loaded.append(data)
        return object()  # stands in for torch's CacheInfo

    def fake_flush():
        state.steps.append("flush")

    monkeypatch.setattr(rbln_mega_cache, "set_dir", state.set_dirs.append)
    monkeypatch.setattr(rbln_mega_cache, "flush_to_bundle", fake_flush)
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", fake_save)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", fake_load)
    return state


def _tmp_leftovers(path: str) -> list[str]:
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        return []
    return [name for name in os.listdir(directory) if name.endswith(".tmp")]


def _bundle_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class TestSaveLoad:
    def test_round_trip(self, bundle):
        mega_cache.save(MODEL, SIG)
        assert _bundle_bytes(bundle.path) == bundle.artifact

        mega_cache.load(MODEL, SIG)
        assert bundle.loaded == [bundle.artifact]

    def test_save_flushes_staged_blobs_first(self, bundle):
        # Disk-staged .rbln blobs enter the bundle only via flush_to_bundle(),
        # so flushing after serializing would drop every one of them.
        mega_cache.save(MODEL, SIG)
        assert bundle.steps == ["flush", "serialize"]

    def test_both_point_rebel_at_the_cache_root(self, bundle):
        mega_cache.save(MODEL, SIG)
        mega_cache.load(MODEL, SIG)
        assert bundle.set_dirs == [mega_cache.cache_root()] * 2

    def test_save_leaves_no_tmp_file(self, bundle):
        mega_cache.save(MODEL, SIG)
        assert not _tmp_leftovers(bundle.path)

    def test_load_miss_is_silent(self, bundle):
        mega_cache.load(MODEL, SIG)  # nothing saved yet
        assert bundle.loaded == []

    def test_signature_miss_does_not_read_another_bundle(self, bundle):
        mega_cache.save(MODEL, SIG)
        mega_cache.load(MODEL, "other-sig")
        assert bundle.loaded == []

    def test_a_run_does_not_read_another_max_num_seqs_bundle(self, bundle):
        # The reported failure: two runs differing only here shared a bundle,
        # so prefill hit while decode missed.
        sig1 = mega_cache.config_signature(make_vllm_config(max_num_seqs=1))
        sig4 = mega_cache.config_signature(make_vllm_config(max_num_seqs=4))
        mega_cache.save(MODEL, sig1)
        mega_cache.load(MODEL, sig4)
        assert bundle.loaded == []

    def test_rank_miss_does_not_read_another_rank(self, bundle, monkeypatch):
        mega_cache.save(MODEL, SIG)
        monkeypatch.setenv("LOCAL_RANK", "1")
        mega_cache.load(MODEL, SIG)
        assert bundle.loaded == []

    def test_resave_replaces_in_place(self, bundle):
        mega_cache.save(MODEL, SIG)
        bundle.save_result = (b"second-bundle", object())
        mega_cache.save(MODEL, SIG)
        assert _bundle_bytes(bundle.path) == b"second-bundle"

    def test_nothing_new_compiled_keeps_the_bundle(self, bundle):
        # torch returns None when the run recorded no new artifact -- i.e. every
        # graph came out of the loaded bundle. Re-saving must not empty it.
        mega_cache.save(MODEL, SIG)
        bundle.save_result = None
        mega_cache.save(MODEL, SIG)
        assert _bundle_bytes(bundle.path) == bundle.artifact

    def test_nothing_to_save_writes_no_bundle(self, bundle):
        bundle.save_result = None
        mega_cache.save(MODEL, SIG)
        assert not os.path.exists(bundle.path)

    def test_save_failure_leaves_no_bundle(self, bundle):
        bundle.save_raises = RuntimeError("boom")
        mega_cache.save(MODEL, SIG)  # must not propagate
        assert not os.path.exists(bundle.path)

    def test_save_failure_keeps_the_previous_bundle(self, bundle):
        mega_cache.save(MODEL, SIG)
        bundle.save_raises = RuntimeError("boom")
        mega_cache.save(MODEL, SIG)
        assert _bundle_bytes(bundle.path) == bundle.artifact

    def test_out_of_space_leaves_no_partial_bundle(self, bundle, monkeypatch):
        real_open = open

        def full_disk(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            if str(path).endswith(".tmp"):
                handle.write(b"partial")
                handle.close()
                raise OSError(errno.ENOSPC, "No space left on device")
            return handle

        with monkeypatch.context() as m:
            m.setattr("builtins.open", full_disk)
            mega_cache.save(MODEL, SIG)  # must not propagate
        assert not os.path.exists(bundle.path)
        assert not _tmp_leftovers(bundle.path)

    def test_out_of_space_keeps_the_previous_bundle(self, bundle, monkeypatch):
        mega_cache.save(MODEL, SIG)
        bundle.save_result = (b"second-bundle", object())
        with monkeypatch.context() as m:
            m.setattr(mega_cache.os, "replace", _raiser(OSError(errno.ENOSPC, "boom")))
            mega_cache.save(MODEL, SIG)
        assert _bundle_bytes(bundle.path) == bundle.artifact
        assert not _tmp_leftovers(bundle.path)

    def test_out_of_space_is_logged_at_error(self, bundle, monkeypatch, caplog):
        # Every restart recompiles from here on, so it cannot be a debug line.
        monkeypatch.setattr(
            mega_cache.os, "replace", _raiser(OSError(errno.ENOSPC, "no space"))
        )
        with caplog.at_level(logging.ERROR, logger=mega_cache.logger.name):
            mega_cache.save(MODEL, SIG)
        assert "out of disk space" in caplog.text

    def test_fsync_failure_is_survivable(self, bundle, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(mega_cache.os, "fsync", _raiser(OSError(errno.EIO, "io error")))
            mega_cache.save(MODEL, SIG)  # must not propagate
        assert not os.path.exists(bundle.path)
        assert not _tmp_leftovers(bundle.path)

    def test_corrupt_bundle_warns_and_recompiles(self, bundle, monkeypatch, caplog):
        os.makedirs(os.path.dirname(bundle.path), exist_ok=True)
        with open(bundle.path, "wb") as f:
            f.write(b"garbage")
        monkeypatch.setattr(
            torch.compiler, "load_cache_artifacts", _raiser(RuntimeError("bad bundle"))
        )
        with caplog.at_level(logging.WARNING, logger=mega_cache.logger.name):
            mega_cache.load(MODEL, SIG)  # must not propagate
        assert "bad bundle" in caplog.text

    def test_unreadable_bundle_warns_and_skips(self, bundle, monkeypatch, caplog):
        mega_cache.save(MODEL, SIG)
        monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda _: None)
        with caplog.at_level(logging.WARNING, logger=mega_cache.logger.name):
            mega_cache.load(MODEL, SIG)
        assert "unreadable" in caplog.text

    def test_disabled_compile_cache_is_a_no_op(self, bundle, monkeypatch):
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        mega_cache.save(MODEL, SIG)
        assert not os.path.exists(bundle.path)
        mega_cache.load(MODEL, SIG)
        assert bundle.loaded == []
        assert bundle.set_dirs == []


@pytest.mark.maybe_use_device
class TestWarmupWiring:
    """What binds this module into production: the bundle is loaded before
    warm-up compiles anything and written only once warm-up has succeeded."""

    @pytest.fixture
    def calls(self, make_model_runner, monkeypatch):
        runner = make_model_runner()
        steps: list[tuple] = []

        # The sampling-side warm-up is gated on the last PP rank; nothing here
        # initializes a distributed group.
        monkeypatch.setattr(
            rbln_model_runner,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True),
        )
        monkeypatch.setattr(runner, "offload_context", contextlib.nullcontext)
        monkeypatch.setattr(
            runner, "_dummy_run", lambda *a, **kw: steps.append(("compile",))
        )
        monkeypatch.setattr(runner, "_dummy_sampler_run", lambda *a, **kw: None)
        monkeypatch.setattr(runner, "_warmup_sampler_decode_batches", lambda: None)
        monkeypatch.setattr(
            mega_cache, "load", lambda model, sig: steps.append(("load", model, sig))
        )
        monkeypatch.setattr(
            mega_cache, "save", lambda model, sig: steps.append(("save", model, sig))
        )
        return SimpleNamespace(runner=runner, steps=steps, monkeypatch=monkeypatch)

    def test_load_before_compiling_save_after(self, calls):
        calls.runner.warmup_model()

        kinds = [step[0] for step in calls.steps]
        assert kinds[0] == "load"
        assert kinds[-1] == "save"
        assert "compile" in kinds
        # Same (model, sig) on both ends, or the run writes a bundle it can
        # never load back.
        assert calls.steps[0][1:] == calls.steps[-1][1:]
        assert calls.steps[0][1] == calls.runner.model_config.model

    def test_failed_warm_up_writes_no_bundle(self, calls):
        calls.monkeypatch.setattr(
            calls.runner, "_dummy_run", _raiser(RuntimeError("compile failed"))
        )
        with pytest.raises(RuntimeError, match="compile failed"):
            calls.runner.warmup_model()
        assert [step[0] for step in calls.steps] == ["load"]


class TestConformance:
    """Drift alarms: the save/load path is stubbed everywhere above, so nothing
    else in this file would notice either dependency changing shape."""

    def test_torch_mega_cache_api(self):
        assert callable(torch.compiler.save_cache_artifacts)
        assert callable(torch.compiler.load_cache_artifacts)

    def test_rebel_mega_cache_api(self):
        from rebel.core import mega_cache as rbln_mega_cache

        assert len(inspect.signature(rbln_mega_cache.set_dir).parameters) == 1
        assert not inspect.signature(rbln_mega_cache.flush_to_bundle).parameters

    def test_rbln_artifact_type_is_registered_with_torch(self):
        # rebel registers it at import; without it a bundle's rbln entries
        # deserialize into nothing and every graph recompiles.
        from rebel.core import mega_cache as rbln_mega_cache  # noqa: F401
        from torch.compiler._cache import CacheArtifactFactory

        assert CacheArtifactFactory.create("rbln", "M:probe", b"").type() == "rbln"
