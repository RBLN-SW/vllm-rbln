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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, PaliGemmaConfig, Qwen3Config, WhisperConfig

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.platform.optimum_impl import check_and_update
from vllm_rbln.utils.optimum.converter import dispatch
from vllm_rbln.utils.optimum.converter.dispatch import _generate_model_path_name
from vllm_rbln.utils.optimum.converter.params import (
    load_compiled_rbln_config,
)


@pytest.fixture(autouse=True)
def compile_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(dispatch, "load_compiled_rbln_config", lambda cfg: None)
    monkeypatch.setattr(
        dispatch, "resolve_compiled_model_path", lambda model, cache: None
    )
    monkeypatch.setattr(
        "vllm_rbln.utils.optimum.converter.params._resolve_default_prefill_chunk_size",
        lambda: 128,
    )
    monkeypatch.setattr(
        "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA25"
    )


def _vllm_config(
    user_max_num_batched_tokens=None,
    dtype=torch.bfloat16,
    *,
    prefix_caching=False,
    optimum_overrides=None,
    num_devices_per_local_rank=1,
    mamba_block_size=None,
    hf_config=None,
):
    additional_config = {
        "optimum_overrides": optimum_overrides or {},
        "num_devices_per_local_rank": num_devices_per_local_rank,
    }
    if user_max_num_batched_tokens is not None:
        additional_config["user_max_num_batched_tokens"] = user_max_num_batched_tokens
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="meta-llama/Llama-3.1-8B",
            max_model_len=8192,
            dtype=dtype,
            hf_config=(
                hf_config
                if hf_config is not None
                else LlamaConfig(architectures=["LlamaForCausalLM"])
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
        cache_config=SimpleNamespace(
            block_size=8192,
            gpu_memory_utilization=0.9,
            user_specified_block_size=True,
            enable_prefix_caching=prefix_caching,
            num_gpu_blocks_override=None,
            user_specified_mamba_block_size=False,
            mamba_block_size=mamba_block_size,
        ),
        additional_config=build_optimum_rbln_config(additional_config),
    )

    dispatch.sync_vllm_and_optimum(cfg)
    return cfg


class TestGenerateModelPathName:
    def test_npu_type_changes_hash(self, monkeypatch):
        ca25 = _generate_model_path_name(_vllm_config())
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA22"
        )
        assert _generate_model_path_name(_vllm_config()) != ca25

    def test_npu_name_is_normalized(self, monkeypatch):
        canonical = _generate_model_path_name(_vllm_config())
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: " rbln-ca25 "
        )
        cfg = _vllm_config()
        assert cfg.additional_config.compile_spec["npu_name"] == "RBLN-CA25"
        assert _generate_model_path_name(cfg) == canonical

    def test_resolved_npu_type_is_preserved(self, monkeypatch):
        cfg = _vllm_config()
        original = _generate_model_path_name(cfg)
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA22"
        )
        dispatch.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.compile_spec["npu_name"] == "RBLN-CA25"
        assert _generate_model_path_name(cfg) == original

    @pytest.mark.parametrize("prefix_caching", [False, True])
    def test_paligemma_uses_final_chunk_before_cache_lookup(
        self, monkeypatch, prefix_caching
    ):
        observed = []

        def generate_name(vllm_config):
            observed.append(
                vllm_config.additional_config.compile_spec["rbln_config"].get(
                    "language_model",
                    vllm_config.additional_config.compile_spec["rbln_config"],
                )["prefill_chunk_size"]
            )
            return _generate_model_path_name(vllm_config)

        monkeypatch.setattr(dispatch, "_generate_model_path_name", generate_name)
        cfg = _vllm_config(
            prefix_caching=prefix_caching,
            hf_config=PaliGemmaConfig(
                architectures=["PaliGemmaForConditionalGeneration"]
            ),
        )
        spec = cfg.additional_config.compile_spec["rbln_config"]["language_model"]
        assert observed == [spec["prefill_chunk_size"]] == [8192]
        assert cfg.cache_config.block_size == 8192
        dispatch.sync_vllm_and_optimum(cfg)
        assert cfg.scheduler_config.max_num_batched_tokens == 8192

    def test_paligemma_rejects_user_chunk_that_differs_from_final_spec(self):
        with pytest.raises(ValueError, match="prefill_chunk_size"):
            _vllm_config(
                512,
                hf_config=PaliGemmaConfig(
                    architectures=["PaliGemmaForConditionalGeneration"]
                ),
            )

    def test_whisper_runtime_uses_final_compile_lengths(self):
        cfg = _vllm_config(
            hf_config=WhisperConfig(
                architectures=["WhisperForConditionalGeneration"],
                max_target_positions=8192,
            )
        )
        spec = cfg.additional_config.compile_spec["rbln_config"]
        assert spec["dec_max_seq_len"] == cfg.model_config.max_model_len == 8192
        assert spec["batch_size"] == cfg.scheduler_config.max_num_seqs == 4

    def test_paligemma_cache_hit_preserves_the_final_runtime_settings(
        self, monkeypatch
    ):
        hf_config = PaliGemmaConfig(architectures=["PaliGemmaForConditionalGeneration"])
        cfg = _vllm_config(hf_config=hf_config)
        cache_path = Path(cfg.additional_config.cached_model_path)
        cache_path.mkdir(parents=True)
        metadata = cfg.additional_config.compile_spec["rbln_config"]
        metadata = json.loads(json.dumps(metadata))
        metadata["language_model"].update(
            kvcache_block_size=8192, kvcache_num_blocks=4, dtype="bfloat16"
        )
        (cache_path / "rbln_config.json").write_text(json.dumps(metadata))
        monkeypatch.setattr(
            dispatch, "load_compiled_rbln_config", load_compiled_rbln_config
        )
        monkeypatch.setattr(
            dispatch,
            "resolve_compiled_model_path",
            lambda model, cache: str(cache_path) if cache is not None else None,
        )
        cached = _vllm_config(hf_config=hf_config)
        assert cached.additional_config.cached_model_path == str(cache_path)
        assert (
            cached.scheduler_config.max_num_batched_tokens
            == cfg.scheduler_config.max_num_batched_tokens
            == 8192
        )
        assert cached.cache_config.block_size == cfg.cache_config.block_size == 8192

    @pytest.mark.parametrize(
        ("runner", "reranker", "expected_arch", "expected_cls", "expected_tokens"),
        [
            ("pooling", False, "Qwen3Model", "RBLNQwen3Model", 1024),
            ("pooling", True, "Qwen3ForCausalLM", "RBLNAutoModelForCausalLM", 128),
            ("generate", False, "Qwen3ForCausalLM", "RBLNAutoModelForCausalLM", 128),
        ],
    )
    def test_qwen3_architecture_is_resolved_before_cache_lookup(
        self,
        monkeypatch,
        runner,
        reranker,
        expected_arch,
        expected_cls,
        expected_tokens,
    ):
        hf_config = Qwen3Config(
            architectures=[
                "Qwen3ForSequenceClassification" if reranker else "Qwen3ForCausalLM"
            ],
            is_original_qwen3_reranker=reranker,
            classifier_from_token=["no", "yes"] if reranker else None,
        )
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                model="Qwen/Qwen3-0.6B",
                architectures=hf_config.architectures,
                runner_type=runner,
                hf_config=hf_config,
                hf_text_config=hf_config,
                multimodal_config=None,
                max_model_len=1024,
                dtype=torch.bfloat16,
            ),
            parallel_config=SimpleNamespace(
                worker_cls="auto", tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1, max_num_batched_tokens=2048, async_scheduling=False
            ),
            cache_config=SimpleNamespace(
                block_size=1024,
                gpu_memory_utilization=0.9,
                user_specified_block_size=True,
                enable_prefix_caching=False,
            ),
            additional_config={},
            speculative_config=None,
        )
        observed = []

        def generate_name(vllm_config):
            observed.append(
                (
                    list(vllm_config.model_config.hf_config.architectures),
                    vllm_config.additional_config.compile_spec["model_cls"],
                    vllm_config.additional_config.compile_spec["rbln_config"].get(
                        "prefill_chunk_size", 1024
                    ),
                )
            )
            return _generate_model_path_name(vllm_config)

        monkeypatch.setattr(dispatch, "_generate_model_path_name", generate_name)
        check_and_update(cfg)
        assert observed == [([expected_arch], expected_cls, expected_tokens)]

    def test_user_max_num_batched_tokens_changes_hash(self):
        # Two runs that would compile different prefill chunk sizes must not
        # collide on the same cache key.
        name_512 = _generate_model_path_name(_vllm_config(512))
        name_256 = _generate_model_path_name(_vllm_config(256))
        assert name_512 != name_256

    def test_unset_differs_from_explicit(self):
        name_unset = _generate_model_path_name(_vllm_config(None))
        name_512 = _generate_model_path_name(_vllm_config(512))
        assert name_unset != name_512

    def test_same_value_same_hash(self):
        assert _generate_model_path_name(
            _vllm_config(512)
        ) == _generate_model_path_name(_vllm_config(512))

    def test_dtype_changes_hash(self):
        name_bf16 = _generate_model_path_name(_vllm_config(dtype=torch.bfloat16))
        name_fp32 = _generate_model_path_name(_vllm_config(dtype=torch.float32))
        assert name_bf16 != name_fp32

    @pytest.mark.parametrize(
        "overrides",
        [
            {"num_hidden_layers": 1},
            {"layer_types": ["sliding_attention"]},
            {"rope_parameters": {"rope_type": "linear", "factor": 2.0}},
            {"vision_config": {"num_hidden_layers": 1}},
            {"device": "custom_compile_target"},
        ],
    )
    def test_resolved_hf_overrides_change_hash(self, overrides):
        hf_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        hf_config.update(overrides)
        assert _generate_model_path_name(
            _vllm_config(hf_config=hf_config)
        ) != _generate_model_path_name(_vllm_config())

    def test_equivalent_hf_configs_share_the_compiled_artifact(self):
        hf_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        same_config = LlamaConfig.from_dict(hf_config.to_dict())
        assert _generate_model_path_name(
            _vllm_config(hf_config=hf_config)
        ) == _generate_model_path_name(_vllm_config(hf_config=same_config))

    def test_nested_region_changes_hash(self, monkeypatch):
        monkeypatch.delenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", raising=False)
        default = _generate_model_path_name(_vllm_config())
        monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
        assert _generate_model_path_name(_vllm_config()) != default
        monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "0")
        assert _generate_model_path_name(_vllm_config()) == default

    def test_default_and_explicit_chunk_share_the_compiled_artifact(self):
        assert _generate_model_path_name(_vllm_config()) == _generate_model_path_name(
            _vllm_config(128)
        )

    def test_key_survives_runtime_sync_and_model_path_replacement(self):
        cfg = _vllm_config()
        original = _generate_model_path_name(cfg)
        cfg.model_config.model = cfg.additional_config.cached_model_path
        cfg.cache_config.block_size = 128
        cfg.scheduler_config.max_num_batched_tokens = 512
        cfg.additional_config.optimum_overrides = {}
        cfg.model_config.hf_config.num_hidden_layers = 1
        assert _generate_model_path_name(cfg) == original

    def test_repeated_sync_preserves_the_resolved_spec(self):
        cfg = _vllm_config()
        spec = cfg.additional_config.compile_spec
        path = cfg.additional_config.cached_model_path
        dispatch.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.compile_spec is spec
        assert cfg.additional_config.cached_model_path == path

    @pytest.mark.parametrize("key", ["device", "create_runtimes", "kvcache_num_blocks"])
    def test_runtime_options_do_not_change_the_key(self, key):
        cfg = _vllm_config()
        original = _generate_model_path_name(cfg)
        cfg.additional_config.compile_spec["rbln_config"][key] = 1
        assert _generate_model_path_name(cfg) == original

    def test_prefix_cache_sub_blocks_do_not_change_the_compiled_artifact(self):
        cfg = _vllm_config(prefix_caching=True)
        assert cfg.cache_config.block_size == 128
        assert cfg.additional_config.attn_block_size == 8192
        assert _generate_model_path_name(cfg) == _generate_model_path_name(
            _vllm_config()
        )

    @pytest.mark.parametrize(
        ("user_tokens", "overrides", "expected"),
        [
            (None, {}, 128),
            (None, {"prefill_chunk_size": 256}, 256),
            (512, {}, 512),
            (512, {"prefill_chunk_size": 512}, 512),
        ],
    )
    def test_chunk_size_is_resolved_before_cache_lookup(
        self, monkeypatch, user_tokens, overrides, expected
    ):
        observed = []

        def generate_name(vllm_config):
            observed.append(
                vllm_config.additional_config.compile_spec["rbln_config"].get(
                    "language_model",
                    vllm_config.additional_config.compile_spec["rbln_config"],
                )["prefill_chunk_size"]
            )
            return _generate_model_path_name(vllm_config)

        monkeypatch.setattr(dispatch, "_generate_model_path_name", generate_name)
        cfg = _vllm_config(user_tokens, optimum_overrides=overrides)
        assert observed == [expected]
        assert cfg.scheduler_config.max_num_batched_tokens == expected
        assert (
            cfg.additional_config.compile_spec["rbln_config"]["prefill_chunk_size"]
            == expected
        )

    @pytest.mark.parametrize(
        ("dtype", "devices", "device_overrides", "expected_devices"),
        [
            (torch.bfloat16, 2, {}, 2),
            (torch.bfloat16, 1, {"num_devices": 4}, 4),
            (torch.float32, 3, {}, 3),
        ],
    )
    def test_compile_inputs_are_resolved_before_cache_lookup(
        self, monkeypatch, dtype, devices, device_overrides, expected_devices
    ):
        observed = []

        def generate_name(vllm_config):
            observed.append(
                (
                    vllm_config.additional_config.compile_spec["rbln_config"][
                        "batch_size"
                    ],
                    vllm_config.additional_config.compile_spec["rbln_config"][
                        "max_seq_len"
                    ],
                    vllm_config.additional_config.compile_spec["dtype"],
                    vllm_config.additional_config.compile_spec["rbln_config"][
                        "num_devices"
                    ],
                )
            )
            return _generate_model_path_name(vllm_config)

        monkeypatch.setattr(dispatch, "_generate_model_path_name", generate_name)
        cfg = _vllm_config(
            dtype=dtype,
            num_devices_per_local_rank=devices,
            optimum_overrides={
                "batch_size": 2,
                "max_seq_len": 4096,
                "kvcache_block_size": 4096,
                **device_overrides,
            },
        )
        assert observed == [(2, 4096, dtype, expected_devices)]
        spec = cfg.additional_config.compile_spec
        assert spec["dtype"] == dtype
        assert (
            spec["rbln_config"]["batch_size"] == cfg.scheduler_config.max_num_seqs == 2
        )
        assert (
            spec["rbln_config"]["max_seq_len"] == cfg.model_config.max_model_len == 4096
        )
        assert spec["rbln_config"]["num_devices"] == expected_devices
        assert cfg.additional_config.num_devices_per_local_rank == expected_devices

    def test_repeated_sync_preserves_the_attention_block_size(self):
        cfg = _vllm_config(prefix_caching=True)
        dispatch.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.attn_block_size == 8192
        assert cfg.cache_config.block_size == 128

    def test_cache_metadata_is_loaded_without_replacing_the_model_id(self, monkeypatch):
        cfg = _vllm_config()
        source_model = cfg.model_config.model
        cache_path = Path(cfg.additional_config.cached_model_path)
        cache_path.mkdir(parents=True)
        metadata = {"prefill_chunk_size": 256}
        (cache_path / "rbln_config.json").write_text(json.dumps(metadata))
        monkeypatch.setattr(
            dispatch, "load_compiled_rbln_config", load_compiled_rbln_config
        )
        monkeypatch.setattr(
            dispatch,
            "resolve_compiled_model_path",
            lambda model, cache: str(cache_path) if cache is not None else None,
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                dispatch,
                "load_compiled_rbln_config",
                lambda path: pytest.fail(
                    "path resolution must not read rbln_config.json"
                ),
            )
            assert dispatch._resolve_cached_model_path(cfg) == str(cache_path)
        assert cfg.model_config.model == source_model

        synced = []
        monkeypatch.setattr(
            dispatch, "sync_runtime_config", lambda cfg, data, **kw: synced.append(data)
        )
        monkeypatch.setattr(
            dispatch,
            "resolve_compile_spec",
            lambda cfg: pytest.fail(
                "cached metadata must bypass compile input parsing"
            ),
        )
        dispatch.sync_vllm_and_optimum(cfg)
        assert synced == [metadata]
        assert cfg.model_config.model == source_model


def test_initial_cache_hit_applies_runtime_settings_once(monkeypatch):
    from vllm_rbln.utils.optimum.converter import from_optimum

    calls = []
    original = from_optimum.update_block_size

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(from_optimum, "update_block_size", record)
    metadata = {
        "batch_size": 2,
        "max_seq_len": 4096,
        "kvcache_block_size": 4096,
        "kvcache_num_blocks": 2,
        "prefill_chunk_size": 256,
        "dtype": "float32",
        "num_devices": 2,
    }
    monkeypatch.setattr(dispatch, "load_compiled_rbln_config", lambda path: metadata)
    cfg = _vllm_config(
        mamba_block_size=8192,
        optimum_overrides={"device": [0], "create_runtimes": False},
    )
    assert len(calls) == 1
    assert cfg.scheduler_config.max_num_seqs == 2
    assert cfg.scheduler_config.max_num_batched_tokens == 256
    assert cfg.model_config.max_model_len == 4096
    assert cfg.model_config.dtype == torch.float32
    assert cfg.additional_config.num_devices_per_local_rank == 2
    assert cfg.additional_config.optimum_overrides == {"device": [0]}
    assert cfg.cache_config.mamba_block_size == 4096
    assert cfg.cache_config.num_gpu_blocks == 3
    dispatch.sync_vllm_and_optimum(cfg)
    assert cfg.cache_config.num_gpu_blocks == 3


def test_resolution_does_not_apply_runtime_settings(monkeypatch):
    cfg = _vllm_config()
    cfg.additional_config.compile_spec = None
    cfg.additional_config.optimum_overrides = {"batch_size": 2, "max_seq_len": 4096}
    dispatch.resolve_compile_spec(cfg)
    assert cfg.scheduler_config.max_num_seqs == 4
    assert cfg.model_config.max_model_len == 8192
    assert cfg.scheduler_config.max_num_batched_tokens == 128
    assert cfg.additional_config.compile_spec is not None
    assert cfg.additional_config.compile_spec["rbln_config"]["batch_size"] == 2


def test_compile_snapshot_does_not_share_mutable_overrides():
    overrides = {"custom_options": {"values": [1, 2]}}
    cfg = _vllm_config(optimum_overrides=overrides)
    overrides["custom_options"]["values"].append(3)
    assert cfg.additional_config.compile_spec["rbln_config"]["custom_options"] == {
        "values": [1, 2]
    }


def test_metadata_loader_reads_only_the_selected_path(tmp_path):
    first = tmp_path / "source"
    second = tmp_path / "cache"
    for path, value in ((first, 1), (second, 2)):
        path.mkdir()
        (path / "rbln_config.json").write_text(json.dumps({"batch_size": value}))
    assert load_compiled_rbln_config(str(first)) == {"batch_size": 1}
    assert load_compiled_rbln_config(str(second)) == {"batch_size": 2}
    assert load_compiled_rbln_config(str(tmp_path / "absent")) is None


def test_explicit_compiled_directory_takes_precedence(tmp_path):
    from vllm_rbln.utils.optimum.paths import resolve_compiled_model_path

    source = tmp_path / "source"
    cache = tmp_path / "cache"
    for path in (source, cache):
        path.mkdir()
        (path / "rbln_config.json").write_text("{}")
    assert resolve_compiled_model_path(str(source), str(cache)) == str(source)
    assert resolve_compiled_model_path(str(tmp_path / "missing"), str(cache)) == str(
        cache
    )
