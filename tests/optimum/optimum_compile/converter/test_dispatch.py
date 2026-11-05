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

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.utils.optimum.converter import dispatch
from vllm_rbln.utils.optimum.converter.dispatch import _generate_model_path_name
from vllm_rbln.utils.optimum.converter.params import (
    RBLNParams,
    load_compiled_rbln_config,
)


@pytest.fixture(autouse=True)
def compile_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(dispatch, "load_compiled_rbln_config", lambda cfg: None)
    monkeypatch.setattr(dispatch, "is_compiled_dir", lambda path: False)
    monkeypatch.setattr(
        "vllm_rbln.utils.optimum.converter.params._resolve_default_prefill_chunk_size",
        lambda: 128,
    )


def _vllm_config(
    user_max_num_batched_tokens=None,
    dtype=torch.bfloat16,
    *,
    prefix_caching=False,
    optimum_overrides=None,
    num_devices_per_local_rank=1,
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
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
        cache_config=SimpleNamespace(
            block_size=8192,
            gpu_memory_utilization=0.9,
            user_specified_block_size=True,
            enable_prefix_caching=prefix_caching,
        ),
        additional_config=build_optimum_rbln_config(additional_config),
    )

    dispatch.sync_vllm_and_optimum(cfg)
    return cfg


class TestGenerateModelPathName:
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
            observed.append(vllm_config.scheduler_config.max_num_batched_tokens)
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
                    vllm_config.scheduler_config.max_num_seqs,
                    vllm_config.model_config.max_model_len,
                    vllm_config.model_config.dtype,
                    vllm_config.additional_config.num_devices_per_local_rank,
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

    @pytest.mark.parametrize("prefix_caching", [False, True])
    def test_compile_inputs_are_parsed_once(self, monkeypatch, prefix_caching):
        parsed = []
        parse = RBLNParams.from_rbln_config

        def capture_params(cls, *args, **kwargs):
            params = parse(*args, **kwargs)
            parsed.append(params)
            return params

        monkeypatch.setattr(RBLNParams, "from_rbln_config", classmethod(capture_params))
        cfg = _vllm_config(prefix_caching=prefix_caching)
        assert len(parsed) == 1
        spec = cfg.additional_config.compile_spec["rbln_config"]
        assert spec["batch_size"] == parsed[0].batch_size
        assert spec["max_seq_len"] == parsed[0].max_seq_len
        assert spec["num_devices"] == parsed[0].num_devices
        assert spec["prefill_chunk_size"] == parsed[0].prefill_chunk_size
        assert parsed[0].kvcache_block_size == 8192

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
            dispatch, "is_compiled_dir", lambda path: path == str(cache_path)
        )
        assert dispatch._resolve_cached_rbln_config(cfg, RBLNParams()) == metadata
        assert cfg.model_config.model == source_model

        synced = []
        monkeypatch.setattr(
            dispatch, "sync_from_optimum", lambda cfg, data: synced.append(data)
        )
        monkeypatch.setattr(
            dispatch,
            "sync_from_vllm",
            lambda cfg: pytest.fail(
                "cached metadata must bypass compile input parsing"
            ),
        )
        dispatch.sync_vllm_and_optimum(cfg)
        assert synced == [metadata]
        assert cfg.model_config.model == source_model
