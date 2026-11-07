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

import pytest
import torch
from transformers import LlamaConfig

from vllm_rbln.utils.optimum import setup
from vllm_rbln.utils.optimum.artifacts import (
    generate_compile_cache_name,
    load_compiled_rbln_config,
)

pytestmark = pytest.mark.usefixtures("compile_environment")


class TestCompileCacheName:
    def test_npu_type_changes_hash(self, monkeypatch, vllm_config_factory):
        ca25 = generate_compile_cache_name(
            vllm_config_factory().additional_config.compile_spec
        )
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA22"
        )
        assert (
            generate_compile_cache_name(
                vllm_config_factory().additional_config.compile_spec
            )
            != ca25
        )

    def test_npu_name_is_normalized(self, monkeypatch, vllm_config_factory):
        canonical = generate_compile_cache_name(
            vllm_config_factory().additional_config.compile_spec
        )
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: " rbln-ca25 "
        )
        cfg = vllm_config_factory()
        assert cfg.additional_config.compile_spec.npu_name == "RBLN-CA25"
        assert (
            generate_compile_cache_name(cfg.additional_config.compile_spec) == canonical
        )

    def test_resolved_npu_type_is_preserved(self, monkeypatch, vllm_config_factory):
        cfg = vllm_config_factory()
        original = generate_compile_cache_name(cfg.additional_config.compile_spec)
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA22"
        )
        setup.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.compile_spec.npu_name == "RBLN-CA25"
        assert (
            generate_compile_cache_name(cfg.additional_config.compile_spec) == original
        )

    def test_user_max_num_batched_tokens_changes_hash(self, vllm_config_factory):
        # Two runs that would compile different prefill chunk sizes must not
        # collide on the same cache key.
        name_512 = generate_compile_cache_name(
            vllm_config_factory(512).additional_config.compile_spec
        )
        name_256 = generate_compile_cache_name(
            vllm_config_factory(256).additional_config.compile_spec
        )
        assert name_512 != name_256

    def test_unset_differs_from_explicit(self, vllm_config_factory):
        name_unset = generate_compile_cache_name(
            vllm_config_factory(None).additional_config.compile_spec
        )
        name_512 = generate_compile_cache_name(
            vllm_config_factory(512).additional_config.compile_spec
        )
        assert name_unset != name_512

    def test_same_value_same_hash(self, vllm_config_factory):
        assert generate_compile_cache_name(
            vllm_config_factory(512).additional_config.compile_spec
        ) == generate_compile_cache_name(
            vllm_config_factory(512).additional_config.compile_spec
        )

    def test_dtype_changes_hash(self, vllm_config_factory):
        name_bf16 = generate_compile_cache_name(
            vllm_config_factory(dtype=torch.bfloat16).additional_config.compile_spec
        )
        name_fp32 = generate_compile_cache_name(
            vllm_config_factory(dtype=torch.float32).additional_config.compile_spec
        )
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
    def test_resolved_hf_overrides_change_hash(self, overrides, vllm_config_factory):
        hf_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        hf_config.update(overrides)
        assert generate_compile_cache_name(
            vllm_config_factory(hf_config=hf_config).additional_config.compile_spec
        ) != generate_compile_cache_name(
            vllm_config_factory().additional_config.compile_spec
        )

    def test_equivalent_hf_configs_share_the_compiled_artifact(
        self, vllm_config_factory
    ):
        hf_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        same_config = LlamaConfig.from_dict(hf_config.to_dict())
        assert generate_compile_cache_name(
            vllm_config_factory(hf_config=hf_config).additional_config.compile_spec
        ) == generate_compile_cache_name(
            vllm_config_factory(hf_config=same_config).additional_config.compile_spec
        )

    def test_nested_region_changes_hash(self, monkeypatch, vllm_config_factory):
        monkeypatch.delenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", raising=False)
        default = generate_compile_cache_name(
            vllm_config_factory().additional_config.compile_spec
        )
        monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
        assert (
            generate_compile_cache_name(
                vllm_config_factory().additional_config.compile_spec
            )
            != default
        )
        monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "0")
        assert (
            generate_compile_cache_name(
                vllm_config_factory().additional_config.compile_spec
            )
            == default
        )

    def test_default_and_explicit_chunk_share_the_compiled_artifact(
        self, vllm_config_factory
    ):
        assert generate_compile_cache_name(
            vllm_config_factory().additional_config.compile_spec
        ) == generate_compile_cache_name(
            vllm_config_factory(128).additional_config.compile_spec
        )

    def test_key_survives_runtime_sync_and_model_path_replacement(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory()
        original = generate_compile_cache_name(cfg.additional_config.compile_spec)
        cfg.model_config.model = cfg.additional_config.cached_model_path
        cfg.cache_config.block_size = 128
        cfg.scheduler_config.max_num_batched_tokens = 512
        cfg.additional_config.optimum_overrides = {}
        cfg.model_config.hf_config.num_hidden_layers = 1
        assert (
            generate_compile_cache_name(cfg.additional_config.compile_spec) == original
        )

    @pytest.mark.parametrize("key", ["device", "create_runtimes", "kvcache_num_blocks"])
    def test_runtime_options_do_not_change_the_key(self, key, vllm_config_factory):
        cfg = vllm_config_factory()
        original = generate_compile_cache_name(cfg.additional_config.compile_spec)
        cfg.additional_config.compile_spec.rbln_config[key] = 1
        assert (
            generate_compile_cache_name(cfg.additional_config.compile_spec) == original
        )

    def test_prefix_cache_sub_blocks_do_not_change_the_compiled_artifact(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(prefix_caching=True)
        assert cfg.cache_config.block_size == 128
        assert cfg.additional_config.attn_block_size == 8192
        assert generate_compile_cache_name(
            cfg.additional_config.compile_spec
        ) == generate_compile_cache_name(
            vllm_config_factory().additional_config.compile_spec
        )


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
    from vllm_rbln.utils.optimum.artifacts import resolve_compiled_model_path

    source = tmp_path / "source"
    cache = tmp_path / "cache"
    for path in (source, cache):
        path.mkdir()
        (path / "rbln_config.json").write_text("{}")
    assert resolve_compiled_model_path(str(source), str(cache)) == str(source)
    assert resolve_compiled_model_path(str(tmp_path / "missing"), str(cache)) == str(
        cache
    )
