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

from types import SimpleNamespace

import pytest
import torch
from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import GptOssConfig, LlamaConfig, PretrainedConfig, Qwen3Config

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.utils.optimum.configuration import sync
from vllm_rbln.utils.optimum.configuration.inputs import compile_cache_name
from vllm_rbln.utils.optimum.configuration.multimodal.qwen import (
    build_qwen3_5_config,
)
from vllm_rbln.utils.optimum.configuration.prepare import prepare_compile_inputs


@pytest.fixture(autouse=True)
def npu_name(monkeypatch):
    monkeypatch.setattr(
        "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA25"
    )


def _vllm_config(hf_config, *, optimum_overrides=None, user_chunk=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            dtype=torch.bfloat16,
            max_model_len=8192,
            hf_config=hf_config,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        cache_config=SimpleNamespace(
            block_size=8192,
            user_specified_block_size=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.9,
        ),
        additional_config=build_optimum_rbln_config(
            {
                "optimum_overrides": optimum_overrides or {},
                "user_max_num_batched_tokens": user_chunk,
            }
        ),
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "config_cls, architecture",
    [(GptOssConfig, "GptOssForCausalLM"), (Qwen3Config, "Qwen3ForCausalLM")],
)
def test_decoder_region_model_selection(monkeypatch, enabled, config_cls, architecture):
    monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", str(int(enabled)))
    config = config_cls(architectures=[architecture])
    inputs = prepare_compile_inputs(_vllm_config(config))
    assert inputs.model_cls == RBLNAutoModelForCausalLM.__name__
    assert inputs.rbln_config.get("nested_compile_region", False) is enabled


@pytest.mark.parametrize(
    "architecture, key",
    [
        ("Gemma3ForConditionalGeneration", "language_model"),
        ("Idefics3ForConditionalGeneration", "text_model"),
        ("Qwen3VLForConditionalGeneration", None),
    ],
)
def test_multimodal_decoder_option(monkeypatch, architecture, key):
    monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
    config = PretrainedConfig(architectures=[architecture])
    inputs = prepare_compile_inputs(_vllm_config(config))
    decoder = inputs.rbln_config[key] if key else inputs.rbln_config
    assert decoder["nested_compile_region"] is True
    if key:
        assert "nested_compile_region" not in inputs.rbln_config


def test_flag_conflicts_with_decoder_override(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
    with pytest.raises(ValueError, match="nested_compile_region"):
        prepare_compile_inputs(
            _vllm_config(
                Qwen3Config(architectures=["Qwen3ForCausalLM"]),
                optimum_overrides={"nested_compile_region": False},
            )
        )


def test_paligemma_compile_config_uses_full_prompt_prefill():
    cfg = _vllm_config(
        PretrainedConfig(architectures=["PaliGemmaForConditionalGeneration"])
    )
    inputs = prepare_compile_inputs(cfg)
    assert inputs.rbln_config["language_model"]["prefill_chunk_size"] == 8192
    assert cfg.additional_config.optimum_overrides == {}


def test_dtype_override_points_to_vllm_dtype():
    cfg = _vllm_config(
        Qwen3Config(architectures=["Qwen3ForCausalLM"]),
        optimum_overrides={"dtype": "bfloat16"},
    )
    with pytest.raises(ValueError, match="`dtype` cannot be set"):
        prepare_compile_inputs(cfg)


def test_user_chunk_conflicting_with_override_is_rejected():
    cfg = _vllm_config(
        Qwen3Config(architectures=["Qwen3ForCausalLM"]),
        optimum_overrides={"prefill_chunk_size": 256},
        user_chunk=512,
    )
    with pytest.raises(ValueError, match="Conflicting prefill chunk size"):
        prepare_compile_inputs(cfg)


class TestQwen35ConfigBuilder:
    """Qwen3.5 takes the same parameters as the other Qwen VLMs: a block_size below
    max_model_len partitions the cache and selects flash attention, and a block_size
    equal to it leaves optimum's default, eager."""

    cfg = {
        "batch_size": 1,
        "num_devices": 1,
        "memory_budget": 0.9,
        "prefill_chunk_size": 128,
    }

    def test_valid_partition_sets_kvcache_partition_len(self):
        param = build_qwen3_5_config(max_model_len=4096, block_size=1024, **self.cfg)
        assert param["attn_impl"] == "flash_attn"
        # block_size is actually carried into the compiled model.
        assert param["kvcache_partition_len"] == 1024
        assert param["max_seq_len"] == 4096

    def test_block_size_equal_max_model_len_leaves_eager(self):
        # One partition is not a partition: nothing is set, so optimum keeps its
        # default and the whole cache is one block.
        param = build_qwen3_5_config(max_model_len=4096, block_size=4096, **self.cfg)
        assert "attn_impl" not in param
        assert "kvcache_partition_len" not in param
        assert param["max_seq_len"] == 4096


@pytest.mark.usefixtures("compile_environment")
class TestCompileCacheName:
    def test_npu_type_changes_hash(self, monkeypatch, vllm_config_factory):
        ca25 = compile_cache_name(
            vllm_config_factory().additional_config.compile_inputs
        )
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA22"
        )
        assert (
            compile_cache_name(vllm_config_factory().additional_config.compile_inputs)
            != ca25
        )

    def test_npu_name_is_normalized(self, monkeypatch, vllm_config_factory):
        canonical = compile_cache_name(
            vllm_config_factory().additional_config.compile_inputs
        )
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: " rbln-ca25 "
        )
        cfg = vllm_config_factory()
        assert cfg.additional_config.compile_inputs.npu_name == "RBLN-CA25"
        assert compile_cache_name(cfg.additional_config.compile_inputs) == canonical

    def test_resolved_npu_type_is_preserved(self, monkeypatch, vllm_config_factory):
        cfg = vllm_config_factory()
        original = compile_cache_name(cfg.additional_config.compile_inputs)
        monkeypatch.setattr(
            "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA22"
        )
        sync.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.compile_inputs.npu_name == "RBLN-CA25"
        assert compile_cache_name(cfg.additional_config.compile_inputs) == original

    def test_user_max_num_batched_tokens_changes_hash(self, vllm_config_factory):
        # Two runs that would compile different prefill chunk sizes must not
        # collide on the same cache key.
        name_512 = compile_cache_name(
            vllm_config_factory(512).additional_config.compile_inputs
        )
        name_256 = compile_cache_name(
            vllm_config_factory(256).additional_config.compile_inputs
        )
        assert name_512 != name_256

    def test_unset_differs_from_explicit(self, vllm_config_factory):
        name_unset = compile_cache_name(
            vllm_config_factory(None).additional_config.compile_inputs
        )
        name_512 = compile_cache_name(
            vllm_config_factory(512).additional_config.compile_inputs
        )
        assert name_unset != name_512

    def test_same_value_same_hash(self, vllm_config_factory):
        assert compile_cache_name(
            vllm_config_factory(512).additional_config.compile_inputs
        ) == compile_cache_name(
            vllm_config_factory(512).additional_config.compile_inputs
        )

    def test_dtype_changes_hash(self, vllm_config_factory):
        name_bf16 = compile_cache_name(
            vllm_config_factory(dtype=torch.bfloat16).additional_config.compile_inputs
        )
        name_fp32 = compile_cache_name(
            vllm_config_factory(dtype=torch.float32).additional_config.compile_inputs
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
        assert compile_cache_name(
            vllm_config_factory(hf_config=hf_config).additional_config.compile_inputs
        ) != compile_cache_name(vllm_config_factory().additional_config.compile_inputs)

    def test_equivalent_hf_configs_share_the_compiled_artifact(
        self, vllm_config_factory
    ):
        hf_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        same_config = LlamaConfig.from_dict(hf_config.to_dict())
        assert compile_cache_name(
            vllm_config_factory(hf_config=hf_config).additional_config.compile_inputs
        ) == compile_cache_name(
            vllm_config_factory(hf_config=same_config).additional_config.compile_inputs
        )

    def test_nested_region_changes_hash(self, monkeypatch, vllm_config_factory):
        monkeypatch.delenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", raising=False)
        default = compile_cache_name(
            vllm_config_factory().additional_config.compile_inputs
        )
        monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
        assert (
            compile_cache_name(vllm_config_factory().additional_config.compile_inputs)
            == default
        )
        monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "0")
        assert (
            compile_cache_name(vllm_config_factory().additional_config.compile_inputs)
            != default
        )

    def test_default_and_explicit_chunk_share_the_compiled_artifact(
        self, vllm_config_factory
    ):
        assert compile_cache_name(
            vllm_config_factory().additional_config.compile_inputs
        ) == compile_cache_name(
            vllm_config_factory(128).additional_config.compile_inputs
        )

    def test_key_survives_sync_and_model_path_replacement(self, vllm_config_factory):
        cfg = vllm_config_factory()
        original = compile_cache_name(cfg.additional_config.compile_inputs)
        cfg.model_config.model = cfg.additional_config.artifact_path
        cfg.cache_config.block_size = 128
        cfg.scheduler_config.max_num_batched_tokens = 512
        cfg.additional_config.optimum_overrides = {}
        cfg.model_config.hf_config.num_hidden_layers = 1
        assert compile_cache_name(cfg.additional_config.compile_inputs) == original

    @pytest.mark.parametrize("key", ["device", "create_runtimes", "kvcache_num_blocks"])
    def test_load_only_options_do_not_change_the_key(self, key, vllm_config_factory):
        cfg = vllm_config_factory()
        original = compile_cache_name(cfg.additional_config.compile_inputs)
        cfg.additional_config.compile_inputs.rbln_config[key] = 1
        assert compile_cache_name(cfg.additional_config.compile_inputs) == original

    def test_prefix_cache_sub_blocks_do_not_change_the_compiled_artifact(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(prefix_caching=True)
        assert cfg.cache_config.block_size == 128
        assert cfg.additional_config.attn_block_size == 8192
        assert compile_cache_name(
            cfg.additional_config.compile_inputs
        ) == compile_cache_name(vllm_config_factory().additional_config.compile_inputs)


@pytest.mark.parametrize(
    "architecture, overrides, user_chunk, expected_chunk, default_calls",
    [
        ("LlamaForCausalLM", {}, None, 256, 1),
        ("LlamaForCausalLM", {"prefill_chunk_size": 384}, None, 384, 0),
        ("LlamaForCausalLM", {}, 512, 512, 0),
        (
            "Gemma3ForConditionalGeneration",
            {"language_model": {"prefill_chunk_size": 384}},
            None,
            384,
            0,
        ),
        ("PaliGemmaForConditionalGeneration", {}, None, 8192, 0),
    ],
)
def test_compile_inputs_resolve_chunk_before_considering_the_npu_default(
    monkeypatch, architecture, overrides, user_chunk, expected_chunk, default_calls
):
    from vllm_rbln.utils.optimum.configuration import prepare as prepare_module

    calls = []

    def default_chunk():
        calls.append(True)
        return 256

    monkeypatch.setattr(prepare_module, "default_prefill_chunk_size", default_chunk)
    cfg = _vllm_config(
        PretrainedConfig(architectures=[architecture]),
        optimum_overrides=overrides,
        user_chunk=user_chunk,
    )
    inputs = prepare_compile_inputs(cfg)
    decoder_config = inputs.rbln_config.get("language_model", inputs.rbln_config)
    assert decoder_config["prefill_chunk_size"] == expected_chunk
    assert len(calls) == default_calls


@pytest.mark.parametrize(
    "architecture, overrides, expected",
    [
        ("LlamaForCausalLM", {}, (1, 8192, 8192, 1)),
        (
            "LlamaForCausalLM",
            {
                "batch_size": 2,
                "max_seq_len": 4096,
                "kvcache_partition_len": 1024,
                "num_devices": 2,
            },
            (2, 4096, 1024, 2),
        ),
        ("Qwen3Model", {}, (1, 8192, 8192, 1)),
        (
            "Gemma3ForConditionalGeneration",
            {
                "language_model": {
                    "batch_size": 2,
                    "max_seq_len": 4096,
                    "kvcache_block_size": 1024,
                    "device": [0, 1],
                }
            },
            (2, 4096, 1024, 2),
        ),
        ("WhisperForConditionalGeneration", {}, (1, 8192, 8192, 1)),
    ],
)
def test_compile_inputs_contain_resolved_values(architecture, overrides, expected):
    hf_config = PretrainedConfig(
        architectures=[architecture], max_target_positions=8192
    )
    inputs = prepare_compile_inputs(
        _vllm_config(hf_config, optimum_overrides=overrides)
    )
    selected = inputs.rbln_config.get("language_model", inputs.rbln_config)
    length_key = (
        "dec_max_seq_len"
        if architecture == "WhisperForConditionalGeneration"
        else "max_seq_len"
    )
    assert (
        selected["batch_size"],
        selected[length_key],
        selected["kvcache_block_size"],
        selected["num_devices"],
    ) == expected


def test_whisper_rejects_a_block_size_that_conflicts_with_decoder_length():
    hf_config = PretrainedConfig(
        architectures=["WhisperForConditionalGeneration"], max_target_positions=8192
    )
    cfg = _vllm_config(hf_config, optimum_overrides={"kvcache_block_size": 4096})
    with pytest.raises(ValueError, match="kvcache_block_size"):
        prepare_compile_inputs(cfg)
