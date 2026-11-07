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

from types import SimpleNamespace

import pytest
import torch
from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import GptOssConfig, PretrainedConfig, Qwen3Config

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.utils.optimum.compilation import build_compile_spec
from vllm_rbln.utils.optimum.compilation.multimodal.qwen import build_qwen3_5_config


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
    spec = build_compile_spec(_vllm_config(config))
    assert spec.model_cls == RBLNAutoModelForCausalLM.__name__
    assert spec.rbln_config.get("nested_compile_region", False) is enabled


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
    spec = build_compile_spec(_vllm_config(config))
    decoder = spec.rbln_config[key] if key else spec.rbln_config
    assert decoder["nested_compile_region"] is True
    if key:
        assert "nested_compile_region" not in spec.rbln_config


def test_flag_conflicts_with_decoder_override(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
    with pytest.raises(ValueError, match="nested_compile_region"):
        build_compile_spec(
            _vllm_config(
                Qwen3Config(architectures=["Qwen3ForCausalLM"]),
                optimum_overrides={"nested_compile_region": False},
            )
        )


def test_paligemma_compile_config_uses_full_prompt_prefill():
    cfg = _vllm_config(
        PretrainedConfig(architectures=["PaliGemmaForConditionalGeneration"])
    )
    spec = build_compile_spec(cfg)
    assert spec.rbln_config["language_model"]["prefill_chunk_size"] == 8192
    assert cfg.additional_config.optimum_overrides == {}


def test_dtype_override_points_to_vllm_dtype():
    cfg = _vllm_config(
        Qwen3Config(architectures=["Qwen3ForCausalLM"]),
        optimum_overrides={"dtype": "bfloat16"},
    )
    with pytest.raises(ValueError, match="`dtype` cannot be set"):
        build_compile_spec(cfg)


def test_user_chunk_conflicting_with_override_is_rejected():
    cfg = _vllm_config(
        Qwen3Config(architectures=["Qwen3ForCausalLM"]),
        optimum_overrides={"prefill_chunk_size": 256},
        user_chunk=512,
    )
    with pytest.raises(ValueError, match="Conflicting prefill chunk size"):
        build_compile_spec(cfg)


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
