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

import pytest
from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import GptOssConfig, PretrainedConfig, Qwen3Config

from vllm_rbln.utils.optimum.compilation import RBLNCompileSpec
from vllm_rbln.utils.optimum.params import RBLNParams


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "config_cls, architecture",
    [(GptOssConfig, "GptOssForCausalLM"), (Qwen3Config, "Qwen3ForCausalLM")],
)
def test_decoder_region_model_selection(monkeypatch, enabled, config_cls, architecture):
    monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", str(int(enabled)))
    config = config_cls(architectures=[architecture])
    spec = RBLNCompileSpec.for_architecture(
        config,
        params=RBLNParams(batch_size=1, max_seq_len=8192, kvcache_block_size=8192),
        memory_budget=0.9,
    )
    assert spec.model_cls is RBLNAutoModelForCausalLM
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
    spec = RBLNCompileSpec.for_architecture(
        config,
        params=RBLNParams(batch_size=1, max_seq_len=8192, kvcache_block_size=8192),
        memory_budget=0.9,
    )
    decoder = spec.rbln_config[key] if key else spec.rbln_config
    assert decoder["nested_compile_region"] is True
    if key:
        assert "nested_compile_region" not in spec.rbln_config


def test_flag_conflicts_with_decoder_override(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_NESTED_COMPILE_REGION", "1")
    with pytest.raises(ValueError, match="nested_compile_region"):
        RBLNCompileSpec.for_architecture(
            Qwen3Config(architectures=["Qwen3ForCausalLM"]),
            params=RBLNParams(batch_size=1, max_seq_len=8192, kvcache_block_size=8192),
            memory_budget=0.9,
            optimum_overrides={"nested_compile_region": False},
        )


def test_paligemma_finalizes_params_before_rendering_config():
    params = RBLNParams(
        batch_size=4,
        max_seq_len=8192,
        kvcache_block_size=8192,
        prefill_chunk_size=128,
    )
    spec = RBLNCompileSpec.for_architecture(
        PretrainedConfig(architectures=["PaliGemmaForConditionalGeneration"]),
        params=params,
        memory_budget=0.9,
    )
    assert params.prefill_chunk_size == 128
    assert spec.params.prefill_chunk_size == 8192
    assert spec.rbln_config["language_model"]["prefill_chunk_size"] == 8192
