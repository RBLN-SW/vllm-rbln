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

from types import SimpleNamespace

import pytest
import torch
from transformers import MistralConfig, Phi3Config

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.utils.optimum.converter import dispatch
from vllm_rbln.utils.optimum.converter.dispatch import (
    _generate_model_path_name,
    disable_redundant_sliding_window,
)


def _vllm_config(user_max_num_batched_tokens=None, dtype=torch.bfloat16):
    additional_config = {}
    if user_max_num_batched_tokens is not None:
        additional_config["user_max_num_batched_tokens"] = user_max_num_batched_tokens
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="meta-llama/Llama-3.1-8B", max_model_len=8192, dtype=dtype
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(block_size=8192, gpu_memory_utilization=0.9),
        additional_config=build_optimum_rbln_config(additional_config),
    )


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


def _phi3(window):
    return Phi3Config(sliding_window=window, architectures=["Phi3ForCausalLM"])


def _mistral(window):
    return MistralConfig(sliding_window=window, architectures=["MistralForCausalLM"])


def _windowed(config, max_model_len, optimum_overrides=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            hf_text_config=config,
            max_model_len=max_model_len,
            disable_sliding_window=False,
        ),
        additional_config=build_optimum_rbln_config(
            {"optimum_overrides": optimum_overrides or {}}
        ),
    )


def _resolve_to(monkeypatch, compiled_rbln_config):
    monkeypatch.setattr(
        dispatch, "_resolve_rbln_config", lambda vllm_config: compiled_rbln_config
    )


class TestDisableRedundantSlidingWindow:
    @pytest.mark.parametrize(
        ("config", "max_model_len", "dropped"),
        [
            # Phi-3.5-mini / Phi-4-mini ship a window wider than any context.
            (_phi3(262144), 8192, True),
            (_mistral(4096), 4096, True),
            (_mistral(4096), 8192, False),
        ],
    )
    def test_before_compile_a_window_covering_the_context_is_dropped(
        self, monkeypatch, config, max_model_len, dropped
    ):
        _resolve_to(monkeypatch, None)
        vllm_config = _windowed(config, max_model_len)
        disable_redundant_sliding_window(vllm_config)
        assert (config.sliding_window is None) is dropped
        assert vllm_config.model_config.disable_sliding_window is dropped

    def test_the_max_seq_len_override_decides(self, monkeypatch):
        _resolve_to(monkeypatch, None)
        config = _mistral(4096)
        disable_redundant_sliding_window(_windowed(config, 4096, {"max_seq_len": 8192}))
        assert config.sliding_window == 4096

    @pytest.mark.parametrize(
        ("cache_impl", "dropped"), [("static", True), ("sliding_window", False)]
    )
    def test_a_compiled_model_keeps_the_cache_it_was_built_with(
        self, monkeypatch, cache_impl, dropped
    ):
        _resolve_to(monkeypatch, {"cache_impl": cache_impl})
        config = _mistral(4096)
        disable_redundant_sliding_window(_windowed(config, 4096))
        assert (config.sliding_window is None) is dropped
