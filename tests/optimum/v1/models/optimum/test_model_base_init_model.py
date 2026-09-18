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
"""Unit tests for what optimum-rbln receives on a cache-miss compile.

vLLM's ``hf_config`` may be a vLLM-private config class (e.g. qwen3_asr) that
transformers' model classes cannot read, so ``init_model`` does not forward it.
It passes only the layer count (and the per-layer attention types that HF
validates against it) as HF config kwargs, nested under ``text_config`` for
composite models. Everything that needs an NPU is faked.
"""

import types

import torch

from vllm_rbln.model_executor.models.optimum import model_base
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase


def _init_model_with(monkeypatch, tmp_path, hf_config) -> dict:
    passed = {}

    class FakeRBLNModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            passed.update(kwargs)
            return types.SimpleNamespace(
                rbln_config=types.SimpleNamespace(), save_pretrained=lambda p: None
            )

    monkeypatch.setattr(
        model_base.RBLNCompileSpec,
        "for_architecture",
        classmethod(
            lambda cls, *a, **k: types.SimpleNamespace(
                model_cls=FakeRBLNModel, rbln_config={}
            )
        ),
    )
    monkeypatch.setattr(model_base, "is_compiled_dir", lambda path: False)
    monkeypatch.setattr(model_base, "get_attn_block_size", lambda cfg: 4096)

    obj = RBLNOptimumModelBase.__new__(RBLNOptimumModelBase)
    obj.model_config = types.SimpleNamespace(
        hf_config=hf_config,
        model="repo",
        max_model_len=4096,
        dtype=torch.float16,
    )
    obj.scheduler_config = types.SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=128
    )
    obj.vllm_config = types.SimpleNamespace(
        additional_config={"cached_model_path": str(tmp_path)},
        model_config=obj.model_config,
        scheduler_config=obj.scheduler_config,
        cache_config=types.SimpleNamespace(gpu_memory_utilization=0.9),
        ec_transfer_config=None,
    )
    obj.init_model()
    return passed


def test_flat_config_passes_layer_count_as_top_level_kwargs(monkeypatch, tmp_path):
    # hf_overrides={"num_hidden_layers": 2} on a text-only model: the override
    # lands on the top-level config, and vLLM's config object stays out.
    hf_config = types.SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
    )
    hf_config.get_text_config = lambda: hf_config

    passed = _init_model_with(monkeypatch, tmp_path, hf_config)

    assert "config" not in passed
    assert passed["num_hidden_layers"] == 2
    assert passed["layer_types"] == ["full_attention", "full_attention"]
    assert "text_config" not in passed


def test_composite_config_nests_layer_count_under_text_config(monkeypatch, tmp_path):
    # A vLLM-private composite config (qwen3_asr keeps the text config under
    # thinker_config): the override must reach HF's text_config sub-config.
    text_config = types.SimpleNamespace(num_hidden_layers=2)
    hf_config = types.SimpleNamespace(
        architectures=["Qwen3ASRForConditionalGeneration"],
        get_text_config=lambda: text_config,
    )

    passed = _init_model_with(monkeypatch, tmp_path, hf_config)

    assert "config" not in passed
    assert passed["text_config"] == {"num_hidden_layers": 2}
    assert "num_hidden_layers" not in passed
