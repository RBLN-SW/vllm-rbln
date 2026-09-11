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
"""Unit test for the config optimum-rbln receives on a cache-miss compile.

vLLM's ``hf_config`` may be a vLLM-private config class (e.g. qwen3_asr) that
transformers' model classes cannot read, so ``init_model`` loads the
checkpoint's own config through the HF model class and carries over only the
layer count. Everything that needs an NPU is faked.
"""

import types

import torch

from vllm_rbln.model_executor.models.optimum import model_base
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase


class _TextConfig:
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


class _CheckpointConfig:
    def __init__(self):
        self.text = _TextConfig(num_hidden_layers=28)

    def get_text_config(self):
        return self.text


def test_init_model_compiles_with_checkpoint_config_and_vllm_layer_count(
    monkeypatch, tmp_path
):
    checkpoint_config = _CheckpointConfig()
    loaded_from = {}

    def load_config(path, trust_remote_code):
        loaded_from["path"] = path
        return checkpoint_config

    hf_class = types.SimpleNamespace(
        config_class=types.SimpleNamespace(from_pretrained=load_config)
    )
    passed = {}

    class FakeRBLNModel:
        @classmethod
        def get_hf_class(cls):
            return hf_class

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

    # vLLM's view of the model: a private config class carrying an
    # hf_overrides={"num_hidden_layers": 2} smoke-compile override.
    vllm_hf_config = types.SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        get_text_config=lambda: _TextConfig(num_hidden_layers=2),
    )
    obj = RBLNOptimumModelBase.__new__(RBLNOptimumModelBase)
    obj.model_config = types.SimpleNamespace(
        hf_config=vllm_hf_config,
        model="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        dtype=torch.float16,
        trust_remote_code=False,
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

    assert loaded_from["path"] == "Qwen/Qwen3-0.6B"
    assert passed["config"] is checkpoint_config
    assert passed["config"] is not vllm_hf_config
    assert checkpoint_config.text.num_hidden_layers == 2
