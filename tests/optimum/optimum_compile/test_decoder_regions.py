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
from optimum.rbln import RBLNAutoModelForCausalLM
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyLayer,
)
from optimum.rbln.transformers.models.gpt_oss.gpt_oss_architecture import (
    RBLNGptOssExperts,
)
from transformers import GptOssConfig, Qwen3Config

from vllm_rbln.model_executor.models.optimum.compilation import RBLNCompileSpec
from vllm_rbln.model_executor.models.optimum.compilation.decoder_regions import (
    RegionExperts,
    with_decoder_regions,
)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "config_cls, architecture",
    [(GptOssConfig, "GptOssForCausalLM"), (Qwen3Config, "Qwen3ForCausalLM")],
)
def test_decoder_region_model_selection(monkeypatch, enabled, config_cls, architecture):
    monkeypatch.setenv("VLLM_RBLN_NESTED_COMPILE_REGION", str(int(enabled)))
    config = config_cls(architectures=[architecture])
    spec = RBLNCompileSpec.for_architecture(
        config,
        batch_size=1,
        block_size=8192,
        max_model_len=8192,
        num_devices=1,
        memory_budget=0.9,
        prefill_chunk_size=128,
    )
    assert spec.model_cls is RBLNAutoModelForCausalLM


def test_region_experts_preserve_weights_without_copy():
    model = SimpleNamespace(
        num_experts=2,
        intermediate_size=64,
        hidden_size=64,
        alpha=1.702,
        limit=7.0,
        gate_up_proj_blocks=torch.randint(256, (2, 128, 2, 16), dtype=torch.uint8),
        gate_up_proj_scales=torch.randint(256, (2, 128, 2), dtype=torch.uint8),
        gate_up_proj_bias=torch.randn(2, 128),
        down_proj_blocks=torch.randint(256, (2, 64, 2, 16), dtype=torch.uint8),
        down_proj_scales=torch.randint(256, (2, 64, 2), dtype=torch.uint8),
        down_proj_bias=torch.randn(2, 64),
    )
    original = RBLNGptOssExperts(model, top_k=1)
    region = RegionExperts(original)
    for name, weight in original.named_buffers():
        actual = getattr(region, name)
        torch.testing.assert_close(actual, weight)
        assert actual.data_ptr() == weight.data_ptr()
        assert actual.stride() == weight.stride()
    assert len(list(region.buffers())) == 6


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_region_export_and_restoration(monkeypatch, enabled, fail):
    class Layer(DecoderOnlyLayer):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, x):
            return x.sin() + 1

    class OtherLayer(Layer):
        def forward(self, x):
            return x.cos() * 2

    model = torch.nn.Sequential(Layer(), OtherLayer(), Layer(), OtherLayer())
    x = torch.randn(2, 3)
    expected = model(x)
    monkeypatch.setenv("VLLM_RBLN_NESTED_COMPILE_REGION", str(int(enabled)))

    def compile(cls, model, **kwargs):
        assert kwargs.get("model_trace_method") == (
            "export_strict" if enabled else None
        )
        exported = torch.export.export(model, (x,), strict=True)
        calls = [n for n in exported.graph.nodes if "invoke_subgraph" in str(n.target)]
        assert len(calls) == (4 if enabled else 0)
        if enabled:
            assert calls[0].args[1] == calls[2].args[1]
            assert calls[1].args[1] == calls[3].args[1]
            assert calls[0].args[1] != calls[1].args[1]
        torch.testing.assert_close(exported.module()(x), expected)
        if fail:
            raise RuntimeError("compile failed")

    compiled = with_decoder_regions(compile)
    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            compiled(None, model)
    else:
        compiled(None, model)
    assert all("forward" not in layer.__dict__ for layer in model)
    assert [type(layer) for layer in model] == [Layer, OtherLayer, Layer, OtherLayer]
    torch.testing.assert_close(model(x), expected)


def test_platform_patch_is_idempotent():
    from optimum.rbln.modeling import RBLNModel
    from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
        RBLNDecoderOnlyModel,
    )

    from vllm_rbln.platform.optimum_impl import patch_upstream

    original = RBLNModel.compile.__func__
    patch_upstream()
    patched = RBLNDecoderOnlyModel.compile.__func__
    patch_upstream()
    assert RBLNDecoderOnlyModel.compile.__func__ is patched
    assert patched is not original
    assert RBLNModel.compile.__func__ is original
