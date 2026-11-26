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

"""skt/A.X-K2-NVFP4 as compressed-tensors (hub commit e1310bba45 onward)."""

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)

from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNCompressedTensorsW4A4Nvfp4MoEMethod,
)

# The checkpoint's quantization_config, trimmed to what dispatch reads.
_NVFP4 = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "tensor_group",
    "group_size": 16,
    "dynamic": False,
}
AXK2_NVFP4_QCFG = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*\\.experts\\..*"],
            "weights": _NVFP4,
            "input_activations": _NVFP4,
        },
        "group_1": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "block",
                "block_structure": [128, 128],
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        },
    },
    "ignore": ["lm_head", "re:.*\\.mlp\\.gate$"],
}

E, HIDDEN, INTER = 2, 32, 16


def _method() -> RBLNCompressedTensorsW4A4Nvfp4MoEMethod:
    cfg = CompressedTensorsConfig.from_config(AXK2_NVFP4_QCFG)
    layer = SimpleNamespace(moe_config=SimpleNamespace(is_act_and_mul=True))
    method = CompressedTensorsMoEMethod.get_moe_method(
        cfg, layer, "model.layers.1.mlp.experts"
    )
    assert isinstance(method, RBLNCompressedTensorsW4A4Nvfp4MoEMethod)
    return method


def _loaded_layer(method) -> torch.nn.Module:
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=E,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    gen = torch.Generator().manual_seed(0)
    for p in layer.parameters():
        if p.dtype == torch.uint8:
            p.data = torch.randint(0, 256, p.shape, dtype=torch.uint8, generator=gen)
        else:
            p.data = (torch.rand(p.shape, generator=gen) + 0.5).to(p.dtype)
    return layer


def test_dispatch_picks_the_rbln_method_instead_of_raising():
    # Upstream raises "No NvFp4 MoE backend supports the deployment
    # configuration." here.
    method = _method()
    assert method.group_size == 16
    assert method.is_monolithic
    assert method.get_fused_moe_quant_config(None) is None


def test_loaded_tensors_take_the_modelopt_names_and_scale_2():
    method = _method()
    layer = _loaded_layer(method)
    packed13 = layer.w13_weight_packed.data.clone()
    packed2 = layer.w2_weight_packed.data.clone()
    block13 = layer.w13_weight_scale.data.clone()
    gscale13 = layer.w13_weight_global_scale.data.clone()
    gscale2 = layer.w2_weight_global_scale.data.clone()

    method.process_weights_after_loading(layer)

    assert torch.equal(layer.w13_weight, packed13)
    assert torch.equal(layer.w2_weight, packed2)
    assert torch.equal(
        layer.w13_weight_scale.view(torch.uint8), block13.view(torch.uint8)
    )
    # compressed-tensors stores the reciprocal of ModelOpt's weight_scale_2,
    # and the gate and up halves keep their own.
    assert layer.w13_weight_scale_2.shape == (E, 2)
    torch.testing.assert_close(layer.w13_weight_scale_2, 1.0 / gscale13)
    torch.testing.assert_close(layer.w2_weight_scale_2, 1.0 / gscale2)
    for gone in (
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_global_scale",
        "w2_weight_global_scale",
    ):
        assert not hasattr(layer, gone)
