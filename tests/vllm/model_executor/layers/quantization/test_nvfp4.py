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

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import get_quantization_config

from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNModelOptNvFp4Config,
    RBLNModelOptNvFp4FusedMoE,
)
from vllm_rbln.patches import apply_registrations

# A legacy hf_quant_config.json, as vLLM normalizes it (quant_method added).
HF_QUANT_CONFIG = {
    "quant_method": "modelopt_fp4",
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "exclude_modules": ["model.layers.2.mlp.experts"],
    },
}


@pytest.fixture
def routed_experts() -> RoutedExperts:
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = MagicMock()
    return layer


def test_modelopt_fp4_routes_quantized_experts_to_the_rbln_method(routed_experts):
    apply_registrations()
    config_cls = get_quantization_config("modelopt_fp4")
    assert config_cls is RBLNModelOptNvFp4Config
    config = config_cls.from_config(HF_QUANT_CONFIG)

    method = config.get_quant_method(routed_experts, "model.layers.3.mlp.experts")
    assert isinstance(method, RBLNModelOptNvFp4FusedMoE)
    # An excluded layer stays unquantized: upstream falls back to its
    # unquantized MoE method when the config returns None.
    assert config.get_quant_method(routed_experts, "model.layers.2.mlp.experts") is None
