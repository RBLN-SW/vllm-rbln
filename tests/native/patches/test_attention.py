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
import pytest
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8KVCacheMethod

from tests.native.vllm_config import make_vllm_config

# The fp8_e5m2 KV-cache gate in Attention.__init__: upstream keys it on the
# checkpoint format, the patch keys it on whether the checkpoint declares KV
# scales. Small config, as in test_flash_attention; nothing is loaded.

PREFIX = "model.layers.0.self_attn.attn"


def _e5m2_attention(quant_config) -> Attention:
    cfg = make_vllm_config(
        max_model_len=32, max_num_batched_tokens=8, kv_cache_dtype="fp8_e5m2"
    )
    with set_current_vllm_config(cfg):
        return Attention(
            num_heads=8,
            head_size=128,
            scale=1.0,
            cache_config=cfg.cache_config,
            quant_config=quant_config,
            prefix=PREFIX,
        )


def test_fp8_checkpoint_without_kv_scales_accepts_e5m2():
    # MiniMax-M2.7's shape: quant_method=fp8, no kv_cache_scheme. Upstream
    # rejects it for being fp8-format; the scales still get their loadable
    # parameters so a checkpoint that does carry them is not silently ignored.
    attn = _e5m2_attention(Fp8Config(is_checkpoint_fp8_serialized=True))

    assert attn.kv_cache_dtype == "fp8_e5m2"
    assert isinstance(attn.quant_method, Fp8KVCacheMethod)
    assert hasattr(attn, "k_scale") and hasattr(attn, "v_scale")


def test_checkpoint_declaring_kv_scales_rejects_e5m2():
    quant_config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="float-quantized",
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "symmetric": True,
        },
    )
    with pytest.raises(ValueError, match="fp8_e5m2"):
        _e5m2_attention(quant_config)
