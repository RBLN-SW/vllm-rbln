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

from pathlib import Path

import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig


def test_platform_plugins():
    import runpy

    repo_root = Path(__file__).resolve().parents[3]
    example_file = (
        repo_root / "examples" / "experimental" / "offline_inference_basic.py"
    )
    runpy.run_path(str(example_file))

    # check if the plugin is loaded correctly
    from vllm.platforms import _init_trace, current_platform

    assert current_platform.plugin_name == "rbln", (
        f"Expected DummyDevice, got {current_platform.plugin_name}, "
        "possibly because current_platform is imported before the plugin"
        f" is loaded. The first import:\n{_init_trace}"
    )


def test_register_ops(vllm_config):
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        # Attention
        from vllm.model_executor.layers.attention.attention import Attention

        attention = Attention(16, 32, 0.125, 16, prefix="layer.0")
        assert hasattr(attention, "layer_index"), (
            f"Expected 'layer_index' in attention.__dict__, got {attention.__dict__}"
        )
        assert isinstance(attention.layer_index, int), (
            f"Expected 'layer_index' in attention.__dict__, got {attention.__dict__}"
        )
        assert attention.layer_index == 0, (
            f"Expected 'layer_index' in attention.__dict__, got {attention.__dict__}"
        )

        # RotaryEmbedding
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        rope = RotaryEmbedding(16, 16, 16, 16, True, torch.float16)
        assert "rope_forward_oot" in str(rope.__dict__["_forward_method"]), (
            f"Expected 'rope_forward_oot' in layer.__dict__['_forward_method'], \
                got {rope.__dict__['_forward_method']}"
        )
        assert isinstance(rope.get_buffer("cos_cache"), torch.Tensor), (
            f"Expected 'cos_cache' in buffer, got {rope.get_buffer('cos_cache')}"
        )
        assert isinstance(rope.get_buffer("sin_cache"), torch.Tensor), (
            f"Expected 'sin_cache' in buffer, got {rope.get_buffer('sin_cache')}"
        )


def test_get_attn_backend_cls():
    from vllm_rbln.platform import RblnPlatform

    attn_backend_cls = RblnPlatform.get_attn_backend_cls(
        AttentionBackendEnum.FLASH_ATTN,
        AttentionSelectorConfig(
            16,  # head_size
            torch.float16,  # dtype
            None,  # kv_cache_dtype
            1024,  # block_size
            False,  # use_mla
            False,  # has_sink
            False,  # use_sparse
            False,  # use_mm_prefix
            AttentionType.DECODER,  # attn_type
        ),
    )
    assert (
        attn_backend_cls
        == "vllm_rbln.v1.attention.backends.flash_attention.RBLNAttentionBackend"
    ), (
        f"Expected 'vllm_rbln.attention.backends.flash_attention.\
        RBLNAttentionBackend', got {attn_backend_cls}"
    )
