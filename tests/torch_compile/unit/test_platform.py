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
        == "vllm_rbln.v1.attention.backends.flash_attention.RBLNFlashAttentionBackend"
    ), (
        f"Expected 'vllm_rbln.attention.backends.flash_attention.\
        RBLNFlashAttentionBackend', got {attn_backend_cls}"
    )


# ---------------------------------------------------------------------------
# check_and_update_config: VLLM_RBLN_USE_DYNAMIC_KV_CACHE needs the vLLM-native
# path. The check has to sit outside the VLLM_RBLN_USE_VLLM_MODEL branch --
# `validate_and_setup_prerequisite` only runs inside it, so a guard placed there
# cannot see the optimum-rbln path this refuses.
# ---------------------------------------------------------------------------
def _thin_config():
    from types import SimpleNamespace

    # Enough to reach the flag check and no further; the method needs a real
    # VllmConfig to finish either branch.
    return SimpleNamespace(
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )


def test_dynamic_kv_without_the_vllm_model_path_is_refused(monkeypatch):
    import pytest

    from vllm_rbln.platform import RblnPlatform

    monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "1")
    monkeypatch.delenv("VLLM_RBLN_USE_VLLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="VLLM_RBLN_USE_VLLM_MODEL=1"):
        RblnPlatform.check_and_update_config(_thin_config())


def test_the_dynamic_kv_flag_check_does_not_overreach(monkeypatch):
    import pytest

    from vllm_rbln.platform import RblnPlatform

    # Both valid pairings must get past this check. The thin config cannot carry
    # either branch to the end, so assert only that this guard is not what
    # stopped it.
    for dynamic_kv, use_vllm_model in (("0", "0"), ("1", "1")):
        monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", dynamic_kv)
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", use_vllm_model)
        with pytest.raises(Exception) as excinfo:
            RblnPlatform.check_and_update_config(_thin_config())
        assert "VLLM_RBLN_USE_DYNAMIC_KV_CACHE" not in str(excinfo.value)
