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

"""Gemma4's attention backend choice on RBLN.

Config objects only -- no checkpoint, no device.
"""

from types import SimpleNamespace

from vllm.model_executor.models.config import Gemma4Config
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_rbln.patches.gemma4 import (
    keep_the_rbln_attention_backend,
    upstream_verify_and_update_config,
)


class _Layers(list):
    """`model_arch_config` as the hook reads it: indexable per layer."""

    @property
    def total_num_hidden_layers(self) -> int:
        return len(self)


def _vllm_config() -> SimpleNamespace:
    """One sliding (256) and one full (512) layer, Gemma4's head sizes."""
    layers = _Layers([SimpleNamespace(head_size=256), SimpleNamespace(head_size=512)])
    model_config = SimpleNamespace(
        model_arch_config=layers,
        hf_text_config=SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"]
        ),
    )
    return SimpleNamespace(
        model_config=model_config,
        attention_config=SimpleNamespace(backend=None, flash_attn_version=None),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def test_upstream_forces_triton_for_mixed_head_sizes():
    # Why the patch exists: without FA4 upstream picks TRITON_ATTN, which the
    # RBLN platform refuses. If upstream stops, the patch has lost its reason.
    config = _vllm_config()
    upstream_verify_and_update_config(config)
    assert config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


def test_the_backend_is_left_to_the_platform():
    config = _vllm_config()
    Gemma4Config.verify_and_update_config(config)
    assert config.attention_config.backend is None


def test_the_patch_is_the_one_installed():
    assert Gemma4Config.verify_and_update_config is keep_the_rbln_attention_backend


def test_the_rbln_model_serves_the_architecture():
    from vllm.model_executor.models import ModelRegistry

    from vllm_rbln.patches.gemma4 import register_gemma4

    register_gemma4()

    model = ModelRegistry.models["Gemma4ForConditionalGeneration"]
    assert (model.module_name, model.class_name) == (
        "vllm_rbln.model_executor.models.gemma4",
        "RBLNGemma4ForConditionalGeneration",
    )
