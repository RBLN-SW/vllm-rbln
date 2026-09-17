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


import pytest
from transformers import Gemma4Config, OPTConfig
from vllm.config import model as vllm_model_config

from vllm_rbln.platform import RblnPlatform


@pytest.fixture
def get_config(monkeypatch):
    """Serve fixed configs from ``vllm.config.model.get_config`` and apply the patch."""
    configs: dict[str, object] = {}
    monkeypatch.setattr(
        vllm_model_config, "get_config", lambda model, **kw: configs[model]
    )
    monkeypatch.setattr(
        vllm_model_config, "_rbln_gemma4_get_config_patched", False, raising=False
    )
    RblnPlatform._allow_gemma4_global_per_layer_attribute_access()

    def load(model, config):
        configs[model] = config
        return vllm_model_config.get_config(model)

    return load


def test_gemma4_text_config_allows_global_head_dim(get_config):
    config = get_config("gemma4", Gemma4Config())
    assert config.text_config.allow_global_per_layer_attribute_access is True
    assert isinstance(config.text_config.head_dim, int)


def test_other_model_types_are_untouched(get_config):
    config = get_config("opt", OPTConfig())
    assert "allow_global_per_layer_attribute_access" not in config.__dict__
