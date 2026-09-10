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


import json
from types import SimpleNamespace

import pytest
from vllm.engine import arg_utils
from vllm.engine.arg_utils import EngineArgs

from vllm_rbln.platform import RblnPlatform

FLAG = "allow_global_per_layer_attribute_access"


@pytest.fixture
def captured_model_config(monkeypatch):
    """Stub ModelConfig to record create_model_config kwargs without loading a model."""
    captured = {}

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(arg_utils, "ModelConfig", fake_model_config)
    return captured


def _model_dir(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(tmp_path)


def _create(model, **engine_kwargs):
    # Apply the injection exactly as pre_register_and_update does on the optimum path.
    RblnPlatform._allow_gemma4_global_per_layer_attribute_access()
    EngineArgs(model=model, **engine_kwargs).create_model_config()


def test_gemma4_gets_text_config_flag(tmp_path, captured_model_config):
    _create(_model_dir(tmp_path, "gemma4"))
    assert captured_model_config["hf_overrides"] == {"text_config": {FLAG: True}}


def test_other_model_types_are_untouched(tmp_path, captured_model_config):
    _create(_model_dir(tmp_path, "opt"))
    assert captured_model_config["hf_overrides"] == {}


def test_user_dict_overrides_are_merged(tmp_path, captured_model_config):
    user = {"architectures": ["X"], "text_config": {"sliding_window": 8}}
    _create(_model_dir(tmp_path, "gemma4"), hf_overrides=user)
    assert captured_model_config["hf_overrides"] == {
        "architectures": ["X"],
        "text_config": {"sliding_window": 8, FLAG: True},
    }
    assert user == {"architectures": ["X"], "text_config": {"sliding_window": 8}}, (
        "the caller's dict must not be mutated"
    )


def test_user_callable_override_is_composed(tmp_path, captured_model_config):
    seen = []

    def user_fn(config):
        seen.append(config)
        return config

    _create(_model_dir(tmp_path, "gemma4"), hf_overrides=user_fn)
    composed = captured_model_config["hf_overrides"]

    config = SimpleNamespace(text_config=SimpleNamespace())
    assert composed(config) is config
    assert seen == [config]
    assert getattr(config.text_config, FLAG) is True

    # vLLM probes hf_overrides_fn with a bare PretrainedConfig that has no text_config.
    bare = SimpleNamespace(model_type="dummy_gemma4")
    assert composed(bare).model_type == "dummy_gemma4"
