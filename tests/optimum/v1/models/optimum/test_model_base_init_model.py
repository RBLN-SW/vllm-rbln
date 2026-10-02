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
"""Test config forwarding to optimum-rbln during compilation on a cache miss.

Configs defined in transformers are passed directly as objects. Other configs
are passed as kwargs containing ``num_hidden_layers`` and, when available,
``layer_types``, nested under ``text_config`` for composite models.
"""

from transformers import Gemma4Config, PretrainedConfig
from vllm.transformers_utils.configs import Qwen3ASRConfig


class FlatConfig(PretrainedConfig):
    """A decoder config defined outside transformers, with no sub-configs."""

    model_type = "flat_decoder_for_test"


def test_flat_config_passes_layer_count_as_top_level_kwargs(export_kwargs):
    hf_config = FlatConfig(
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
    )

    passed = export_kwargs(hf_config)

    assert "config" not in passed
    assert passed["num_hidden_layers"] == 2
    assert passed["layer_types"] == ["full_attention", "full_attention"]
    assert "text_config" not in passed


def test_composite_config_nests_layer_count_under_text_config(export_kwargs):
    # vLLM defines Qwen3-ASR's config, so the object itself does not go through.
    hf_config = Qwen3ASRConfig(
        architectures=["Qwen3ASRForConditionalGeneration"],
        text_config={"num_hidden_layers": 2},
    )

    passed = export_kwargs(hf_config)

    assert "config" not in passed
    assert passed["text_config"] == {"num_hidden_layers": 2}
    assert "num_hidden_layers" not in passed


def test_transformers_config_is_forwarded_as_the_config_object(export_kwargs):
    # gemma4's config carries per-layer state that the kwargs cannot reproduce,
    # and transformers defines the class, so the object itself goes through.
    hf_config = Gemma4Config(architectures=["Gemma4ForConditionalGeneration"])

    passed = export_kwargs(hf_config)

    assert passed["config"] is hf_config
    assert "text_config" not in passed
    assert "num_hidden_layers" not in passed
