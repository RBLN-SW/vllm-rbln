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
"""A Phi-3 whose sliding window covers the context runs as full attention,
matching the static cache optimum-rbln compiles for it."""

from types import SimpleNamespace

import pytest
from transformers import Phi3Config

from vllm_rbln.model_executor.models import optimum as optimum_models
from vllm_rbln.utils.optimum.predicates import is_phi3_full_attention


def _model_config(sliding_window: int | None, max_model_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        model="phi3",
        hf_config=Phi3Config(
            sliding_window=sliding_window, architectures=["Phi3ForCausalLM"]
        ),
        max_model_len=max_model_len,
        runner_type="generate",
    )


@pytest.mark.parametrize(
    ("sliding_window", "max_model_len", "expected"),
    [
        (262144, 8192, True),  # Phi-3.5-mini / Phi-4-mini
        (8192, 8192, True),
        (2047, 4096, False),  # Phi-3-mini-4k: the window is real
        (None, 16384, False),  # phi-4
    ],
)
def test_is_phi3_full_attention(sliding_window, max_model_len, expected):
    model_config = _model_config(sliding_window, max_model_len)
    assert is_phi3_full_attention(model_config) is expected


def test_load_model_runs_wide_window_phi3_as_full_attention(monkeypatch):
    class Loaded:
        def __init__(self, vllm_config):
            pass

        def eval(self):
            return self

    class FullAttention(Loaded):
        pass

    class SlidingWindow(Loaded):
        pass

    monkeypatch.setattr(optimum_models, "RBLNOptimumForCausalLM", FullAttention)
    monkeypatch.setattr(
        optimum_models, "RBLNOptimumSlidingWindowAttentionForCausalLM", SlidingWindow
    )
    vllm_config = SimpleNamespace(
        model_config=_model_config(262144, 8192),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    assert isinstance(optimum_models.load_model(vllm_config), FullAttention)
