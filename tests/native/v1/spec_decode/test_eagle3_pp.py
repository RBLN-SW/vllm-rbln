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

"""What the EAGLE3 pipeline handoff claims to support, and what it rejects.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Model

from vllm_rbln.platform import RblnPlatform
from vllm_rbln.v1.spec_decode.eagle3_pp import (
    EAGLE3_PP_TARGET_ARCHS,
    eagle3_aux_hidden_states_enabled,
)

SUPPORTED = "MiniMaxM2ForCausalLM"
UNSUPPORTED = "LlamaForCausalLM"

# Each allowlisted architecture, with the class whose patched `forward` carries the
# aux states across the split. Growing the allowlist has to touch this map, and the
# assertion below then bites if the patch itself is missing.
PATCHED_FORWARD_OWNERS = {SUPPORTED: MiniMaxM2Model}


def _config(arch: str, pp_size: int, method="eagle3", eagle_config=None):
    spec = (
        None
        if method is None
        else SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_config=eagle_config)
            ),
        )
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        speculative_config=spec,
    )


def test_the_allowlist_and_the_patched_forwards_agree():
    assert set(PATCHED_FORWARD_OWNERS) == set(EAGLE3_PP_TARGET_ARCHS)
    for arch, model_cls in PATCHED_FORWARD_OWNERS.items():
        assert model_cls.forward.__module__.startswith("vllm_rbln."), (
            f"{arch} is allowlisted for EAGLE3 under pipeline parallelism, but "
            f"{model_cls.__name__}.forward is still upstream's"
        )


def test_a_supported_target_is_accepted():
    RblnPlatform._validate_eagle3_pp_config(_config(SUPPORTED, 4))


def test_an_unsupported_target_is_rejected_at_startup():
    with pytest.raises(ValueError, match="EAGLE3 with pipeline_parallel_size"):
        RblnPlatform._validate_eagle3_pp_config(_config(UNSUPPORTED, 2))


def test_an_unsupported_target_passes_when_aux_is_off():
    # With `use_aux_hidden_state` off nothing is captured anywhere, so upstream's
    # unpatched forward is harmless and the split is fine. Rejecting this would
    # block a configuration that works.
    RblnPlatform._validate_eagle3_pp_config(
        _config(UNSUPPORTED, 2, eagle_config={"use_aux_hidden_state": False})
    )


@pytest.mark.parametrize("method", [None, "eagle", "ngram", "medusa"])
def test_only_eagle3_is_gated(method):
    RblnPlatform._validate_eagle3_pp_config(_config(UNSUPPORTED, 4, method=method))


@pytest.mark.parametrize(
    "eagle_config, expected",
    [
        (None, True),
        ({}, True),
        ({"use_aux_hidden_state": True}, True),
        ({"use_aux_hidden_state": False}, False),
        ("not-a-dict", True),
    ],
)
def test_the_aux_flag_reader_matches_the_draft_config(eagle_config, expected):
    # One reader for the runner and the guard: a non-last stage that disagrees with
    # the last one about this captures nothing and the drafter comes up short.
    spec = _config(SUPPORTED, 4, eagle_config=eagle_config).speculative_config

    assert eagle3_aux_hidden_states_enabled(spec) is expected


def test_a_non_eagle3_method_needs_no_aux():
    assert eagle3_aux_hidden_states_enabled(None) is False
    assert (
        eagle3_aux_hidden_states_enabled(
            _config(SUPPORTED, 4, method="eagle").speculative_config
        )
        is False
    )
