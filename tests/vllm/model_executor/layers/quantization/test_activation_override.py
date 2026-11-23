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

from types import SimpleNamespace

import pytest
from vllm.config import set_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec

from vllm_rbln.model_executor.layers.quantization.activation_override import (
    activation_is_fp8,
    linear_w8a8_in_scope,
    validate_activation_override,
)
from vllm_rbln.patches import fp8_linear_method

W8A8 = {"activation": "fp8_per_block_dynamic"}
QKV = "model.layers.0.self_attn.qkv_proj"
O_PROJ = "model.layers.0.self_attn.o_proj"
EXPERTS = "model.layers.3.mlp.experts"
FUSED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


def _args(**kwargs) -> QuantizationConfigArgs:
    return QuantizationConfigArgs(**kwargs)


@pytest.mark.parametrize(
    "args, expected",
    [
        (None, (False, False, False)),
        (_args(linear=W8A8), (True, True, False)),
        (_args(moe=W8A8), (False, False, True)),
        (_args(linear=W8A8, moe=W8A8), (True, True, True)),
        (_args(linear=W8A8, ignore=["*o_proj"]), (True, False, False)),
        (_args(linear={"activation": None}), (False, False, False)),
    ],
)
def test_linear_and_moe_overrides(args, expected):
    got = (
        activation_is_fp8(args, QKV, "linear", FUSED),
        activation_is_fp8(args, O_PROJ, "linear", FUSED),
        activation_is_fp8(args, EXPERTS, "moe"),
    )
    assert got == expected


def test_targets_pick_per_layer():
    # vLLM 0.31 only builds shorthand targets; a QuantSpec value is what newer
    # upstream accepts, so build the args the way it will.
    args = SimpleNamespace(
        linear=None,
        moe=None,
        ignore=[],
        targets={
            "re:.*qkv_proj": QuantSpec(**W8A8),
            "re:.*o_proj": QuantSpec(activation=None),
        },
    )
    assert activation_is_fp8(args, QKV, "linear", FUSED)
    assert not activation_is_fp8(args, O_PROJ, "linear", FUSED)
    assert not activation_is_fp8(args, EXPERTS, "moe")


@pytest.mark.parametrize(
    "args, quantization, match",
    [
        (_args(linear="fp8_per_channel"), "fp8", "no `weight`"),
        (_args(linear={"activation": "fp8_per_token"}), "fp8", "not supported"),
        (_args(linear=W8A8), "modelopt", "FP8 block checkpoints"),
        (_args(linear=W8A8), None, "FP8 block checkpoints"),
    ],
)
def test_refused(args, quantization, match):
    with pytest.raises(ValueError, match=match):
        validate_activation_override(args, quantization)


@pytest.mark.parametrize("args", [None, _args(linear=W8A8, moe={"activation": None})])
def test_accepted(args):
    validate_activation_override(args, "fp8")


@pytest.mark.parametrize(
    "prefix, w8a8", [(QKV, True), (O_PROJ, False)], ids=["qkv", "o_proj"]
)
def test_create_weights_chooses_the_kernel_per_layer(monkeypatch, prefix, w8a8):
    seen = []
    monkeypatch.setattr(
        fp8_linear_method,
        "_original_create_weights",
        lambda self, layer, *a, **kw: seen.append(linear_w8a8_in_scope()),
    )
    method = SimpleNamespace(quant_config=SimpleNamespace(packed_modules_mapping=FUSED))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization_config=_args(linear=W8A8, ignore=["*o_proj"])
        )
    )
    with set_current_vllm_config(config):
        fp8_linear_method.rbln_fp8_linear_create_weights(
            method, SimpleNamespace(prefix=prefix)
        )
    assert seen == [w8a8]
    assert linear_w8a8_in_scope() is False
