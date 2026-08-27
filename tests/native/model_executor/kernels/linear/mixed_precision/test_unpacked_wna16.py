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

# Pure tensor math, no NPU. The device-side arithmetic is covered by
# test_unpacked_wna16_compile_parity.py.

import os
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.scalar_type import scalar_types

from vllm_rbln.model_executor.kernels.linear.mixed_precision.unpacked_wna16 import (
    RBLNUnpackedwNa16LinearKernel,
)

_UNPACK = (
    "vllm_rbln.model_executor.kernels.linear.mixed_precision."
    "unpacked_wna16.unpack_from_int32"
)


def _config(
    *,
    weight_type=scalar_types.uint8b128,
    group_size=-1,
    zero_points=False,
    has_g_idx=False,
    in_features=128,
    out_features=64,
):
    return MPLinearLayerConfig(
        # [in, out]; the kernel hands the transpose to the unpacker, and
        # swapping the two silently mis-shapes every weight.
        full_weight_shape=(in_features, out_features),
        partition_weight_shape=(in_features, out_features),
        weight_type=weight_type,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=has_g_idx,
    )


def _kernel(config):
    return RBLNUnpackedwNa16LinearKernel(
        config, w_q_param_name="weight_packed", w_s_param_name="weight_scale"
    )


@pytest.fixture(autouse=True)
def _contain_quant_bits(monkeypatch):
    # process_weights_after_loading writes RBLN_QUANT_BITS straight into
    # os.environ, which would otherwise leak into the rest of the session.
    monkeypatch.delenv("RBLN_QUANT_BITS", raising=False)


def _layer(w_q, w_s):
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(w_q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
    return layer


class TestCanImplement:
    @pytest.mark.parametrize(
        "weight_type", [scalar_types.uint4b8, scalar_types.uint8b128]
    )
    def test_accepts_the_two_symmetric_int_types(self, weight_type):
        assert RBLNUnpackedwNa16LinearKernel.can_implement(
            _config(weight_type=weight_type)
        ) == (True, None)

    def test_rejects_another_weight_type(self):
        ok, reason = RBLNUnpackedwNa16LinearKernel.can_implement(
            _config(weight_type=scalar_types.uint8)
        )
        assert not ok
        assert "not supported" in reason

    def test_rejects_asymmetric_quantization(self):
        ok, reason = RBLNUnpackedwNa16LinearKernel.can_implement(
            _config(zero_points=True)
        )
        assert not ok
        assert "Asymmetric" in reason

    @pytest.mark.parametrize("group_size", [-1, 64, 128])
    def test_accepts_the_supported_group_sizes(self, group_size):
        assert RBLNUnpackedwNa16LinearKernel.can_implement(
            _config(group_size=group_size)
        ) == (True, None)

    def test_rejects_another_group_size(self):
        ok, reason = RBLNUnpackedwNa16LinearKernel.can_implement(_config(group_size=32))
        assert not ok
        assert "group_size=32" in reason

    def test_rejects_activation_ordering(self):
        ok, reason = RBLNUnpackedwNa16LinearKernel.can_implement(
            _config(has_g_idx=True)
        )
        assert not ok
        assert "ordering" in reason


class TestProcessWeightsAfterLoading:
    @pytest.mark.parametrize(
        ("weight_type", "bits"),
        [(scalar_types.uint4b8, "4"), (scalar_types.uint8b128, "8")],
    )
    def test_publishes_the_bit_width_to_the_compiler(self, weight_type, bits):
        # The graph carries int8 either way, so this env var is the only thing
        # that tells rebel_compiler a weight is 4-bit.
        kernel = _kernel(_config(weight_type=weight_type))
        layer = _layer(torch.zeros(64, 32, dtype=torch.int32), torch.ones(64, 1))
        with patch(_UNPACK, return_value=torch.zeros(64, 128, dtype=torch.int8)):
            kernel.process_weights_after_loading(layer)
        assert os.environ["RBLN_QUANT_BITS"] == bits

    def test_unpacks_to_the_transpose_of_the_full_weight_shape(self):
        kernel = _kernel(_config(in_features=128, out_features=64))
        layer = _layer(torch.zeros(64, 32, dtype=torch.int32), torch.ones(64, 1))
        with patch(
            _UNPACK, return_value=torch.zeros(64, 128, dtype=torch.int8)
        ) as unpack:
            kernel.process_weights_after_loading(layer)
        _, num_bits, shape = unpack.call_args.args
        assert num_bits == 8
        assert shape == torch.Size((64, 128))

    @pytest.mark.parametrize(
        ("group_size", "expected_columns"), [(-1, 2), (64, 2), (128, 4)]
    )
    def test_only_a_128_group_is_split_into_two_64_groups(
        self, group_size, expected_columns
    ):
        # apply_weights always reads 64-wide groups.
        kernel = _kernel(_config(group_size=group_size))
        layer = _layer(
            torch.zeros(64, 32, dtype=torch.int32), torch.tensor([[1.0, 2.0]] * 64)
        )
        with patch(_UNPACK, return_value=torch.zeros(64, 128, dtype=torch.int8)):
            kernel.process_weights_after_loading(layer)
        assert layer.weight_scale.shape == (64, expected_columns)
        if group_size == 128:
            assert torch.equal(
                layer.weight_scale[0], torch.tensor([1.0, 1.0, 2.0, 2.0])
            )


class TestApplyWeights:
    def test_channelwise_scales_every_row_by_one_value(self):
        kernel = _kernel(_config(group_size=-1, in_features=128, out_features=64))
        w_q = torch.full((64, 128), 3, dtype=torch.int8)
        w_s = torch.arange(1, 65, dtype=torch.float16).reshape(64, 1)
        x = torch.ones(1, 128, dtype=torch.float16)
        out = kernel.apply_weights(_layer(w_q, w_s), x)
        # Each output is 128 terms of 3 * that row's scale.
        assert torch.allclose(out, (3 * 128 * w_s).reshape(1, 64))

    @pytest.mark.parametrize("group_size", [64, 128])
    def test_grouped_scales_each_64_wide_group_separately(self, group_size):
        # 64 and 128 differ only upstream: by the time apply_weights sees them
        # the scale already has one column per 64-wide group either way.
        kernel = _kernel(
            _config(group_size=group_size, in_features=128, out_features=64)
        )
        w_q = torch.full((64, 128), 2, dtype=torch.int8)
        w_s = torch.tensor([[1.0, 10.0]] * 64, dtype=torch.float16)
        x = torch.ones(1, 128, dtype=torch.float16)
        out = kernel.apply_weights(_layer(w_q, w_s), x)
        # 64 terms of 2*1 in the first group, 64 of 2*10 in the second.
        expected = torch.full((1, 64), 64 * 2.0 + 64 * 20.0, dtype=torch.float16)
        assert torch.allclose(out, expected)

    def test_bias_is_added(self):
        kernel = _kernel(_config(group_size=-1, in_features=128, out_features=64))
        layer = _layer(
            torch.zeros(64, 128, dtype=torch.int8),
            torch.ones(64, 1, dtype=torch.float16),
        )
        bias = torch.arange(64, dtype=torch.float16)
        out = kernel.apply_weights(layer, torch.ones(1, 128, dtype=torch.float16), bias)
        assert torch.equal(out, bias.reshape(1, 64))


class TestRegistration:
    def test_kernel_is_registered_for_the_oot_platform(self):
        from vllm.model_executor.kernels import linear
        from vllm.platforms import PlatformEnum

        assert (
            RBLNUnpackedwNa16LinearKernel in linear._POSSIBLE_KERNELS[PlatformEnum.OOT]
        )

    def test_kernel_is_the_one_chosen_for_a_w8a16_layer(self):
        # Being in the list is not the same as winning the pick. The config is
        # what RedHatAI/Qwen2.5-0.5B-quantized.w8a16 resolves to.
        from vllm.model_executor.kernels.linear import choose_mp_linear_kernel

        assert choose_mp_linear_kernel(_config()) is RBLNUnpackedwNa16LinearKernel
