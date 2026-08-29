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

# Pure tensor math, no NPU: per-token-group activation quant, block-scaled weight
# dequant, the shape/env guards, and the W8A8/W8A16 forward composition. The base
# can_implement is stubbed so only the RBLN guards decide.

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.kernels.linear.scaled_mm import Fp8BlockScaledMMLinearKernel

import vllm_rbln.envs as envs
from vllm_rbln.model_executor.kernels.linear.block_fp8 import (
    RBLNW8A8BlockFp8LinearKernel,
    RBLNW8A16BlockFp8LinearKernel,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _config(*, block_n, block_k, out=8, in_features=8):
    # Minimal stand-in for FP8ScaledMMLinearLayerConfig: only the fields
    # check_shape reads (the block group shape + weight shape).
    return SimpleNamespace(
        weight_quant_key=SimpleNamespace(
            scale=SimpleNamespace(group_shape=SimpleNamespace(row=block_n, col=block_k))
        ),
        weight_shape=(out, in_features),
    )


def _base_can_implement_ok():
    # Stub the upstream base check so only the RBLN guards decide the outcome.
    return patch.object(
        Fp8BlockScaledMMLinearKernel,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )


def _w8a16(block_n=2, block_k=2) -> Any:
    kernel: Any = object.__new__(RBLNW8A16BlockFp8LinearKernel)
    kernel.weight_group_shape = (block_n, block_k)
    return kernel


def _w8a8(block_n=2, block_k=2, out_dtype=torch.float32) -> Any:
    kernel: Any = object.__new__(RBLNW8A8BlockFp8LinearKernel)
    kernel.weight_group_shape = (block_n, block_k)
    kernel.config = SimpleNamespace(out_dtype=out_dtype)
    return kernel


def _patch_params(params):
    # Fake the base kernel's layer-param lookup so the forward path needs no layer.
    return patch.object(
        RBLNW8A8BlockFp8LinearKernel,
        "_get_layer_params",
        lambda self, layer: params,
    )


class TestPerTokenGroupQuantFp8:
    def test_scale_is_group_amax_over_fp8_max(self):
        # Each contiguous group of `group_size` is scaled by its own abs-max.
        x = torch.tensor([[2.0, -4.0, 1.0, 3.0]])  # groups [2,-4] and [1,3]
        _, scale = RBLNW8A8BlockFp8LinearKernel._per_token_group_quant_fp8(x, 2)
        assert scale.flatten().tolist() == pytest.approx([4 / FP8_MAX, 3 / FP8_MAX])

    def test_quantized_values_stay_in_fp8_range(self):
        x = torch.tensor([[2.0, -4.0, 1.0, 3.0]])
        q, _ = RBLNW8A8BlockFp8LinearKernel._per_token_group_quant_fp8(x, 2)
        assert q.abs().max().item() <= FP8_MAX

    def test_scale_has_one_entry_per_group(self):
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        _, scale = RBLNW8A8BlockFp8LinearKernel._per_token_group_quant_fp8(x, 3)
        assert scale.shape == (2, 2)  # 6 // group_size 3 = 2 groups per row

    def test_quant_then_rescale_recovers_input(self):
        # q * scale (broadcast back over the group) reconstructs x for values that
        # are representable -- the fake-quant round-trip.
        x = torch.tensor([[2.0, -4.0, 1.0, 3.0]])
        q, scale = RBLNW8A8BlockFp8LinearKernel._per_token_group_quant_fp8(x, 2)
        recovered = q * scale.repeat_interleave(2, dim=-1)
        assert recovered.flatten().tolist() == pytest.approx(x.flatten().tolist())

    def test_all_zero_group_avoids_zero_scale(self):
        # amax is clamped to eps, so an all-zero group gets a tiny positive scale
        # (no divide-by-zero) and quantizes to zeros.
        q, scale = RBLNW8A8BlockFp8LinearKernel._per_token_group_quant_fp8(
            torch.zeros(1, 4), 2
        )
        assert q.flatten().tolist() == [0.0, 0.0, 0.0, 0.0]
        assert (scale > 0).all()


class TestDequantizeBlockFp8Weight:
    def test_block_scale_spans_output_rows_and_input_cols(self):
        # block_n=2 -> each scale row covers 2 output rows; block_k=2 -> each scale
        # col covers 2 input cols. With a ones weight, the output is the scale grid.
        kernel = _w8a16(2, 2)
        scale = torch.tensor([[10.0, 100.0], [20.0, 200.0]])
        out = kernel._dequantize_block_fp8_weight(
            torch.ones(4, 4), scale, torch.float32
        )
        assert out.tolist() == [
            [10.0, 10.0, 100.0, 100.0],
            [10.0, 10.0, 100.0, 100.0],
            [20.0, 20.0, 200.0, 200.0],
            [20.0, 20.0, 200.0, 200.0],
        ]

    def test_output_keeps_weight_shape(self):
        kernel = _w8a16(2, 2)
        out = kernel._dequantize_block_fp8_weight(
            torch.ones(4, 4), torch.ones(2, 2), torch.float32
        )
        assert out.shape == (4, 4)


class TestApplyBlockScaledMm:
    def test_w8a16_dequantizes_weight_and_runs_linear(self):
        # W8A16 uses the activation directly (As is discarded); only the weight is
        # block-dequantized before the matmul.
        kernel = _w8a16(2, 2)
        activation = torch.ones(1, 4)
        weight = torch.ones(4, 4)
        weight_scale = torch.full((2, 2), 2.0)  # dequant weight -> all 2.0
        out = kernel.apply_block_scaled_mm(activation, weight, As=None, Bs=weight_scale)
        # linear(ones[1,4], 2*ones[4,4]) -> each output = 4 * 1 * 2 = 8.
        assert out.flatten().tolist() == pytest.approx([8.0, 8.0, 8.0, 8.0])


class TestApplyWeights:
    def test_w8a8_quantizes_activation_dequantizes_weight_then_linear(self):
        kernel = _w8a8(2, 2)
        params = SimpleNamespace(
            weight=torch.ones(4, 4),
            weight_scale=torch.ones(2, 2),
            weight_scale_inv=None,
        )
        x = torch.tensor([[2.0, -4.0, 1.0, 3.0]])
        with _patch_params(params):
            out = kernel.apply_weights(layer=object(), x=x, bias=None)
        # activation round-trips to ~x, weight dequants to ones -> each output col
        # is sum(x) = 2.0.
        assert out.flatten().tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])

    def test_weight_scale_inv_takes_precedence_over_weight_scale(self):
        kernel = _w8a8(2, 2)
        params = SimpleNamespace(
            weight=torch.ones(4, 4),
            weight_scale=torch.ones(2, 2),
            weight_scale_inv=torch.full((2, 2), 3.0),
        )
        x = torch.ones(1, 4)
        with _patch_params(params):
            out = kernel.apply_weights(layer=object(), x=x, bias=None)
        # weight dequant uses scale_inv=3 -> weight 3.0; linear = 4 * 1 * 3 = 12.
        assert out.flatten().tolist() == pytest.approx([12.0, 12.0, 12.0, 12.0])

    def test_bias_is_added(self):
        kernel = _w8a8(2, 2)
        params = SimpleNamespace(
            weight=torch.zeros(4, 4),
            weight_scale=torch.ones(2, 2),
            weight_scale_inv=None,
        )
        bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with _patch_params(params):
            out = kernel.apply_weights(layer=object(), x=torch.ones(1, 4), bias=bias)
        # zero weight -> linear is 0, so the output is exactly the bias.
        assert out.flatten().tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


class TestCheckShape:
    def test_propagates_base_rejection(self):
        # If the upstream base check fails, check_shape returns its verdict as-is
        # before running any RBLN-specific block guard.
        with patch.object(
            Fp8BlockScaledMMLinearKernel,
            "can_implement",
            classmethod(lambda cls, config: (False, "base says no")),
        ):
            ok, reason = RBLNW8A16BlockFp8LinearKernel.check_shape(
                _config(block_n=4, block_k=4)
            )
        assert (ok, reason) == (False, "base says no")

    def test_accepts_valid_block_shape(self):
        with _base_can_implement_ok():
            ok, reason = RBLNW8A16BlockFp8LinearKernel.check_shape(
                _config(block_n=4, block_k=4)
            )
        assert (ok, reason) == (True, None)

    def test_rejects_nonpositive_block(self):
        with _base_can_implement_ok():
            ok, reason = RBLNW8A16BlockFp8LinearKernel.check_shape(
                _config(block_n=0, block_k=4)
            )
        assert ok is False
        assert reason is not None and "positive block size" in reason

    def test_rejects_input_features_not_divisible_by_block_k(self):
        with _base_can_implement_ok():
            ok, reason = RBLNW8A16BlockFp8LinearKernel.check_shape(
                _config(block_n=4, block_k=3, in_features=8)
            )
        assert ok is False
        assert reason is not None and "divisible" in reason


class TestCanImplement:
    def test_w8a16_applies_unless_w8a8_requested(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_USE_W8A8", False)
        with _base_can_implement_ok():
            ok, _ = RBLNW8A16BlockFp8LinearKernel.can_implement(
                _config(block_n=4, block_k=4)
            )
        assert ok is True

    def test_w8a16_rejected_when_w8a8_requested(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_USE_W8A8", True)
        with _base_can_implement_ok():
            ok, reason = RBLNW8A16BlockFp8LinearKernel.can_implement(
                _config(block_n=4, block_k=4)
            )
        assert ok is False
        assert reason is not None and "W8A16" in reason

    def test_w8a8_applies_when_requested(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_USE_W8A8", True)
        with _base_can_implement_ok():
            ok, _ = RBLNW8A8BlockFp8LinearKernel.can_implement(
                _config(block_n=4, block_k=4)
            )
        assert ok is True

    def test_w8a8_rejected_by_default(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_USE_W8A8", False)
        with _base_can_implement_ok():
            ok, reason = RBLNW8A8BlockFp8LinearKernel.can_implement(
                _config(block_n=4, block_k=4)
            )
        assert ok is False
        assert reason is not None and "W8A8" in reason


class TestIsSupported:
    def test_always_supported(self):
        assert RBLNW8A16BlockFp8LinearKernel.is_supported() == (True, None)


class TestRegistration:
    def test_block_fp8_kernels_registered_for_oot_platform(self):
        # The native conftest's plugin load inserts both RBLN block-fp8 kernels
        # into the OOT slot of vLLM's fp8-block-kernel registry.
        from vllm.model_executor.kernels import linear
        from vllm.platforms import PlatformEnum

        oot = linear._POSSIBLE_FP8_BLOCK_KERNELS.get(PlatformEnum.OOT, [])
        assert RBLNW8A16BlockFp8LinearKernel in oot
        assert RBLNW8A8BlockFp8LinearKernel in oot
