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
from typing import Any
from unittest.mock import patch

import pytest
import torch

from vllm_rbln.model_executor.layers.quantization.modelopt_fp8 import (
    RBLNModelOptFp8LinearMethod,
)
from vllm_rbln.model_executor.layers.quantization.modelopt_mxfp8 import (
    RBLNModelOptMxFp8LinearMethod,
)

IN_FEATURES = 4
PARTITIONS = [2, 3]


@pytest.fixture(autouse=True)
def _single_rank():
    # BasevLLMParameter reads the TP group at construction; these tests build
    # parameters outside a distributed environment.
    mod = "vllm.model_executor.parameter"
    with (
        patch(f"{mod}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{mod}.get_tensor_model_parallel_world_size", return_value=1),
    ):
        yield


def _method(serialized: bool = True) -> Any:
    return RBLNModelOptFp8LinearMethod(
        SimpleNamespace(is_checkpoint_fp8_serialized=serialized)
    )


def _built_layer(
    serialized: bool = True, partitions: list[int] = PARTITIONS
) -> torch.nn.Module:
    """A layer carrying exactly the parameters create_weights registers."""
    layer = torch.nn.Module()
    _method(serialized).create_weights(
        layer,
        input_size_per_partition=IN_FEATURES,
        output_partition_sizes=partitions,
        input_size=IN_FEATURES,
        output_size=sum(partitions),
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    return layer


class TestCreateWeights:
    def test_registers_fp8_weight_and_per_partition_scales(self):
        layer = _built_layer()
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert tuple(layer.weight.shape) == (sum(PARTITIONS), IN_FEATURES)
        # One scale per fused partition, so differing halves stay distinguishable.
        assert tuple(layer.weight_scale.shape) == (len(PARTITIONS),)
        assert tuple(layer.input_scale.shape) == (len(PARTITIONS),)

    def test_scales_start_at_the_float32_sentinel(self):
        layer = _built_layer()
        sentinel = torch.finfo(torch.float32).min
        assert bool((layer.weight_scale == sentinel).all())
        assert bool((layer.input_scale == sentinel).all())

    def test_unserialized_checkpoint_keeps_params_dtype_and_no_scales(self):
        layer = _built_layer(serialized=False)
        assert layer.weight.dtype == torch.bfloat16
        assert not hasattr(layer, "weight_scale")
        assert not hasattr(layer, "input_scale")

    def test_records_the_shapes_apply_needs(self):
        layer = _built_layer()
        assert layer.logical_widths == PARTITIONS
        assert layer.input_size_per_partition == IN_FEATURES
        assert layer.output_size_per_partition == sum(PARTITIONS)
        assert layer.orig_dtype == torch.bfloat16


class TestProcessWeightsAfterLoading:
    def test_uniform_scales_are_kept_per_partition(self):
        layer = _built_layer()
        layer.weight_scale.data = torch.tensor([0.5, 0.5])
        layer.input_scale.data = torch.tensor([0.25, 0.25])
        _method().process_weights_after_loading(layer)
        assert tuple(layer.weight_scale.shape) == (len(PARTITIONS),)
        assert layer.weight_scale.tolist() == [0.5, 0.5]

    def test_differing_scales_are_kept_per_partition(self):
        # The whole point of this method: upstream requantises both halves to
        # one max scale, which is lossy. Keep them instead.
        layer = _built_layer()
        layer.weight_scale.data = torch.tensor([0.5, 2.0])
        layer.input_scale.data = torch.tensor([0.25, 0.25])
        _method().process_weights_after_loading(layer)
        assert layer.weight_scale.tolist() == [0.5, 2.0]

    def test_input_scale_collapses_to_the_max(self):
        layer = _built_layer()
        layer.weight_scale.data = torch.tensor([0.5, 0.5])
        layer.input_scale.data = torch.tensor([0.25, 4.0])
        _method().process_weights_after_loading(layer)
        assert float(layer.input_scale) == 4.0


class TestApply:
    def _prepared(
        self, weight_scale: list[float], partitions: list[int] = PARTITIONS
    ) -> torch.nn.Module:
        layer = _built_layer(partitions=partitions)
        layer.weight.data = torch.ones(sum(partitions), IN_FEATURES).to(
            torch.float8_e4m3fn
        )
        layer.weight_scale.data = torch.tensor(weight_scale)
        layer.input_scale.data = torch.ones(len(partitions))
        _method().process_weights_after_loading(layer)
        return layer

    def test_single_partition_scale_dequantises_the_whole_weight(self):
        layer = self._prepared([0.5], partitions=[sum(PARTITIONS)])
        assert layer.weight_scale.ndim == 0
        x = torch.ones(1, IN_FEATURES)
        out = _method().apply(layer, x)
        # Every weight entry is 1.0 * 0.5, summed over IN_FEATURES ones.
        assert torch.allclose(out, torch.full_like(out, 0.5 * IN_FEATURES))

    def test_per_partition_scale_is_applied_to_its_own_rows(self):
        layer = self._prepared([0.5, 2.0])
        x = torch.ones(1, IN_FEATURES)
        out = _method().apply(layer, x)
        expected = torch.tensor(
            [[0.5 * IN_FEATURES] * PARTITIONS[0] + [2.0 * IN_FEATURES] * PARTITIONS[1]]
        )
        assert torch.allclose(out, expected)

    def test_bias_is_added(self):
        # apply() computes in bfloat16, so the bias has to arrive in it too.
        layer = self._prepared([0.5, 0.5])
        x = torch.ones(1, IN_FEATURES)
        bias = torch.arange(sum(PARTITIONS), dtype=torch.bfloat16)
        out = _method().apply(layer, x, bias=bias)
        assert torch.allclose(out, 0.5 * IN_FEATURES + bias.float())

    def test_output_returns_to_the_input_dtype(self):
        layer = self._prepared([0.5, 0.5])
        x = torch.ones(1, IN_FEATURES, dtype=torch.float16)
        assert _method().apply(layer, x).dtype == torch.float16


class TestRegistration:
    def test_fp8_builder_is_installed_in_the_upstream_hook(self):
        # The suite conftest's patch application fills LINEAR_METHOD_BUILDERS,
        # which build_linear_method consults before the generic method. Both the
        # plain FP8 config and the mixed-precision config dispatch through it.
        from vllm.model_executor.layers.quantization.modelopt import (
            LINEAR_METHOD_BUILDERS,
            build_linear_method,
        )

        assert "FP8" in LINEAR_METHOD_BUILDERS
        built = build_linear_method(
            SimpleNamespace(is_checkpoint_fp8_serialized=True), "FP8", "model.layer"
        )
        assert isinstance(built, RBLNModelOptFp8LinearMethod)


class TestModelOptMxFp8:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_loaded_block_scales_and_bias_determine_output(self, dtype):
        from vllm.model_executor.layers.quantization.modelopt import build_linear_method
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        method = build_linear_method(SimpleNamespace(), "MXFP8", "model.layer")
        assert isinstance(method, RBLNModelOptMxFp8LinearMethod)
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[1, 1],
            input_size=64,
            output_size=2,
            params_dtype=dtype,
            weight_loader=default_weight_loader,
        )
        layer.weight.weight_loader(
            layer.weight,
            torch.tensor([[1.0] * 64, [2.0] * 64]).to(torch.float8_e4m3fn),
        )
        layer.weight_scale.weight_loader(
            layer.weight_scale,
            torch.tensor([[126, 129], [128, 127]], dtype=torch.uint8),
        )
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            layer.weight_scale,
            torch.tensor([[0.5, 4.0], [2.0, 1.0]], dtype=torch.bfloat16),
        )
        x = torch.tensor([[1.0] * 64, [-1.0] * 64], dtype=dtype)
        output = method.apply(layer, x, bias=torch.tensor([1.0, -2.0], dtype=dtype))
        torch.testing.assert_close(
            output, torch.tensor([[145.0, 190.0], [-143.0, -194.0]], dtype=dtype)
        )

    def test_rejects_partial_scale_groups(self):
        with pytest.raises(ValueError, match="multiple of the group size"):
            RBLNModelOptMxFp8LinearMethod().create_weights(
                torch.nn.Module(),
                input_size_per_partition=33,
                output_partition_sizes=[2],
                input_size=33,
                output_size=2,
                params_dtype=torch.bfloat16,
            )
