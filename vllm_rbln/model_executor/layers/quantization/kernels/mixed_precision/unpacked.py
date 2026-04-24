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

import os

import torch
from compressed_tensors.compressors.quantized_compressors import unpack_from_int32
from vllm.model_executor.kernels.linear import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.scalar_type import scalar_types


class RBLNInt8UnpackedLinearKernel(MPLinearKernel):
    """
    Torch native implementation of mixed precision Linear, based on
    compressed_tensors' dequantize() function. rebel_compiler detects this
    pattern and maps it to the actual kernel.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type not in (scalar_types.uint4b8, scalar_types.uint8b128):
            return False, f"Weight type {c.weight_type} not supported"
        if c.zero_points:
            return False, "Asymmetric quantization not supported"
        if c.group_size not in (-1, 64, 128):
            return False, f"Group size {c.group_size} not supported"
        if c.has_g_idx:
            return False, "Group/dynamic activation ordering not supported"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        bits = c.weight_type.mantissa

        os.environ["RBLN_QUANT_BITS"] = str(bits)

        if c.has_g_idx:
            w_gidx = getattr(layer, self.w_gidx_name)
            layer.perm = torch.argsort(w_gidx)

        def transform_w_q(x: BasevLLMParameter):
            in_features, out_features = c.full_weight_shape
            x.data = unpack_from_int32(
                x.data, bits, torch.Size((out_features, in_features))
            )
            if c.has_g_idx:
                x.data = x.data[:, layer.perm]
            return x

        def transform_w_s(x: BasevLLMParameter):
            if c.group_size == 128:
                # Currently we only support group size 64 natively. So
                # duplicate scale to break a group into two groups of size 64.
                x.data = x.data.repeat_interleave(2, dim=-1)
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        in_features, out_features = self.config.full_weight_shape

        if self.config.has_g_idx:
            x = x[..., layer.perm]

        w_q, w_s, _, _ = self._get_weight_params(layer)
        if self.config.group_size > 0:
            w_q = w_q.view(out_features, in_features // 64, 64)  # see transform_w_s
            w_fp = w_q.type(x.dtype) * w_s.unsqueeze(-1)
            w_fp = w_fp.view(out_features, in_features)
        else:
            w_fp = w_q.type(x.dtype) * w_s

        return torch.nn.functional.linear(x, w_fp, bias)
