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

"""RBLN-native RMSNorm that bypasses vLLM's ``ir.ops`` layer.

vLLM 0.22 routes ``RMSNorm.forward_native`` through ``ir.ops.rms_norm`` /
``ir.ops.fused_add_rms_norm.maybe_inplace``. Under the RBLN torch.compile /
rebel-compiler path that indirection produces a single graph output for the
fused-add op, so the model code's ``x, residual = norm(x, residual)`` unpack
fails with "not enough values to unpack (expected 2, got 1)". The ``ir`` layer
is also slated for deprecation upstream.

We override ``RMSNorm`` to compute the normalization directly with plain torch
ops (the exact same math rebel-compiler pattern-matches), returning a single
tensor when there is no residual and a ``(out, residual)`` tuple otherwise.
"""

import torch
import vllm.model_executor.layers.layernorm as upstream


def _rbln_rms_norm_forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.to(torch.float32)

    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    var_size = self.variance_size_override
    x_var = x if var_size is None else x[..., :var_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)

    if self.weight is not None:
        x = x.to(self.weight.dtype) * self.weight
    x = x.to(orig_dtype)

    if residual is None:
        return x
    return x, residual


# Patch both the native and OOT entry points. RBLN is an out-of-tree platform,
# so CustomOp dispatch selects forward_oot when the op is enabled
# (custom_ops="all"); forward_native covers the not-enabled fallback. Both now
# avoid ir.ops entirely.
upstream.RMSNorm.forward_native = _rbln_rms_norm_forward
upstream.RMSNorm.forward_oot = _rbln_rms_norm_forward
