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

import vllm.model_executor.kernels.linear as linear
from vllm.platforms import PlatformEnum

from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (  # noqa: E501
    RBLNInt8UnpackedLinearKernel,
)

# vLLM 0.22 selects mixed-precision (WNA16) kernels from a per-platform registry
# (``_POSSIBLE_KERNELS``) keyed by ``PlatformEnum`` inside
# ``choose_mp_linear_kernel``. The RBLN platform is OOT and has no built-in
# entry, so the lookup raises ``KeyError: <PlatformEnum.OOT>``. Register our
# torch-native unpacked kernel for the OOT platform via the public API rather
# than monkeypatching ``choose_mp_linear_kernel``: callers such as
# ``compressed_tensors_wNa16`` import that function *by name* at module load, so
# a late attribute swap on the module would never be observed by them.
#
# ``choose_mp_linear_kernel`` only consults ``kernel.get_min_capability()`` when
# the platform reports a compute capability; RBLN's ``get_device_capability()``
# returns ``None``, so our kernel's ``NotImplementedError`` there is never hit.
if RBLNInt8UnpackedLinearKernel not in linear._POSSIBLE_KERNELS.get(
    PlatformEnum.OOT, []
):
    linear.register_linear_kernel(
        RBLNInt8UnpackedLinearKernel,
        PlatformEnum.OOT,
        kernel_type="mp",
    )
