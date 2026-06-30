# Copyright 2025 Rebellions Inc. All rights reserved.
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

import vllm.model_executor.kernels.linear as linear
from vllm.platforms import PlatformEnum

from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (  # noqa: E501
    RBLNInt8UnpackedLinearKernel,
)

if RBLNInt8UnpackedLinearKernel not in linear._POSSIBLE_KERNELS.get(
    PlatformEnum.OOT, []
):
    linear.register_linear_kernel(
        RBLNInt8UnpackedLinearKernel,
        PlatformEnum.OOT,
        kernel_type="mp",
    )
