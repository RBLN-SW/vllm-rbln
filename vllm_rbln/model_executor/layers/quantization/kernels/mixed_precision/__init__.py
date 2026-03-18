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

# NOTE: This module should be imported before creating model config,
# because the function we're patching is imported at that moment.

import vllm.envs as envs
import vllm.model_executor.kernels.linear as linear
from vllm.model_executor.kernels.linear.mixed_precision import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)

from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (  # noqa: E501
    RBLNInt8UnpackedLinearKernel,
)

choose_mp_linear_kernel_original = linear.choose_mp_linear_kernel

_POSSIBLE_KERNELS: list[type[MPLinearKernel]] = [
    RBLNInt8UnpackedLinearKernel,
]


def choose_mp_linear_kernel_rbln(
    config: MPLinearLayerConfig,
    compute_capability: int | None = None,
) -> type[MPLinearKernel]:
    from vllm.platforms import current_platform

    if "rbln" not in current_platform.get_device_name().lower():
        return choose_mp_linear_kernel_original(config, compute_capability)

    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS:
        if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel.__name__} disabled by environment variable"
            )
            continue

        can_implement, failure_reason = kernel.can_implement(config)
        if can_implement:
            return kernel
        else:
            failure_reasons.append(
                f" {kernel.__name__} cannot implement due to: {failure_reason}"
            )

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "WNA16 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


linear.choose_mp_linear_kernel = choose_mp_linear_kernel_rbln
