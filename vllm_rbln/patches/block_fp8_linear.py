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

from vllm.model_executor.kernels import linear
from vllm.platforms import PlatformEnum

from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.kernels.linear.block_fp8 import (
    RBLNW8A16BlockFp8LinearKernel,
)
from vllm_rbln.patches import add_registration

logger = init_logger(__name__)


@add_registration(reason="Register RBLN block FP8 linear kernel for vLLM OOT platform.")
def register_rbln_fp8_kernels() -> None:
    block_kernels = linear._POSSIBLE_FP8_BLOCK_KERNELS.setdefault(PlatformEnum.OOT, [])
    if RBLNW8A16BlockFp8LinearKernel not in block_kernels:
        block_kernels.insert(0, RBLNW8A16BlockFp8LinearKernel)
