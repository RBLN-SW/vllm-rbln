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

from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNCompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm_rbln.patches import register_patch

register_patch(
    target=(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4."
        "CompressedTensorsW4A4Nvfp4MoEMethod"
    ),
    reason=(
        "Upstream's NVFP4 MoE backends are CUDA kernels with no OOT entry, so "
        "a compressed-tensors NVFP4 checkpoint (skt/A.X-K2-NVFP4 from hub "
        "commit e1310bba45) fails with 'No NvFp4 MoE backend supports the "
        "deployment configuration'. get_moe_method imports this name at call "
        "time; replacing it routes the experts to RBLN custom_moe."
    ),
)(RBLNCompressedTensorsW4A4Nvfp4MoEMethod)
