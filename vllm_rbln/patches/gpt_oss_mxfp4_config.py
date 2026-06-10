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

from vllm.model_executor.layers.quantization import register_quantization_config

from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
    RBLNGptOssMxfp4Config,
)
from vllm_rbln.patches import add_registration


@add_registration(
    reason=(
        "Override the built-in GPT-OSS MXFP4 quantization config so GPT-OSS "
        "MoE layers use the RBLN MXFP4 custom-op method while preserving "
        "upstream quantization detection and non-MoE fallbacks."
    )
)
def register_rbln_gpt_oss_mxfp4_config() -> None:
    register_quantization_config("gpt_oss_mxfp4")(RBLNGptOssMxfp4Config)
