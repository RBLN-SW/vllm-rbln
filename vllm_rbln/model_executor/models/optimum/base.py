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
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from vllm.multimodal.inputs import BatchedTensorInputs


# FIXME(eunji): In original vLLM, this dataclasss is located in model_runner.
# And it makes available to decouple the vllm logic and hf model logic
@dataclass(frozen=True)
class ModelInputForRBLN:
    """
    Used by the RBLNModelRunner.
    """
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    block_tables: torch.Tensor
    running_requests_ids: List[str]
    finished_requests_ids: List[str]
    is_prompt: bool = False
    cached_block_tables: List[int] = field(
        default_factory=list)  # for prefix caching
    cached_lengths: List[int] = field(
        default_factory=list)  # for prefix caching
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    dummy_block: Optional[int] = None  # for prefix caching


version_error = RuntimeError(
    "Incompatible vLLM version detected. "
    "This vLLM version is not compatible with optimum-rbln. "
    "Please verify that you are using a supported version.")
