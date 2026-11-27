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

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM


class RBLNNemotronHForCausalLM(NemotronHForCausalLM):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype]:
        return (vllm_config.model_config.dtype,)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...]]:
        hf_config = vllm_config.model_config.hf_config
        return (
            (
                hf_config.mamba_num_heads
                // vllm_config.parallel_config.tensor_parallel_size,
                hf_config.ssm_state_size,
                hf_config.mamba_head_dim,
            ),
        )
