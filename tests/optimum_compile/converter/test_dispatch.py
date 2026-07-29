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

from types import SimpleNamespace

import torch

from vllm_rbln.utils.optimum.converter.common import (
    USER_MAX_NUM_BATCHED_TOKENS_KEY,
)
from vllm_rbln.utils.optimum.converter.dispatch import _generate_model_path_name


def _vllm_config(user_max_num_batched_tokens=None, dtype=torch.bfloat16):
    additional_config = {}
    if user_max_num_batched_tokens is not None:
        additional_config[USER_MAX_NUM_BATCHED_TOKENS_KEY] = user_max_num_batched_tokens
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="meta-llama/Llama-3.1-8B", max_model_len=8192, dtype=dtype
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(block_size=8192, gpu_memory_utilization=0.9),
        additional_config=additional_config,
    )


class TestGenerateModelPathName:
    def test_user_max_num_batched_tokens_changes_hash(self):
        # Two runs that would compile different prefill chunk sizes must not
        # collide on the same cache key.
        name_512 = _generate_model_path_name(_vllm_config(512))
        name_256 = _generate_model_path_name(_vllm_config(256))
        assert name_512 != name_256

    def test_unset_differs_from_explicit(self):
        name_unset = _generate_model_path_name(_vllm_config(None))
        name_512 = _generate_model_path_name(_vllm_config(512))
        assert name_unset != name_512

    def test_same_value_same_hash(self):
        assert _generate_model_path_name(
            _vllm_config(512)
        ) == _generate_model_path_name(_vllm_config(512))

    def test_dtype_changes_hash(self):
        name_bf16 = _generate_model_path_name(_vllm_config(dtype=torch.bfloat16))
        name_fp32 = _generate_model_path_name(_vllm_config(dtype=torch.float32))
        assert name_bf16 != name_fp32
