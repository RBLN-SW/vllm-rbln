# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the multimodal compile-param builders (no NPU required)."""

import pytest

from vllm_rbln.model_executor.models.optimum.compilation.multimodal.qwen import (
    get_param_qwen3_5,
)

_COMMON = dict(batch_size=1, num_devices=1, memory_budget=0.9, prefill_chunk_size=128)


class TestGetParamQwen35:
    """Qwen3.5 forces flash attention, so block_size must partition
    max_model_len into >= 2 even parts; otherwise the config is rejected."""

    def test_valid_partition_sets_kvcache_partition_len(self):
        param = get_param_qwen3_5(max_model_len=4096, block_size=1024, **_COMMON)
        assert param["attn_impl"] == "flash_attn"
        # block_size is actually carried into the compiled model.
        assert param["kvcache_partition_len"] == 1024
        assert param["max_seq_len"] == 4096

    def test_block_size_equal_max_model_len_raises(self):
        # Only 1 partition -> flash attention has no partition to work with.
        with pytest.raises(ValueError, match="block_size"):
            get_param_qwen3_5(max_model_len=4096, block_size=4096, **_COMMON)
