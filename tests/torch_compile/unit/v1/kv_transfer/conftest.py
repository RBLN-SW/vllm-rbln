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

from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def rbln_params():
    return {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "dtype": torch.float16,
    }


@pytest.fixture
def kv_cache_names():
    return [f"model.layers.{i}.self_attn.kv_cache" for i in range(32)]


@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.fetch_kv_cache = MagicMock()
    runtime.update_kv_cache = MagicMock()
    return runtime


@pytest.fixture
def mock_memory_obj():
    def _create(shape, dtype=torch.float16):
        obj = MagicMock()
        obj.tensor = torch.randn(shape, dtype=dtype)
        return obj

    return _create
