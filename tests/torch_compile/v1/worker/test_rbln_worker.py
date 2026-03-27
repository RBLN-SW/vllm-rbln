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

"""Tests for RBLNWorker.determine_available_memory with kv_cache_memory_bytes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

ESTIMATED_AVAILABLE_MEMORY = 4 * (1 << 30)  # 4 GiB


@pytest.fixture
def mock_worker():
    """Create a minimal mock of RBLNWorker with just the attributes
    needed by determine_available_memory."""
    worker = MagicMock()

    # model_runner.model.named_parameters -> one bf16 param
    param = torch.zeros(1, dtype=torch.bfloat16)
    worker.model_runner.model.named_parameters.return_value = [("dummy", param)]
    worker.model_runner.specialized_moe_decode = False
    worker.model_runner.bucketing_manager.decode_batch_buckets_count = 1

    # model_config
    worker.model_config = SimpleNamespace(quantization=None)

    # parallel_config (unused by mock but needed by estimate_available_memory)
    worker.parallel_config = SimpleNamespace()

    # cache_config defaults
    worker.cache_config = SimpleNamespace(
        gpu_memory_utilization=0.9,
        kv_cache_memory_bytes=None,
    )

    return worker


def _call_determine_available_memory(worker):
    """Call the real determine_available_memory with mocked internals."""
    from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

    # Bind the real method to our mock worker
    return RBLNWorker.determine_available_memory.__wrapped__(worker)


@patch(
    "vllm_rbln.v1.worker.rbln_worker.current_platform",
    **{"get_device_name.return_value": "RBLN-CA12"},
)
@patch(
    "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
    return_value=ESTIMATED_AVAILABLE_MEMORY,
)
class TestDetermineAvailableMemory:
    """Test kv_cache_memory_bytes handling in determine_available_memory."""

    def test_no_kv_cache_memory_bytes(self, mock_estimate, mock_platform, mock_worker):
        """When kv_cache_memory_bytes is not set, return the estimate."""
        result = _call_determine_available_memory(mock_worker)
        assert result == ESTIMATED_AVAILABLE_MEMORY

    def test_kv_cache_memory_bytes_within_limit(
        self, mock_estimate, mock_platform, mock_worker
    ):
        """When kv_cache_memory_bytes <= available, return the requested value."""
        requested = 2 * (1 << 30)  # 2 GiB
        mock_worker.cache_config.kv_cache_memory_bytes = requested

        result = _call_determine_available_memory(mock_worker)
        assert result == requested

    def test_kv_cache_memory_bytes_exceeds_available(
        self, mock_estimate, mock_platform, mock_worker
    ):
        """When kv_cache_memory_bytes > available, clamp to available."""
        requested = 8 * (1 << 30)  # 8 GiB, exceeds 4 GiB
        mock_worker.cache_config.kv_cache_memory_bytes = requested

        result = _call_determine_available_memory(mock_worker)
        assert result == ESTIMATED_AVAILABLE_MEMORY

    def test_kv_cache_memory_bytes_equal_to_available(
        self, mock_estimate, mock_platform, mock_worker
    ):
        """When kv_cache_memory_bytes == available, return the requested value."""
        mock_worker.cache_config.kv_cache_memory_bytes = ESTIMATED_AVAILABLE_MEMORY

        result = _call_determine_available_memory(mock_worker)
        assert result == ESTIMATED_AVAILABLE_MEMORY
