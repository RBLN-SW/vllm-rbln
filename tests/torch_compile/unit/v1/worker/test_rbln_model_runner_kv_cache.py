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

"""Unit tests for KV cache related logic in RBLNModelRunner.

Tests _add_dummy_requests, _make_dummy_scheduler_outputs,
select_common_block_size, _prepare_kernel_block_sizes,
_allocate_kv_cache_tensors, _process_kv_cache_copy_ops,
and _propagate_runtime_holder.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_rbln.v1.core.rbln_kv_cache_manager import KVCacheCopyOp
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner


def _make_runner_stub(**overrides):
    """Create a minimal RBLNModelRunner stub for KV cache tests."""
    runner = object.__new__(RBLNModelRunner)
    runner.device = torch.device("cpu")
    runner.cache_config = MagicMock()
    runner.cache_config.block_size = 16
    defaults = {
        "pin_memory": False,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(runner, k, v)
    return runner


# ============================================================
# _allocate_kv_cache_tensors Tests
# ============================================================


class TestAllocateKvCacheTensors:
    """Test _allocate_kv_cache_tensors: creates zero tensors for KV cache."""

    def _bind(self, runner):
        runner._allocate_kv_cache_tensors = (
            RBLNModelRunner._allocate_kv_cache_tensors.__get__(runner)
        )

    def test_basic_allocation(self):
        runner = _make_runner_stub()
        runner.runner_only_attn_layers = set()
        self._bind(runner)

        kv_tensor = MagicMock()
        kv_tensor.size = 1024
        kv_tensor.shared_by = ["layer_0", "layer_1"]

        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_tensors = [kv_tensor]
        kv_cache_config.kv_cache_groups = [
            MagicMock(layer_names=["layer_0", "layer_1"])
        ]

        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_USE_CUSTOM_KERNEL = True
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            result = runner._allocate_kv_cache_tensors(kv_cache_config)

        assert "layer_0" in result
        assert "layer_1" in result
        # Both should share the same tensor
        assert result["layer_0"] is result["layer_1"]
        assert result["layer_0"].shape == (1024,)
        assert result["layer_0"].dtype == torch.int8

    def test_meta_device_when_compile(self):
        """When VLLM_RBLN_USE_CUSTOM_KERNEL=False and COMPILE_MODEL=True,
        tensors are on meta device."""
        runner = _make_runner_stub()
        runner.runner_only_attn_layers = set()
        self._bind(runner)

        kv_tensor = MagicMock()
        kv_tensor.size = 512
        kv_tensor.shared_by = ["layer_0"]

        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_tensors = [kv_tensor]
        kv_cache_config.kv_cache_groups = [MagicMock(layer_names=["layer_0"])]

        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_USE_CUSTOM_KERNEL = False
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            result = runner._allocate_kv_cache_tensors(kv_cache_config)

        assert result["layer_0"].device.type == "meta"

    def test_multiple_kv_cache_tensors(self):
        """Multiple KV cache tensor configs for different layer groups."""
        runner = _make_runner_stub()
        runner.runner_only_attn_layers = set()
        self._bind(runner)

        kv_tensor_0 = MagicMock()
        kv_tensor_0.size = 1024
        kv_tensor_0.shared_by = ["layer_0"]

        kv_tensor_1 = MagicMock()
        kv_tensor_1.size = 2048
        kv_tensor_1.shared_by = ["layer_1"]

        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_tensors = [kv_tensor_0, kv_tensor_1]
        kv_cache_config.kv_cache_groups = [
            MagicMock(layer_names=["layer_0", "layer_1"])
        ]

        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_USE_CUSTOM_KERNEL = True
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            result = runner._allocate_kv_cache_tensors(kv_cache_config)

        assert result["layer_0"].shape == (1024,)
        assert result["layer_1"].shape == (2048,)
        assert result["layer_0"] is not result["layer_1"]


# ============================================================
# _process_kv_cache_copy_ops Tests
# ============================================================


@pytest.mark.skip(reason="Requires vLLM RBLN runtime support for KV cache copy ops.")
class TestProcessKvCacheCopyOps:
    def _bind(self, runner):
        runner._process_kv_cache_copy_ops = (
            RBLNModelRunner._process_kv_cache_copy_ops.__get__(runner)
        )

    def test_device_tensor_mode_uses_torch_copy(self):
        runner = _make_runner_stub()
        runner.model_config = MagicMock(enforce_eager=False)
        runner.runtime_holder = []
        kv_cache = torch.zeros((2, 3, 1, 1, 4, 1), dtype=torch.int64)
        kv_cache[:, 0, :, :, :2, :] = 7
        runner.kv_caches = [kv_cache]
        self._bind(runner)

        op = KVCacheCopyOp(group_id=0, src_block_id=0, dst_block_id=1, num_tokens=2)
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            runner._process_kv_cache_copy_ops([op])

        assert torch.equal(kv_cache[:, 1, :, :, :2, :], kv_cache[:, 0, :, :, :2, :])
        assert torch.all(kv_cache[:, 1, :, :, 2:, :] == 0)


# ============================================================
# _propagate_runtime_holder Tests
# ============================================================


class _ConnectorWithHolder:
    """Stub for an RBLN-aware connector exposing set_runtime_holder."""

    def __init__(self) -> None:
        self.holder: list | None = None

    def set_runtime_holder(self, runtime_holder: list) -> None:
        self.holder = runtime_holder


class _PlainConnector:
    """Stub for a connector without set_runtime_holder (e.g. NIXL)."""


class _MultiConnectorStub:
    """Stub mimicking vLLM MultiConnector: has _connectors, no set_runtime_holder."""

    def __init__(self, children: list[object]) -> None:
        self._connectors = children
