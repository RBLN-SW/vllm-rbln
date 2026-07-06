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
_allocate_kv_cache_tensors, and _process_kv_cache_copy_ops.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData

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
# _add_dummy_requests Tests
# ============================================================


class TestMakeDummyRequest:
    """Test _add_dummy_requests: block_ids computation for dummy requests.

    This method computes num_blocks from total_tokens and block_size,
    and creates block_ids tuples for each KV cache group.
    """

    def _bind(self, runner):
        runner._add_dummy_requests = RBLNModelRunner._add_dummy_requests.__get__(runner)

    def test_block_count_basic(self):
        """Basic block count: 100 tokens / block_size=16 = 7 blocks."""
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=100,
            num_computed_tokens=0,
            num_kv_cache_groups=1,
        )

        assert len(requests) == 1
        req = requests[0]
        # ceil(100/16) = 7 blocks
        assert len(req.block_ids) == 1  # 1 kv cache group
        assert len(req.block_ids[0]) == 7
        assert all(b == 0 for b in req.block_ids[0])  # null_block_id=0

    def test_block_count_exact_divisible(self):
        """When total_tokens is exactly divisible by block_size."""
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=64,
            num_computed_tokens=0,
            num_kv_cache_groups=1,
        )

        assert len(requests[0].block_ids[0]) == 4  # 64/16 = 4

    def test_multiple_kv_cache_groups(self):
        """block_ids tuple is replicated for each KV cache group."""
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=32,
            num_computed_tokens=0,
            num_kv_cache_groups=3,
        )

        assert len(requests[0].block_ids) == 3
        for group_blocks in requests[0].block_ids:
            assert len(group_blocks) == 2  # 32/16 = 2

    def test_num_scheduled_tokens_prefill(self):
        """For prefill: scheduled = total_tokens - num_computed_tokens."""
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=100,
            num_computed_tokens=50,
            num_kv_cache_groups=1,
        )

        req_id = requests[0].req_id
        assert num_scheduled[req_id] == 50  # 100 - 50

    def test_num_scheduled_tokens_decode(self):
        """For decode (computed == total): scheduled = 1."""
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=100,
            num_computed_tokens=100,
            num_kv_cache_groups=1,
        )

        req_id = requests[0].req_id
        assert num_scheduled[req_id] == 1

    def test_multiple_requests_unique_ids(self):
        """Each dummy request gets a unique ID."""
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=16,
            num_computed_tokens=0,
            num_kv_cache_groups=1,
        )
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=32,
            num_computed_tokens=0,
            num_kv_cache_groups=1,
        )

        assert requests[0].req_id != requests[1].req_id
        assert len(num_scheduled) == 2


# ============================================================
# _make_dummy_scheduler_outputs Tests
# ============================================================


class TestMakeDummySchedulerOutputs:
    """Test _make_dummy_scheduler_outputs: creates sched + cleanup outputs."""

    def _bind(self, runner):
        runner._make_dummy_scheduler_outputs = (
            RBLNModelRunner._make_dummy_scheduler_outputs.__get__(runner)
        )
        runner._add_dummy_requests = RBLNModelRunner._add_dummy_requests.__get__(runner)

    def test_basic_scheduler_output(self):
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=32,
            num_computed_tokens=0,
            num_kv_cache_groups=2,
        )

        sched, cleanup = runner._make_dummy_scheduler_outputs(
            requests,
            num_scheduled,
            num_kv_cache_groups=2,
        )

        assert len(sched.scheduled_new_reqs) == 1
        assert sched.total_num_scheduled_tokens == 32
        assert len(sched.num_common_prefix_blocks) == 2
        assert all(b == 0 for b in sched.num_common_prefix_blocks)

    def test_cleanup_output_finishes_all_requests(self):
        runner = _make_runner_stub()
        runner.cache_config.block_size = 16
        self._bind(runner)

        requests: list[NewRequestData] = []
        num_scheduled: dict[str, int] = {}
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=16,
            num_computed_tokens=0,
            num_kv_cache_groups=1,
        )
        runner._add_dummy_requests(
            requests,
            num_scheduled,
            total_tokens=32,
            num_computed_tokens=0,
            num_kv_cache_groups=1,
        )

        _, cleanup = runner._make_dummy_scheduler_outputs(
            requests,
            num_scheduled,
            num_kv_cache_groups=1,
        )

        # Cleanup should finish all request IDs
        assert len(cleanup.finished_req_ids) == 2
        assert cleanup.total_num_scheduled_tokens == 0
        assert len(cleanup.scheduled_new_reqs) == 0


# ============================================================
# select_common_block_size Tests
# ============================================================


class TestSelectCommonBlockSize:
    """Test select_common_block_size static method.

    This method selects a block size supported by all attention backends
    that is also a factor of kv_manager_block_size.
    """

    def _make_backend(self, supported_sizes):
        """Create a mock attention backend with given supported sizes."""
        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = supported_sizes
        return backend

    def _make_attn_group(self, backend):
        group = MagicMock()
        group.backend = backend
        return group

    def test_kv_manager_block_size_supported(self):
        """If kv_manager_block_size is supported by all backends, return it."""
        backend = self._make_backend([16, 32, 64])
        groups = [self._make_attn_group(backend)]

        result = RBLNModelRunner.select_common_block_size(64, groups)
        assert result == 64

    def test_fallback_to_largest_supported(self):
        """If kv_manager_block_size not supported, find largest int factor."""
        backend = self._make_backend([8, 16])
        groups = [self._make_attn_group(backend)]

        # kv_manager_block_size=64, not in [8, 16]
        # 64 % 16 == 0, so 16 is returned
        result = RBLNModelRunner.select_common_block_size(64, groups)
        assert result == 16

    def test_multiple_backends_intersection(self):
        """Block size must be supported by ALL backends."""
        backend1 = self._make_backend([8, 16, 32])
        backend2 = self._make_backend([16, 32])
        groups = [
            self._make_attn_group(backend1),
            self._make_attn_group(backend2),
        ]

        # kv=64 not in either, 32 is factor of 64 and in both
        result = RBLNModelRunner.select_common_block_size(64, groups)
        assert result == 32

    def test_no_common_block_size_raises(self):
        """Raises ValueError when no valid block size found."""
        backend = self._make_backend([7])  # 7 is not a factor of 64
        groups = [self._make_attn_group(backend)]

        with pytest.raises(ValueError, match="No common block size"):
            RBLNModelRunner.select_common_block_size(64, groups)

    def test_multiple_of_support(self):
        """MultipleOf format in supported sizes."""
        from vllm.v1.attention.backend import MultipleOf

        backend = self._make_backend([MultipleOf(16)])
        groups = [self._make_attn_group(backend)]

        # kv_manager_block_size=64, 64 % 16 == 0 → supported
        result = RBLNModelRunner.select_common_block_size(64, groups)
        assert result == 64


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
            mock_envs.VLLM_RBLN_USE_DEVICE_TENSOR = False
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
            mock_envs.VLLM_RBLN_USE_DEVICE_TENSOR = False
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
            mock_envs.VLLM_RBLN_USE_DEVICE_TENSOR = False
            mock_envs.VLLM_RBLN_USE_CUSTOM_KERNEL = True
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            result = runner._allocate_kv_cache_tensors(kv_cache_config)

        assert result["layer_0"].shape == (1024,)
        assert result["layer_1"].shape == (2048,)
        assert result["layer_0"] is not result["layer_1"]

    def test_device_tensor_allocates_on_self_device(self):
        """When VLLM_RBLN_USE_DEVICE_TENSOR=True, tensors are allocated on
        self.device regardless of CUSTOM_KERNEL / COMPILE_MODEL."""
        runner = _make_runner_stub()
        runner.runner_only_attn_layers = set()
        self._bind(runner)

        kv_tensor = MagicMock()
        kv_tensor.size = 768
        kv_tensor.shared_by = ["layer_0"]

        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_tensors = [kv_tensor]
        kv_cache_config.kv_cache_groups = [MagicMock(layer_names=["layer_0"])]

        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_USE_DEVICE_TENSOR = True
            mock_envs.VLLM_RBLN_USE_CUSTOM_KERNEL = False
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            result = runner._allocate_kv_cache_tensors(kv_cache_config)

        assert result["layer_0"].device == runner.device
        assert result["layer_0"].shape == (768,)
        assert result["layer_0"].dtype == torch.int8


# ============================================================
# _process_kv_cache_copy_ops Tests
# ============================================================


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
            mock_envs.VLLM_RBLN_USE_DEVICE_TENSOR = True
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            runner._process_kv_cache_copy_ops([op])

        assert torch.equal(kv_cache[:, 1, :, :, :2, :], kv_cache[:, 0, :, :, :2, :])
        assert torch.all(kv_cache[:, 1, :, :, 2:, :] == 0)

    def test_compiled_non_device_tensor_uses_runtime_copy(self):
        # It's tricky to test the actual copy, so we only check that the
        # runtime method is called with correct args.
        runner = _make_runner_stub()
        runner.model_config = MagicMock(enforce_eager=False)
        runtime = MagicMock()
        runner.runtime_holder = [runtime]
        kv_cache = torch.zeros((2, 3, 1, 1, 4, 1), dtype=torch.int64)
        kv_cache[:, 0, :, :, :2, :] = 7
        runner.kv_caches = [kv_cache]
        self._bind(runner)

        op = KVCacheCopyOp(group_id=0, src_block_id=0, dst_block_id=1, num_tokens=2)
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_USE_DEVICE_TENSOR = False
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True

            runner._process_kv_cache_copy_ops([op])

        runtime._copy_kv_cache.assert_called_once_with(0, 1, 2)


# ============================================================
# _select_canonical_kv_layers_per_pool Tests
# ============================================================
#
# Pick one Full-preferred layer per HMA pool. `mark_static_address` is
# last-write-wins on storage->name, and the NIXL connector uses the
# chosen layer's view as its descriptor stride, so both need a single
# canonical layer per pool with `cache.shape[0] == num_blocks` (logical).


def _make_kv_cache_config(pools, layer_specs):
    """Build a minimal KVCacheConfig-like object.

    `pools`: list of `shared_by` lists, one per HMA pool.
    `layer_specs`: dict layer_name -> KVCacheSpec.
    """
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_tensors = [MagicMock(shared_by=pool) for pool in pools]
    # Drive `_attn_group_iterator` via attn_groups[i][j].
    # Each AttentionGroup needs `layer_names` and `kv_cache_spec`.
    grouped: dict[int, dict] = {}
    for layer_name, spec in layer_specs.items():
        grouped.setdefault(id(spec), {"spec": spec, "layers": []})["layers"].append(
            layer_name
        )
    attn_groups = [
        [MagicMock(layer_names=g["layers"], kv_cache_spec=g["spec"])]
        for g in grouped.values()
    ]
    return kv_cache_config, attn_groups


class TestSelectCanonicalKvLayersPerPool:
    """`_select_canonical_kv_layers_per_pool` picks one layer per HMA
    pool, preferring Full-attention (whose view is at logical block
    granularity)."""

    def _bind(self, attn_groups):
        runner = object.__new__(RBLNModelRunner)
        runner.attn_groups = attn_groups
        # `_attn_group_iterator` is itertools.chain.from_iterable(attn_groups);
        # bind it via the same class so the implementation under test is reached.
        runner._attn_group_iterator = RBLNModelRunner._attn_group_iterator.__get__(
            runner
        )
        runner._select_canonical_kv_layers_per_pool = (
            RBLNModelRunner._select_canonical_kv_layers_per_pool.__get__(runner)
        )
        return runner

    def test_prefers_full_attention_layer(self):
        """A pool with both Full and SWA layers picks the Full one — its
        view has `cache.shape[-2] == block_size` (logical)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        full_spec = MagicMock(spec=FullAttentionSpec)
        swa_spec = MagicMock(spec=SlidingWindowSpec)
        kv_cache_config, attn_groups = _make_kv_cache_config(
            pools=[["layer.swa", "layer.full"]],
            layer_specs={"layer.full": full_spec, "layer.swa": swa_spec},
        )
        runner = self._bind(attn_groups)

        chosen = runner._select_canonical_kv_layers_per_pool(kv_cache_config)

        assert chosen == {"layer.full"}

    def test_falls_back_to_first_layer_when_no_full(self):
        """Pure-SWA pool (no Full layer) falls back to the first layer in
        `shared_by`."""
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        swa_spec = MagicMock(spec=SlidingWindowSpec)
        kv_cache_config, attn_groups = _make_kv_cache_config(
            pools=[["layer.swa.0", "layer.swa.1"]],
            layer_specs={"layer.swa.0": swa_spec, "layer.swa.1": swa_spec},
        )
        runner = self._bind(attn_groups)

        chosen = runner._select_canonical_kv_layers_per_pool(kv_cache_config)

        assert chosen == {"layer.swa.0"}

    def test_skips_pool_with_empty_shared_by(self):
        """A pool whose `shared_by` is empty doesn't contribute a layer."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        full_spec = MagicMock(spec=FullAttentionSpec)
        kv_cache_config, attn_groups = _make_kv_cache_config(
            pools=[[], ["layer.full"]],
            layer_specs={"layer.full": full_spec},
        )
        runner = self._bind(attn_groups)

        chosen = runner._select_canonical_kv_layers_per_pool(kv_cache_config)

        assert chosen == {"layer.full"}

    def test_one_canonical_layer_per_pool(self):
        """Multiple HMA pools, each Full+SWA — one Full per pool, no overlap."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        full_spec = MagicMock(spec=FullAttentionSpec)
        swa_spec = MagicMock(spec=SlidingWindowSpec)
        kv_cache_config, attn_groups = _make_kv_cache_config(
            pools=[
                ["pool0.swa", "pool0.full"],
                ["pool1.swa", "pool1.full"],
            ],
            layer_specs={
                "pool0.full": full_spec,
                "pool0.swa": swa_spec,
                "pool1.full": full_spec,
                "pool1.swa": swa_spec,
            },
        )
        runner = self._bind(attn_groups)

        chosen = runner._select_canonical_kv_layers_per_pool(kv_cache_config)

        assert chosen == {"pool0.full", "pool1.full"}
