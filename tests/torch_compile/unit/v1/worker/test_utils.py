# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for vllm_rbln.v1.worker.utils module.

Covers estimate_available_memory, get_autobind_cpu_ids,
set_cpu_affinity, and set_omp_num_threads.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.platforms import CpuArchEnum
from vllm.platforms.cpu import LogicalCPUInfo

from vllm_rbln.v1.worker.utils import (
    estimate_available_memory,
    get_autobind_cpu_ids,
    set_cpu_affinity,
    set_omp_num_threads,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    num_layers=32,
    vocab_size=32000,
    hidden_size=4096,
    num_kv_heads=8,
):
    """Create a minimal model config stub."""
    cfg = SimpleNamespace(
        _num_layers=num_layers,
        _vocab_size=vocab_size,
        _hidden_size=hidden_size,
        _num_kv_heads=num_kv_heads,
    )
    cfg.get_num_layers = lambda pc: cfg._num_layers
    cfg.get_vocab_size = lambda: cfg._vocab_size
    cfg.get_hidden_size = lambda: cfg._hidden_size
    cfg.get_num_kv_heads = lambda pc: cfg._num_kv_heads
    return cfg


def _make_parallel_config(tp_size=1):
    return SimpleNamespace(
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        world_size=tp_size,
        world_size_across_dp=tp_size,
        data_parallel_rank=0,
    )


def _make_cpu(cpu_id, physical_core, numa_node):
    return LogicalCPUInfo(id=cpu_id, physical_core=physical_core, numa_node=numa_node)


# ---------------------------------------------------------------------------
# estimate_available_memory
# ---------------------------------------------------------------------------
class TestEstimateAvailableMemory:
    """Test DRAM estimation for ATOM and REBEL devices."""

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_atom_device_basic(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config(tp_size=1)

        result = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=1 * 2**30,  # 1GB kernel
            gpu_memory_utilization=0.9,
        )
        assert result > 0

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_rebel_device_basic(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CR100"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config(tp_size=1)

        result = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=1 * 2**30,
            gpu_memory_utilization=0.9,
        )
        assert result > 0

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_unknown_device_raises(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-XX99"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match="invalid RBLN architecture"):
            estimate_available_memory(
                model_cfg,
                parallel_cfg,
                kernel_size=1 * 2**30,
            )

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_both_params_and_kernel_raises(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match="Both.*cannot be specified"):
            estimate_available_memory(
                model_cfg,
                parallel_cfg,
                n_model_params=1_000_000,
                kernel_size=2**30,
            )

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_estimated_kernel_size_from_params(self, mock_envs, mock_platform):
        """When kernel_size is None, it should be estimated from n_model_params."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        result = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            n_model_params=7_000_000_000,
            nbits_per_param=16,
        )
        assert result > 0

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_no_params_no_kernel_raises(self, mock_envs, mock_platform):
        """If neither kernel_size nor n_model_params given, should raise."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match="n_model_params.*should be specified"):
            estimate_available_memory(model_cfg, parallel_cfg)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_oom_raises_memory_error(self, mock_envs, mock_platform):
        """Huge kernel_size should exhaust available memory."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(MemoryError, match="Insufficient DRAM"):
            estimate_available_memory(
                model_cfg,
                parallel_cfg,
                kernel_size=100 * 2**30,  # 100GB — exceeds 16GB ATOM
            )

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_atom_tp4_scales_memory(self, mock_envs, mock_platform):
        """TP=4 on ATOM should give ~4x the memory of TP=1."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"

        model_cfg = _make_model_config()
        kernel = 1 * 2**30

        mock_envs.VLLM_RBLN_TP_SIZE = 1
        mem_tp1 = estimate_available_memory(
            model_cfg,
            _make_parallel_config(tp_size=1),
            kernel_size=kernel,
        )

        mock_envs.VLLM_RBLN_TP_SIZE = 4
        mem_tp4 = estimate_available_memory(
            model_cfg,
            _make_parallel_config(tp_size=4),
            kernel_size=kernel,
        )
        # TP=4 should give significantly more memory
        assert mem_tp4 > mem_tp1

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_rebel_requires_tp1(self, mock_envs, mock_platform):
        """REBEL (CR) device asserts tp_size==1."""
        mock_platform.get_device_name.return_value = "RBLN-CR100"
        mock_envs.VLLM_RBLN_TP_SIZE = 2

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config(tp_size=1)

        with pytest.raises(AssertionError):
            estimate_available_memory(
                model_cfg,
                parallel_cfg,
                kernel_size=1 * 2**30,
            )

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_gpu_memory_utilization_effect(self, mock_envs, mock_platform):
        """Lower utilization should give less available memory."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()
        kernel = 1 * 2**30

        mem_90 = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=kernel,
            gpu_memory_utilization=0.9,
        )
        mem_50 = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=kernel,
            gpu_memory_utilization=0.5,
        )
        assert mem_90 > mem_50

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_custom_buffer(self, mock_envs, mock_platform):
        """Explicit buffer should reduce available memory compared to default."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()
        kernel = 1 * 2**30

        mem_default = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=kernel,
        )
        mem_big_buffer = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=kernel,
            buffer=4 * 2**30,  # 4GB buffer
        )
        assert mem_default > mem_big_buffer

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_rsd_replicas_for_large_kv_heads(self, mock_envs, mock_platform):
        """When kv_heads < rsd_size, rsd_replicas > 1 reduces memory."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_TP_SIZE = 4  # rsd_size = 4

        # num_kv_heads=2, rsd_size=4 → rsd_replicas = 4//2 = 2
        model_cfg_few_heads = _make_model_config(num_kv_heads=2)
        # num_kv_heads=8, rsd_size=4 → rsd_replicas = 4//8 = 0 → max(0,1) = 1
        model_cfg_many_heads = _make_model_config(num_kv_heads=8)

        parallel_cfg = _make_parallel_config(tp_size=4)
        kernel = 1 * 2**30

        mem_few = estimate_available_memory(
            model_cfg_few_heads,
            parallel_cfg,
            kernel_size=kernel,
        )
        mem_many = estimate_available_memory(
            model_cfg_many_heads,
            parallel_cfg,
            kernel_size=kernel,
        )
        # Fewer kv_heads → more replicas → less memory per replica
        assert mem_few < mem_many


# ---------------------------------------------------------------------------
# get_autobind_cpu_ids
# ---------------------------------------------------------------------------
class TestGetAutobindCpuIds:
    """Test NUMA-aware CPU binding logic."""

    def _simple_cpu_list(self):
        """8 CPUs, 2 NUMA nodes, 2 physical cores per node, HT (2 threads/core)."""
        return [
            _make_cpu(0, 0, 0),
            _make_cpu(4, 0, 0),  # NUMA 0, core 0
            _make_cpu(1, 1, 0),
            _make_cpu(5, 1, 0),  # NUMA 0, core 1
            _make_cpu(2, 2, 1),
            _make_cpu(6, 2, 1),  # NUMA 1, core 2
            _make_cpu(3, 3, 1),
            _make_cpu(7, 3, 1),  # NUMA 1, core 3
        ]

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_basic_single_rank(self, mock_cpu_platform):
        cpus = self._simple_cpu_list()
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0, 1], cpus)

        parallel_cfg = _make_parallel_config(tp_size=1)
        result = get_autobind_cpu_ids(
            rank=0,
            local_rank=0,
            parallel_config=parallel_cfg,
            cpu_selector=lambda cpus: cpus,  # take all
        )

        # rank 0 → NUMA 0, should get CPUs from NUMA node 0
        cpu_ids = [int(x) for x in result.split(",")]
        assert all(
            any(c.id == cid and c.numa_node == 0 for c in cpus) for cid in cpu_ids
        )

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_rank_round_robins_numa_nodes(self, mock_cpu_platform):
        cpus = self._simple_cpu_list()
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0, 1], cpus)
        parallel_cfg = _make_parallel_config(tp_size=2)

        r0 = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        r1 = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)

        # Different ranks should get different NUMA nodes
        r0_ids = set(int(x) for x in r0.split(","))
        r1_ids = set(int(x) for x in r1.split(","))
        assert r0_ids.isdisjoint(r1_ids), "Ranks should not share CPUs"

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_no_available_numa_returns_all(self, mock_cpu_platform):
        """If allowed NUMA nodes don't have CPUs, return 'all'."""
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([], [])

        parallel_cfg = _make_parallel_config()
        result = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        assert result == "all"

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_cpu_selector_filters_threads(self, mock_cpu_platform):
        """cpu_selector=lambda cpus: cpus[:1] should pick one thread per core."""
        cpus = self._simple_cpu_list()
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0, 1], cpus)

        parallel_cfg = _make_parallel_config(tp_size=1)
        result = get_autobind_cpu_ids(
            0,
            0,
            parallel_cfg,
            cpu_selector=lambda cpus: cpus[:1],  # 1 thread per core
        )
        cpu_ids = [int(x) for x in result.split(",")]
        # NUMA 0 has 2 cores, should get 2 CPUs (one per core)
        assert len(cpu_ids) == 2

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_multiple_ranks_same_numa_exclusive_allocation(self, mock_cpu_platform):
        """When 2 ranks map to the same NUMA node, CPUs are split."""
        # Single NUMA node with 4 cores, 1 thread each
        cpus = [_make_cpu(i, i, 0) for i in range(4)]
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0], cpus)

        parallel_cfg = _make_parallel_config(tp_size=2)

        r0 = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        r1 = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)

        r0_ids = set(int(x) for x in r0.split(","))
        r1_ids = set(int(x) for x in r1.split(","))
        assert r0_ids.isdisjoint(r1_ids)
        assert len(r0_ids) + len(r1_ids) == 4

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_uneven_cpu_split(self, mock_cpu_platform):
        """3 CPUs split between 2 ranks: one gets 2, other gets 1."""
        cpus = [_make_cpu(i, i, 0) for i in range(3)]
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0], cpus)

        parallel_cfg = _make_parallel_config(tp_size=2)

        r0 = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        r1 = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)

        r0_count = len(r0.split(","))
        r1_count = len(r1.split(","))
        assert {r0_count, r1_count} == {1, 2}

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_dp_rank_affects_binding(self, mock_cpu_platform):
        """Data parallelism changes rank_across_dp calculation."""
        cpus = [_make_cpu(i, i, 0) for i in range(8)]
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0], cpus)

        dp_cfg = SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=2,
            world_size=1,
            world_size_across_dp=2,
            data_parallel_rank=1,
        )

        result = get_autobind_cpu_ids(0, 0, dp_cfg, lambda cpus: cpus)
        cpu_ids = [int(x) for x in result.split(",")]
        # dp_rank=1, rank_across_dp = 1*1 + 0 = 1
        # With single NUMA node, both ranks share, so rank 1 gets second half
        assert len(cpu_ids) == 4

    @patch("vllm_rbln.v1.worker.utils.CpuPlatform")
    def test_empty_allocation_returns_all(self, mock_cpu_platform):
        """If cpu_selector returns empty lists, should fallback to 'all'."""
        cpus = [_make_cpu(0, 0, 0)]
        mock_cpu_platform.get_allowed_cpu_core_node_list.return_value = ([0], cpus)

        # 2 ranks but only 1 CPU in the only NUMA node
        parallel_cfg = _make_parallel_config(tp_size=2)

        # rank 1 should get no CPUs (rank 0 gets the 1 CPU)
        result = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)
        assert result == "all"


# ---------------------------------------------------------------------------
# set_cpu_affinity
# ---------------------------------------------------------------------------
class TestSetCpuAffinity:
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_nobind_when_numa_disabled(self, mock_platform_mod, mock_envs):
        """When VLLM_RBLN_NUMA is False, should skip binding."""
        mock_envs.VLLM_RBLN_NUMA = False
        mock_platform_mod.system.return_value = "Linux"

        parallel_cfg = _make_parallel_config()
        # Should not raise
        set_cpu_affinity(0, 0, parallel_cfg)

    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_nobind_on_non_linux(self, mock_platform_mod, mock_envs):
        """Non-Linux systems should skip binding."""
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Darwin"

        parallel_cfg = _make_parallel_config()
        set_cpu_affinity(0, 0, parallel_cfg)

    @patch("vllm_rbln.v1.worker.utils.os.sched_setaffinity")
    @patch("vllm_rbln.v1.worker.utils.os.sched_getaffinity")
    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_sets_affinity_on_x86(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
        mock_get_aff,
        mock_set_aff,
    ):
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.X86
        mock_autobind.return_value = "0,1,2,3"
        mock_get_aff.return_value = {0, 1, 2, 3}

        parallel_cfg = _make_parallel_config()
        set_cpu_affinity(0, 0, parallel_cfg)

        mock_set_aff.assert_called_once_with(0, [0, 1, 2, 3])

    @patch("vllm_rbln.v1.worker.utils.os.sched_setaffinity")
    @patch("vllm_rbln.v1.worker.utils.os.sched_getaffinity")
    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_affinity_mismatch_warns(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
        mock_get_aff,
        mock_set_aff,
    ):
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.X86
        mock_autobind.return_value = "0,1"
        # Simulate kernel restricting CPUs
        mock_get_aff.return_value = {0}

        parallel_cfg = _make_parallel_config()
        with patch("vllm_rbln.v1.worker.utils.logger") as mock_logger:
            set_cpu_affinity(0, 0, parallel_cfg)
            mock_logger.warning.assert_called_once()
            assert "mismatch" in str(mock_logger.warning.call_args).lower()

    @patch("vllm_rbln.v1.worker.utils.os.sched_setaffinity")
    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_os_error_propagates(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
        mock_set_aff,
    ):
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.X86
        mock_autobind.return_value = "999"
        mock_set_aff.side_effect = OSError("Invalid CPU")

        parallel_cfg = _make_parallel_config()
        with pytest.raises(OSError, match="Invalid CPU"):
            set_cpu_affinity(0, 0, parallel_cfg)

    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_nobind_for_arm_arch(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
    ):
        """ARM architecture is not handled — falls through to 'nobind'."""
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.ARM

        parallel_cfg = _make_parallel_config()
        # Should not raise — nobind means skip
        set_cpu_affinity(0, 0, parallel_cfg)
        # get_autobind_cpu_ids should not be called for ARM
        mock_autobind.assert_not_called()

    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_powerpc_uses_smt_selector(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
    ):
        """PowerPC should call get_autobind_cpu_ids with SMT-specific selector."""
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.POWERPC
        mock_autobind.return_value = "nobind"

        parallel_cfg = _make_parallel_config()
        set_cpu_affinity(0, 0, parallel_cfg)

        mock_autobind.assert_called_once()
        # Verify the selector function filters by cpu.id % 8 < 4
        selector = (
            mock_autobind.call_args[1].get("cpu_selector")
            or mock_autobind.call_args[0][3]
        )
        test_cpus = [_make_cpu(i, 0, 0) for i in range(8)]
        selected = selector(test_cpus)
        selected_ids = [c.id for c in selected]
        assert selected_ids == [0, 1, 2, 3]

    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_x86_uses_first_thread_selector(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
    ):
        """x86 selector should pick cpus[:1] — first thread per core."""
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.X86
        mock_autobind.return_value = "nobind"

        parallel_cfg = _make_parallel_config()
        set_cpu_affinity(0, 0, parallel_cfg)

        selector = (
            mock_autobind.call_args[1].get("cpu_selector")
            or mock_autobind.call_args[0][3]
        )
        test_cpus = [_make_cpu(0, 0, 0), _make_cpu(4, 0, 0)]  # 2 threads, same core
        selected = selector(test_cpus)
        assert len(selected) == 1
        assert selected[0].id == 0

    @patch("vllm_rbln.v1.worker.utils.get_autobind_cpu_ids")
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @patch("vllm_rbln.v1.worker.utils.platform")
    def test_autobind_returns_all_no_sched_call(
        self,
        mock_platform_mod,
        mock_envs,
        mock_cur_platform,
        mock_autobind,
    ):
        """When autobind returns 'all', sched_setaffinity should NOT be called."""
        mock_envs.VLLM_RBLN_NUMA = True
        mock_platform_mod.system.return_value = "Linux"
        mock_cur_platform.get_cpu_architecture.return_value = CpuArchEnum.X86
        mock_autobind.return_value = "all"

        parallel_cfg = _make_parallel_config()
        with patch("vllm_rbln.v1.worker.utils.os.sched_setaffinity") as mock_set:
            set_cpu_affinity(0, 0, parallel_cfg)
            mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# set_omp_num_threads
# ---------------------------------------------------------------------------
class TestSetOmpNumThreads:
    @patch("torch.set_num_threads")
    def test_default_threads(self, mock_set):
        env = os.environ.copy()
        env.pop("RBLN_NUM_THREADS", None)
        with patch.dict(os.environ, env, clear=True):
            set_omp_num_threads(0, 0)
            mock_set.assert_called_once_with(2)
            assert os.environ["RBLN_NUM_THREADS"] == "2"

    @patch("torch.set_num_threads")
    def test_env_override(self, mock_set):
        with patch.dict(os.environ, {"RBLN_NUM_THREADS": "8"}):
            set_omp_num_threads(0, 0)
            mock_set.assert_called_once_with(8)

    @patch("torch.set_num_threads")
    def test_custom_default(self, mock_set):
        env = os.environ.copy()
        env.pop("RBLN_NUM_THREADS", None)
        with patch.dict(os.environ, env, clear=True):
            set_omp_num_threads(0, 0, default_num_threads=4)
            mock_set.assert_called_once_with(4)
            assert os.environ["RBLN_NUM_THREADS"] == "4"

    @patch("torch.set_num_threads")
    def test_env_takes_precedence_over_default(self, mock_set):
        """Even with default_num_threads=4, env var should win."""
        with patch.dict(os.environ, {"RBLN_NUM_THREADS": "16"}):
            set_omp_num_threads(0, 0, default_num_threads=4)
            mock_set.assert_called_once_with(16)
