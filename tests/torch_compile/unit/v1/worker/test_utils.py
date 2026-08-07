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

"""Tests for vllm_rbln.v1.worker.utils module.

Covers estimate_available_memory, get_autobind_cpu_ids,
set_cpu_affinity, and set_omp_num_threads.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.platforms import CpuArchEnum
from vllm.utils.cpu_resource_utils import LogicalCPUInfo

from vllm_rbln.v1.worker.utils import (
    REBEL_DRAM_NBYTES,
    chiplet_replication_factor,
    divide_by_chiplet_replication,
    estimate_available_memory,
    estimate_model_kernel_size,
    get_autobind_cpu_ids,
    get_rbln_owned_card_indices,
    get_rbln_visible_card_indices,
    read_rbln_card_dram_total_bytes,
    read_rbln_card_dram_used_bytes,
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
class TestEstimateModelKernelSize:
    @patch("vllm_rbln.v1.worker.utils.current_platform")
    def test_bytes_path_estimates_kernel_size(self, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CA12"

        model_cfg = _make_model_config(num_layers=2, vocab_size=64, hidden_size=32)
        parallel_cfg = _make_parallel_config(tp_size=1)

        result = estimate_model_kernel_size(
            model_cfg,
            parallel_cfg,
            n_model_bytes=12_288,
        )

        assert result == 6_291_456

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "Either `n_model_params` or `n_model_bytes`"),
            (
                {"n_model_params": 1_000_000, "n_model_bytes": 2_000_000},
                "Only one of `n_model_params` or `n_model_bytes`",
            ),
            (
                {"n_model_params": 1_000_000},
                "`nbits_per_param` should be specified",
            ),
        ],
    )
    def test_validates_input_combinations(self, mock_platform, kwargs, match):
        mock_platform.get_device_name.return_value = "RBLN-CA12"

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match=match):
            estimate_model_kernel_size(model_cfg, parallel_cfg, **kwargs)


class TestEstimateAvailableMemory:
    """Test DRAM estimation for ATOM and REBEL devices."""

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_atom_device_basic(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match="cannot both be specified"):
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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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
    def test_estimated_kernel_size_from_bytes(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

        model_cfg = _make_model_config(num_layers=2, vocab_size=64, hidden_size=32)
        parallel_cfg = _make_parallel_config(tp_size=1)

        result = estimate_available_memory(
            model_cfg,
            parallel_cfg,
            n_model_bytes=12_288,
        )
        assert result > 0

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_no_params_no_kernel_raises(self, mock_envs, mock_platform):
        """If neither kernel_size nor n_model_params given, should raise."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match="n_model_params.*should be specified"):
            estimate_available_memory(model_cfg, parallel_cfg)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_n_model_params_requires_nbits(self, mock_envs, mock_platform):
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

        model_cfg = _make_model_config()
        parallel_cfg = _make_parallel_config()

        with pytest.raises(ValueError, match="`nbits_per_param` should be specified"):
            estimate_available_memory(
                model_cfg,
                parallel_cfg,
                n_model_params=1_000_000,
            )

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_oom_raises_memory_error(self, mock_envs, mock_platform):
        """Huge kernel_size should exhaust available memory."""
        mock_platform.get_device_name.return_value = "RBLN-CA12"
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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

        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1
        mem_tp1 = estimate_available_memory(
            model_cfg,
            _make_parallel_config(tp_size=1),
            kernel_size=kernel,
        )

        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 4
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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 2

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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1

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
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 4  # rsd_size = 4

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
# chiplet replication factor
# ---------------------------------------------------------------------------
class TestChipletReplicationFactor:
    """Pin the KV-cache chiplet replication factor to exact values.

    Each chiplet stores ceil(kvh / rsd_size) KV heads, so the rsd_size chiplets
    together hold rsd_size * ceil(kvh / rsd_size) heads where the logical cache
    has kvh. The ratio is what the DRAM budget must be divided by.

    This one expression replaces two partial ones that must not be stacked:
    `rsd_size // kvh` (right only when kvh divides rsd_size) and
    `rsd_size * ceil(kvh/rsd_size) / kvh` applied as a *correction* on top of it
    (which halves the budget a second time for kvh < rsd_size -- the 844-block
    state).
    """

    @pytest.mark.parametrize(
        ("num_kv_heads", "rsd_size", "expected"),
        [
            (2, 4, 2.0),
            (10, 4, 1.2),
            (8, 4, 1.0),
            (4, 4, 1.0),
            (1, 4, 4.0),
            (3, 4, 4 / 3),
            (2, 1, 1.0),
        ],
    )
    def test_factor_values(self, num_kv_heads, rsd_size, expected):
        assert chiplet_replication_factor(num_kv_heads, rsd_size) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        ("num_kv_heads", "rsd_size"),
        [(2, 4), (10, 4), (8, 4), (4, 4), (1, 4), (3, 4), (7, 4), (2, 1)],
    )
    def test_division_is_exact_integer_floor(self, num_kv_heads, rsd_size):
        nbytes = 127_538_298_880  # the real post-kernel/buffer budget
        expected = int(nbytes // chiplet_replication_factor(num_kv_heads, rsd_size))
        got = divide_by_chiplet_replication(nbytes, num_kv_heads, rsd_size)
        assert got == expected

    def test_replicated_kv_heads_halve_the_budget(self):
        """kvh=2, rsd_size=4 → replication factor 2 → 59.3896 GiB.

        Matches the `available_memory_estimate = 59.39 GiB` line and the 1689
        blocks derived from it on dev.
        """
        nbytes = 127_538_298_880
        got = divide_by_chiplet_replication(nbytes, 2, 4)
        assert got == 63_769_149_440
        assert got / 2**30 == pytest.approx(59.3896, abs=1e-4)
        # vLLM turns bytes into blocks with page_size_bytes = 36 MiB.
        assert got // 37_748_736 == 1689

    def test_degenerate_inputs_are_a_no_op(self):
        assert chiplet_replication_factor(0, 4) == 1.0
        assert chiplet_replication_factor(2, 0) == 1.0
        assert divide_by_chiplet_replication(100, 0, 4) == 100
        assert divide_by_chiplet_replication(100, 2, 0) == 100


# ---------------------------------------------------------------------------
# who gets the exact replication factor
# ---------------------------------------------------------------------------
class TestReplicationFactorIsGated:
    """The exact replication factor applies only under dynamic KV.

    Measured on RBLN-CR03 (kernel 6 GiB, gmu 0.9); the figures are in DISAGREE below.
    """

    KERNEL = 6 * 2**30
    # kvh where the two formulas disagree -> (released GiB, exact GiB)
    DISAGREE = {
        3: (118.0, 88.5),
        5: (118.0, 73.75),
        6: (118.0, 88.5),
        7: (118.0, 103.25),
        9: (118.0, 88.5),
        10: (118.0, 98.3333),
    }
    AGREE = (1, 2, 4, 8, 12, 16)

    def _measure(
        self,
        mock_envs,
        mock_platform,
        num_kv_heads,
        dynamic_kv,
        sysfs_total=REBEL_DRAM_NBYTES,
        sysfs_error=None,
    ):
        """Measure with the card DRAM capacity pinned, not read off the host.

        The CI fleet is ATOM (~15.7 GiB), so letting the RBLN-CR branch read the
        real card makes every figure in this class ~9x too small. `sysfs_error`
        exists because a caller's own patch of the reader would lose to the one
        applied here.
        """
        mock_platform.get_device_name.return_value = "RBLN-CR03"
        mock_envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = 1
        mock_envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE = dynamic_kv

        def read_card_dram_total_bytes() -> int | None:
            if sysfs_error is not None:
                raise sysfs_error
            return sysfs_total

        with patch(
            "vllm_rbln.v1.worker.utils.read_rbln_card_dram_total_bytes",
            read_card_dram_total_bytes,
        ):
            return estimate_available_memory(
                _make_model_config(num_kv_heads=num_kv_heads),
                _make_parallel_config(tp_size=1),
                kernel_size=self.KERNEL,
                gpu_memory_utilization=0.9,
            )

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @pytest.mark.parametrize("num_kv_heads", sorted(DISAGREE))
    def test_default_path_keeps_the_released_number(
        self, mock_envs, mock_platform, num_kv_heads
    ):
        got = self._measure(mock_envs, mock_platform, num_kv_heads, False)
        released, exact = self.DISAGREE[num_kv_heads]
        assert got / 2**30 == pytest.approx(released, abs=1e-3)
        # ... and is emphatically NOT the exact one.
        assert got / 2**30 != pytest.approx(exact, abs=1e-3)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @pytest.mark.parametrize("num_kv_heads", sorted(DISAGREE))
    def test_flag_applies_the_exact_factor(
        self, mock_envs, mock_platform, num_kv_heads
    ):
        got = self._measure(mock_envs, mock_platform, num_kv_heads, True)
        _released, exact = self.DISAGREE[num_kv_heads]
        assert got / 2**30 == pytest.approx(exact, abs=1e-3)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @pytest.mark.parametrize("num_kv_heads", AGREE)
    def test_flag_changes_nothing_where_the_formulas_agree(
        self, mock_envs, mock_platform, num_kv_heads
    ):
        """Covers the kvh=2 case, where replication applies."""
        off = self._measure(mock_envs, mock_platform, num_kv_heads, False)
        on = self._measure(mock_envs, mock_platform, num_kv_heads, True)
        assert off == on

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_the_released_number_is_never_the_smaller_one(
        self, mock_envs, mock_platform
    ):
        """The flag can only shrink the estimate, so it cannot introduce an OOM."""
        for num_kv_heads in sorted(self.DISAGREE) + list(self.AGREE):
            off = self._measure(mock_envs, mock_platform, num_kv_heads, False)
            on = self._measure(mock_envs, mock_platform, num_kv_heads, True)
            assert on <= off, num_kv_heads

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_disagreement_is_reported_not_silent(
        self, mock_envs, mock_platform, caplog
    ):
        """Only the flag path warns -- it is the only one that acts on this."""
        with caplog.at_level("WARNING"):
            self._measure(mock_envs, mock_platform, 5, True)
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelname == "WARNING" and "KV cache replication" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "num_key_value_heads=5" in warnings[0]
        assert "118.000 GiB released" in warnings[0]
        assert "73.750 GiB exact" in warnings[0]

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_the_default_path_does_not_warn_about_an_estimate_it_keeps(
        self, mock_envs, mock_platform, caplog
    ):
        """With the flag off this function changes nothing, so it says nothing.

        The disagreement is still recorded at debug level -- it is the reason to
        turn the flag on, not a fault in the run that did not.
        """
        with caplog.at_level("DEBUG", logger="vllm_rbln.v1.worker.utils"):
            self._measure(mock_envs, mock_platform, 5, False)
        by_level = [
            (r.levelname, r.getMessage())
            for r in caplog.records
            if "KV cache replication" in r.getMessage()
        ]
        assert [level for level, _ in by_level] == ["DEBUG"]

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_no_warning_when_the_formulas_agree(self, mock_envs, mock_platform, caplog):
        with caplog.at_level("WARNING"):
            self._measure(mock_envs, mock_platform, 2, False)
        assert not [
            r for r in caplog.records if "KV cache replication" in r.getMessage()
        ]

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_heterogeneous_cards_do_not_break_the_static_path(
        self, mock_envs, mock_platform, caplog
    ):
        """The per-chiplet reader refuses to answer; this path must not die.

        `read_rbln_card_dram_total_bytes` raises when the visible cards report
        different capacities, because a single per-chiplet budget would be
        meaningless. This estimate only wants one card's capacity, so it falls
        back instead of turning a mixed host into a start-up failure.
        """
        with caplog.at_level("WARNING"):
            got = self._measure(
                mock_envs,
                mock_platform,
                8,
                True,
                sysfs_error=RuntimeError(
                    "visible RBLN cards report different dram_total"
                ),
            )
        # 144 GiB - 4 GiB, the literal, == what sysfs reports on a uniform host.
        assert got / 2**30 == pytest.approx(118.0, abs=1e-3)
        assert any("different dram_total" in m for m in caplog.messages)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @pytest.mark.parametrize(
        "sysfs_total",
        [
            None,  # no sysfs at all -> the constant
            150_323_855_360,  # 140.0 GiB, what current drivers report
            REBEL_DRAM_NBYTES,  # exactly the ceiling
        ],
    )
    def test_sysfs_capacity_never_raises_the_estimate(
        self, mock_envs, mock_platform, sysfs_total
    ):
        """The reader clamps, so nothing above the ceiling can reach here."""
        got = self._measure(mock_envs, mock_platform, 8, True, sysfs_total=sysfs_total)
        assert got / 2**30 == pytest.approx(118.0, abs=1e-3)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    def test_a_smaller_card_does_lower_the_estimate(self, mock_envs, mock_platform):
        """The clamp is one-directional: less DRAM must still shrink the estimate.

        Without this, reverting to "just use the constant" would keep every other
        test in this class green -- and reporting *less* is the whole reason to
        read the device in the first place.
        """
        got = self._measure(
            mock_envs,
            mock_platform,
            8,
            True,
            sysfs_total=REBEL_DRAM_NBYTES - 4 * 2**30,
        )
        assert got / 2**30 == pytest.approx(118.0 - 4 * 0.9, abs=1e-3)

    @patch("vllm_rbln.v1.worker.utils.current_platform")
    @patch("vllm_rbln.v1.worker.utils.envs")
    @pytest.mark.parametrize(
        "sysfs_total", [None, REBEL_DRAM_NBYTES - 4 * 2**30, REBEL_DRAM_NBYTES]
    )
    def test_the_default_path_never_reads_the_device(
        self, mock_envs, mock_platform, sysfs_total
    ):
        """With the flag off the estimate is the constant, whatever sysfs says.

        The two figures agree on every card measured, so reading the driver here
        would change nothing today -- but it would tie the default path's KV size
        to a driver release. That is not this feature's call to make.
        """
        got = self._measure(mock_envs, mock_platform, 8, False, sysfs_total=sysfs_total)
        assert got / 2**30 == pytest.approx(118.0, abs=1e-3)


# ---------------------------------------------------------------------------
# sysfs DRAM readers
# ---------------------------------------------------------------------------
class TestRblnSysfsReaders:
    def test_visible_indices_from_env(self):
        with patch.dict(os.environ, {"RBLN_DEVICES": "2,0,1"}):
            assert get_rbln_visible_card_indices() == [0, 1, 2]

    def test_owned_indices_resolve_container_local_numbering(self, tmp_path):
        """`RBLN_DEVICES` indexes the container's devices, not sysfs card names.

        Measured on a container exposing /dev/rbln4..7: `RBLN_DEVICES=4,5,6,7`
        enumerates zero logical devices, `0,1,2,3` works, and holding `rbln:0`
        there put the context on physical rbln4. So entry `i` is the `i`-th
        present device and must not be used as a sysfs name.
        """
        for index in (4, 5, 6, 7):
            (tmp_path / f"rbln{index}").touch()
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0,1,2,3"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(tmp_path)),
        ):
            assert get_rbln_owned_card_indices() == [4, 5, 6, 7]

    def test_owned_indices_accept_physical_names_too(self, tmp_path):
        """On a container that does hold card 0 the two spellings coincide, and
        an entry that is already a physical name must not be dropped."""
        for index in (4, 5, 6, 7):
            (tmp_path / f"rbln{index}").touch()
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "4,5,6,7"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(tmp_path)),
        ):
            assert get_rbln_owned_card_indices() == [4, 5, 6, 7]

    def test_owned_indices_fall_back_without_device_nodes(self, tmp_path):
        """No /dev/rbln* (unit env, host without the driver) keeps the previous
        behaviour exactly rather than guessing."""
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "1,2"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(tmp_path)),
        ):
            assert get_rbln_owned_card_indices() == [1, 2]

    def test_dram_used_ignores_cards_we_do_not_own(self, tmp_path):
        """The regression this guards: a neighbouring container's card was read
        and charged against this worker, cutting a PP4 run from 1024 blocks to
        308 and, with a larger neighbour, driving the budget negative."""
        dev = tmp_path / "dev"
        dev.mkdir()
        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        # We own rbln4-7 (idle). rbln0-3 belong to someone else and are busy.
        for index in (4, 5, 6, 7):
            (dev / f"rbln{index}").touch()
        for index in (0, 1, 2, 3):
            card = sysfs / f"rbln{index}"
            card.mkdir()
            (card / "dram_used").write_text("13461618688\n")
        for index in (4, 5, 6, 7):
            card = sysfs / f"rbln{index}"
            card.mkdir()
            (card / "dram_used").write_text("0\n")
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0,1,2,3"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(dev)),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(sysfs)),
        ):
            assert read_rbln_card_dram_used_bytes() == 0

    def test_dram_used_still_sees_a_tenant_on_our_own_card(self, tmp_path):
        """The scope fix must not make the reader blind: a tenant sharing one of
        our own cards still has to be charged."""
        dev = tmp_path / "dev"
        dev.mkdir()
        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        for index in (4, 5):
            (dev / f"rbln{index}").touch()
            card = sysfs / f"rbln{index}"
            card.mkdir()
        (sysfs / "rbln4" / "dram_used").write_text("0\n")
        (sysfs / "rbln5" / "dram_used").write_text("2048\n")
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0,1"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(dev)),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(sysfs)),
        ):
            assert read_rbln_card_dram_used_bytes() == 2048

    def test_visible_indices_fall_back_to_sysfs_listing(self, tmp_path):
        for index in (0, 3, 1):
            (tmp_path / f"rbln{index}").mkdir()
        (tmp_path / "rsd0").mkdir()
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": ""}),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(tmp_path)),
        ):
            assert get_rbln_visible_card_indices() == [0, 1, 3]

    def test_dram_total_is_none_without_sysfs(self, tmp_path):
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": ""}),
            patch(
                "vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR",
                str(tmp_path / "missing"),
            ),
        ):
            assert read_rbln_card_dram_total_bytes() is None
            assert read_rbln_card_dram_used_bytes() == 0

    def test_dram_total_reads_uniform_capacity(self, tmp_path):
        # RBLN_DEV_DIR must be patched too: both readers resolve RBLN_DEVICES
        # against the device nodes actually present, so leaving /dev alone makes
        # the result depend on the host's card numbering. On a container exposing
        # /dev/rbln4..7 "0,1" resolves to cards 4 and 5, which this fake sysfs
        # does not have, and the assertions below read 0.
        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        dev = tmp_path / "dev"
        dev.mkdir()
        for index in (0, 1):
            card = sysfs / f"rbln{index}"
            card.mkdir()
            (card / "dram_total").write_text("150323855360\n")
            (card / "dram_used").write_text(f"{index * 1024}\n")
            (dev / f"rbln{index}").touch()
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0,1"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(dev)),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(sysfs)),
        ):
            total = read_rbln_card_dram_total_bytes()
            assert total == 150_323_855_360
            # 140.0 GiB exactly, i.e. the value the old literal encoded; / 4
            # chiplets = the 35.0 GiB per-chiplet capacity.
            assert total / 2**30 == 140.0
            assert total // 4 == 35 * 2**30
            # dram_used is reported per card; take the worst case.
            assert read_rbln_card_dram_used_bytes() == 1024

    def test_heterogeneous_capacity_is_rejected(self, tmp_path):
        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        dev = tmp_path / "dev"
        dev.mkdir()
        for index, size in ((0, 150323855360), (1, 75161927680)):
            card = sysfs / f"rbln{index}"
            card.mkdir()
            (card / "dram_total").write_text(f"{size}\n")
            (dev / f"rbln{index}").touch()
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0,1"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(dev)),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(sysfs)),
            pytest.raises(RuntimeError, match="different dram_total"),
        ):
            read_rbln_card_dram_total_bytes()

    def test_dram_total_ignores_cards_we_do_not_own(self, tmp_path):
        """Capacity must come from our own cards, like `dram_used`.

        A container holding /dev/rbln4..7 with RBLN_DEVICES=0,1 owns physical
        cards 4 and 5. Reading the raw entry instead would report card 0's
        capacity -- somebody else's card, and a different SKU here.
        """
        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        dev = tmp_path / "dev"
        dev.mkdir()
        for index in (4, 5, 6, 7):
            card = sysfs / f"rbln{index}"
            card.mkdir()
            (card / "dram_total").write_text("150323855360\n")
            (dev / f"rbln{index}").touch()
        # Cards we do NOT own, deliberately a different capacity.
        for index in (0, 1):
            card = sysfs / f"rbln{index}"
            card.mkdir()
            (card / "dram_total").write_text("75161927680\n")

        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0,1"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(dev)),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(sysfs)),
        ):
            # Ours (4, 5), not the raw entries (0, 1). Reading the raw entries
            # would return 75161927680 instead.
            assert read_rbln_card_dram_total_bytes() == 150_323_855_360

    @pytest.mark.parametrize(
        ("reported", "expected"),
        [
            (150_323_855_360, 150_323_855_360),  # today's driver, unchanged
            (REBEL_DRAM_NBYTES, REBEL_DRAM_NBYTES),  # exactly the ceiling
            (144 * 2**30, REBEL_DRAM_NBYTES),  # raw capacity -> clamped
            (200 * 2**30, REBEL_DRAM_NBYTES),  # anything larger -> clamped
            (100 * 2**30, 100 * 2**30),  # a smaller card passes through
        ],
    )
    def test_dram_total_is_clamped_in_the_reader(self, tmp_path, reported, expected):
        """The clamp belongs to the reader, not to one call site.

        Both callers size real allocations from this number: the static estimate
        every model takes, and the per-chiplet dynamic-KV budget that feeds
        `max_num_blocks`. A clamp applied at only one of them leaves the other
        over-committing the device.
        """
        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        dev = tmp_path / "dev"
        dev.mkdir()
        card = sysfs / "rbln0"
        card.mkdir()
        (card / "dram_total").write_text(f"{reported}\n")
        (dev / "rbln0").touch()
        with (
            patch.dict(os.environ, {"RBLN_DEVICES": "0"}),
            patch("vllm_rbln.v1.worker.utils.RBLN_DEV_DIR", str(dev)),
            patch("vllm_rbln.v1.worker.utils.RBLN_SYSFS_CLASS_DIR", str(sysfs)),
        ):
            assert read_rbln_card_dram_total_bytes() == expected


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

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_basic_single_rank(self, mock_nodes, mock_cpus):
        cpus = self._simple_cpu_list()
        mock_nodes.return_value = [0, 1]
        mock_cpus.return_value = cpus

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

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_rank_round_robins_numa_nodes(self, mock_nodes, mock_cpus):
        cpus = self._simple_cpu_list()
        mock_nodes.return_value = [0, 1]
        mock_cpus.return_value = cpus
        parallel_cfg = _make_parallel_config(tp_size=2)

        r0 = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        r1 = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)

        # Different ranks should get different NUMA nodes
        r0_ids = set(int(x) for x in r0.split(","))
        r1_ids = set(int(x) for x in r1.split(","))
        assert r0_ids.isdisjoint(r1_ids), "Ranks should not share CPUs"

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_no_available_numa_returns_all(self, mock_nodes, mock_cpus):
        """If allowed NUMA nodes don't have CPUs, return 'all'."""
        mock_nodes.return_value = []
        mock_cpus.return_value = []

        parallel_cfg = _make_parallel_config()
        result = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        assert result == "all"

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_cpu_selector_filters_threads(self, mock_nodes, mock_cpus):
        """cpu_selector=lambda cpus: cpus[:1] should pick one thread per core."""
        cpus = self._simple_cpu_list()
        mock_nodes.return_value = [0, 1]
        mock_cpus.return_value = cpus

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

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_multiple_ranks_same_numa_exclusive_allocation(self, mock_nodes, mock_cpus):
        """When 2 ranks map to the same NUMA node, CPUs are split."""
        # Single NUMA node with 4 cores, 1 thread each
        cpus = [_make_cpu(i, i, 0) for i in range(4)]
        mock_nodes.return_value = [0]
        mock_cpus.return_value = cpus

        parallel_cfg = _make_parallel_config(tp_size=2)

        r0 = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        r1 = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)

        r0_ids = set(int(x) for x in r0.split(","))
        r1_ids = set(int(x) for x in r1.split(","))
        assert r0_ids.isdisjoint(r1_ids)
        assert len(r0_ids) + len(r1_ids) == 4

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_uneven_cpu_split(self, mock_nodes, mock_cpus):
        """3 CPUs split between 2 ranks: one gets 2, other gets 1."""
        cpus = [_make_cpu(i, i, 0) for i in range(3)]
        mock_nodes.return_value = [0]
        mock_cpus.return_value = cpus

        parallel_cfg = _make_parallel_config(tp_size=2)

        r0 = get_autobind_cpu_ids(0, 0, parallel_cfg, lambda cpus: cpus)
        r1 = get_autobind_cpu_ids(1, 1, parallel_cfg, lambda cpus: cpus)

        r0_count = len(r0.split(","))
        r1_count = len(r1.split(","))
        assert {r0_count, r1_count} == {1, 2}

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_dp_rank_affects_binding(self, mock_nodes, mock_cpus):
        """Data parallelism changes rank_across_dp calculation."""
        cpus = [_make_cpu(i, i, 0) for i in range(8)]
        mock_nodes.return_value = [0]
        mock_cpus.return_value = cpus

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

    @patch("vllm_rbln.v1.worker.utils.get_allowed_cpu_list")
    @patch("vllm_rbln.v1.worker.utils.get_visible_memory_node")
    def test_empty_allocation_returns_all(self, mock_nodes, mock_cpus):
        """If cpu_selector returns empty lists, should fallback to 'all'."""
        cpus = [_make_cpu(0, 0, 0)]
        mock_nodes.return_value = [0]
        mock_cpus.return_value = cpus

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
