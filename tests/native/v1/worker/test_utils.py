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

# Configs are SimpleNamespace stubs. For the environment-dependent functions
# (device DRAM, NUMA, CPU affinity) only the inputs are mocked and the real
# computed values asserted.

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.platforms import CpuArchEnum, current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.cpu_resource_utils import LogicalCPUInfo
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import vllm_rbln.envs as envs
import vllm_rbln.v1.worker.utils as worker_utils
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec
from vllm_rbln.v1.worker.utils import (
    REBEL_DRAM_NBYTES,
    chiplet_replication_factor,
    compute_rbln_local_omp_cpuid,
    divide_by_chiplet_replication,
    estimate_available_memory,
    estimate_model_kernel_size,
    get_autobind_cpu_ids,
    get_kv_cache_names,
    get_rbln_owned_card_indices,
    get_rbln_planned_affinity_cpu_count,
    get_rbln_visible_card_indices,
    prepare_kernel_block_sizes,
    read_rbln_card_dram_total_bytes,
    read_rbln_card_dram_used_bytes,
    reorder_input_batch,
    set_cpu_affinity,
    set_omp_num_threads,
)

_GB = 2**30


def _make_model_config(
    num_layers=32, vocab_size=32000, hidden_size=4096, num_kv_heads=8
):
    # Duck-typed stub exposing the config getters the functions read.
    cfg = SimpleNamespace()
    cfg.get_num_layers = lambda pc: num_layers
    cfg.get_vocab_size = lambda: vocab_size
    cfg.get_hidden_size = lambda: hidden_size
    cfg.get_num_kv_heads = lambda pc: num_kv_heads
    return cfg


def _make_parallel_config(
    tp_size=1,
    dp_size=1,
    world_size=None,
    world_size_across_dp=None,
    data_parallel_rank=0,
    local_rank=0,
):
    ws = tp_size if world_size is None else world_size
    return SimpleNamespace(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        world_size=ws,
        world_size_across_dp=ws
        if world_size_across_dp is None
        else world_size_across_dp,
        data_parallel_rank=data_parallel_rank,
    )


def _make_cpu(cpu_id, physical_core, numa_node):
    return LogicalCPUInfo(id=cpu_id, physical_core=physical_core, numa_node=numa_node)


def _cpus_8():
    # 8 CPUs, 2 NUMA nodes, 2 physical cores/node, 2 threads/core.
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


def _kv_config(*specs):
    # prepare_kernel_block_sizes only reads kv_cache_groups[i].kv_cache_spec, so
    # a duck-typed config avoids KVCacheConfig's cross-group validation.
    groups = [SimpleNamespace(kv_cache_spec=s) for s in specs]
    return SimpleNamespace(kv_cache_groups=groups)


def _backend(sizes):
    # A backend type exposing the classmethod select_common_block_size calls.
    return type(
        "FakeBackend",
        (),
        {"get_supported_kernel_block_sizes": classmethod(lambda cls: list(sizes))},
    )


def _full_attn(block_size):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.float16
    )


class TestEstimateModelKernelSize:
    def test_bytes_path_exact(self):
        # n_model_bytes path, default bits passed explicitly (no platform mock).
        result = estimate_model_kernel_size(
            _make_model_config(num_layers=2, vocab_size=64, hidden_size=32),
            _make_parallel_config(tp_size=1),
            n_model_bytes=12_288,
            default_bits_per_param=16,
        )
        assert result == 6_291_456

    def test_params_path_exact(self):
        # n_model_params + nbits_per_param path.
        result = estimate_model_kernel_size(
            _make_model_config(num_layers=4, vocab_size=128, hidden_size=64),
            _make_parallel_config(tp_size=1),
            n_model_params=1_000_000,
            nbits_per_param=16,
            default_bits_per_param=16,
        )
        assert result == 10_485_760

    def test_tp_scaling(self):
        # tp_size changes lm_heads_nbytes (align per-tp, then multiply by tp).
        mc = _make_model_config(num_layers=2, vocab_size=64, hidden_size=32)
        common = dict(n_model_bytes=12_288, default_bits_per_param=16)
        r1 = estimate_model_kernel_size(mc, _make_parallel_config(tp_size=1), **common)
        r2 = estimate_model_kernel_size(mc, _make_parallel_config(tp_size=2), **common)
        assert r1 == 6_291_456
        assert r2 == 8_388_608
        assert r2 != r1

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "Either `n_model_params` or `n_model_bytes`"),
            (
                {"n_model_params": 1_000_000, "n_model_bytes": 2_000_000},
                "Only one of",
            ),
            ({"n_model_params": 1_000_000}, "nbits_per_param"),
        ],
    )
    def test_validates_input_combinations(self, kwargs, match):
        # neither params/bytes, both, or params without nbits -> ValueError.
        with pytest.raises(ValueError, match=match):
            estimate_model_kernel_size(
                _make_model_config(),
                _make_parallel_config(),
                default_bits_per_param=16,
                **kwargs,
            )

    def test_default_bits_from_platform(self, monkeypatch):
        # With default_bits_per_param=None the bits come from the device name:
        # ATOM/REBEL -> 16, unknown RBLN -> ValueError, non-rbln -> assert.
        mc = _make_model_config(num_layers=2, vocab_size=64, hidden_size=32)
        pc = _make_parallel_config(tp_size=1)

        def _name(name):
            monkeypatch.setattr(current_platform, "get_device_name", lambda: name)

        _name("RBLN-CA25")
        assert estimate_model_kernel_size(mc, pc, n_model_bytes=12_288) == 6_291_456
        _name("RBLN-CR13")
        assert estimate_model_kernel_size(mc, pc, n_model_bytes=12_288) == 6_291_456
        _name("RBLN-XX99")
        with pytest.raises(ValueError, match="invalid RBLN architecture"):
            estimate_model_kernel_size(mc, pc, n_model_bytes=12_288)
        _name("cpu")
        with pytest.raises(AssertionError):
            estimate_model_kernel_size(mc, pc, n_model_bytes=12_288)


class TestGetKvCacheNames:
    @staticmethod
    def _caches(*names):
        return {n: torch.empty(0) for n in names}

    def test_orders_by_layer_index(self):
        # Returned ordered by extracted layer index, not dict insertion order.
        caches = self._caches(
            "model.layers.2.self_attn",
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
        )
        assert get_kv_cache_names(caches) == [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
            "model.layers.2.self_attn",
        ]

    def test_multi_layer_same_index_raises(self, monkeypatch):
        # Two layers at one index on a non-cpu/cuda/xpu platform -> not supported.
        monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
        monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
        monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
        caches = self._caches("model.layers.0.self_attn", "model.layers.0.cross_attn")
        with pytest.raises(NotImplementedError):
            get_kv_cache_names(caches)

    def test_multi_layer_same_index_allowed_on_cpu(self, monkeypatch):
        # The other side of the branch: on cpu/cuda/xpu both names are returned.
        monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
        monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
        monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
        caches = self._caches("model.layers.0.self_attn", "model.layers.0.cross_attn")
        names = get_kv_cache_names(caches)
        assert set(names) == set(caches)
        assert len(names) == 2


class TestPrepareKernelBlockSizes:
    def test_sliding_window_uses_window(self):
        # RBLNSlidingWindowSpec group -> kernel block size is its sliding_window.
        sw = RBLNSlidingWindowSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
            sliding_window=16,
        )
        assert prepare_kernel_block_sizes(_kv_config(sw), [[]]) == [16]

    def test_attention_uses_select_common_block_size(self):
        # AttentionSpec group -> select_common_block_size splits 32 to a backend-
        # supported 16.
        result = prepare_kernel_block_sizes(
            _kv_config(_full_attn(32)),
            [[SimpleNamespace(backend=_backend([16]))]],
        )
        assert result == [16]

    def test_mamba_uses_block_size(self):
        # MambaSpec group -> its own block_size (no virtual splitting).
        mamba = MambaSpec(block_size=8, shapes=((64,),), dtypes=(torch.float16,))
        assert prepare_kernel_block_sizes(_kv_config(mamba), [[]]) == [8]

    def test_encoder_only_skipped(self):
        # EncoderOnlyAttentionSpec group is skipped (absent from the result).
        enc = EncoderOnlyAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
        )
        assert prepare_kernel_block_sizes(_kv_config(enc), [[]]) == []

    def test_uniform_type_unwraps(self):
        # UniformTypeKVCacheSpecs unwraps to its inner FullAttentionSpec; the
        # block_size still comes from the (uniform) group spec.
        uni = UniformTypeKVCacheSpecs(
            block_size=32, kv_cache_specs={"l0": _full_attn(32)}
        )
        result = prepare_kernel_block_sizes(
            _kv_config(uni), [[SimpleNamespace(backend=_backend([16]))]]
        )
        assert result == [16]

    def test_unknown_spec_raises(self):
        # A spec matching none of the branches -> NotImplementedError.
        with pytest.raises(NotImplementedError):
            prepare_kernel_block_sizes(_kv_config(SimpleNamespace(block_size=8)), [[]])

    def test_mixed_groups_ordered(self):
        # Several groups -> block sizes returned in group order.
        sw = RBLNSlidingWindowSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
            sliding_window=16,
        )
        mamba = MambaSpec(block_size=8, shapes=((64,),), dtypes=(torch.float16,))
        result = prepare_kernel_block_sizes(
            _kv_config(sw, mamba, _full_attn(64)),
            [[], [], [SimpleNamespace(backend=_backend([64]))]],
        )
        assert result == [16, 8, 64]


def _input_batch(num_reqs=4):
    ib = InputBatch(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=1000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
    )
    for i in range(num_reqs):
        ib.add_request(
            CachedRequestState(
                req_id=f"r{i}",
                prompt_token_ids=[i * 10 + j for j in range(3 + i)],
                mm_features=None,
                sampling_params=SamplingParams(
                    temperature=0.4 * (i + 1), top_p=0.1 * (i + 1)
                ),
                pooling_params=None,
                generator=None,
                block_ids=([i, i + 1],),
                num_computed_tokens=0,
                output_token_ids=[],
            )
        )
    return ib


class TestReorderInputBatch:
    # perm=[1,0,3,2] is two disjoint transpositions, so reorder(perm) must equal
    # swap_states(0,1) followed by swap_states(2,3).
    _PERM = np.array([1, 0, 3, 2])

    def test_reorder_matches_swap_states(self):
        # One vectorized reorder(perm) must leave every per-request field equal to
        # the swap_states sequence. token_ids is only reindexed over the
        # meaningful columns, so compare just that region.
        a, b = _input_batch(4), _input_batch(4)
        reorder_input_batch(a, self._PERM)
        b.swap_states(0, 1)
        b.swap_states(2, 3)

        n = 4
        assert a._req_ids[:n] == b._req_ids[:n]
        assert a.req_id_to_index == b.req_id_to_index
        assert a.spec_token_ids[:n] == b.spec_token_ids[:n]
        for name in (
            "temperature_cpu",
            "top_p_cpu",
            "num_prompt_tokens",
            "num_tokens_no_spec",
            "num_computed_tokens_cpu",
            "request_lora_mapping",
        ):
            assert np.array_equal(getattr(a, name)[:n], getattr(b, name)[:n]), name
        valid = int(a.num_tokens_no_spec[:n].max())
        assert np.array_equal(a.token_ids_cpu[:n, :valid], b.token_ids_cpu[:n, :valid])
        for bt_a, bt_b in zip(a.block_table.block_tables, b.block_table.block_tables):
            assert np.array_equal(bt_a.block_table.np[:n], bt_b.block_table.np[:n])
            assert np.array_equal(
                bt_a.num_blocks_per_row[:n], bt_b.num_blocks_per_row[:n]
            )

    def test_updates_req_id_to_index(self):
        # req_id_to_index is rebuilt so each request id maps to its new slot.
        a = _input_batch(4)
        reorder_input_batch(a, self._PERM)
        assert a.req_id_to_index == {"r1": 0, "r0": 1, "r3": 2, "r2": 3}

    def test_pooling_model_skips_sampling_fields(self):
        # is_pooling_model short-circuits: request fields are permuted but the
        # sampling fields (temperature) are left untouched.
        a = _input_batch(4)
        before_temp = a.temperature_cpu[:4].copy()
        a.is_pooling_model = True
        reorder_input_batch(a, self._PERM)
        assert a._req_ids[:4] == ["r1", "r0", "r3", "r2"]
        assert np.array_equal(a.temperature_cpu[:4], before_temp)

    def test_index_keyed_dicts_remapped(self):
        # Index-keyed dicts (generators, bad_words_token_ids -- same remap pattern
        # as req_prompt_embeds): new slot k inherits old slot perm[k]'s entry.
        a = _input_batch(4)
        g0, g2 = object(), object()
        a.generators = {0: g0, 2: g2}
        a.bad_words_token_ids = {0: [[1, 2]]}
        reorder_input_batch(a, self._PERM)
        assert a.generators == {1: g0, 3: g2}
        assert a.bad_words_token_ids == {1: [[1, 2]]}


class TestEstimateAvailableMemory:
    @pytest.fixture
    def rbln(self, monkeypatch):
        # Mock only the environment inputs (device name + devices-per-rank).
        def _set(device_name, rsd=1):
            monkeypatch.setattr(
                current_platform, "get_device_name", lambda: device_name
            )
            monkeypatch.setattr(envs, "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", rsd)

        return _set

    def test_atom_exact(self, rbln):
        rbln("RBLN-CA25", rsd=1)
        assert (
            estimate_available_memory(
                _make_model_config(), _make_parallel_config(), kernel_size=_GB
            )
            == 13_579_478_630
        )

    def test_rebel_exact(self, rbln):
        rbln("RBLN-CR13", rsd=1)
        assert (
            estimate_available_memory(
                _make_model_config(), _make_parallel_config(), kernel_size=_GB
            )
            == 132_070_244_352
        )

    def test_rebel_requires_rsd_1(self, rbln):
        rbln("RBLN-CR13", rsd=2)
        with pytest.raises(AssertionError):
            estimate_available_memory(
                _make_model_config(), _make_parallel_config(), kernel_size=_GB
            )

    def test_unknown_device_raises(self, rbln):
        rbln("RBLN-XX99", rsd=1)
        with pytest.raises(ValueError, match="invalid RBLN architecture"):
            estimate_available_memory(
                _make_model_config(), _make_parallel_config(), kernel_size=_GB
            )

    def test_gpu_memory_utilization_effect(self, rbln):
        rbln("RBLN-CA25", rsd=1)
        mc, pc = _make_model_config(), _make_parallel_config()
        high = estimate_available_memory(
            mc, pc, kernel_size=_GB, gpu_memory_utilization=0.9
        )
        low = estimate_available_memory(
            mc, pc, kernel_size=_GB, gpu_memory_utilization=0.45
        )
        assert low < high

    def test_oom_raises_memory_error(self, rbln):
        rbln("RBLN-CA25", rsd=1)
        with pytest.raises(MemoryError):
            estimate_available_memory(
                _make_model_config(), _make_parallel_config(), kernel_size=100 * _GB
            )

    def test_rsd_replicas_for_large_kv_heads(self, rbln):
        # num_kv_heads only feeds rsd_replicas = max(1, rsd // num_kv_heads).
        rbln("RBLN-CA25", rsd=4)
        pc = _make_parallel_config()
        replica2 = estimate_available_memory(
            _make_model_config(num_kv_heads=2), pc, kernel_size=_GB
        )
        replica1 = estimate_available_memory(
            _make_model_config(num_kv_heads=8), pc, kernel_size=_GB
        )
        assert replica2 == replica1 // 2

    def test_buffer_default_vs_explicit(self, rbln):
        rbln("RBLN-CA25", rsd=1)
        mc, pc = _make_model_config(), _make_parallel_config()
        default = estimate_available_memory(mc, pc, kernel_size=_GB)
        no_buffer = estimate_available_memory(mc, pc, kernel_size=_GB, buffer=0)
        # Default buffer = 256MB/runtime * num_runtimes (2 * rsd = 2) = 2**29.
        assert no_buffer - default == 2**29

    def test_validation_combinations(self, rbln):
        rbln("RBLN-CA25", rsd=1)
        mc, pc = _make_model_config(), _make_parallel_config()
        with pytest.raises(ValueError, match="cannot both be"):
            estimate_available_memory(mc, pc, kernel_size=_GB, n_model_params=1_000_000)
        with pytest.raises(ValueError, match="Either"):
            estimate_available_memory(mc, pc)
        with pytest.raises(ValueError, match="nbits_per_param"):
            estimate_available_memory(mc, pc, n_model_params=1_000_000)

    def test_estimates_kernel_when_not_given(self, rbln):
        rbln("RBLN-CA25", rsd=1)
        result = estimate_available_memory(
            _make_model_config(), _make_parallel_config(), n_model_bytes=10 * _GB
        )
        assert result > 0


class TestGetAutobindCpuIds:
    @pytest.fixture
    def cpus(self, monkeypatch):
        cpu_list = _cpus_8()
        monkeypatch.setattr(worker_utils, "get_visible_memory_node", lambda: [0, 1])
        monkeypatch.setattr(worker_utils, "get_allowed_cpu_list", lambda: cpu_list)
        return cpu_list

    @staticmethod
    def _all(cpus):
        return cpus

    def test_basic_single_rank(self, cpus):
        # rank 0 -> NUMA 0 -> all of NUMA-0's CPUs (sorted by id).
        result = get_autobind_cpu_ids(0, 0, _make_parallel_config(tp_size=1), self._all)
        assert result == "0,1,4,5"

    def test_rank_round_robins_numa_nodes(self, cpus):
        # rank 1 -> NUMA 1 (rank_across_dp % num_numa).
        result = get_autobind_cpu_ids(1, 0, _make_parallel_config(tp_size=1), self._all)
        assert result == "2,3,6,7"

    def test_no_available_numa_returns_all(self, monkeypatch):
        # No allowed NUMA node overlaps the CPU list -> "all".
        monkeypatch.setattr(worker_utils, "get_visible_memory_node", lambda: [9])
        monkeypatch.setattr(worker_utils, "get_allowed_cpu_list", lambda: _cpus_8())
        assert get_autobind_cpu_ids(0, 0, _make_parallel_config(), self._all) == "all"

    def test_cpu_selector_filters_threads(self, cpus):
        # cpus[:1] keeps one logical CPU per physical core.
        result = get_autobind_cpu_ids(
            0, 0, _make_parallel_config(tp_size=1), lambda c: c[:1]
        )
        assert result == "0,1"

    def test_multiple_ranks_same_numa_exclusive(self, cpus):
        # tp=4: ranks 0 and 2 share NUMA 0 -> disjoint halves.
        pc = _make_parallel_config(tp_size=4, world_size=4, world_size_across_dp=4)
        r0 = get_autobind_cpu_ids(0, 0, pc, self._all)
        r2 = get_autobind_cpu_ids(2, 0, pc, self._all)
        assert r0 == "0,1"
        assert r2 == "4,5"
        assert set(r0.split(",")).isdisjoint(r2.split(","))

    def test_uneven_split_distributes_remainder(self, monkeypatch):
        # NUMA 0 with 3 CPUs shared by 2 ranks -> 2 + 1 (remainder to the first).
        cpu_list = [
            _make_cpu(0, 0, 0),
            _make_cpu(1, 1, 0),
            _make_cpu(2, 2, 0),
            _make_cpu(3, 3, 1),
        ]
        monkeypatch.setattr(worker_utils, "get_visible_memory_node", lambda: [0, 1])
        monkeypatch.setattr(worker_utils, "get_allowed_cpu_list", lambda: cpu_list)
        pc = _make_parallel_config(tp_size=3, world_size=3, world_size_across_dp=3)
        r0 = get_autobind_cpu_ids(0, 0, pc, self._all)  # ranks 0,2 -> NUMA 0
        r2 = get_autobind_cpu_ids(2, 0, pc, self._all)
        assert len(r0.split(",")) == 2
        assert len(r2.split(",")) == 1

    def test_dp_rank_affects_binding(self, cpus):
        # data_parallel_rank shifts rank_across_dp -> different NUMA node.
        pc0 = _make_parallel_config(
            tp_size=1,
            dp_size=2,
            world_size=1,
            world_size_across_dp=2,
            data_parallel_rank=0,
        )
        pc1 = _make_parallel_config(
            tp_size=1,
            dp_size=2,
            world_size=1,
            world_size_across_dp=2,
            data_parallel_rank=1,
        )
        assert get_autobind_cpu_ids(0, 0, pc0, self._all) != get_autobind_cpu_ids(
            0, 0, pc1, self._all
        )

    def test_empty_allocation_returns_all(self, monkeypatch):
        # A selector that drops every CPU -> "all" fallback.
        monkeypatch.setattr(worker_utils, "get_visible_memory_node", lambda: [0, 1])
        monkeypatch.setattr(worker_utils, "get_allowed_cpu_list", lambda: _cpus_8())
        assert (
            get_autobind_cpu_ids(0, 0, _make_parallel_config(), lambda c: []) == "all"
        )


class TestComputeRblnLocalOmpCpuid:
    @pytest.fixture
    def cpus(self, monkeypatch):
        monkeypatch.setattr(worker_utils, "get_visible_memory_node", lambda: [0, 1])
        monkeypatch.setattr(worker_utils, "get_allowed_cpu_list", lambda: _cpus_8())

    def _linux_arch(self, monkeypatch, arch):
        monkeypatch.setattr(envs, "VLLM_RBLN_NUMA", True)
        monkeypatch.setattr(worker_utils.platform, "system", lambda: "Linux")
        monkeypatch.setattr(current_platform, "get_cpu_architecture", lambda: arch)

    def test_numa_disabled_returns_nobind(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_NUMA", False)
        monkeypatch.setattr(worker_utils.platform, "system", lambda: "Linux")
        assert compute_rbln_local_omp_cpuid(0, 0, _make_parallel_config()) == "nobind"

    def test_non_linux_returns_nobind(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_NUMA", True)
        monkeypatch.setattr(worker_utils.platform, "system", lambda: "Darwin")
        assert compute_rbln_local_omp_cpuid(0, 0, _make_parallel_config()) == "nobind"

    def test_x86_uses_first_thread_selector(self, cpus, monkeypatch):
        # x86 -> 1 CPU per physical core.
        self._linux_arch(monkeypatch, CpuArchEnum.X86)
        assert compute_rbln_local_omp_cpuid(0, 0, _make_parallel_config()) == "0,1"

    def test_powerpc_uses_smt_selector(self, monkeypatch):
        # POWERPC SMT: id % 8 < 4 keeps multiple threads per core here (unlike
        # x86's one-per-core), so both threads of each core survive.
        cpu_list = [
            _make_cpu(0, 0, 0),
            _make_cpu(1, 0, 0),  # core 0
            _make_cpu(2, 1, 0),
            _make_cpu(3, 1, 0),  # core 1
        ]
        monkeypatch.setattr(worker_utils, "get_visible_memory_node", lambda: [0])
        monkeypatch.setattr(worker_utils, "get_allowed_cpu_list", lambda: cpu_list)
        self._linux_arch(monkeypatch, CpuArchEnum.POWERPC)
        assert compute_rbln_local_omp_cpuid(0, 0, _make_parallel_config()) == "0,1,2,3"

    def test_other_arch_returns_nobind(self, cpus, monkeypatch):
        self._linux_arch(monkeypatch, CpuArchEnum.ARM)
        assert compute_rbln_local_omp_cpuid(0, 0, _make_parallel_config()) == "nobind"


class TestGetRblnPlannedAffinityCpuCount:
    def test_bound_list_counts_ids(self, monkeypatch):
        monkeypatch.setattr(
            worker_utils, "compute_rbln_local_omp_cpuid", lambda r, lr, pc: "0,1,2,3"
        )
        assert get_rbln_planned_affinity_cpu_count(0, 0, _make_parallel_config()) == 4

    def test_nobind_or_all_uses_current_affinity(self, monkeypatch):
        monkeypatch.setattr(
            worker_utils, "compute_rbln_local_omp_cpuid", lambda r, lr, pc: "nobind"
        )
        monkeypatch.setattr(worker_utils.os, "sched_getaffinity", lambda pid: {0, 1, 2})
        assert get_rbln_planned_affinity_cpu_count(0, 0, _make_parallel_config()) == 3


class TestSetCpuAffinity:
    @staticmethod
    def _cpuid(monkeypatch, value):
        monkeypatch.setattr(
            worker_utils, "compute_rbln_local_omp_cpuid", lambda r, lr, pc: value
        )

    def test_nobind_makes_no_sched_call(self, monkeypatch):
        # cpuid "nobind" -> no syscall. (Its causes -- NUMA off / non-Linux /
        # unhandled arch -- are covered in TestComputeRblnLocalOmpCpuid.)
        self._cpuid(monkeypatch, "nobind")
        calls = []
        monkeypatch.setattr(
            worker_utils.os, "sched_setaffinity", lambda *a: calls.append(a)
        )
        set_cpu_affinity(0, 0, _make_parallel_config())
        assert calls == []

    def test_all_result_makes_no_sched_call(self, monkeypatch):
        self._cpuid(monkeypatch, "all")
        calls = []
        monkeypatch.setattr(
            worker_utils.os, "sched_setaffinity", lambda *a: calls.append(a)
        )
        set_cpu_affinity(0, 0, _make_parallel_config())
        assert calls == []

    def test_sets_affinity_with_computed_ids(self, monkeypatch):
        self._cpuid(monkeypatch, "0,1,2,3")
        calls = []
        monkeypatch.setattr(
            worker_utils.os,
            "sched_setaffinity",
            lambda pid, ids: calls.append((pid, list(ids))),
        )
        monkeypatch.setattr(
            worker_utils.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}
        )
        set_cpu_affinity(0, 0, _make_parallel_config())
        assert calls == [(0, [0, 1, 2, 3])]

    def test_affinity_mismatch_warns(self, monkeypatch):
        self._cpuid(monkeypatch, "0,1,2,3")
        monkeypatch.setattr(worker_utils.os, "sched_setaffinity", lambda pid, ids: None)
        # Applied mask differs from requested -> warns, does not raise.
        monkeypatch.setattr(worker_utils.os, "sched_getaffinity", lambda pid: {0, 1})
        set_cpu_affinity(0, 0, _make_parallel_config())

    def test_os_error_propagates(self, monkeypatch):
        self._cpuid(monkeypatch, "0,1,2,3")

        def _boom(pid, ids):
            raise OSError("denied")

        monkeypatch.setattr(worker_utils.os, "sched_setaffinity", _boom)
        with pytest.raises(OSError):
            set_cpu_affinity(0, 0, _make_parallel_config())


class TestSetOmpNumThreads:
    @pytest.fixture(autouse=True)
    def _restore(self, monkeypatch):
        # set_omp_num_threads mutates the process thread count + env; snapshot and
        # restore both so we can assert the real torch.get_num_threads() readback.
        saved = torch.get_num_threads()
        monkeypatch.delenv("RBLN_NUM_THREADS", raising=False)
        yield
        torch.set_num_threads(saved)

    def test_default_threads(self):
        set_omp_num_threads(0, 0)
        assert torch.get_num_threads() == 2
        assert os.environ["RBLN_NUM_THREADS"] == "2"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RBLN_NUM_THREADS", "8")
        set_omp_num_threads(0, 0)
        assert torch.get_num_threads() == 8

    def test_custom_default(self):
        set_omp_num_threads(0, 0, default_num_threads=4)
        assert torch.get_num_threads() == 4
        assert os.environ["RBLN_NUM_THREADS"] == "4"

    def test_env_takes_precedence_over_default(self, monkeypatch):
        monkeypatch.setenv("RBLN_NUM_THREADS", "16")
        set_omp_num_threads(0, 0, default_num_threads=4)
        assert torch.get_num_threads() == 16


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
