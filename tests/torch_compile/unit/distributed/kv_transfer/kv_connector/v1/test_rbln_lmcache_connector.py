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

"""Tests for rbln_lmcache_connector: RBLNConnector, RBLNLMCacheManager,
RBLNLMCacheConnectorV1Impl, RBLNLMCacheConnectorV1, and helpers."""

import gc
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import (
    CreateRBLNConnector,
    RBLNConnector,
)
from vllm_rbln.distributed.utils import calculate_local_rank_and_world_size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slot_mapping(num_tokens, offset=0):
    return torch.arange(offset, offset + num_tokens, dtype=torch.long)


def _init_xfer_buffers(conn, num_blocks=4):
    kv_caches = {
        name: torch.zeros(
            2, num_blocks, conn.num_kv_heads, 1,
            conn.block_size, conn.head_dim, dtype=conn.dtype,
        )
        for name in conn.kv_cache_names
    }
    conn.initialize_xfer_buffers(kv_caches)


def _make_connector(num_layers=2, block_size=16):
    runtime = MagicMock()
    kv_names = [f"layer_{i}" for i in range(num_layers)]
    conn = RBLNConnector(
        num_layers=num_layers, num_kv_heads=4, head_dim=64,
        block_size=block_size, dtype=torch.float16,
        kv_cache_names=kv_names, runtime_holder=[runtime],
    )
    kv_caches = {
        name: torch.zeros(2, 4, 4, 1, block_size, 64, dtype=torch.float16)
        for name in kv_names
    }
    conn.initialize_xfer_buffers(kv_caches)
    return conn


def _get_rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0


# ===================================================================
# RBLNConnector: init, shape, xfer buffers
# ===================================================================


class TestRBLNConnectorInit:
    def test_default_init(self):
        conn = RBLNConnector(
            num_layers=32, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
        )
        assert conn.runtime is None
        assert conn.kv_cache_names == []

    def test_hidden_dim_computation(self):
        conn = RBLNConnector(
            num_layers=32, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
        )
        assert conn.hidden_dim_size == 1024

    def test_factory_returns_correct_type(self):
        conn = CreateRBLNConnector(
            num_layers=32, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
        )
        assert isinstance(conn, RBLNConnector)


class TestGetShape:
    def test_shape_format(self):
        conn = RBLNConnector(
            num_layers=32, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
        )
        assert conn.get_shape(64) == torch.Size([2, 32, 64, 1024])

    def test_shape_single_token(self):
        conn = RBLNConnector(
            num_layers=32, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
        )
        assert conn.get_shape(1) == torch.Size([2, 32, 1, 1024])

    @pytest.mark.parametrize("num_kv_heads,head_dim", [
        (1, 64), (4, 256), (32, 128), (128, 32),
    ])
    def test_shape_various_head_configs(self, num_kv_heads, head_dim):
        conn = RBLNConnector(
            num_layers=16, num_kv_heads=num_kv_heads, head_dim=head_dim,
            block_size=16, dtype=torch.float16,
        )
        assert conn.get_shape(32)[3] == num_kv_heads * head_dim

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_shape_dtype_independent(self, dtype):
        conn = RBLNConnector(
            num_layers=4, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=dtype,
        )
        assert conn.get_shape(32) == torch.Size([2, 4, 32, 1024])


class TestXferBuffers:
    @staticmethod
    def _init_buffers(conn, kv_names, num_blocks=4):
        kv_caches = {
            name: torch.zeros(
                2, num_blocks, conn.num_kv_heads, 1,
                conn.block_size, conn.head_dim, dtype=conn.dtype,
            )
            for name in kv_names
        }
        conn.initialize_xfer_buffers(kv_caches)

    def test_xfer_buffers_are_4096_aligned(self):
        kv_names = ["layer_0", "layer_1"]
        conn = RBLNConnector(
            num_layers=2, num_kv_heads=8, head_dim=128,
            block_size=256, dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names)
        for name, buf in conn._xfer_buffers.items():
            assert buf.data_ptr() % 4096 == 0

    def test_xfer_buffers_per_layer(self):
        kv_names = ["layer_0", "layer_1", "layer_2"]
        conn = RBLNConnector(
            num_layers=3, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names)
        assert len(conn._xfer_buffers) == 3

    def test_xfer_buffer_shape(self):
        kv_names = ["layer_0"]
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names)
        assert conn._xfer_buffers["layer_0"].shape == (2, 4, 4, 1, 16, 64)

    def test_xfer_buffers_reused(self):
        conn = _make_connector()
        assert conn._xfer_buffers is conn._xfer_buffers


# ===================================================================
# RBLNConnector: slot grouping
# ===================================================================


class TestSlotGrouping:
    def test_contiguous_single_block(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        groups = conn._group_slots_by_block(torch.arange(16, dtype=torch.long))
        assert len(groups) == 1 and len(groups[0]) == 16

    def test_cross_block_boundary(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        groups = conn._group_slots_by_block(torch.arange(14, 18, dtype=torch.long))
        assert len(groups) == 2

    def test_scattered_slots(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        groups = conn._group_slots_by_block(torch.tensor([0, 32, 64], dtype=torch.long))
        assert len(groups) == 3


# ===================================================================
# RBLNConnector: D2H (from_gpu)
# ===================================================================


class TestD2H:
    def test_fetch_called_per_layer_per_block(self):
        runtime = MagicMock()
        kv_names = [f"layer_{i}" for i in range(4)]
        conn = RBLNConnector(
            num_layers=4, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
            kv_cache_names=kv_names, runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(32), dtype=torch.float16)
        conn.from_gpu(mem, 0, 32, slot_mapping=_make_slot_mapping(32))
        assert runtime.fetch_kv_cache.call_count == 8

    def test_fetch_passes_correct_params(self):
        runtime = MagicMock()
        kv_names = ["k_cache_0", "k_cache_1"]
        conn = RBLNConnector(
            num_layers=2, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=kv_names, runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)
        conn.from_gpu(mem, 0, 16, slot_mapping=_make_slot_mapping(16))
        assert runtime.fetch_kv_cache.call_count == 2
        for call, expected_name in zip(runtime.fetch_kv_cache.call_args_list, kv_names):
            assert call[0][2] == 0
            assert call[0][3] == 16
            assert call[0][4] == expected_name

    def test_missing_slot_mapping_raises(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=["layer_0"], runtime_holder=[MagicMock()],
        )
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)
        with pytest.raises(AssertionError, match="slot_mapping"):
            conn.from_gpu(mem, 0, 16)

    def test_none_tensor_raises(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=["layer_0"], runtime_holder=[MagicMock()],
        )
        mem = MagicMock()
        mem.tensor = None
        with pytest.raises(AssertionError):
            conn.from_gpu(mem, 0, 16, slot_mapping=_make_slot_mapping(16))


# ===================================================================
# RBLNConnector: H2D (to_gpu)
# ===================================================================


class TestH2D:
    def test_update_called_per_layer_per_block(self):
        runtime = MagicMock()
        kv_names = [f"layer_{i}" for i in range(4)]
        conn = RBLNConnector(
            num_layers=4, num_kv_heads=8, head_dim=128,
            block_size=16, dtype=torch.float16,
            kv_cache_names=kv_names, runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        mem = MagicMock()
        mem.tensor = torch.randn(conn.get_shape(48), dtype=torch.float16)
        conn.to_gpu(mem, 0, 48, slot_mapping=_make_slot_mapping(48))
        assert runtime.update_kv_cache.call_count == 12
        assert runtime.fetch_kv_cache.call_count == 12

    def test_partial_block(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=["layer_0"], runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        mem = MagicMock()
        mem.tensor = torch.randn(conn.get_shape(5), dtype=torch.float16)
        conn.to_gpu(mem, 0, 5, slot_mapping=_make_slot_mapping(5))
        assert runtime.update_kv_cache.call_count == 1
        assert runtime.fetch_kv_cache.call_count == 1

    def test_read_modify_write_order(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=1024, dtype=torch.float16,
            kv_cache_names=["layer_0"], runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        mem = MagicMock()
        mem.tensor = torch.randn(conn.get_shape(256), dtype=torch.float16)
        conn.to_gpu(mem, 0, 256, slot_mapping=_make_slot_mapping(256))
        all_calls = runtime.method_calls
        fetch_idx = next(i for i, c in enumerate(all_calls) if c[0] == "fetch_kv_cache")
        update_idx = next(i for i, c in enumerate(all_calls) if c[0] == "update_kv_cache")
        assert fetch_idx < update_idx


# ===================================================================
# RBLNConnector: batched operations
# ===================================================================


class TestBatchedOperations:
    def test_batched_from_gpu_calls_per_item(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
            kv_cache_names=["layer_0"], runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        mems = [MagicMock() for _ in range(3)]
        for m in mems:
            m.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)
        conn.batched_from_gpu(mems, [0, 16, 32], [16, 32, 48], slot_mapping=_make_slot_mapping(48))
        assert runtime.fetch_kv_cache.call_count == 3

    def test_batched_to_gpu_none_inputs(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        conn.batched_to_gpu(None, None, None)


# ===================================================================
# RBLNConnector: setters
# ===================================================================


class TestSetters:
    def test_set_runtime_holder_lazy(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        holder = []
        conn.set_runtime_holder(holder)
        assert conn.runtime is None
        runtime = MagicMock()
        holder.append(runtime)
        assert conn.runtime is runtime

    def test_set_runtime_eager(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        runtime = MagicMock()
        conn.set_runtime(runtime)
        assert conn.runtime is runtime

    def test_set_kv_cache_names(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        conn.set_kv_cache_names(["layer_0", "layer_1"])
        assert conn.kv_cache_names == ["layer_0", "layer_1"]

    def test_initialize_kvcaches_ptr(self):
        conn = RBLNConnector(
            num_layers=1, num_kv_heads=4, head_dim=64,
            block_size=16, dtype=torch.float16,
        )
        runtime = MagicMock()
        conn.initialize_kvcaches_ptr(runtime_holder=[runtime], kv_cache_names=["layer_0"])
        assert conn.runtime is runtime
        assert conn.kv_cache_names == ["layer_0"]


# ===================================================================
# Memory leak detection
# ===================================================================


class TestMemoryLeak:
    def test_repeated_from_gpu_no_tensor_accumulation(self):
        conn = _make_connector()
        shape, slot_mapping = conn.get_shape(16), _make_slot_mapping(16)
        gc.collect()
        before = len([o for o in gc.get_objects() if isinstance(o, torch.Tensor)])
        for _ in range(100):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            conn.from_gpu(mem, 0, 16, slot_mapping=slot_mapping)
        gc.collect()
        assert len([o for o in gc.get_objects() if isinstance(o, torch.Tensor)]) - before < 50

    def test_repeated_to_gpu_no_tensor_accumulation(self):
        conn = _make_connector()
        shape, slot_mapping = conn.get_shape(16), _make_slot_mapping(16)
        gc.collect()
        before = len([o for o in gc.get_objects() if isinstance(o, torch.Tensor)])
        for _ in range(100):
            mem = MagicMock()
            mem.tensor = torch.randn(shape, dtype=torch.float16)
            conn.to_gpu(mem, 0, 16, slot_mapping=slot_mapping)
        gc.collect()
        assert len([o for o in gc.get_objects() if isinstance(o, torch.Tensor)]) - before < 50

    def test_repeated_ops_stable_rss(self):
        conn = _make_connector()
        shape, slot_mapping = conn.get_shape(32), _make_slot_mapping(32)
        for _ in range(10):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            conn.from_gpu(mem, 0, 32, slot_mapping=slot_mapping)
            conn.to_gpu(mem, 0, 32, slot_mapping=slot_mapping)
        gc.collect()
        rss_before = _get_rss_mb()
        if rss_before == 0:
            pytest.skip("Cannot read RSS")
        for _ in range(500):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            conn.from_gpu(mem, 0, 32, slot_mapping=slot_mapping)
            conn.to_gpu(mem, 0, 32, slot_mapping=slot_mapping)
        gc.collect()
        assert _get_rss_mb() - rss_before < 50


# ===================================================================
# _rbln_calculate_local_rank_and_world_size
# ===================================================================


class TestRBLNCalculateLocalRank:
    def _make_mock_config(self, rank=0, world_size=1, tp_size=1, pp_size=1):
        config = MagicMock()
        config.parallel_config.rank = rank
        config.parallel_config.world_size = world_size
        config.parallel_config.tensor_parallel_size = tp_size
        config.parallel_config.pipeline_parallel_size = pp_size
        return config

    def test_single_device(self):
        assert calculate_local_rank_and_world_size(self._make_mock_config()) == (0, 1)

    def test_multi_device(self):
        mock_rebel = MagicMock()
        mock_rebel.get_npu_count.return_value = 4
        with patch.dict("sys.modules", {"rebel": mock_rebel}):
            result = calculate_local_rank_and_world_size(
                self._make_mock_config(rank=2, world_size=4, tp_size=4, pp_size=1)
            )
            assert result == (2, 4)

    def test_rebel_unavailable_defaults_to_1(self):
        assert calculate_local_rank_and_world_size(self._make_mock_config()) == (0, 1)


# ===================================================================
# RBLNVllmServiceFactory (requires lmcache)
# ===================================================================


class TestRBLNVllmServiceFactory:
    def test_is_subclass_of_vllm_service_factory(self):
        from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNVllmServiceFactory
        assert issubclass(RBLNVllmServiceFactory, VllmServiceFactory)

    def test_overrides_get_or_create_lmcache_engine(self):
        from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNVllmServiceFactory
        assert (
            RBLNVllmServiceFactory.get_or_create_lmcache_engine
            is not VllmServiceFactory.get_or_create_lmcache_engine
        )

    def test_scheduler_metadata_creation(self):
        """Step 1: metadata should be created without get_vllm_torch_dev."""
        import os
        os.environ["LMCACHE_CHUNK_SIZE"] = "64"
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_ENABLE_SCHEDULER_BYPASS_LOOKUP"] = "True"
        try:
            from lmcache.integration.vllm.utils import lmcache_get_or_create_config
            from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNVllmServiceFactory

            config = lmcache_get_or_create_config()
            mock_vllm_config = MagicMock()
            mock_vllm_config.model_config.get_num_layers.return_value = 32
            mock_vllm_config.model_config.get_num_kv_heads.return_value = 8
            mock_vllm_config.model_config.get_head_size.return_value = 128
            mock_vllm_config.model_config.model = "test-model"
            mock_vllm_config.model_config.served_model_name = "test"
            mock_vllm_config.model_config.dtype = "float16"
            mock_vllm_config.model_config.hf_config.model_type = "llama"
            mock_vllm_config.parallel_config.rank = 0
            mock_vllm_config.parallel_config.world_size = 1
            mock_vllm_config.parallel_config.tensor_parallel_size = 1
            mock_vllm_config.parallel_config.pipeline_parallel_size = 1
            mock_vllm_config.cache_config.cache_dtype = "auto"
            mock_vllm_config.cache_config.block_size = 128
            mock_vllm_config.kv_transfer_config = None

            factory = RBLNVllmServiceFactory(config, mock_vllm_config, "scheduler")
            metadata = factory.get_or_create_metadata()
            assert metadata is not None
            print(f"metadata: local_worker_id={metadata.local_worker_id}, "
                  f"local_world_size={metadata.local_world_size}")
        finally:
            os.environ.pop("LMCACHE_CHUNK_SIZE", None)
            os.environ.pop("LMCACHE_LOCAL_CPU", None)
            os.environ.pop("LMCACHE_ENABLE_SCHEDULER_BYPASS_LOOKUP", None)

    def test_scheduler_engine_creation_with_bypass(self):
        """Step 2: scheduler with bypass_lookup should create an engine."""
        import os
        os.environ["LMCACHE_CHUNK_SIZE"] = "64"
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_ENABLE_SCHEDULER_BYPASS_LOOKUP"] = "True"
        try:
            from lmcache.integration.vllm.utils import lmcache_get_or_create_config
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNVllmServiceFactory

            # Clear any cached engine
            for k in list(LMCacheEngineBuilder._instances):
                LMCacheEngineBuilder.destroy(k)

            config = lmcache_get_or_create_config()
            assert config.enable_scheduler_bypass_lookup is True, (
                f"bypass_lookup should be True, got {config.enable_scheduler_bypass_lookup}"
            )

            mock_vllm_config = MagicMock()
            mock_vllm_config.model_config.get_num_layers.return_value = 32
            mock_vllm_config.model_config.get_num_kv_heads.return_value = 8
            mock_vllm_config.model_config.get_head_size.return_value = 128
            mock_vllm_config.model_config.model = "test-model"
            mock_vllm_config.model_config.served_model_name = "test"
            mock_vllm_config.model_config.dtype = "float16"
            mock_vllm_config.model_config.hf_config.model_type = "llama"
            mock_vllm_config.parallel_config.rank = 0
            mock_vllm_config.parallel_config.world_size = 1
            mock_vllm_config.parallel_config.tensor_parallel_size = 1
            mock_vllm_config.parallel_config.pipeline_parallel_size = 1
            mock_vllm_config.cache_config.cache_dtype = "auto"
            mock_vllm_config.cache_config.block_size = 128
            mock_vllm_config.kv_transfer_config = None

            factory = RBLNVllmServiceFactory(config, mock_vllm_config, "scheduler")
            engine = factory.get_or_create_lmcache_engine()
            assert engine is not None, "Scheduler with bypass_lookup should create an engine"
            print(f"engine created: {type(engine).__name__}")

            # Step 3: lookup client should be created
            lookup_client = factory.maybe_create_lookup_client()
            assert lookup_client is not None, "Scheduler should have a lookup client"
            print(f"lookup_client: {type(lookup_client).__name__}")

            for k in list(LMCacheEngineBuilder._instances):
                LMCacheEngineBuilder.destroy(k)
        finally:
            os.environ.pop("LMCACHE_CHUNK_SIZE", None)
            os.environ.pop("LMCACHE_LOCAL_CPU", None)
            os.environ.pop("LMCACHE_ENABLE_SCHEDULER_BYPASS_LOOKUP", None)


# ===================================================================
# Integration: import, inheritance, interface
# ===================================================================


class TestConnectorImport:
    def test_connector_v1_importable(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1
        assert RBLNLMCacheConnectorV1 is not None

    def test_connector_impl_importable(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1Impl
        assert RBLNLMCacheConnectorV1Impl is not None

    def test_service_factory_importable(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNVllmServiceFactory
        assert RBLNVllmServiceFactory is not None


class TestConnectorInheritance:
    def test_impl_is_subclass_of_upstream(self):
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1Impl
        assert issubclass(RBLNLMCacheConnectorV1Impl, LMCacheConnectorV1Impl)

    def test_impl_overrides_init(self):
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1Impl
        assert RBLNLMCacheConnectorV1Impl.__init__ is not LMCacheConnectorV1Impl.__init__

    def test_v1_is_subclass_of_kv_connector_base(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1
        assert issubclass(RBLNLMCacheConnectorV1, KVConnectorBase_V1)

    def test_v1_has_all_required_methods(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1
        for method in ["register_kv_caches", "start_load_kv", "wait_for_layer_load",
                        "save_kv_layer", "wait_for_save", "get_finished", "shutdown",
                        "get_num_new_matched_tokens", "update_state_after_alloc",
                        "build_connector_meta", "request_finished"]:
            assert hasattr(RBLNLMCacheConnectorV1, method), f"Missing: {method}"


# ===================================================================
# register_kv_caches: 6D -> 5D normalization
# ===================================================================


def _rbln_kv_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (2, num_blocks, num_kv_heads, 1, block_size, head_size)


def _make_rbln_kv_caches(num_layers=28, num_blocks=8, block_size=1024,
                          num_kv_heads=8, head_size=128, dtype=torch.float16):
    shape = _rbln_kv_shape(num_blocks, block_size, num_kv_heads, head_size)
    return {
        f"model.layers.{i}.self_attn.kv_cache": torch.zeros(shape, dtype=dtype)
        for i in range(num_layers)
    }


def _make_impl(num_layers=28, num_kv_heads=8, head_size=128,
               chunk_size=256, kv_dtype=torch.float16):
    from lmcache.v1.metadata import LMCacheMetadata
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import RBLNLMCacheConnectorV1Impl

    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl.kv_caches = {}
    metadata = LMCacheMetadata(
        model_name="test", world_size=1, local_world_size=1, worker_id=0,
        local_worker_id=0, kv_dtype=kv_dtype,
        kv_shape=(num_layers, 2, chunk_size, num_kv_heads, head_size),
        use_mla=False, role="worker", served_model_name="test", chunk_size=chunk_size,
    )
    manager = MagicMock()
    manager.lmcache_engine.metadata = metadata
    manager.lmcache_engine.gpu_connector = MagicMock()
    impl._manager = manager
    return impl


class TestRegisterKVCaches:
    def test_register_does_not_raise(self):
        _make_impl().register_kv_caches(_make_rbln_kv_caches())

    def test_layer_groups_built(self):
        impl = _make_impl()
        impl.register_kv_caches(_make_rbln_kv_caches(num_layers=28))
        mgr = impl.lmcache_engine.metadata.kv_layer_groups_manager
        assert mgr.num_groups == 1 and mgr.kv_layer_groups[0].num_layers == 28

    def test_hidden_dim_size_correct(self):
        impl = _make_impl(num_kv_heads=8, head_size=128)
        impl.register_kv_caches(_make_rbln_kv_caches(num_kv_heads=8, head_size=128))
        assert impl.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups[0].hidden_dim_size == 1024

    def test_get_shapes_matches_connector(self):
        impl = _make_impl(num_layers=28, num_kv_heads=8, head_size=128, chunk_size=256)
        impl.register_kv_caches(_make_rbln_kv_caches(num_layers=28, num_kv_heads=8, head_size=128))
        shapes = impl.lmcache_engine.metadata.get_shapes(256)
        assert shapes[0] == torch.Size([2, 28, 256, 1024])

    def test_kv_cache_names_forwarded(self):
        impl = _make_impl()
        impl.register_kv_caches(_make_rbln_kv_caches(num_layers=4))
        names = impl.lmcache_engine.gpu_connector.set_kv_cache_names.call_args[0][0]
        assert names == [f"model.layers.{i}.self_attn.kv_cache" for i in range(4)]

    def test_normalized_shape_is_5d(self):
        impl = _make_impl()
        impl.register_kv_caches(_make_rbln_kv_caches(num_blocks=8, block_size=1024, num_kv_heads=8, head_size=128))
        for t in impl.kv_caches.values():
            assert t.shape == torch.Size([2, 8, 1024, 8, 128])

    def test_standard_5d_passthrough(self):
        impl = _make_impl(num_layers=4)
        kv = {
            f"model.layers.{i}.self_attn.kv_cache": torch.zeros(2, 8, 16, 8, 128, dtype=torch.float16)
            for i in range(4)
        }
        impl.register_kv_caches(kv)
        for t in impl.kv_caches.values():
            assert t.shape == torch.Size([2, 8, 16, 8, 128])

    def test_different_head_configs_create_multiple_groups(self):
        impl = _make_impl(num_layers=4)
        kv = {}
        for i in range(2):
            kv[f"model.layers.{i}.self_attn.kv_cache"] = torch.zeros(*_rbln_kv_shape(8, 1024, 8, 128), dtype=torch.float16)
        for i in range(2, 4):
            kv[f"model.layers.{i}.self_attn.kv_cache"] = torch.zeros(*_rbln_kv_shape(8, 1024, 4, 256), dtype=torch.float16)
        impl.register_kv_caches(kv)
        assert impl.lmcache_engine.metadata.kv_layer_groups_manager.num_groups == 2

    def test_bfloat16_dtype_preserved(self):
        impl = _make_impl(kv_dtype=torch.bfloat16)
        impl.register_kv_caches(_make_rbln_kv_caches(num_layers=4, dtype=torch.bfloat16))
        assert impl.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups[0].dtype == torch.bfloat16
