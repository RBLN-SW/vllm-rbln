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

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache import RBLNConnector


def _make_slot_mapping(num_tokens, offset=0):
    return torch.arange(offset, offset + num_tokens, dtype=torch.long)


def _init_xfer_buffers(conn, num_blocks=4):
    kv_caches = {
        name: torch.zeros(
            2,
            num_blocks,
            conn.num_kv_heads,
            1,
            conn.block_size,
            conn.head_dim,
            dtype=conn.dtype,
        )
        for name in conn.kv_cache_names
    }
    conn.initialize_xfer_buffers(kv_caches)


class TestRBLNConnectorInit:
    def test_default_init(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.runtime is None
        assert conn.kv_cache_names == []
        assert conn._xfer_buffer_all is None

    def test_hidden_dim_computation(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.hidden_dim_size == 1024

    def test_direct_construction(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert isinstance(conn, RBLNConnector)


class TestGetShape:
    def test_shape_format(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.get_shape(64) == torch.Size([2, 32, 64, 1024])

    def test_shape_single_token(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.get_shape(1) == torch.Size([2, 32, 1, 1024])

    @pytest.mark.parametrize(
        "num_kv_heads,head_dim",
        [
            (1, 64),
            (4, 256),
            (32, 128),
            (128, 32),
        ],
    )
    def test_shape_various_head_configs(self, num_kv_heads, head_dim):
        conn = RBLNConnector(
            num_layers=16,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype=torch.float16,
        )
        shape = conn.get_shape(32)
        assert shape[3] == num_kv_heads * head_dim


class TestXferBuffers:
    @staticmethod
    def _init_buffers(conn, kv_names, num_blocks=4):
        kv_caches = {
            name: torch.zeros(
                2,
                num_blocks,
                conn.num_kv_heads,
                1,
                conn.block_size,
                conn.head_dim,
                dtype=conn.dtype,
            )
            for name in kv_names
        }
        conn.initialize_xfer_buffers(kv_caches)

    def test_all_layer_buffer_is_4096_aligned(self):
        kv_names = ["layer_0", "layer_1"]
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=8,
            head_dim=128,
            block_size=256,
            dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names)
        assert conn._xfer_buffer_all is not None
        assert conn._xfer_buffer_all.data_ptr() % 4096 == 0

    def test_all_layer_buffer_size_equals_sum_of_layer_views(self):
        kv_names = ["layer_0", "layer_1", "layer_2"]
        conn = RBLNConnector(
            num_layers=3,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names, num_blocks=4)
        expected_total = sum(v.numel() for v in conn._xfer_layer_views)
        assert conn._xfer_buffer_all.numel() == expected_total

    def test_layer_views_count_matches_layers(self):
        kv_names = ["layer_0", "layer_1", "layer_2"]
        conn = RBLNConnector(
            num_layers=3,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names)
        assert len(conn._xfer_layer_views) == 3

    def test_layer_views_shapes_match_kv_caches(self):
        kv_names = ["layer_0", "layer_1"]
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=kv_names,
        )
        self._init_buffers(conn, kv_names, num_blocks=4)
        for view in conn._xfer_layer_views:
            assert view.shape == (2, 4, 4, 1, 16, 64)


class TestSlotGrouping:
    def test_contiguous_single_block(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        slots = torch.arange(16, dtype=torch.long)
        groups = conn._group_slots_by_block(slots)
        assert len(groups) == 1
        assert 0 in groups
        assert len(groups[0]) == 16

    def test_cross_block_boundary(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        slots = torch.arange(14, 18, dtype=torch.long)
        groups = conn._group_slots_by_block(slots)
        assert len(groups) == 2
        assert 0 in groups
        assert 1 in groups

    def test_scattered_slots(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        slots = torch.tensor([0, 32, 64], dtype=torch.long)
        groups = conn._group_slots_by_block(slots)
        assert len(groups) == 3


class TestD2HWithRuntime:
    def test_fetch_called_once_per_block(self):
        runtime = MagicMock()
        num_layers = 4
        kv_names = [f"layer_{i}" for i in range(num_layers)]
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=kv_names,
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(32)
        mem = MagicMock()
        mem.tensor = torch.zeros(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(32)

        conn.from_gpu(mem, 0, 32, slot_mapping=slot_mapping)

        assert runtime.fetch_kv_cache.call_count == 2

    def test_fetch_passes_none_layer_name(self):
        runtime = MagicMock()
        kv_names = ["k_cache_0", "k_cache_1"]
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=kv_names,
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(16)
        mem = MagicMock()
        mem.tensor = torch.zeros(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(16)

        conn.from_gpu(mem, 0, 16, slot_mapping=slot_mapping)

        assert runtime.fetch_kv_cache.call_count == 1
        call_args = runtime.fetch_kv_cache.call_args[0]
        assert call_args[2] == 0
        assert call_args[3] == 16
        assert call_args[4] is None

    def test_missing_slot_mapping_raises(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=["layer_0"],
            runtime_holder=[runtime],
        )
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)

        with pytest.raises(AssertionError, match="slot_mapping"):
            conn.from_gpu(mem, 0, 16)

    def test_none_tensor_raises(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=["layer_0"],
            runtime_holder=[runtime],
        )
        mem = MagicMock()
        mem.tensor = None

        with pytest.raises(AssertionError):
            conn.from_gpu(mem, 0, 16, slot_mapping=_make_slot_mapping(16))


class TestH2DWithRuntime:
    def test_full_block_skips_fetch(self):
        runtime = MagicMock()
        num_layers = 4
        kv_names = [f"layer_{i}" for i in range(num_layers)]
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=kv_names,
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(48)
        mem = MagicMock()
        mem.tensor = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(48)

        conn.to_gpu(mem, 0, 48, slot_mapping=slot_mapping)

        assert runtime.update_kv_cache.call_count == 3
        assert runtime.fetch_kv_cache.call_count == 0

    def test_partial_block_does_rmw(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=["layer_0"],
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(5)
        mem = MagicMock()
        mem.tensor = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(5)

        conn.to_gpu(mem, 0, 5, slot_mapping=slot_mapping)

        assert runtime.update_kv_cache.call_count == 1
        assert runtime.fetch_kv_cache.call_count == 1

    def test_read_modify_write_when_block_larger_than_chunk(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=1024,
            dtype=torch.float16,
            kv_cache_names=["layer_0"],
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        chunk_size = 256
        shape = conn.get_shape(chunk_size)
        mem = MagicMock()
        mem.tensor = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(chunk_size)

        conn.to_gpu(mem, 0, chunk_size, slot_mapping=slot_mapping)

        assert runtime.fetch_kv_cache.call_count == 1
        assert runtime.update_kv_cache.call_count == 1

        all_calls = runtime.method_calls
        fetch_idx = next(i for i, c in enumerate(all_calls) if c[0] == "fetch_kv_cache")
        update_idx = next(
            i for i, c in enumerate(all_calls) if c[0] == "update_kv_cache"
        )
        assert fetch_idx < update_idx

    def test_update_passes_none_layer_name(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=["layer_0", "layer_1"],
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(16)
        mem = MagicMock()
        mem.tensor = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(16)

        conn.to_gpu(mem, 0, 16, slot_mapping=slot_mapping)

        assert runtime.update_kv_cache.call_count == 1
        call_args = runtime.update_kv_cache.call_args[0]
        assert call_args[4] is None

    def test_mixed_full_and_partial_blocks(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=["layer_0", "layer_1"],
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(20)
        mem = MagicMock()
        mem.tensor = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(20)

        conn.to_gpu(mem, 0, 20, slot_mapping=slot_mapping)

        assert runtime.update_kv_cache.call_count == 2
        assert runtime.fetch_kv_cache.call_count == 1


class TestBatchedOperations:
    def test_batched_from_gpu_calls_per_item(self):
        runtime = MagicMock()
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            kv_cache_names=["layer_0"],
            runtime_holder=[runtime],
        )
        _init_xfer_buffers(conn)
        shape = conn.get_shape(16)
        mems = [MagicMock() for _ in range(3)]
        for m in mems:
            m.tensor = torch.zeros(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(48)

        conn.batched_from_gpu(
            mems,
            [0, 16, 32],
            [16, 32, 48],
            slot_mapping=slot_mapping,
        )
        assert runtime.fetch_kv_cache.call_count == 3

    def test_batched_to_gpu_none_inputs(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        conn.batched_to_gpu(None, None, None)


class TestSetters:
    def test_set_runtime_holder(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        holder = []
        conn.set_runtime_holder(holder)
        assert conn.runtime is None

        runtime = MagicMock()
        holder.append(runtime)
        assert conn.runtime is runtime

    def test_set_runtime_legacy(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        runtime = MagicMock()
        conn.set_runtime(runtime)
        assert conn.runtime is runtime

    def test_set_kv_cache_names(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        names = ["layer_0", "layer_1"]
        conn.set_kv_cache_names(names)
        assert conn.kv_cache_names == names

    def test_initialize_kvcaches_ptr_runtime_holder(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        runtime = MagicMock()
        holder = [runtime]
        conn.initialize_kvcaches_ptr(
            runtime_holder=holder,
            kv_cache_names=["layer_0"],
        )
        assert conn.runtime is runtime
        assert conn.kv_cache_names == ["layer_0"]

    def test_initialize_kvcaches_ptr_runtime_legacy(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        runtime = MagicMock()
        conn.initialize_kvcaches_ptr(
            runtime_holder=[runtime],
            kv_cache_names=["layer_0"],
        )
        assert conn.runtime is runtime
        assert conn.kv_cache_names == ["layer_0"]


class TestDtypeHandling:
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ],
    )
    def test_get_shape_dtype_independent(self, dtype):
        conn = RBLNConnector(
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=dtype,
        )
        assert conn.get_shape(32) == torch.Size([2, 4, 32, 1024])


class TestDataCorrectness:
    @staticmethod
    def _make_conn_with_data(num_layers=2, num_kv_heads=4, head_dim=64, block_size=16):
        kv_names = [f"layer_{i}" for i in range(num_layers)]
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=torch.float16,
            kv_cache_names=kv_names,
        )

        num_blocks = 4
        kv_caches = {
            name: torch.zeros(
                2,
                num_blocks,
                num_kv_heads,
                1,
                block_size,
                head_dim,
                dtype=torch.float16,
            )
            for name in kv_names
        }
        conn.initialize_xfer_buffers(kv_caches)
        return conn

    def test_d2h_round_trip_full_block_6d(self):
        conn = self._make_conn_with_data()
        shape = conn.get_shape(16)
        original_data = torch.randn(shape, dtype=torch.float16)

        captured_buffers = {}

        def mock_fetch(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    view.copy_(
                        captured_buffers.get((block_idx, i), torch.zeros_like(view))
                    )

        def mock_update(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    captured_buffers[(block_idx, i)] = view.clone()

        runtime = MagicMock()
        runtime.fetch_kv_cache = mock_fetch
        runtime.update_kv_cache = mock_update
        conn.set_runtime(runtime)

        mem_write = MagicMock()
        mem_write.tensor = original_data.clone()
        slot_mapping = _make_slot_mapping(16)
        conn.to_gpu(mem_write, 0, 16, slot_mapping=slot_mapping)

        mem_read = MagicMock()
        mem_read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(mem_read, 0, 16, slot_mapping=slot_mapping)

        assert torch.allclose(original_data, mem_read.tensor, atol=1e-3)

    def test_d2h_round_trip_partial_block_6d(self):
        conn = self._make_conn_with_data()
        num_tokens = 5
        shape = conn.get_shape(num_tokens)
        original_data = torch.randn(shape, dtype=torch.float16)

        captured_buffers = {}

        def mock_fetch(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    stored = captured_buffers.get(
                        (block_idx, i), torch.zeros_like(view)
                    )
                    view.copy_(stored)

        def mock_update(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    captured_buffers[(block_idx, i)] = view.clone()

        runtime = MagicMock()
        runtime.fetch_kv_cache = mock_fetch
        runtime.update_kv_cache = mock_update
        conn.set_runtime(runtime)

        slot_mapping = _make_slot_mapping(num_tokens)

        mem_write = MagicMock()
        mem_write.tensor = original_data.clone()
        conn.to_gpu(mem_write, 0, num_tokens, slot_mapping=slot_mapping)

        mem_read = MagicMock()
        mem_read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(mem_read, 0, num_tokens, slot_mapping=slot_mapping)

        assert torch.allclose(original_data, mem_read.tensor, atol=1e-3)

    @pytest.mark.parametrize("mem_dtype", [torch.bfloat16, torch.float32])
    def test_d2h_round_trip_cross_dtype(self, mem_dtype):
        conn = self._make_conn_with_data()
        shape = conn.get_shape(16)
        original_data = torch.randn(shape, dtype=mem_dtype)

        captured_buffers = {}

        def mock_fetch(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    view.copy_(
                        captured_buffers.get((block_idx, i), torch.zeros_like(view))
                    )

        def mock_update(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    captured_buffers[(block_idx, i)] = view.clone()

        runtime = MagicMock()
        runtime.fetch_kv_cache = mock_fetch
        runtime.update_kv_cache = mock_update
        conn.set_runtime(runtime)

        mem_write = MagicMock()
        mem_write.tensor = original_data.clone()
        slot_mapping = _make_slot_mapping(16)
        conn.to_gpu(mem_write, 0, 16, slot_mapping=slot_mapping)

        mem_read = MagicMock()
        mem_read.tensor = torch.zeros(shape, dtype=mem_dtype)
        conn.from_gpu(mem_read, 0, 16, slot_mapping=slot_mapping)

        expected = original_data.to(torch.float16).to(mem_dtype)
        assert torch.allclose(expected, mem_read.tensor, atol=1e-3)

    def test_d2h_round_trip_partial_cross_dtype(self):
        conn = self._make_conn_with_data()
        num_tokens = 5
        shape = conn.get_shape(num_tokens)
        original_data = torch.randn(shape, dtype=torch.bfloat16)

        captured_buffers = {}

        def mock_fetch(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    stored = captured_buffers.get(
                        (block_idx, i), torch.zeros_like(view)
                    )
                    view.copy_(stored)

        def mock_update(ptr, block_idx, offset, block_size, layer_name):
            if layer_name is None:
                for i, view in enumerate(conn._xfer_layer_views):
                    captured_buffers[(block_idx, i)] = view.clone()

        runtime = MagicMock()
        runtime.fetch_kv_cache = mock_fetch
        runtime.update_kv_cache = mock_update
        conn.set_runtime(runtime)

        slot_mapping = _make_slot_mapping(num_tokens)

        mem_write = MagicMock()
        mem_write.tensor = original_data.clone()
        conn.to_gpu(mem_write, 0, num_tokens, slot_mapping=slot_mapping)

        mem_read = MagicMock()
        mem_read.tensor = torch.zeros(shape, dtype=torch.bfloat16)
        conn.from_gpu(mem_read, 0, num_tokens, slot_mapping=slot_mapping)

        expected = original_data.to(torch.float16).to(torch.bfloat16)
        assert torch.allclose(expected, mem_read.tensor, atol=1e-3)
