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

import torch
from lmcache.v1.metadata import LMCacheMetadata

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import (
    RBLNLMCacheConnectorV1Impl,
)


def _rbln_get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    return (2, num_blocks, num_kv_heads, 1, block_size, head_size)


def _make_rbln_kv_caches(
    num_layers: int = 28,
    num_blocks: int = 8,
    block_size: int = 1024,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    shape = _rbln_get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
    return {
        f"model.layers.{i}.self_attn.kv_cache": torch.zeros(shape, dtype=dtype)
        for i in range(num_layers)
    }


def _make_impl_for_register(
    num_layers: int = 28,
    num_kv_heads: int = 8,
    head_size: int = 128,
    chunk_size: int = 256,
    kv_dtype: torch.dtype = torch.float16,
) -> RBLNLMCacheConnectorV1Impl:
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)

    impl.kv_caches = {}

    kv_shape = (num_layers, 2, chunk_size, num_kv_heads, head_size)
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=False,
        role="worker",
        served_model_name="test",
        chunk_size=chunk_size,
    )

    manager = MagicMock()
    manager.lmcache_engine.metadata = metadata
    manager.lmcache_engine.gpu_connector = MagicMock()
    impl._manager = manager

    return impl


class TestRegisterRBLNKVCaches:
    def test_register_does_not_raise(self):
        impl = _make_impl_for_register()
        kv_caches = _make_rbln_kv_caches()
        impl.register_kv_caches(kv_caches)

    def test_layer_groups_built(self):
        impl = _make_impl_for_register()
        kv_caches = _make_rbln_kv_caches(num_layers=28)
        impl.register_kv_caches(kv_caches)

        mgr = impl.lmcache_engine.metadata.kv_layer_groups_manager
        assert mgr.num_groups == 1
        assert mgr.kv_layer_groups[0].num_layers == 28

    def test_hidden_dim_size_correct(self):
        impl = _make_impl_for_register(num_kv_heads=8, head_size=128)
        kv_caches = _make_rbln_kv_caches(num_kv_heads=8, head_size=128)
        impl.register_kv_caches(kv_caches)

        group = impl.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups[0]
        assert group.hidden_dim_size == 8 * 128

    def test_get_shapes_matches_connector(self):
        num_layers, num_kv_heads, head_size = 28, 8, 128
        chunk_size = 256
        impl = _make_impl_for_register(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            chunk_size=chunk_size,
        )
        kv_caches = _make_rbln_kv_caches(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        impl.register_kv_caches(kv_caches)

        shapes = impl.lmcache_engine.metadata.get_shapes(chunk_size)
        assert len(shapes) == 1
        assert shapes[0] == torch.Size(
            [2, num_layers, chunk_size, num_kv_heads * head_size]
        )

    def test_kv_cache_names_forwarded_to_connector(self):
        impl = _make_impl_for_register()
        kv_caches = _make_rbln_kv_caches(num_layers=4)
        impl.register_kv_caches(kv_caches)

        impl.lmcache_engine.gpu_connector.set_kv_cache_names.assert_called_once()
        names = impl.lmcache_engine.gpu_connector.set_kv_cache_names.call_args[0][0]
        assert names == [f"model.layers.{i}.self_attn.kv_cache" for i in range(4)]

    def test_normalized_shape_is_5d(self):
        impl = _make_impl_for_register()
        kv_caches = _make_rbln_kv_caches(
            num_blocks=8, block_size=1024, num_kv_heads=8, head_size=128
        )
        impl.register_kv_caches(kv_caches)

        for tensor in impl.kv_caches.values():
            assert tensor.dim() == 5
            assert tensor.shape == torch.Size([2, 8, 1024, 8, 128])


class TestRegisterStandardKVCaches:
    def test_standard_5d_passthrough(self):
        impl = _make_impl_for_register(num_layers=4)
        kv_caches = {
            f"model.layers.{i}.self_attn.kv_cache": torch.zeros(
                2, 8, 16, 8, 128, dtype=torch.float16
            )
            for i in range(4)
        }
        impl.register_kv_caches(kv_caches)

        for tensor in impl.kv_caches.values():
            assert tensor.dim() == 5
            assert tensor.shape == torch.Size([2, 8, 16, 8, 128])

        group = impl.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups[0]
        assert group.hidden_dim_size == 8 * 128


class TestRegisterEdgeCases:
    def test_different_head_configs_create_multiple_groups(self):
        impl = _make_impl_for_register(num_layers=4)

        kv_caches = {}
        for i in range(2):
            kv_caches[f"model.layers.{i}.self_attn.kv_cache"] = torch.zeros(
                *_rbln_get_kv_cache_shape(8, 1024, 8, 128),
                dtype=torch.float16,
            )
        for i in range(2, 4):
            kv_caches[f"model.layers.{i}.self_attn.kv_cache"] = torch.zeros(
                *_rbln_get_kv_cache_shape(8, 1024, 4, 256),
                dtype=torch.float16,
            )

        impl.register_kv_caches(kv_caches)

        mgr = impl.lmcache_engine.metadata.kv_layer_groups_manager
        assert mgr.num_groups == 2

        groups = sorted(mgr.kv_layer_groups, key=lambda g: g.layer_indices[0])
        assert groups[0].hidden_dim_size == 8 * 128
        assert groups[1].hidden_dim_size == 4 * 256

    def test_bfloat16_dtype_preserved(self):
        impl = _make_impl_for_register(kv_dtype=torch.bfloat16)
        kv_caches = _make_rbln_kv_caches(num_layers=4, dtype=torch.bfloat16)
        impl.register_kv_caches(kv_caches)

        group = impl.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups[0]
        assert group.dtype == torch.bfloat16
