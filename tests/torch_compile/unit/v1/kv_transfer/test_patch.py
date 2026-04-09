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

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector import (
    RBLNLMCacheConnectorV1,
    RBLNLMCacheConnectorV1Impl,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.rbln_connector import (
    RBLNConnector,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.rbln_lmcache_service_factory import (
    RBLNServiceFactory,
)

LMCACHE_CONNECTOR_MODULE = (
    "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_lmcache_connector"
)
LMCACHE_SERVICE_FACTORY_MODULE = (
    "vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache"
    ".rbln_lmcache_service_factory"
)


class TestRBLNConnectorImport:
    def test_connector_importable(self):
        assert RBLNLMCacheConnectorV1 is not None

    def test_connector_impl_importable(self):
        assert RBLNLMCacheConnectorV1Impl is not None

    def test_service_factory_importable(self):
        assert RBLNServiceFactory is not None


class TestRBLNConnectorV1Impl:
    def test_is_subclass_of_upstream_impl(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl,
        )

        assert issubclass(RBLNLMCacheConnectorV1Impl, LMCacheConnectorV1Impl)

    def test_overrides_init(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl,
        )

        assert (
            RBLNLMCacheConnectorV1Impl.__init__ is not LMCacheConnectorV1Impl.__init__
        )


class TestRBLNConnectorV1:
    def test_is_subclass_of_kv_connector_base(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        assert issubclass(RBLNLMCacheConnectorV1, KVConnectorBase_V1)

    def test_has_all_required_methods(self):
        required_methods = [
            "register_kv_caches",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
            "get_finished",
            "shutdown",
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "build_connector_meta",
            "request_finished",
        ]
        for method in required_methods:
            assert hasattr(RBLNLMCacheConnectorV1, method), f"Missing method: {method}"

    def test_has_set_runtime_holder(self):
        assert hasattr(RBLNLMCacheConnectorV1, "set_runtime_holder")

    def test_has_set_runtime(self):
        assert hasattr(RBLNLMCacheConnectorV1, "set_runtime")

    def test_set_runtime_holder_propagates_to_connector(self):
        gpu_connector = MagicMock(spec=RBLNConnector)
        manager = MagicMock()
        manager.lmcache_engine.gpu_connector = gpu_connector

        impl = object.__new__(RBLNLMCacheConnectorV1Impl)
        impl._manager = manager

        v1 = object.__new__(RBLNLMCacheConnectorV1)
        v1._lmcache_engine = impl

        holder = [MagicMock()]
        v1.set_runtime_holder(holder)
        gpu_connector.set_runtime_holder.assert_called_once_with(holder)

    def test_set_runtime_propagates_to_connector(self):
        gpu_connector = MagicMock(spec=RBLNConnector)
        manager = MagicMock()
        manager.lmcache_engine.gpu_connector = gpu_connector

        impl = object.__new__(RBLNLMCacheConnectorV1Impl)
        impl._manager = manager

        v1 = object.__new__(RBLNLMCacheConnectorV1)
        v1._lmcache_engine = impl

        runtime = MagicMock()
        v1.set_runtime(runtime)
        gpu_connector.set_runtime.assert_called_once_with(runtime)


class TestRBLNCalculateLocalRank:
    def _make_mock_vllm_config(self, rank=0, world_size=1, tp_size=1, pp_size=1):
        config = MagicMock()
        config.parallel_config.rank = rank
        config.parallel_config.world_size = world_size
        config.parallel_config.tensor_parallel_size = tp_size
        config.parallel_config.pipeline_parallel_size = pp_size
        return config

    def test_single_device(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.rbln_lmcache_service_factory import (
            _rbln_calculate_local_rank_and_world_size,
        )

        config = self._make_mock_vllm_config()
        local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(config)
        assert local_rank == 0
        assert local_world_size == 1

    def test_multi_device(self):
        from unittest.mock import patch

        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.rbln_lmcache_service_factory import (
            _rbln_calculate_local_rank_and_world_size,
        )

        mock_rebel = MagicMock()
        mock_rebel.get_npu_count.return_value = 4
        with patch.dict("sys.modules", {"rebel": mock_rebel}):
            config = self._make_mock_vllm_config(
                rank=2, world_size=4, tp_size=4, pp_size=1
            )
            local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(
                config
            )
            assert local_rank == 2
            assert local_world_size == 4

    def test_rebel_unavailable_defaults_to_1(self):
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.rbln_lmcache_service_factory import (
            _rbln_calculate_local_rank_and_world_size,
        )

        config = self._make_mock_vllm_config()
        local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(config)
        assert local_rank == 0
        assert local_world_size == 1
