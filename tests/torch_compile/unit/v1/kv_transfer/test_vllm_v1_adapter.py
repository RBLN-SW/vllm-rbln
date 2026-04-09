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

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache.rbln_lmcache_service_factory import (
    RBLNServiceFactory,
)

FACTORY_MODULE = (
    "vllm_rbln.distributed.kv_transfer.kv_connector.v1.lmcache"
    ".rbln_lmcache_service_factory"
)


class TestRBLNServiceFactoryInheritance:
    def test_is_subclass_of_vllm_service_factory(self):
        from lmcache.integration.vllm.vllm_service_factory import (
            VllmServiceFactory,
        )

        assert issubclass(RBLNServiceFactory, VllmServiceFactory)

    def test_overrides_get_or_create_metadata(self):
        assert "get_or_create_metadata" in RBLNServiceFactory.__dict__

    def test_overrides_get_or_create_lmcache_engine(self):
        assert "get_or_create_lmcache_engine" in RBLNServiceFactory.__dict__


class TestRBLNServiceFactoryEngine:
    def _make_factory(self, role="worker"):
        mock_config = MagicMock()
        mock_config.chunk_size = 256
        mock_config.enable_scheduler_bypass_lookup = False

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.get_num_layers.return_value = 32
        mock_vllm_config.model_config.get_num_kv_heads.return_value = 8
        mock_vllm_config.model_config.get_head_size.return_value = 128
        mock_vllm_config.model_config.model = "test-model"
        mock_vllm_config.model_config.served_model_name = "test"
        mock_vllm_config.parallel_config.rank = 0
        mock_vllm_config.parallel_config.world_size = 1
        mock_vllm_config.parallel_config.tensor_parallel_size = 1
        mock_vllm_config.parallel_config.pipeline_parallel_size = 1
        mock_vllm_config.cache_config.cache_dtype = "auto"
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.kv_transfer_config = None

        factory = RBLNServiceFactory(mock_config, mock_vllm_config, role)
        return factory

    def test_returns_existing_engine(self):
        mock_engine = MagicMock()
        factory = self._make_factory()

        factory.metadata = MagicMock()
        with patch(
            f"{FACTORY_MODULE}.LMCacheEngineBuilder.get",
            return_value=mock_engine,
        ):
            result = factory.get_or_create_lmcache_engine()
            assert result is mock_engine

    def _metadata_patches(self):
        return (
            patch(
                "lmcache.integration.vllm.utils.mla_enabled",
                return_value=False,
            ),
            patch(
                "lmcache.integration.vllm.utils.validate_mla_config",
            ),
            patch(
                "lmcache.integration.vllm.utils.calculate_draft_layers",
                return_value=0,
            ),
            patch(
                "vllm.utils.torch_utils.get_kv_cache_torch_dtype",
                return_value=torch.float16,
            ),
        )

    def test_scheduler_role_creates_no_connector(self):
        mock_engine = MagicMock()
        factory = self._make_factory(role="scheduler")
        factory.lmcache_config.enable_scheduler_bypass_lookup = True

        with ExitStack() as stack:
            for p in self._metadata_patches():
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    f"{FACTORY_MODULE}.LMCacheEngineBuilder.get",
                    return_value=None,
                )
            )
            mock_create = stack.enter_context(
                patch(
                    f"{FACTORY_MODULE}.LMCacheEngineBuilder.get_or_create",
                    return_value=mock_engine,
                )
            )

            factory.get_or_create_lmcache_engine()

            call_args = mock_create.call_args
            connector_arg = call_args[0][3]
            assert connector_arg is None

    def test_worker_role_creates_rbln_connector(self):
        mock_engine = MagicMock()
        mock_connector = MagicMock()
        factory = self._make_factory(role="worker")

        with ExitStack() as stack:
            for p in self._metadata_patches():
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    f"{FACTORY_MODULE}.LMCacheEngineBuilder.get",
                    return_value=None,
                )
            )
            mock_create = stack.enter_context(
                patch(
                    f"{FACTORY_MODULE}.LMCacheEngineBuilder.get_or_create",
                    return_value=mock_engine,
                )
            )
            stack.enter_context(
                patch(
                    "vllm.distributed.parallel_state.get_tp_group",
                    return_value=MagicMock(),
                )
            )
            mock_create_connector = stack.enter_context(
                patch(
                    f"{FACTORY_MODULE}.RBLNConnector",
                    return_value=mock_connector,
                )
            )

            factory.get_or_create_lmcache_engine()

            mock_create_connector.assert_called_once()
            call_args = mock_create.call_args
            connector_arg = call_args[0][3]
            assert connector_arg is mock_connector
