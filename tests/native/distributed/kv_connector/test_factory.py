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

# What can break here is a stale module path or class name after a rename. NIXL is
# resolved all the way to its class; LMCache only by presence, since its module
# imports `lmcache_rbln` at load time.

# Importing the factory module fires the register_connector side effects, exactly
# as register_ops() does under VLLM_RBLN_USE_VLLM_MODEL in production.
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa: F401


def test_native_backend_registers_both_connectors():
    registry = KVConnectorFactory._registry
    assert "RblnNixlConnector" in registry
    assert "RBLNLMCacheConnectorV1" in registry


def test_nixl_connector_resolves_to_class():
    # Calling the stored loader imports the module and pulls the class; a stale
    # module path (ImportError) or class name (AttributeError) would fail here.
    cls = KVConnectorFactory._registry["RblnNixlConnector"]()
    assert cls.__name__ == "RblnNixlConnector"
