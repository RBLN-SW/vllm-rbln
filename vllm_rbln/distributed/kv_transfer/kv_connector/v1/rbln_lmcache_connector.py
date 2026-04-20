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

"""RBLN LMCache KV connector for vLLM — thin re-export shim.

All RBLN-specific logic lives in the ``lmcache_rbln`` package.
This module re-exports ``RBLNLMCacheConnectorV1`` so that vllm-rbln's
factory registration can reference it via a stable path inside the
``vllm_rbln`` namespace.

Usage in vLLM config::

    kv_connector = (
        "vllm_rbln.distributed.kv_transfer"
        ".kv_connector.v1.rbln_lmcache_connector"
        ".RBLNLMCacheConnectorV1"
    )
"""

from lmcache_rbln.integration.vllm.connector import RBLNLMCacheConnectorV1

__all__ = ["RBLNLMCacheConnectorV1"]
