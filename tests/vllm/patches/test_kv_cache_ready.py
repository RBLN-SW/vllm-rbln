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

"""The backported ``on_kv_cache_ready`` connector hook (vllm-project/vllm#66574).

Stub children only; no worker or engine is started.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.patches.kv_cache_ready import (
    multi_connector_on_kv_cache_ready,
    on_kv_cache_ready,
)


def test_every_connector_has_the_hook():
    # LMCache's multiprocess connector refuses a worker whose base class lacks
    # the attribute, so its presence is the contract, not an implementation.
    assert KVConnectorBase_V1.on_kv_cache_ready is on_kv_cache_ready
    assert MultiConnector.on_kv_cache_ready is multi_connector_on_kv_cache_ready


def test_multi_connector_tells_every_child_in_order():
    calls: list[int] = []
    multi = SimpleNamespace(
        _connectors=[
            SimpleNamespace(on_kv_cache_ready=lambda i=i: calls.append(i))
            for i in range(3)
        ]
    )

    multi_connector_on_kv_cache_ready(multi)

    assert calls == [0, 1, 2]
