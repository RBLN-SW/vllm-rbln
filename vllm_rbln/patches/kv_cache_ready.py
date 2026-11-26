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

"""Backport the connector half of vllm-project/vllm#66574.

``KVConnectorBase_V1.on_kv_cache_ready`` tells a worker connector that the KV
cache is final and the first request has not arrived. The engine half lives in
``dynamic_kv.py``, which already owns ``EngineCore._initialize_kv_caches``, and
the worker half is ``RBLNWorker.notify_kv_cache_ready``.
"""

from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.patches import register_patch

# The engine call has to move after the dynamic-KV resize, which upstream's
# position precedes, so an upgrade must revisit it rather than stack two calls.
if hasattr(KVConnectorBase_V1, "on_kv_cache_ready"):
    raise RuntimeError(
        "vLLM ships KVConnectorBase_V1.on_kv_cache_ready (vllm-project/vllm#66574); "
        "delete vllm_rbln/patches/kv_cache_ready.py and its engine and worker halves."
    )

_REASON = (
    "vllm-project/vllm#66574 adds the startup-ready connector hook after vLLM "
    "0.31.0; LMCache's multiprocess connector refuses a worker without it. "
    "TODO(vllm-rbln): delete when the pinned vLLM includes that PR."
)


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.base."
        "KVConnectorBase_V1.on_kv_cache_ready"
    ),
    reason=_REASON,
    key="vllm_rbln.patches.kv_cache_ready.base",
)
def on_kv_cache_ready(self: KVConnectorBase_V1) -> None:
    """Finish deferred registration after final KV sizing.

    Called once on each worker before the first request, after preceding
    device work. Exceptions fail startup. Not called on sleep/wake.
    """
    return


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector."
        "MultiConnector.on_kv_cache_ready"
    ),
    reason=_REASON,
    key="vllm_rbln.patches.kv_cache_ready.multi_connector",
)
def multi_connector_on_kv_cache_ready(self: MultiConnector) -> None:
    for child in self._connectors:
        child.on_kv_cache_ready()
