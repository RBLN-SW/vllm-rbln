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

"""Forward a child connector's worker-side KV events through ``MultiConnector``.

The model runner collects worker-side KV events only through
``get_kv_connector_kv_cache_events``. ``MultiConnector`` already hands the
aggregated output back to every child on the scheduler side and chains their
``take_events``; only this worker-side hop is missing upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.patches import register_patch

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVConnectorKVEvents


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector."
        "MultiConnector.get_kv_connector_kv_cache_events"
    ),
    reason=(
        "MultiConnector leaves get_kv_connector_kv_cache_events unimplemented "
        "(a TODO; vllm-project/vllm#31811 closed unmerged), so a child's "
        "worker-side KV events never reach the scheduler: the RDS events of "
        "RBLNLMCacheConnectorV1 under MultiConnector(NIXL + LMCache) are "
        "dropped. TODO(vllm-rbln): delete once upstream MultiConnector "
        "forwards them."
    ),
    key="vllm_rbln.patches.multi_connector.get_kv_connector_kv_cache_events",
    owner_module="vllm_rbln.patches.multi_connector",
)
def get_kv_connector_kv_cache_events(
    self: MultiConnector,
) -> KVConnectorKVEvents | None:
    # Every child is called each step: a child may drain its worker-side queue
    # in this call even on the workers that report nothing.
    events = [
        child_events
        for child in self._connectors
        if (child_events := child.get_kv_connector_kv_cache_events()) is not None
    ]
    if len(events) > 1:
        raise NotImplementedError(
            "MultiConnector cannot merge KV events from more than one child "
            f"connector; got them from {len(events)} children"
        )
    return events[0] if events else None
