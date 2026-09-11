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
"""Let ``MultiConnector`` collect KV-cache events from its children.

``MultiConnector`` never implements ``get_kv_connector_kv_cache_events``, so it
inherits the base method that returns ``None``. A child connector's own
implementation is therefore never called, and its events never reach vLLM's
event publisher. Upstream knows -- ``multi_connector.py`` carries the TODO:

    # TODO: Add a generic implementation of 'get_kv_connector_kv_cache_events'
    # method for the MultiConnector. It should be able to get events from
    # multiple connectors, handling the case where only a subset of the
    # requested connectors implements the ...
    # WIP: https://github.com/vllm-project/vllm/pull/31811

That WIP PR was closed without being merged (2026-02-16) and ``main`` still has
the TODO, so there is nothing to wait for.

The neighbouring methods on the same class already fan out to the children --
``update_connector_output`` and ``take_events`` both iterate ``_connectors`` --
so only this one is missing. Everything downstream of it is already wired:

    kv_connector_model_runner_mixin.py  calls this method after model execution
    MultiConnector.update_connector_output  hands the worker output to children
    MultiConnector.take_events              collects from children
    Scheduler.update_from_output            merges and publishes

Deployed impact: with ``MultiConnector`` wrapping NIXL plus an offload
connector (PD disaggregation needs NIXL, so this is the normal shape), the
offload tier's KV events are dropped wholesale. Measured on a PD-disaggregated
MiniMax-M2.7 deployment: the endpoint picker's KV-block index admitted 0 blocks
before this patch and 640 blocks per request after it, matching the number of
blocks the offload tier actually stored.

Scope: this returns the first child that produces events rather than merging
across children, because the event container is connector-specific and cannot
be combined generically. That covers the "only a subset implements it" case the
upstream TODO describes, which is the shape we deploy. If a second child ever
produces events too, we log once and keep the first -- a generic merge belongs
upstream.
"""

import logging
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_rbln.patches import register_patch

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorKVEvents

logger = logging.getLogger(__name__)

# Fails loudly on the upgrade that makes this patch redundant.
assert (
    MultiConnector.get_kv_connector_kv_cache_events
    is KVConnectorBase_V1.get_kv_connector_kv_cache_events
), (
    "upstream MultiConnector now implements get_kv_connector_kv_cache_events; "
    "delete this module, its entry in vllm_rbln/patches/__init__.py and "
    "tests/native/patches/test_multi_connector_kv_events.py."
)

_warned_multiple = False


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "multi_connector.MultiConnector.get_kv_connector_kv_cache_events"
    ),
    reason=(
        "MultiConnector inherits the base no-op, so a child connector's KV events "
        "never reach the publisher. Upstream TODO with no merged fix "
        "(vllm#31811 closed unmerged). TODO: delete once upstream implements it."
    ),
)
def patched_get_kv_connector_kv_cache_events(
    self: MultiConnector,
) -> "KVConnectorKVEvents | None":
    global _warned_multiple
    found = None
    for connector in self._connectors:
        events = connector.get_kv_connector_kv_cache_events()
        if events is None:
            continue
        if found is None:
            found = events
            continue
        if not _warned_multiple:
            _warned_multiple = True
            logger.warning(
                "More than one sub-connector produced KV cache events; keeping "
                "the first (%s) and dropping %s. Merging across connectors needs "
                "an upstream generic implementation.",
                type(found).__name__,
                type(events).__name__,
            )
    return found
