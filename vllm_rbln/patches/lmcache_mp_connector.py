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
"""Backport of vllm-project/vllm#47505 (open, unmerged as of writing).

A non-chosen ``MultiConnector`` sub-connector calls ``update_state_after_alloc``
with ``num_external_tokens=0``, but ``LMCacheMPConnectorUpstream`` decides
whether to load purely from ``needs_retrieve()``, which only looks at block
counts. It wrongly waits to load and leaks lookup locks (nothing outside
``end_session`` releases them). This backport adds
``and num_external_tokens > 0`` to that condition, matching upstream's fix.

TODO(vllm-pr-47505): delete this module and its entry in
``patches/__init__.py`` once vllm#47505 merges and the vllm-rbln pin picks it
up.
"""

import importlib.util
from typing import TYPE_CHECKING

from vllm_rbln.patches import register_patch

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
        LMCacheMPConnectorUpstream,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


def _lmcache_available() -> bool:
    """True when the optional ``lmcache`` package is installed.

    The vLLM-vendored target module exists in the pin, but importing it
    does ``from lmcache.utils import ...`` at module scope, so without
    ``lmcache`` the import raises ``ModuleNotFoundError`` and the registry
    then AttributeErrors on the parent package. Native CI does not install
    ``lmcache``.
    """
    return importlib.util.find_spec("lmcache") is not None


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector."
        "LMCacheMPConnectorUpstream.update_state_after_alloc"
    ),
    reason=(
        "Backport vllm#47505 (open, unmerged): a non-chosen MultiConnector "
        "sub-connector sees num_external_tokens=0 but needs_retrieve() ignores "
        "that, so it wrongly waits to load and leaks lookup locks."
    ),
    condition=_lmcache_available,
)
def patched_update_state_after_alloc(
    self: "LMCacheMPConnectorUpstream",
    request: "Request",
    blocks: "KVCacheBlocks",
    num_external_tokens: int,
) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
        LMCacheMPRequestState,
        logger,
        reformat_block_ids,
    )

    tracker = self._get_request_tracker(request.request_id)
    block_ids = reformat_block_ids(blocks.get_block_ids())

    existing_count = len(tracker.allocated_block_ids)
    new_block_ids = block_ids[existing_count:]
    if new_block_ids:
        tracker.append_block_ids(new_block_ids)

    # Exact patch vs upstream — the rest of this method is a copy.
    # upstream:  condition = tracker.needs_retrieve()
    condition = tracker.needs_retrieve() and num_external_tokens > 0
    if tracker.state == LMCacheMPRequestState.PREFETCHING:
        tracker.state = (
            LMCacheMPRequestState.WAITING_FOR_LOAD
            if condition
            else LMCacheMPRequestState.READY
        )
        self.scheduler_adapter.cleanup_lookup_result(request.request_id)

        if tracker.num_lmcache_hit_blocks > 0:
            if not condition:
                free_end = tracker.num_lmcache_hit_blocks * self.vllm_block_size
            else:
                free_end = tracker.num_vllm_hit_blocks * self.vllm_block_size

            if free_end > 0:
                self.scheduler_adapter.free_lookup_locks(
                    token_ids=list(tracker.all_token_ids),
                    start=0,
                    end=free_end,
                    request_id=request.request_id,
                )
                logger.debug(
                    "Free locks of tokens %d-%d since it is cached by vLLM.",
                    0,
                    free_end,
                )
