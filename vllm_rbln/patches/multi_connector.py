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
"""Backport of vllm-project/vllm#46865 — give every sub-connector the request's
real blocks in ``MultiConnector.update_state_after_alloc``.

TODO(vllm>=0.26.0): delete this module, its registration in
``vllm_rbln/patches/__init__.py`` and ``tests/native/patches/test_multi_connector.py``.
The fix is upstream (merged to vLLM ``main`` as ``2285cfc`` on 2026-07-09) and
ships in 0.26.0; it is **not** in 0.24.0 — verified by reading the installed
source. The ``condition`` below already makes the patch a no-op on a fixed vLLM,
so bumping before the cleanup is safe — the module just becomes dead weight.

Why it matters
--------------
``update_state_after_alloc`` is the scheduler-side hook vLLM calls right after
it admits a request and allocates its KV blocks
(``vllm/v1/core/sched/scheduler.py``)::

    new_blocks = self.kv_cache_manager.allocate_slots(request, ...)
    if self.connector is not None:
        self.connector.update_state_after_alloc(
            request,
            self.kv_cache_manager.get_blocks(request_id),   # ALL blocks
            num_external_computed_tokens,
        )

vLLM documents the hook as "the connector uses this info to determine if a load
is needed", so ``MultiConnector`` felt free to hand *empty* blocks to every
sub-connector that did not win the lookup::

    chosen = self._requests_to_connector.get(request.request_id, -1)
    empty_blocks = blocks.new_empty()
    for i, c in enumerate(self._connectors):
        c.update_state_after_alloc(
            request, blocks if i == chosen else empty_blocks, ...)

But an *offload* connector (LMCache) also uses that argument to record **which
physical blocks hold the request's KV**, which its store path then reads:
``allocated_tokens`` bounds how much may be stored, and the recorded block ids
become the ``LoadStoreOp.block_ids`` of the STORE op. Blank blocks therefore
mean "cannot store", not merely "will not load".

``_requests_to_connector`` is only populated for a connector whose
``get_num_new_matched_tokens`` returned a non-zero count. On a prefill instance
with a cold offload cache **nobody** returns non-zero — the NIXL child is the
producer and has nothing to pull, and the offload child has an empty cache — so
``chosen`` stays ``-1`` and *every* child gets empty blocks. That closes a cycle
the deployment cannot break out of::

    cold cache -> lookup returns 0 -> not chosen -> empty blocks
      -> no block ids recorded -> allocated_tokens == 0 -> no STORE op
      -> cache stays cold

Observed on R100 PD-disagg (MiniMax-M2.5, DP4, NIXL + LMCache mp under one
``MultiConnector``): ``lmcache describe kvcache`` reported ``cached_objects: 0``
indefinitely while lookups, worker registration and the SHM transport were all
healthy. Instrumenting ``GetStoreMetadata`` showed ``num_scheduled_tokens``
climbing to 12,443 and ``len(all_token_ids)`` at 17,347 while
``allocated_tokens`` never left 0.
"""

import inspect
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _upstream_still_blanks_blocks() -> bool:
    """True while the installed vLLM hands empty blocks to non-chosen children.

    Read from the source rather than comparing ``vllm.__version__`` against
    0.26.0: vllm-rbln pins vLLM through its own lock, so a build can sit on a
    ``main`` snapshot that carries the fix without carrying the version, and a
    version test would then re-patch it. Reading the source is exact either way.
    If the source cannot be read we assume the bug is present — applying the
    patch on an already-fixed build is behaviour-neutral (the replacement *is*
    that fix).
    """
    try:
        source = inspect.getsource(MultiConnector.update_state_after_alloc)
    except (OSError, TypeError):
        logger.warning(
            "Could not read MultiConnector.update_state_after_alloc source; "
            "applying the vllm#46865 backport unconditionally."
        )
        return True
    return "empty_blocks" in source


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "multi_connector.MultiConnector.update_state_after_alloc"
    ),
    reason=(
        "Backport vllm-project/vllm#46865: MultiConnector handed empty blocks "
        "to every sub-connector that did not win the lookup, which starves an "
        "offload connector's store path — it records the request's block ids "
        "from this argument. On a prefill instance with a cold cache nobody "
        "wins, so no child ever sees the real blocks and the cache can never "
        "bootstrap. TODO(vllm>=0.26.0): delete — upstream fix is 2285cfc."
    ),
    condition=_upstream_still_blanks_blocks,
)
def patched_update_state_after_alloc(
    self: MultiConnector,
    request: "Request",
    blocks: "KVCacheBlocks",
    num_external_tokens: int,
) -> None:
    """Hand the real blocks to every sub-connector.

    Only ``num_external_tokens`` distinguishes the chosen connector now: it is
    the one performing the load, so the others get ``0``. They still learn
    which blocks belong to the request, which is what a connector that *stores*
    needs.
    """
    chosen_connector = self._requests_to_connector.get(request.request_id, -1)
    for i, c in enumerate(self._connectors):
        c.update_state_after_alloc(
            request,
            blocks,
            num_external_tokens if i == chosen_connector else 0,
        )
