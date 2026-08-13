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
"""Backport of vllm-project/vllm#46865 (``2285cfc``).

``MultiConnector`` handed empty blocks to every sub-connector that did not win
the lookup. An offload connector records those block ids for its store path, so
blank blocks mean "cannot store". On a prefill instance with a cold cache nobody
wins, so no child ever sees the real blocks and the cache never bootstraps.

TODO(vllm>=0.26.0): delete this module, its entry in ``patches/__init__.py`` and
``tests/native/patches/test_multi_connector.py``.
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
    # Source check, not a version compare: vllm-rbln pins vLLM by lock, so a
    # build can report < 0.26.0 while already carrying the fix.
    try:
        source = inspect.getsource(MultiConnector.update_state_after_alloc)
    except (OSError, TypeError):
        logger.warning("Cannot read MultiConnector source; applying vllm#46865 backport.")
        return True
    return "empty_blocks" in source


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "multi_connector.MultiConnector.update_state_after_alloc"
    ),
    reason=(
        "Backport vllm#46865 (2285cfc): empty blocks for non-chosen sub-connectors "
        "starve an offload connector's store path. TODO(vllm>=0.26.0): delete."
    ),
    condition=_upstream_still_blanks_blocks,
)
def patched_update_state_after_alloc(
    self: MultiConnector,
    request: "Request",
    blocks: "KVCacheBlocks",
    num_external_tokens: int,
) -> None:
    chosen_connector = self._requests_to_connector.get(request.request_id, -1)
    for i, c in enumerate(self._connectors):
        c.update_state_after_alloc(
            request,
            blocks,
            num_external_tokens if i == chosen_connector else 0,
        )
