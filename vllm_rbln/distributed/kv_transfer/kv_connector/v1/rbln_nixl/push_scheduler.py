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

from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPushConnectorScheduler,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class RblnNixlPushConnectorScheduler(RblnNixlSchedulerBase, NixlPushConnectorScheduler):
    """Scheduler side of the write path."""

    def request_finished(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        """Give a request turned away before it was scheduled the field the
        metadata builder will read from it.

        NOTE(RBLN): a request the serving layer rejects -- a prompt past the
        context length, a client that left -- reaches upstream still flagged for
        a remote prefill, and upstream registers an empty receive for it so the
        producer stops holding the blocks it pinned. Building that receive reads
        `remote_block_ids`, which on this direction is filled by
        `update_state_after_alloc`, the one call a rejected request never makes,
        and the engine dies on the missing key. The read path is unaffected:
        there the field arrives with the producer's own reply.
        """
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            params.setdefault("remote_block_ids", ())
        return super().request_finished(request, block_ids)
