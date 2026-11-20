# Copyright 2025 Rebellions Inc. All rights reserved.
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

from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorScheduler,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.v1.kv_cache import REMOTE_STATE_BOUNDARY

if TYPE_CHECKING:
    from vllm.v1.request import Request


class RblnNixlPullConnectorScheduler(RblnNixlSchedulerBase, NixlPullConnectorScheduler):
    """Scheduler side of the read path."""

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free_blocks, params = super().request_finished(request, block_ids)
        if (
            not delay_free_blocks
            or params is None
            or not params["do_remote_prefill"]
            or not self._has_mamba
        ):
            return delay_free_blocks, params
        remote_block_ids, boundary = self._select_boundary_states(
            request, block_ids, params["remote_block_ids"]
        )
        params["remote_block_ids"] = remote_block_ids
        if boundary is not None:
            params[REMOTE_STATE_BOUNDARY] = boundary
        return delay_free_blocks, params
