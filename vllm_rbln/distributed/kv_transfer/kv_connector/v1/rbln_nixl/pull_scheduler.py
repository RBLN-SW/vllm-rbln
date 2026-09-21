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

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorScheduler,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

#: Attribute the read metadata carries the abort notifications under.
ABORT_NOTIFY_ATTR = "rbln_abort_notify_reqs"


class RblnNixlPullConnectorScheduler(RblnNixlSchedulerBase, NixlPullConnectorScheduler):
    """Scheduler side of the read path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        #: Requests queued for a read that only frees the producer's blocks,
        #: collected for the metadata being built next.
        self._abort_notify_reqs: set[str] = set()

    def request_finished(
        self,
        request: "Request",
        block_ids: "BlockIds",
    ) -> tuple[bool, dict[str, Any] | None]:
        """Mark a read that exists only to release an aborted request's blocks.

        A request aborted before it was scheduled still has to tell the producer
        to free what it prefilled, and upstream arranges that as a read with no
        blocks -- indistinguishable at the worker from a full prefix hit, which
        is a live request that must be reported. This one must not be: the
        scheduler has already dropped it, and reporting it trips the
        `assert req_id in self.requests` in `_update_from_kv_xfer_finished`
        (ICR-47).
        """
        params = request.kv_transfer_params
        notify_only = bool(params and params.get("do_remote_prefill"))

        delay_free, transfer_params = super().request_finished(request, block_ids)

        if notify_only and request.request_id in self._reqs_need_recv:
            self._abort_notify_reqs.add(request.request_id)
        return delay_free, transfer_params

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> "KVConnectorMetadata":
        meta = super().build_connector_meta(scheduler_output)
        setattr(meta, ABORT_NOTIFY_ATTR, self._abort_notify_reqs)
        self._abort_notify_reqs = set()
        return meta
