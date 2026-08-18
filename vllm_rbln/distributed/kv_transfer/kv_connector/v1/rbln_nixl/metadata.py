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

"""Connector metadata carrying each request's trace context to the worker.

The KV-transfer spans (``remote_fetch``, ``nixl.wait_for_transfer``) are emitted
in the worker, which has no per-request trace context: vLLM's
``_get_smart_context()`` falls back to the ``traceparent`` injected into
``os.environ`` at worker spawn, so those spans became roots of their own traces
and were invisible in the request waterfall. Only ``ca.request.id`` tied them
back to a request.

The scheduler *does* hold the context — ``Request.trace_headers`` carries the
``traceparent`` the sidecar sent. Connector metadata is the existing
scheduler→worker channel, so the headers ride along with the block IDs they
belong to instead of needing a second transport.

``KVConnectorMetadata`` travels by pickle (``MessageQueue`` shm_broadcast) under
the multiproc executor and by reference under uniproc, so a plain field is
enough; no encoder hook is required.
"""

from __future__ import annotations

from collections.abc import Mapping

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId


class RblnNixlConnectorMetadata(NixlConnectorMetadata):
    """Upstream metadata plus the trace headers of the requests it describes.

    ``NixlConnector.start_load_kv`` asserts ``isinstance(…,
    NixlConnectorMetadata)``, which a subclass satisfies, so nothing upstream has
    to know this exists.
    """

    def __init__(self) -> None:
        super().__init__()
        #: req_id → the request's inbound trace headers. Only populated for
        #: requests whose ``Request.trace_headers`` is set, which vLLM fills in
        #: solely when tracing is configured — absent entries are the normal
        #: untraced case, not an error.
        self.trace_headers: dict[ReqId, Mapping[str, str]] = {}
