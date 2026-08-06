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

"""RBLN-specific NIXL metadata: pipeline-parallel (PP) extensions.

Isolated from upstream ``NixlAgentMetadata`` so the base struct and its
compatibility hash stay untouched. Both P and D are RBLN, so there is no
cross-vendor interop concern -- a private RBLN version tag folded into the NIXL
compatibility hash (``rbln_pp_compat_hash``) is enough to gate PP-aware peers
against mismatched ones.

Under pipeline parallelism the producer advertises, per (pp_rank, tp_rank)
shard, the registered KV-cache layer names it owns. After the side-channel
handshake the consumer matches each producer stage's shard to its own local KV
regions by name (the owned layer range is derived locally on each side, not sent
over the wire).

Also home to :class:`RblnNixlConnectorMetadata`, which carries each request's
trace context on the scheduler→worker channel — a different kind of metadata that
belongs to the same module for the same reason: an RBLN extension of an upstream
NIXL struct kept out of the upstream one.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId

# Bump on any incompatible change to the RBLN PP metadata schema/semantics.
# Folded into the NIXL compatibility hash so a PP-aware producer and a
# mismatched consumer fail the handshake cleanly (both ends are RBLN).
RBLN_NIXL_PP_VERSION: int = 1


@dataclass
class RblnNixlAgentMetadata(NixlAgentMetadata):
    """``NixlAgentMetadata`` + this producer stage's PP identity and the layer
    names it owns.

    New fields default to the ``pp_size == 1`` (no-PP) values so a blob decoded
    by the base/older consumer (which uses ``NixlAgentMetadata`` and ignores the
    extra fields) degrades to single-stage behavior. A PP-aware consumer decodes
    with this type to read them and matches ``registered_layer_names`` against
    its own local layers to place each shard's regions.
    """

    pp_rank: int = 0
    pp_size: int = 1
    # Registered KV-cache layer names, ordered as kv_caches_base_addr / block_lens.
    registered_layer_names: list[str] = field(default_factory=list)


def rbln_pp_compat_hash(base_hash: str) -> str:
    """Fold the RBLN PP schema version into the upstream NIXL compat hash.

    Keeps PP-aware RBLN peers from handshaking with PP-unaware / mismatched
    ones without touching upstream ``compute_nixl_compatibility_hash``.
    """
    return hash_factors(
        {"base": base_hash, "rbln_nixl_pp_version": RBLN_NIXL_PP_VERSION}
    )


class RblnNixlConnectorMetadata(NixlConnectorMetadata):
    """Upstream connector metadata plus the trace headers of its requests.

    The KV-transfer spans (``remote_fetch``, ``nixl.wait_for_transfer``) are
    emitted in the worker, which has no per-request trace context: vLLM's
    ``_get_smart_context()`` falls back to the ``traceparent`` injected into
    ``os.environ`` at worker spawn, so those spans became roots of their own
    traces and were invisible in the request waterfall. Only ``ca.request.id``
    tied them back to a request.

    The scheduler *does* hold the context — ``Request.trace_headers`` carries the
    ``traceparent`` the sidecar sent. Connector metadata is the existing
    scheduler→worker channel, so the headers ride along with the block IDs they
    belong to instead of needing a second transport.

    ``NixlConnector.start_load_kv`` asserts ``isinstance(…,
    NixlConnectorMetadata)``, which a subclass satisfies, so nothing upstream has
    to know this exists. Connector metadata travels by pickle (``MessageQueue``
    shm_broadcast) under the multiproc executor and by reference under uniproc,
    so a plain field is enough; no encoder hook is required.
    """

    def __init__(self) -> None:
        super().__init__()
        #: req_id → the request's inbound trace headers. Only populated for
        #: requests whose ``Request.trace_headers`` is set, which vLLM fills in
        #: solely when tracing is configured — absent entries are the normal
        #: untraced case, not an error.
        self.trace_headers: dict[ReqId, Mapping[str, str]] = {}
