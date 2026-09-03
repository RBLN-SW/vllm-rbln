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

"""What an RBLN producer advertises beyond upstream's ``NixlAgentMetadata``.

Peers pair by what each holds rather than by position, on two axes, and each
axis needs one thing upstream's struct does not carry: the layer names a shard
registered, and the chiplet geometry its regions expanded into. Both describe
the sender; the receiver derives its own side and matches.

Kept in a subclass so upstream's struct and its compatibility hash stay
untouched. Both ends are RBLN, so folding a private version tag into that hash
(``rbln_compat_hash``) is enough to keep peers speaking different schemas from
completing a handshake.
"""

from dataclasses import dataclass, field

from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata

# Bump on any incompatible change to the RBLN metadata schema or semantics.
# Folded into the NIXL compatibility hash so an RBLN peer speaking a different
# schema fails the handshake cleanly (both ends are RBLN). Upstream keeps its
# own counterpart the same way (``NIXL_CONNECTOR_VERSION``).
#   1: pp_rank / pp_size / registered_layer_names (the layer axis)
#   2: + kv_areas / kv_slices (the head axis: chiplet geometry)
#   3: + the transfer direction in the hash
RBLN_NIXL_CONNECTOR_VERSION: int = 3


@dataclass
class RblnNixlAgentMetadata(NixlAgentMetadata):
    """``NixlAgentMetadata`` + which layers and which KV heads this shard holds.

    New fields default to the single-shard, single-area values, so a blob decoded
    by upstream (which uses ``NixlAgentMetadata`` and ignores the extra fields)
    degrades to the shape upstream assumes.
    """

    pp_rank: int = 0
    pp_size: int = 1
    # Registered KV-cache layer names, ordered as kv_caches_base_addr / block_lens.
    registered_layer_names: list[str] = field(default_factory=list)
    # Physical areas one logical region expanded into, and how many of them are
    # DISTINCT rather than replicas (see `_slice_head_bounds`).
    kv_areas: int = 1
    kv_slices: int = 1


def rbln_compat_hash(base_hash: str, *, writes_into_peer: bool) -> str:
    """Fold the RBLN schema version and the transfer direction into the upstream
    NIXL compat hash.

    An extension rather than a change to ``compute_nixl_compatibility_hash``,
    which stays upstream's. The direction belongs in it because the read and the
    write path move bytes by protocols that do not meet: a producer that writes
    into a consumer expecting to read finds a peer whose every length check
    passes. This vLLM hashes nothing that separates them -- the connector name
    is not a factor -- so this is the only place it can be settled.
    """
    return hash_factors(
        {
            "base": base_hash,
            "rbln_nixl_connector_version": RBLN_NIXL_CONNECTOR_VERSION,
            "rbln_writes_into_peer": writes_into_peer,
        }
    )
