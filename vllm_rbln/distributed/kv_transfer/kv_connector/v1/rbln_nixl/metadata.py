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
compatibility hash stay untouched. Both P and D are RBLN, so a private version
tag folded into the compat hash (``rbln_pp_compat_hash``) is enough to gate
PP-aware peers against mismatched ones. Under PP the producer advertises, per
(pp_rank, tp_rank) shard, the KV-cache layer names it owns; the consumer matches
each shard to its local regions by name (owned range derived locally, not sent).
"""

from dataclasses import dataclass, field

from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata

# Bump on any incompatible change to the RBLN metadata schema/semantics.
# Folded into the NIXL compatibility hash so an RBLN peer speaking a different
# schema fails the handshake cleanly (both ends are RBLN).
#   1: pp_rank / pp_size / registered_layer_names
#   2: + kv_areas / kv_slices (chiplet geometry, for asymmetric-TP matching)
RBLN_NIXL_PP_VERSION: int = 2


@dataclass
class RblnNixlAgentMetadata(NixlAgentMetadata):
    """``NixlAgentMetadata`` + this producer shard's PP identity, the layer
    names it owns, and its chiplet geometry.

    New fields default to the single-shard, single-area values so a blob decoded
    by the base/older consumer (which uses ``NixlAgentMetadata`` and ignores the
    extra fields) degrades to single-stage behavior. An RBLN consumer decodes
    with this type to read them: ``registered_layer_names`` places each shard's
    regions on the layer axis, and ``kv_areas`` / ``kv_slices`` do the same on
    the head axis.
    """

    pp_rank: int = 0
    pp_size: int = 1
    # Registered KV-cache layer names, ordered as kv_caches_base_addr / block_lens.
    registered_layer_names: list[str] = field(default_factory=list)
    # Chiplet geometry of one KV entry on this shard. kv_areas is how many
    # physical areas each logical region expanded into; kv_slices is how many of
    # them are DISTINCT (the rest are replicas of a KV head the compiler had to
    # duplicate because the shard owns fewer heads than the device has
    # chiplets). Sent rather than derived: deriving kv_areas from
    # len(kv_caches_base_addr) assumes exactly two regions per layer, which
    # stops being true for MLA / blocks-first layouts.
    kv_areas: int = 1
    kv_slices: int = 1


def rbln_pp_compat_hash(base_hash: str) -> str:
    """Fold the RBLN PP schema version into the upstream NIXL compat hash.

    Keeps PP-aware RBLN peers from handshaking with PP-unaware / mismatched
    ones without touching upstream ``compute_nixl_compatibility_hash``.
    """
    return hash_factors(
        {"base": base_hash, "rbln_nixl_pp_version": RBLN_NIXL_PP_VERSION}
    )
