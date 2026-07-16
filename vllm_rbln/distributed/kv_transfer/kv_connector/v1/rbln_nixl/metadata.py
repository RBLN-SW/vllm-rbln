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
cross-vendor interop concern — a private RBLN version tag folded into the NIXL
compatibility hash (``rbln_pp_compat_hash``) is enough to gate PP-aware peers
against mismatched ones.

Under pipeline parallelism the producer advertises, per (pp_rank, tp_rank)
shard, the registered KV-cache layer names it owns. After the side-channel
handshake the consumer matches each producer stage's shard to its own local KV
regions by name (the owned layer range is derived locally on each side, not sent
over the wire).
"""

from dataclasses import dataclass, field

from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata

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
