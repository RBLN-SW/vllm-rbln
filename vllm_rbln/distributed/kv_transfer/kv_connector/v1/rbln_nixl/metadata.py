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
axis needs one thing upstream's struct does not carry: which PP stage a shard
is, and the chiplet geometry its regions expanded into. Both describe the
sender; the receiver derives its own side and matches.

Kept in a subclass so upstream's struct and its compatibility hash stay
untouched. Both ends are RBLN, so folding a private version tag into that hash
(``rbln_compat_hash``) is enough to keep peers speaking different schemas from
completing a handshake.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

# Bump on any incompatible change to the RBLN metadata schema or semantics, as
# upstream does with ``NIXL_CONNECTOR_VERSION``. Folded into the NIXL compat
# hash so an RBLN peer on another schema fails the handshake cleanly; earlier
# bumps are `git log -L` on this line.
#   7: the read path's completion notification carries a count, not a TP size
RBLN_NIXL_CONNECTOR_VERSION: int = 7


class KVSplitAxis(Enum):
    """Which axis the compiler cut a KV entry on across the chiplets.

    ``HEAD`` means an area holds some of the shard's KV heads over every token
    of a block; ``NON_HEAD`` means every head over some of the tokens.
    """

    HEAD = 0
    NON_HEAD = 1


@dataclass
class RblnNixlAgentMetadata(NixlAgentMetadata):
    """``NixlAgentMetadata`` + which PP stage a shard is and which slice of
    the KV cache it holds.

    New fields default to the single-shard, single-area values, so a blob decoded
    by upstream (which uses ``NixlAgentMetadata`` and ignores the extra fields)
    degrades to the shape upstream assumes.
    """

    pp_rank: int = 0
    pp_size: int = 1
    # Physical areas one logical region expanded into, and how many of them are
    # DISTINCT rather than replicas (see `_slice_head_bounds`).
    kv_areas: int = 1
    kv_slices: int = 1
    # The default keeps a blob without this field meaning what versions 2 and 3
    # meant by the two counts above.
    kv_split_axis: KVSplitAxis = KVSplitAxis.HEAD
    # What a block holds (`_kv_per_block`). A process picks the layout, not the
    # build, so two peers off one build can differ and the version cannot tell
    # them apart. The default is the layout every version through 4 had.
    kv_per_block: int = 1

    @property
    def registered_layer_names(self) -> list[str]:
        """The layers this shard registered, in region order.

        vllm 0.30 put ``region_names`` on the wire, which names a layer per
        region -- the same fact this used to carry as its own field. A layer
        owns a run of consecutive regions, so the run boundaries give the list
        back and a peer that trims regions trims this with them.
        """
        return list(dict.fromkeys(self.region_names or ()))


def rbln_compat_hash(
    base_hash: str,
    *,
    writes_into_peer: bool,
    cross_layers_blocks: bool,
    speculative_config: "SpeculativeConfig | None" = None,
) -> str:
    """Fold the RBLN schema version and the factors upstream's hash drops into
    ``compute_nixl_compatibility_hash``'s result.

    An extension rather than a change to upstream's, which stays theirs. Each
    factor here is one upstream cannot express: the transfer direction, because
    a producer that writes into a consumer expecting to read passes every
    length check; the draft model, because its attention layers are members of
    the KV cache this connector registers while upstream's factors describe the
    target alone; and ``cross_layers_blocks``, which scales the page by the
    KV-cache tensor count and which 0.30 dropped from upstream's own factors.

    Model-level values only -- the hash is compared across every shard, so a
    per-rank quantity would differ between PP stages.
    """
    factors: dict[str, object] = {
        "base": base_hash,
        "rbln_nixl_connector_version": RBLN_NIXL_CONNECTOR_VERSION,
        "rbln_writes_into_peer": writes_into_peer,
        "rbln_cross_layers_blocks": cross_layers_blocks,
    }
    if speculative_config is not None and (
        speculative_config.use_eagle() or speculative_config.uses_draft_model()
    ):
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        factors |= {
            "rbln_spec_method": speculative_config.method,
            "rbln_draft_model": draft_model_config.model,
            "rbln_draft_revision": draft_model_config.revision,
            "rbln_draft_code_revision": draft_model_config.code_revision,
        }
    return hash_factors(factors)
