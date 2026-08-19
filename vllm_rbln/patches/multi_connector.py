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
"""Patches for ``MultiConnector``'s fan-out over its sub-connectors.

Both patches below fix the same shape of bug: ``MultiConnector`` hands every
member the same thing, and a member that cannot use it takes the whole stack
down with it (or silently stops working).

1. ``update_state_after_alloc`` — backport of vllm-project/vllm#46865
   (``2285cfc``).
2. ``set_xfer_handshake_metadata_pp_aware`` — no upstream fix filed yet; see the
   comment on the patch itself.
"""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.version import __version_tuple__ as VLLM_VERSION

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Fails loudly on the upgrade that makes the vllm#46865 backport redundant.
assert VLLM_VERSION < (0, 26), (
    f"vLLM {VLLM_VERSION} already ships vllm#46865; delete "
    "patched_update_state_after_alloc and the tests covering it in "
    "tests/native/patches/test_multi_connector.py. The PP-aware handshake "
    "patch below is independent — keep it until upstream stops requiring "
    "every sub-connector to be PP-aware."
)


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "multi_connector.MultiConnector.update_state_after_alloc"
    ),
    reason=(
        "Backport vllm#46865 (2285cfc): empty blocks for non-chosen sub-connectors "
        "starve an offload connector's store path. TODO(vllm>=0.26.0): delete."
    ),
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


# The base implementation refuses metadata it cannot interpret rather than
# guessing, which is right for a lone connector. Identity comparison against it
# is how we tell "implements the PP-aware hook" from "inherits the refusal".
_BASE_PP_AWARE_HOOK = KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware


def _implements_pp_aware_handshake(connector: KVConnectorBase_V1) -> bool:
    return (
        getattr(type(connector), "set_xfer_handshake_metadata_pp_aware", None)
        is not _BASE_PP_AWARE_HOOK
    )


@register_patch(
    target=(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "multi_connector.MultiConnector.set_xfer_handshake_metadata_pp_aware"
    ),
    reason=(
        "Upstream forwards pp_rank > 0 handshake metadata to every sub-connector, "
        "so one member that does not implement the PP-aware hook fails engine "
        "startup for the whole stack — even when it performs no cross-instance "
        "transfer at all."
    ),
)
def patched_set_xfer_handshake_metadata_pp_aware(
    self: MultiConnector,
    metadata: "dict[tuple[int, int], KVConnectorHandshakeMetadata]",
) -> None:
    """Give each member the metadata it can actually read.

    Upstream fans the full ``{(pp_rank, tp_rank): metadata}`` map out to every
    member, and ``KVConnectorBase_V1``'s default hook raises on any key with
    ``pp_rank > 0``:

        ValueError: RBLNLMCacheConnectorV1 received pp_rank > 0 handshake
                    metadata but does not support PP-disaggregated KV transfer.

    That kills the engine core before it serves a request, and it kills it for
    a member that does not transfer anything across instances. Our production
    P/D shape is exactly that: ``RblnNixlConnector`` moves KV from prefill to
    decode, while ``RBLNLMCacheConnectorV1`` is an offload/reuse tier local to
    each pod. Running prefill with ``--pipeline-parallel-size 4`` therefore
    fails at startup even though nothing about the transfer changed.

    So the full map goes to members that implement the hook, and the
    ``pp_rank == 0`` slice goes to the rest — which is what those members
    received before PP existed. A member that does need every producer stage
    must implement the hook to be handed it; being non-PP-aware is the claim
    that it does not.
    """
    pp0 = {key: meta for key, meta in metadata.items() if key[0] == 0}
    downgraded: list[str] = []

    for c in self._connectors:
        if _implements_pp_aware_handshake(c):
            c.set_xfer_handshake_metadata_pp_aware(metadata)
        else:
            c.set_xfer_handshake_metadata_pp_aware(pp0)
            downgraded.append(type(c).__name__)

    if downgraded and len(pp0) != len(metadata):
        logger.warning(
            "MultiConnector: sub-connectors without the PP-aware KV handshake "
            "were given the pp_rank=0 shards only (%d of %d): %s. That is "
            "correct for a connector that does not transfer KV across "
            "instances; one that does must implement "
            "set_xfer_handshake_metadata_pp_aware.",
            len(pp0),
            len(metadata),
            ", ".join(downgraded),
        )
