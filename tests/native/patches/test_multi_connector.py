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

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

from vllm_rbln.patches.multi_connector import (
    patched_set_xfer_handshake_metadata_pp_aware,
    patched_update_state_after_alloc,
)


class _RecordingConnector:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append((blocks, num_external_tokens))


class _Blocks:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def _multi(*connectors: _RecordingConnector, chosen: dict[str, int] | None = None):
    return SimpleNamespace(
        _connectors=list(connectors),
        _requests_to_connector=chosen if chosen is not None else {},
    )


def test_non_chosen_connector_gets_real_blocks_with_zero_external():
    winner, loser = _RecordingConnector(), _RecordingConnector()
    multi = _multi(winner, loser, chosen={"req-0": 0})
    blocks = _Blocks("real")

    patched_update_state_after_alloc(
        multi, SimpleNamespace(request_id="req-0"), blocks, 128
    )

    assert winner.calls == [(blocks, 128)]
    assert loser.calls == [(blocks, 0)]


def test_all_connectors_get_real_blocks_when_nobody_is_chosen():
    # The deployed path: a cold offload cache means no child returns a non-zero
    # lookup, so chosen stays -1.
    first, second = _RecordingConnector(), _RecordingConnector()
    multi = _multi(first, second)
    blocks = _Blocks("real")

    patched_update_state_after_alloc(
        multi, SimpleNamespace(request_id="req-cold"), blocks, 0
    )

    for connector in (first, second):
        assert connector.calls == [(blocks, 0)]


class _PPAwareConnector:
    """Stands in for RblnNixlConnector: implements the hook, so it gets it all."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        self.seen.append(metadata)


class _LegacyConnector:
    """Stands in for RBLNLMCacheConnectorV1: inherits the base refusal.

    Bound to the real base implementation rather than a stub, so the test
    exercises the actual guard and the actual (pp, tp) -> tp flattening.
    """

    set_xfer_handshake_metadata_pp_aware = (
        KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware
    )

    def __init__(self) -> None:
        self.flattened: list[dict] = []

    def set_xfer_handshake_metadata(self, metadata):
        self.flattened.append(metadata)


def _handshake_metadata(*ranks):
    return {rank: f"meta{i}" for i, rank in enumerate(ranks)}


def test_legacy_member_survives_pp4_prefill():
    # The deployed failure: prefill at --pipeline-parallel-size 4 with
    # MultiConnector[RblnNixlConnector, RBLNLMCacheConnectorV1]. Before the
    # patch the legacy member raised and took the engine core down.
    nixl, lmcache = _PPAwareConnector(), _LegacyConnector()
    metadata = _handshake_metadata((0, 0), (1, 0), (2, 0), (3, 0))

    patched_set_xfer_handshake_metadata_pp_aware(_multi(nixl, lmcache), metadata)

    assert nixl.seen == [metadata]
    # Flattened to {tp_rank: meta}, stage 0 only — what it saw before PP.
    assert lmcache.flattened == [{0: "meta0"}]


def test_single_stage_reaches_every_member_unchanged():
    nixl, lmcache = _PPAwareConnector(), _LegacyConnector()
    metadata = _handshake_metadata((0, 0), (0, 1))

    patched_set_xfer_handshake_metadata_pp_aware(_multi(nixl, lmcache), metadata)

    assert nixl.seen == [metadata]
    assert lmcache.flattened == [{0: "meta0", 1: "meta1"}]
