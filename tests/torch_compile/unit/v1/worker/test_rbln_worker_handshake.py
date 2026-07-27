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

"""Unit coverage for RBLNWorker.get_kv_connector_handshake_metadata.

Verifies the handshake dict is keyed by a flat intra-engine global rank
(pp_rank * tp_size + tp_rank) so pipeline-parallel stages don't collide when
EngineCore merges the per-worker dicts, while non-PP behavior (key == tp_rank)
is unchanged.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import vllm_rbln.v1.worker.rbln_worker as rw
from vllm_rbln.v1.worker.rbln_worker import RBLNWorker


def _handshake(
    *,
    pp_rank,
    tp_rank,
    tp_size,
    has_group=True,
    metadata="META",
):
    """Call get_kv_connector_handshake_metadata with the parallel-state and
    KV-transfer module functions mocked."""
    worker = object.__new__(RBLNWorker)
    tp_group = MagicMock()
    tp_group.rank_in_group = tp_rank
    tp_group.world_size = tp_size
    pp_group = MagicMock()
    pp_group.rank_in_group = pp_rank
    connector = MagicMock()
    connector.get_handshake_metadata.return_value = metadata
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(rw, "has_kv_transfer_group", return_value=has_group)
        )
        stack.enter_context(
            patch.object(rw, "get_kv_transfer_group", return_value=connector)
        )
        stack.enter_context(patch.object(rw, "get_tp_group", return_value=tp_group))
        stack.enter_context(patch.object(rw, "get_pp_group", return_value=pp_group))
        return worker.get_kv_connector_handshake_metadata()


class TestHandshakeMetadataKeying:
    def test_non_pp_key_is_tp_rank(self):
        """pp_rank == 0 → global_rank == tp_rank (no regression)."""
        assert _handshake(pp_rank=0, tp_rank=0, tp_size=1) == {0: "META"}
        assert _handshake(pp_rank=0, tp_rank=2, tp_size=4) == {2: "META"}

    def test_pp_stage_offset(self):
        """global_rank = pp_rank * tp_size + tp_rank."""
        # TP=1: each PP stage gets its pp_rank as the key.
        assert _handshake(pp_rank=0, tp_rank=0, tp_size=1) == {0: "META"}
        assert _handshake(pp_rank=1, tp_rank=0, tp_size=1) == {1: "META"}
        # TP=2, PP=2: (pp,tp) -> pp*2+tp.
        assert _handshake(pp_rank=0, tp_rank=1, tp_size=2) == {1: "META"}
        assert _handshake(pp_rank=1, tp_rank=0, tp_size=2) == {2: "META"}
        assert _handshake(pp_rank=1, tp_rank=1, tp_size=2) == {3: "META"}

    def test_pp_stages_do_not_collide_on_merge(self):
        """Merging per-stage dicts (EngineCore content.update) preserves every
        stage — the collision that {tp_rank: ...} caused under PP is gone."""
        merged = {}
        for pp_rank in range(2):  # pp_size=2, tp_size=1 -> keys 0 and 1
            merged.update(
                _handshake(
                    pp_rank=pp_rank,
                    tp_rank=0,
                    tp_size=1,
                    metadata=f"stage{pp_rank}",
                )
            )
        assert merged == {0: "stage0", 1: "stage1"}

    def test_returns_none_without_kv_transfer_group(self):
        assert _handshake(pp_rank=0, tp_rank=0, tp_size=1, has_group=False) is None

    def test_returns_none_when_connector_has_no_metadata(self):
        assert _handshake(pp_rank=1, tp_rank=0, tp_size=1, metadata=None) is None
