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

# Unit coverage for the RBLN NIXL metadata extension: the layer axis and the
# chiplet geometry a shard advertises.
#
# Self-contained: exercises only ``rbln_nixl.metadata`` (base vLLM NIXL +
# msgspec + hash), so it does not require the ``nixl-rbln`` install or a worker.

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl import metadata as md
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RBLN_NIXL_CONNECTOR_VERSION,
    RblnNixlAgentMetadata,
    rbln_compat_hash,
)

_BASE_FIELDS = dict(
    engine_id="engine-0",
    agent_metadata=b"agent-bytes",
    kv_caches_base_addr=[0x1000, 0x2000],
    device_id=0,
    num_blocks=64,
    block_lens=[8192, 8192],
    kv_cache_layout="HND",
    block_size=16,
    ssm_sizes=(0, 0),
    attn_backend_name="RBLN_FLASH_ATTN",
    physical_blocks_per_logical_kv_block=1,
)


def _make(**pp):
    return RblnNixlAgentMetadata(**_BASE_FIELDS, **pp)


class TestRblnNixlAgentMetadata:
    def test_defaults_are_single_stage(self):
        # Omitting the new fields yields the single-shard, single-area values a
        # blob from an older schema decodes to.
        m = _make()
        assert (m.pp_rank, m.pp_size) == (0, 1)
        assert m.registered_layer_names == []
        assert (m.kv_areas, m.kv_slices) == (1, 1)

    def test_roundtrip_preserves_pp_fields(self):
        # msgspec encode→decode with the RBLN type preserves PP descriptors.
        m = _make(
            pp_rank=1,
            pp_size=2,
            registered_layer_names=["model.layers.14", "model.layers.15"],
        )
        enc = msgspec.msgpack.Encoder().encode(m)
        back = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(enc)
        assert back == m
        assert back.pp_rank == 1
        assert back.pp_size == 2
        assert back.registered_layer_names == [
            "model.layers.14",
            "model.layers.15",
        ]

    def test_upstream_consumer_ignores_pp_fields(self):
        # A blob encoded with the RBLN type decodes cleanly as upstream's
        # ``NixlAgentMetadata`` (PP fields ignored) — backward compatible.
        enc = msgspec.msgpack.Encoder().encode(
            _make(pp_rank=1, pp_size=2, kv_areas=4, kv_slices=2)
        )
        base = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(enc)
        assert base.engine_id == "engine-0"
        assert base.num_blocks == 64
        assert base.block_lens == [8192, 8192]
        assert not hasattr(base, "pp_rank")
        assert not hasattr(base, "kv_areas")

    def test_roundtrip_preserves_chiplet_geometry(self):
        # The head axis rides the same blob as the layer axis. Replication is the
        # case that matters: areas > slices has to survive, or the peer reads the
        # byte arithmetic off the wrong divisor.
        m = _make(kv_areas=4, kv_slices=2)
        back = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            msgspec.msgpack.Encoder().encode(m)
        )
        assert (back.kv_areas, back.kv_slices) == (4, 2)

    def test_registered_layer_names_order_preserved(self):
        names = [f"model.layers.{i}" for i in range(14, 28)]
        m = _make(pp_rank=1, pp_size=2, registered_layer_names=names)
        back = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            msgspec.msgpack.Encoder().encode(m)
        )
        assert back.registered_layer_names == names


class TestRblnCompatHash:
    def test_deterministic(self):
        assert rbln_compat_hash("BASE", writes_into_peer=False) == rbln_compat_hash(
            "BASE", writes_into_peer=False
        )

    def test_differs_from_base(self):
        # Folding our version in must move the hash, or a peer that speaks only
        # upstream's schema would match one that speaks ours.
        assert rbln_compat_hash("BASE", writes_into_peer=False) != "BASE"

    def test_the_two_transfer_directions_do_not_share_a_hash(self):
        # A producer that writes into the consumer and one the consumer reads
        # from describe the same bytes, so every length check on the handshake
        # passes and only this separates them.
        assert rbln_compat_hash("BASE", writes_into_peer=True) != rbln_compat_hash(
            "BASE", writes_into_peer=False
        )

    def test_distinguishes_base_hashes(self):
        assert rbln_compat_hash("hash-a", writes_into_peer=False) != rbln_compat_hash(
            "hash-b", writes_into_peer=False
        )

    def test_version_is_folded(self, monkeypatch):
        # Bumping RBLN_NIXL_CONNECTOR_VERSION changes the hash (gates schema drift).
        h1 = md.rbln_compat_hash("BASE", writes_into_peer=False)
        monkeypatch.setattr(
            md, "RBLN_NIXL_CONNECTOR_VERSION", RBLN_NIXL_CONNECTOR_VERSION + 1
        )
        assert md.rbln_compat_hash("BASE", writes_into_peer=False) != h1
