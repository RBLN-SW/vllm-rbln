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

"""Unit coverage for the RBLN NIXL PP metadata extension.

Self-contained: exercises only ``rbln_nixl.metadata`` (base vLLM NIXL +
msgspec + hash), so it does not require the ``nixl-rbln`` install or a worker.
"""

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RBLN_NIXL_PP_VERSION,
    RblnNixlAgentMetadata,
    rbln_pp_compat_hash,
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
        """Omitting PP fields yields the pp_size==1 (no-PP) values."""
        m = _make()
        assert (m.pp_rank, m.pp_size) == (0, 1)
        assert m.registered_layer_names == []

    def test_roundtrip_preserves_pp_fields(self):
        """msgspec encode→decode with the RBLN type preserves PP descriptors."""
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

    def test_base_consumer_ignores_pp_fields(self):
        """A blob encoded with the RBLN type decodes cleanly as the base
        ``NixlAgentMetadata`` (PP fields ignored) — backward compatible."""
        enc = msgspec.msgpack.Encoder().encode(_make(pp_rank=1, pp_size=2))
        base = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(enc)
        assert base.engine_id == "engine-0"
        assert base.num_blocks == 64
        assert base.block_lens == [8192, 8192]
        assert not hasattr(base, "pp_rank")

    def test_registered_layer_names_order_preserved(self):
        names = [f"model.layers.{i}" for i in range(14, 28)]
        m = _make(pp_rank=1, pp_size=2, registered_layer_names=names)
        back = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            msgspec.msgpack.Encoder().encode(m)
        )
        assert back.registered_layer_names == names


class TestRblnPpCompatHash:
    def test_deterministic(self):
        assert rbln_pp_compat_hash("BASE") == rbln_pp_compat_hash("BASE")

    def test_differs_from_base(self):
        """Folding the PP version must change the hash, so a PP-aware producer
        and a PP-unaware peer (base hash) do not match."""
        assert rbln_pp_compat_hash("BASE") != "BASE"

    def test_distinguishes_base_hashes(self):
        assert rbln_pp_compat_hash("hash-a") != rbln_pp_compat_hash("hash-b")

    def test_version_is_folded(self, monkeypatch):
        """Bumping RBLN_NIXL_PP_VERSION changes the hash (gates schema drift)."""
        from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl import (
            metadata as md,
        )

        h1 = md.rbln_pp_compat_hash("BASE")
        monkeypatch.setattr(md, "RBLN_NIXL_PP_VERSION", RBLN_NIXL_PP_VERSION + 1)
        assert md.rbln_pp_compat_hash("BASE") != h1
