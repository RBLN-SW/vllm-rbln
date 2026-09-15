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

# 0.26 standardized on a blocks-first cache whose K and V share one region and
# asserts that layout in TransferTopology.__post_init__. RBLN's attention cache
# is K/V-first and fails it; its MLA cache passes. Upstream's own registration
# path builds the topology, so every test here constructs one directly.

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_transfer_topology import (
    RblnTransferTopology,
)
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionBackend
from vllm_rbln.v1.attention.backends.mla.flashattn_mla import RBLNFlashAttnMLABackend


class TestRblnTransferTopology:
    @staticmethod
    def _topology(backend, *, is_mla=False, is_mamba=False, tensor_shape=None):
        return RblnTransferTopology(
            tp_rank=0,
            tp_size=2,
            block_size=64,
            engine_id="e",
            is_mla=is_mla,
            is_mamba=is_mamba,
            total_num_kv_heads=8,
            attn_backends=[backend],
            tensor_shape=tensor_shape,
        )

    def test_the_rbln_attention_layout_builds(self):
        topo = self._topology(RBLNFlashAttentionBackend)

        assert topo.cross_layers_blocks is False
        assert topo.local_physical_heads == 4

    def test_k_and_v_come_back_as_separate_regions(self):
        # The caller divides the page size by how many regions come back, so
        # one region here would double the per-block stride.
        topo = self._topology(RBLNFlashAttentionBackend)
        cache = torch.zeros(2, 4, 8, 1, 64, 64)

        regions = topo.get_transfer_cache_regions(cache, object())

        assert len(regions) == 2
        assert all(region.shape[0] == 4 for region in regions)

    def test_an_mla_layer_stays_one_region(self):
        topo = self._topology(RBLNFlashAttnMLABackend, is_mla=True)
        cache = torch.zeros(4, 64, 576)

        assert len(topo.get_transfer_cache_regions(cache, object())) == 1

    @pytest.mark.parametrize(
        "shape",
        [
            # FlashInfer-style: blocks first, K and V packed behind them.
            lambda n, b, h, d: (n, 2, h, b, d),
            # K/V first, but tokens where the blocks axis belongs.
            lambda n, b, h, d: (2, b, h, n, d),
        ],
        ids=["blocks_first", "tokens_before_blocks"],
    )
    def test_a_cache_the_descriptors_cannot_read_is_rejected(self, shape):
        # The arithmetic reads K and V off the leading dim and the blocks off
        # the next, so a cache shaped otherwise has to stop here rather than
        # transfer halves of itself.
        class Backend:
            @staticmethod
            def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
                return shape(num_blocks, block_size, num_kv_heads, head_size)

        with pytest.raises(AssertionError, match="attention"):
            self._topology(Backend)

    def test_the_engine_map_is_there_for_upstream_to_fill(self):
        # Upstream's own register/lookup read this map; no other test here
        # would notice it missing.
        topo = self._topology(RBLNFlashAttentionBackend)
        info = EngineTransferInfo(
            remote_tp_size=1,
            remote_block_len=8,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
        )

        topo.register_remote_engine("peer", info)

        assert topo.get_engine_info("peer") is info

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"is_mamba": True},
            {"tensor_shape": torch.Size((28, 2, 4, 8, 1, 64, 64))},
        ],
    )
    def test_the_other_two_layouts_stay_on_upstream_s_path(self, kwargs):
        # A Mamba state and cross-layer blocks are upstream's own shapes; the
        # split this class restores is for the K/V-first attention cache alone.
        # Registration walks a Mamba layer before the descriptor path refuses
        # it, so that arm is reached in production.
        topo = self._topology(RBLNFlashAttentionBackend, **kwargs)
        cache = torch.zeros(2, 4, 8, 1, 64, 64)

        assert len(topo.get_transfer_cache_regions(cache, object())) == 1

    def test_a_mamba_topology_never_asks_the_backend_for_a_shape(self):
        # A Mamba cache is a (conv, ssm) pair and the connector hands it no
        # tensor shape, so the attention-shaped query has no answer to give.
        class Refuses:
            @staticmethod
            def get_kv_cache_shape(**kwargs):
                raise AssertionError("asked a Mamba topology for an attention shape")

        topo = self._topology(Refuses, is_mamba=True)

        assert topo.cross_layers_blocks is False

    def test_it_sets_what_upstream_s_post_init_sets(self):
        # __post_init__ is reimplemented rather than extended, so a field
        # upstream adds to it is simply absent here and nothing fails until
        # the first handshake reads it.
        class BlocksFirst:
            @staticmethod
            def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
                return (num_blocks, num_kv_heads, block_size, head_size)

        (upstream,) = RblnTransferTopology.__bases__
        theirs = upstream(
            tp_rank=0,
            tp_size=2,
            block_size=64,
            engine_id="e",
            is_mla=False,
            is_mamba=False,
            total_num_kv_heads=8,
            attn_backends=[BlocksFirst],
        )

        ours = self._topology(RBLNFlashAttentionBackend)

        assert vars(ours).keys() == vars(theirs).keys()

    def test_only_a_mamba_state_asks_for_the_block_split(self):
        # The connector doubles its region count off this upstream property,
        # which reads the _cross_layers_blocks that this __post_init__ sets --
        # including on the arm that returns before the shape query.
        attention = self._topology(RBLNFlashAttentionBackend)
        mamba = self._topology(RBLNFlashAttentionBackend, is_mamba=True)

        assert attention.virtually_split_kv_in_blocks is False
        assert mamba.virtually_split_kv_in_blocks is True

    def test_cross_layer_blocks_are_read_off_the_tensor_shape(self):
        topo = self._topology(
            RBLNFlashAttentionBackend, tensor_shape=torch.Size((28, 2, 4, 8, 1, 64, 64))
        )

        assert topo.cross_layers_blocks is True
