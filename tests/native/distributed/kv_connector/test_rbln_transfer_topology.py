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

# RBLN's attention cache is K/V-first on the rbln_triton_ops kernels, where K
# and V become two regions, and blocks-first on rbln_custom_ops, where they
# share a block the descriptor path cannot cut in two; the MLA and Mamba caches
# are upstream's own shapes and register either way. Upstream's own
# registration path builds the topology, so every test here constructs one
# directly.

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo

from tests.native.vllm_config import make_vllm_config
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

    @staticmethod
    def _kernel_config(use_custom_kernel):
        # What the attention cache's layout follows.
        return set_current_vllm_config(
            make_vllm_config(additional_config={"use_custom_kernel": use_custom_kernel})
        )

    def test_the_blocks_first_attention_cache_is_refused(self):
        # K and V share a block there, so the descriptors have no second region
        # to name; refusing here beats transferring halves of a block.
        with (
            self._kernel_config(False),
            pytest.raises(NotImplementedError, match="interleaves inside each"),
        ):
            self._topology(RBLNFlashAttentionBackend)

    def test_the_kv_first_attention_cache_splits_k_from_v(self):
        # Upstream packs K and V into one region; this layout keeps them apart,
        # and the caller divides the page size by how many regions come back.
        with self._kernel_config(True):
            topo = self._topology(RBLNFlashAttentionBackend)
        cache = torch.zeros(2, 4, 1, 1, 64, 8)

        assert len(topo.get_transfer_cache_regions(cache, object())) == 2

    def test_the_mla_layout_builds(self):
        topo = self._topology(RBLNFlashAttnMLABackend, is_mla=True)

        assert topo.cross_layers_blocks is False
        assert topo.local_physical_heads == 4

    def test_an_mla_layer_stays_one_region(self):
        topo = self._topology(RBLNFlashAttnMLABackend, is_mla=True)
        cache = torch.zeros(4, 64, 576)

        assert len(topo.get_transfer_cache_regions(cache, object())) == 1

    def test_an_mla_cache_the_descriptors_cannot_read_is_rejected(self):
        # The arithmetic reads the blocks off the leading dim, so a latent
        # cache shaped otherwise has to stop here.
        class Backend:
            @staticmethod
            def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
                return (block_size, num_blocks, head_size)

        with pytest.raises(AssertionError, match="MLA cache"):
            self._topology(Backend, is_mla=True)

    def test_the_engine_map_is_there_for_upstream_to_fill(self):
        # Upstream's own register/lookup read this map; no other test here
        # would notice it missing.
        topo = self._topology(RBLNFlashAttnMLABackend, is_mla=True)
        info = EngineTransferInfo(
            remote_tp_size=1,
            remote_block_len=8,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
        )

        topo.register_remote_engine("peer", info)

        assert topo.get_engine_info("peer") is info

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

        ours = self._topology(RBLNFlashAttnMLABackend, is_mla=True)

        assert vars(ours).keys() == vars(theirs).keys()

    def test_only_a_mamba_state_asks_for_the_block_split(self):
        # The connector doubles its region count off this upstream property,
        # which reads the two layout fields this __post_init__ sets.
        mla = self._topology(RBLNFlashAttnMLABackend, is_mla=True)
        mamba = self._topology(RBLNFlashAttentionBackend, is_mamba=True)

        assert mla.virtually_split_kv_in_blocks is False
        assert mamba.virtually_split_kv_in_blocks is True

    def test_cross_layer_blocks_are_read_off_the_tensor_shape(self):
        topo = self._topology(
            RBLNFlashAttnMLABackend,
            is_mla=True,
            tensor_shape=torch.Size((28, 4, 64, 576)),
        )

        assert topo.cross_layers_blocks is True
