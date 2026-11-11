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

"""The GatedDeltaNet metadata builder describes a step the way the RBLN kernels
read it: the state blocks a request loads and stores come from the group's
block table, and the conv ring is read and stored by the same blocks as the
state."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionBackend
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionMetadataBuilder,
    gdn_state_shape,
    grouped_qkv_rows,
)

LAYER = "model.layers.1.linear_attn"
MAX_LEN = 8192


def test_grouped_qkv_rows_interleaves_each_key_heads_q_k_and_values():
    # 2 key heads, 2 value heads each, 1-wide heads: [q0 q1 | k0 k1 | v0 v1 v2 v3]
    assert grouped_qkv_rows(2, 2, 1).tolist() == [0, 2, 4, 5, 1, 3, 6, 7]
    # 2-wide heads: each head's rows stay adjacent
    assert grouped_qkv_rows(1, 1, 2).tolist() == [0, 1, 2, 3, 4, 5]
    assert grouped_qkv_rows(2, 1, 2).tolist() == [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]


def test_state_block_is_the_flash_backends_kv_block():
    # A GDN layer and an attention layer on one buffer are one graph input only
    # while both declare a block the same way.
    with set_current_vllm_config(make_vllm_config()):
        cache = RBLNFlashAttentionBackend.get_kv_cache_shape(1, 256, 4, 256)
    assert gdn_state_shape(4, 256, 256) == cache[1:]


def _builder(block_size: int = MAX_LEN, mamba_cache_mode: str = "none"):
    vllm_config = make_vllm_config(
        max_model_len=MAX_LEN, max_num_batched_tokens=128, max_num_seqs=4
    )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    spec = MambaSpec(
        block_size=block_size,
        shapes=(gdn_state_shape(1, 256, 128),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode=mamba_cache_mode,
    )
    # The flash builder it extends reads the config it is built under.
    with set_current_vllm_config(vllm_config):
        return RBLNGDNAttentionMetadataBuilder(
            spec, [LAYER], vllm_config, torch.device("cpu")
        )


@pytest.fixture
def builder():
    return _builder()


@pytest.fixture
def aligned():
    """Prefix caching: a state a 256-token block ("align")."""
    return _builder(block_size=256, mamba_cache_mode="align")


def _common(
    query_lens: list[int], seq_lens: list[int], block_table: list[list[int]]
) -> SimpleNamespace:
    starts = [0]
    for n in query_lens:
        starts.append(starts[-1] + n)
    return SimpleNamespace(
        num_reqs=len(query_lens),
        num_actual_tokens=starts[-1],
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32),
    )


def _assert_conv_rows(
    md, blocks, seq_idx, query_len, has_initial_state, store_blocks=None
):
    """kda_conv's per-row operands: the ring is read and stored where the state is."""
    want = {
        "conv_state_index": blocks,
        "conv_seq_idx": seq_idx,
        "conv_query_len": query_len,
        "conv_has_initial_state": has_initial_state,
        "conv_store_index": blocks if store_blocks is None else store_blocks,
    }
    for name, values in want.items():
        got = getattr(md, name)
        assert got.dtype == torch.int32 and got.tolist() == values, name


class TestBuilder:
    def test_first_prefill_chunk(self, builder):
        positions = torch.arange(128)
        md = builder.build(_common([70], [70], [[7]]), positions, 1, True)
        assert (md.num_prefills, md.num_prefill_tokens, md.num_decodes) == (1, 70, 0)
        assert md.is_prefill
        assert md.num_actual_tokens == 70
        assert md.batch_size == 1 and md.seq_len == 128
        # A fresh request names its block for both, and the kernel reads none.
        # The chunk kernel's inputs are the step's, as upstream names them.
        assert md.prefill_state_indices is md.non_spec_state_indices_tensor
        assert md.prefill_has_initial_state is md.has_initial_state
        assert md.prefill_state_indices.dtype == torch.int16
        assert md.prefill_state_indices.tolist() == [7]
        assert md.store_state_indices.tolist() == [7]
        assert md.prefill_has_initial_state.tolist() == [0]
        _assert_conv_rows(md, [7], [0], [70], [0])
        assert md.valid_mask.shape == (1, 1, 128)
        assert md.valid_mask.sum() == 70 and md.valid_mask[..., :70].all()

    def test_later_prefill_chunk_carries_state(self, builder):
        positions = torch.arange(70, 70 + 58)
        md = builder.build(_common([58], [128], [[7]]), positions, 1, True)
        assert md.num_prefill_tokens == 58
        assert md.prefill_state_indices.tolist() == [7]
        assert md.prefill_has_initial_state.tolist() == [1]
        _assert_conv_rows(md, [7], [70], [58], [1])

    def test_decode_pads_the_batch_onto_the_null_block(self, builder):
        positions = torch.tensor([128, 5])
        md = builder.build(_common([1, 1], [129, 6], [[7], [3]]), positions, 4, False)
        assert (md.num_prefills, md.num_decodes, md.num_decode_tokens) == (0, 2, 2)
        assert not md.is_prefill
        assert md.batch_size == 4 and md.seq_len == 1
        assert md.non_spec_state_indices_tensor.tolist() == [7, 3, 0, 0]
        assert md.store_state_indices.tolist() == [7, 3, 0, 0]
        # The chunk kernel's inputs belong to a prefill step.
        assert md.has_initial_state is None and md.prefill_state_indices is None
        _assert_conv_rows(md, [7, 3, 0, 0], [128, 5, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0])
        assert md.valid_mask is None

    def test_staging_buffers_are_reused(self, builder):
        positions = torch.arange(128)
        first = builder.build(_common([70], [70], [[7]]), positions, 1, True)
        second = builder.build(_common([58], [128], [[7]]), positions, 1, True)
        assert first.conv_state_index.data_ptr() == second.conv_state_index.data_ptr()
        _assert_conv_rows(second, [7], [0], [58], [1])

    @pytest.mark.parametrize(
        ("computed", "query_len", "load", "store"),
        [
            (0, 128, 10, 10),  # fresh: nothing read, written in block 0
            (128, 128, 10, 10),  # ends on the boundary: block 0 updated in place
            (256, 128, 10, 11),  # enters block 1 from block 0's finished state
            (512, 100, 11, 12),  # after a two-block hit: block 1 is only read
        ],
    )
    def test_aligned_prefill_reads_the_computed_block_and_writes_the_last_tokens(
        self, aligned, computed, query_len, load, store
    ):
        positions = torch.arange(computed, computed + 128)
        common = _common([query_len], [computed + query_len], [[10, 11, 12, 13]])
        md = aligned.build(common, positions, 1, True)
        assert md.prefill_state_indices.tolist() == [load]
        assert md.store_state_indices.tolist() == [store]
        _assert_conv_rows(
            md, [load], [computed], [query_len], [int(computed > 0)], [store]
        )

    def test_aligned_decode_enters_a_new_block_from_the_finished_one(self, aligned):
        # Token 255 ends block 0 in place; token 256 starts block 1 from block 0.
        positions = torch.tensor([255, 256])
        common = _common([1, 1], [256, 257], [[10, 11], [20, 21]])
        md = aligned.build(common, positions, 4, False)
        assert md.non_spec_state_indices_tensor.tolist() == [10, 20, 0, 0]
        assert md.store_state_indices.tolist() == [10, 21, 0, 0]
        _assert_conv_rows(
            md,
            [10, 20, 0, 0],
            [255, 256, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [10, 21, 0, 0],
        )
