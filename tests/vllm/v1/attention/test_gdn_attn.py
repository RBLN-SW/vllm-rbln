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
block table, and the conv ring is read and stored by the same blocks."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionMetadataBuilder,
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


def _builder(max_num_batched_tokens: int = 128, **engine_args):
    vllm_config = make_vllm_config(
        max_model_len=MAX_LEN,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=4,
        **engine_args,
    )
    spec = MambaSpec(
        block_size=MAX_LEN,
        shapes=((4, 128, 128),),
        dtypes=(torch.bfloat16,),
    )
    # The flash builder it extends reads the config it is built under.
    with set_current_vllm_config(vllm_config):
        return RBLNGDNAttentionMetadataBuilder(
            spec, [LAYER], vllm_config, torch.device("cpu")
        )


@pytest.fixture
def builder():
    return _builder()


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


def _assert_conv_rows(md, blocks, seq_idx, query_len, has_initial_state):
    """kda_conv's per-row operands: the ring is stored where it was read."""
    want = {
        "conv_state_index": blocks,
        "conv_seq_idx": seq_idx,
        "conv_query_len": query_len,
        "conv_has_initial_state": has_initial_state,
        "conv_store_index": blocks,
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

    def test_a_prefill_step_takes_the_shortest_chunk_size_that_holds_it(self):
        builder = _builder(256, additional_config={"prefill_chunk_size": [128, 256]})
        for tokens, chunk in ((70, 128), (128, 128), (129, 256)):
            md = builder.build(
                _common([tokens], [tokens], [[7]]), torch.arange(chunk), 1, True
            )
            assert md.seq_len == chunk
            assert md.valid_mask.shape == (1, 1, chunk)
            assert md.valid_mask.sum() == tokens

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
