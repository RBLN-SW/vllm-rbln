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

"""A request's GatedDeltaNet state block maps to a backend row for as long as
the request lives, and the metadata builder describes a step the way the RBLN
kernels read it."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.gdn import (
    RBLNGDNAttentionMetadataBuilder,
    RBLNGDNStateRows,
    gdn_state_rows,
)

LAYER = "model.layers.1.linear_attn"


class TestStateRows:
    def test_blocks_keep_their_rows_while_live(self):
        rows = RBLNGDNStateRows(num_live=3)
        assert rows.rows([7], live={7}, batch=1) == [0]
        assert rows.rows([7, 9], live={7, 9}, batch=2) == [0, 1]
        # 9 was prefilled alone; both come back on the rows they took.
        assert rows.rows([9, 7], live={7, 9}, batch=4) == [1, 0, 5, 6]

    def test_a_finished_block_frees_its_row(self):
        rows = RBLNGDNStateRows(num_live=2)
        rows.rows([7, 9], live={7, 9}, batch=2)
        # 7 finished: its row goes to the next new block.
        assert rows.rows([9, 4], live={9, 4}, batch=2) == [1, 0]
        assert rows.row_of == {9: 1, 4: 0}

    def test_the_null_block_takes_a_scratch_row_per_position(self):
        # A dummy run schedules block 0 on every position; a decode pads with it.
        rows = RBLNGDNStateRows(num_live=2)
        assert rows.rows([0, 0], live=set(), batch=2) == [2, 3]
        assert rows.row_of == {}

    def test_no_free_row_is_an_error(self):
        rows = RBLNGDNStateRows(num_live=1)
        rows.rows([7], live={7}, batch=1)
        with pytest.raises(RuntimeError, match="no free GatedDeltaNet state row"):
            rows.rows([9], live={7, 9}, batch=1)


def _layer():
    return SimpleNamespace(
        tp_size=1,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=2 * 256 + 512,
        conv_kernel_size=4,
    )


@pytest.fixture
def builder():
    vllm_config = make_vllm_config(
        max_num_batched_tokens=128, max_num_seqs=4, dtype="float32"
    )
    vllm_config.compilation_config.static_forward_context[LAYER] = _layer()
    spec = MambaSpec(
        block_size=vllm_config.model_config.max_model_len,
        shapes=((64, 512),),
        dtypes=(torch.uint8,),
    )
    return RBLNGDNAttentionMetadataBuilder(
        spec, [LAYER], vllm_config, torch.device("cpu")
    )


def _common(
    query_lens: list[int], seq_lens: list[int], blocks: list[int]
) -> SimpleNamespace:
    starts = [0]
    for n in query_lens:
        starts.append(starts[-1] + n)
    return SimpleNamespace(
        num_reqs=len(query_lens),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.tensor(blocks, dtype=torch.int32).view(-1, 1),
    )


class TestBuilder:
    def test_state_shapes_follow_the_kernels(self, builder):
        state = builder.states[LAYER]
        # 4 live rows + 4 scratch rows; 4 fp32 heads are 128 R1 rows each
        assert state.conv_state.shape == (8, 3, 16, 64)
        assert state.ssm_state.shape == (8, gdn_state_rows(4, torch.float32), 512)
        assert state.ssm_state.dtype == torch.uint8
        assert state.conv_width == 3

    def test_first_prefill_chunk(self, builder):
        positions = torch.arange(128)
        md = builder.build(_common([70], [70], [7]), positions, 1, True, [7])
        assert md.is_prefill and md.batch_size == 1 and md.seq_len == 128
        assert md.state_index.tolist() == [0]
        assert md.has_initial_state.tolist() == [0]
        assert md.seq_idx.tolist() == [0] and md.query_len.tolist() == [70]
        assert md.valid_mask.shape == (1, 1, 128)
        assert md.valid_mask.sum() == 70 and md.valid_mask[..., :70].all()

    def test_later_prefill_chunk_carries_state(self, builder):
        positions = torch.arange(70, 70 + 58)
        md = builder.build(_common([58], [128], [7]), positions, 1, True, [7])
        assert md.has_initial_state.tolist() == [1]
        assert md.seq_idx.tolist() == [70] and md.query_len.tolist() == [58]

    def test_decode_pads_the_batch_onto_scratch_rows(self, builder):
        builder.build(_common([70], [70], [7]), torch.arange(128), 1, True, [7])
        builder.build(_common([5], [5], [3]), torch.arange(128), 1, True, [7, 3])
        positions = torch.tensor([128, 5])
        md = builder.build(
            _common([1, 1], [129, 6], [7, 3]), positions, 4, False, [7, 3]
        )
        assert not md.is_prefill and md.batch_size == 4 and md.seq_len == 1
        assert md.state_index.tolist() == [0, 1, 6, 7]
        assert md.seq_idx.tolist() == [128, 5, 0, 0]
        assert md.query_len.tolist() == [1, 1, 0, 0]
        assert md.has_initial_state.tolist() == [1, 1, 0, 0]
        assert md.valid_mask is None

    def test_staging_buffers_are_reused(self, builder):
        positions = torch.arange(128)
        first = builder.build(_common([70], [70], [7]), positions, 1, True, [7])
        second = builder.build(_common([58], [128], [7]), positions, 1, True, [7])
        assert first.query_len.data_ptr() == second.query_len.data_ptr()
        assert second.query_len.tolist() == [58]
