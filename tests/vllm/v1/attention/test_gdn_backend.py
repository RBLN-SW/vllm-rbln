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

"""The GatedDeltaNet state rows follow their requests onto the batch rows, and
the metadata builder describes a step the way the RBLN kernels read it."""

from types import SimpleNamespace

import pytest
import torch

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.gdn import (
    RBLNGDNAttentionMetadataBuilder,
    RBLNGDNStateRows,
)

LAYER = "model.layers.1.linear_attn"


def _rows(num_rows: int) -> tuple[RBLNGDNStateRows, torch.Tensor]:
    # Row r starts as the value r, so a row's content names where it came from.
    tensor = torch.arange(num_rows + 1, dtype=torch.float32).view(-1, 1).repeat(1, 3)
    return RBLNGDNStateRows(num_rows, [tensor]), tensor


def _content(tensor: torch.Tensor, row: int) -> float:
    return float(tensor[row, 0])


class TestStateRows:
    def test_new_requests_take_their_rows_without_copies(self):
        rows, tensor = _rows(6)
        rows.place(["A", "B"], batch=2)
        assert rows.owner[:2] == ["A", "B"]
        assert rows.row_of == {"A": 0, "B": 1}
        assert torch.equal(
            tensor, torch.arange(7, dtype=torch.float32).view(-1, 1).repeat(1, 3)
        )

    def test_a_prefill_parks_the_request_it_displaces(self):
        rows, tensor = _rows(6)
        rows.place(["A", "B"], batch=2)
        rows.place(["C"], batch=1)
        # A left row 0 for a free parking row; B stays on row 1, which is parking
        # for a batch of one.
        assert rows.owner[0] == "C"
        assert rows.row_of["B"] == 1
        assert _content(tensor, rows.row_of["A"]) == 0.0
        assert rows.row_of["A"] >= 1

    def test_requests_come_back_to_their_batch_rows(self):
        rows, tensor = _rows(6)
        rows.place(["A", "B"], batch=2)
        rows.place(["C"], batch=1)
        rows.place(["A", "B", "C"], batch=4)
        assert rows.owner[:4] == ["A", "B", "C", None]
        # A's state is back on row 0; C's state is whatever row 0 held when C was
        # prefilled there (the former free row's content, 2.0), now moved to row 2.
        assert [_content(tensor, r) for r in range(3)] == [0.0, 1.0, 2.0]
        assert rows.row_of == {"A": 0, "B": 1, "C": 2}

    def test_padded_rows_hold_no_live_state(self):
        rows, _ = _rows(6)
        rows.place(["A", "B", "C"], batch=4)
        rows.place(["A"], batch=4)
        assert rows.owner[0] == "A"
        assert rows.owner[1:4] == [None, None, None]
        assert {rows.row_of["B"], rows.row_of["C"]} <= set(range(4, 6))

    def test_release_frees_the_row(self):
        rows, _ = _rows(6)
        rows.place(["A", "B"], batch=2)
        rows.release(["A", "never-seen"])
        assert rows.owner[0] is None
        assert "A" not in rows.row_of
        rows.place(["C", "B"], batch=2)
        assert rows.owner[:2] == ["C", "B"]

    def test_no_parking_row_is_an_error(self):
        rows, _ = _rows(2)
        rows.place(["A", "B"], batch=2)
        with pytest.raises(RuntimeError, match="no free GatedDeltaNet state row"):
            rows.place(["C"], batch=2)


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
    return RBLNGDNAttentionMetadataBuilder(
        None, [LAYER], vllm_config, torch.device("cpu"), max_batch_pad=4
    )


def _common(query_lens: list[int], seq_lens: list[int]) -> SimpleNamespace:
    starts = [0]
    for n in query_lens:
        starts.append(starts[-1] + n)
    return SimpleNamespace(
        num_reqs=len(query_lens),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )


class TestBuilder:
    def test_state_shapes_follow_the_kernels(self, builder):
        state = builder.states[LAYER]
        # 4 batch rows + 2 x 4 parked + 1 scratch
        assert state.conv_state.shape == (13, 3, 16, 64)
        assert state.ssm_state.shape == (13, 2, 2, 2, 128, 64)
        assert state.conv_width == 3

    def test_first_prefill_chunk(self, builder):
        positions = torch.arange(128)
        md = builder.build(
            _common([70], [70]), positions, batch_pad=1, is_prefill=True, req_ids=["A"]
        )
        assert md.is_prefill and md.batch_size == 1 and md.seq_len == 128
        assert md.slot.dtype == torch.int16 and int(md.slot) == 0
        assert md.has_initial_state.tolist() == [0.0]
        assert md.has_initial_state_int.tolist() == [0]
        assert md.seq_idx.tolist() == [0] and md.query_len.tolist() == [70]
        assert md.valid_mask.shape == (1, 1, 1, 128)
        assert md.valid_mask.sum() == 70 and md.valid_mask[..., :70].all()

    def test_later_prefill_chunk_carries_state(self, builder):
        positions = torch.arange(70, 70 + 58)
        md = builder.build(
            _common([58], [128]), positions, batch_pad=1, is_prefill=True, req_ids=["A"]
        )
        assert md.has_initial_state.tolist() == [1.0]
        assert md.seq_idx.tolist() == [70] and md.query_len.tolist() == [58]

    def test_decode_pads_the_batch(self, builder):
        positions = torch.tensor([128, 4])
        md = builder.build(
            _common([1, 1], [129, 5]),
            positions,
            batch_pad=4,
            is_prefill=False,
            req_ids=["A", "B"],
        )
        assert not md.is_prefill and md.batch_size == 4 and md.seq_len == 1
        assert md.state_index.tolist() == [0, 1, 2, 3]
        assert md.seq_idx.tolist() == [128, 4, 0, 0]
        assert md.query_len.tolist() == [1, 1, 0, 0]
        assert md.has_initial_state_int.tolist() == [1, 1, 0, 0]
        assert md.valid_mask is None
        assert builder.rows.owner[:2] == ["A", "B"]

    def test_staging_buffers_are_reused(self, builder):
        positions = torch.arange(128)
        first = builder.build(
            _common([70], [70]), positions, batch_pad=1, is_prefill=True, req_ids=["A"]
        )
        second = builder.build(
            _common([58], [128]), positions, batch_pad=1, is_prefill=True, req_ids=["A"]
        )
        assert first.query_len.data_ptr() == second.query_len.data_ptr()
        assert second.query_len.tolist() == [58]
