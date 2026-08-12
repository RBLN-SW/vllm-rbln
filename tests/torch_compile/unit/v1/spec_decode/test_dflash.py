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

import torch

from vllm_rbln.v1.spec_decode.dflash import build_block_inputs_cpu

BLOCK_LEN = 4  # 1 bonus + 3 speculative tokens
MAX_MODEL_LEN = 32


def test_block_positions_start_at_anchor():
    anchors = torch.tensor([5, 0, 17], dtype=torch.int64)
    positions, _ = build_block_inputs_cpu(
        anchor_positions=anchors,
        num_reqs_padded=4,
        block_len=BLOCK_LEN,
        max_model_len=MAX_MODEL_LEN,
        mask_dtype=torch.float32,
    )
    assert positions.shape == (4, BLOCK_LEN)
    assert positions[0].tolist() == [5, 6, 7, 8]
    assert positions[1].tolist() == [0, 1, 2, 3]
    assert positions[2].tolist() == [17, 18, 19, 20]
    # Padded request behaves like anchor 0.
    assert positions[3].tolist() == [0, 1, 2, 3]


def test_block_positions_clamped_to_max_model_len():
    anchors = torch.tensor([MAX_MODEL_LEN - 2], dtype=torch.int64)
    positions, _ = build_block_inputs_cpu(
        anchor_positions=anchors,
        num_reqs_padded=1,
        block_len=BLOCK_LEN,
        max_model_len=MAX_MODEL_LEN,
        mask_dtype=torch.float32,
    )
    assert positions.max().item() == MAX_MODEL_LEN - 1


def test_block_mask_covers_context_and_block():
    # Request 0: anchor 5 -> may attend [0, 5 + BLOCK_LEN) = [0, 9).
    # Everything at or beyond column 9 (rejected/stale KV) is masked out.
    anchors = torch.tensor([5], dtype=torch.int64)
    _, mask = build_block_inputs_cpu(
        anchor_positions=anchors,
        num_reqs_padded=2,
        block_len=BLOCK_LEN,
        max_model_len=MAX_MODEL_LEN,
        mask_dtype=torch.float32,
    )
    assert mask.shape == (2, 1, 1, 1, MAX_MODEL_LEN)
    row = mask[0, 0, 0, 0]
    assert row[: 5 + BLOCK_LEN].eq(1.0).all()
    assert row[5 + BLOCK_LEN :].eq(0.0).all()

    # Padded request row: attends only its own block (no empty softmax rows).
    padded_row = mask[1, 0, 0, 0]
    assert padded_row[:BLOCK_LEN].eq(1.0).all()
    assert padded_row[BLOCK_LEN:].eq(0.0).all()


def test_block_mask_end_clamped_to_max_model_len():
    anchors = torch.tensor([MAX_MODEL_LEN - 1], dtype=torch.int64)
    _, mask = build_block_inputs_cpu(
        anchor_positions=anchors,
        num_reqs_padded=1,
        block_len=BLOCK_LEN,
        max_model_len=MAX_MODEL_LEN,
        mask_dtype=torch.float32,
    )
    assert mask[0, 0, 0, 0].eq(1.0).all()


def test_mask_dtype_is_respected():
    anchors = torch.tensor([0], dtype=torch.int64)
    _, mask = build_block_inputs_cpu(
        anchor_positions=anchors,
        num_reqs_padded=1,
        block_len=BLOCK_LEN,
        max_model_len=MAX_MODEL_LEN,
        mask_dtype=torch.float16,
    )
    assert mask.dtype == torch.float16
