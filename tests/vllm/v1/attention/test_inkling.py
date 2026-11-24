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

"""The Inkling builder adds kda_conv's operands to the flash metadata: a row
loads its short-convolution ring from the KV block holding its last computed
token and stores it to the block holding its last scheduled token."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.inkling import (
    RBLNInklingAttentionMetadata,
    RBLNInklingAttentionMetadataBuilder,
)

BLOCK = 256


@pytest.fixture
def builder():
    vllm_config = make_vllm_config(
        max_model_len=4 * BLOCK,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        additional_config={"prefill_chunk_size": [128]},
    )
    spec = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    with set_current_vllm_config(vllm_config):
        return RBLNInklingAttentionMetadataBuilder(
            spec, ["model.layers.0.attn"], vllm_config, torch.device("cpu")
        )


def _common(query_lens, seq_lens, block_table):
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


def _conv_rows(md: RBLNInklingAttentionMetadata) -> dict[str, list[int]]:
    names = (
        "conv_load_blocks",
        "conv_store_blocks",
        "conv_start_positions",
        "conv_query_lens",
        "has_initial_state",
    )
    for name in names:
        assert getattr(md, name).dtype == torch.int32, name
    return {name: getattr(md, name).tolist() for name in names}


def test_decode_rings_follow_each_rows_blocks_and_pad_with_null(builder):
    # Row 0 stays in its second block; row 1's token is the first of its
    # second block, so it loads the ring of block 30 and stores to block 31.
    md = builder.build(
        _common([1, 1], [300, BLOCK + 1], [[10, 11, 0], [30, 31, 0]]),
        torch.tensor([299, BLOCK]),
        batch_pad=4,
        is_prefill=False,
    )
    assert isinstance(md, RBLNInklingAttentionMetadata)
    assert _conv_rows(md) == {
        "conv_load_blocks": [11, 30, 0, 0],
        "conv_store_blocks": [11, 31, 0, 0],
        "conv_start_positions": [299, BLOCK, 0, 0],
        "conv_query_lens": [1, 1, 0, 0],
        "has_initial_state": [1, 1, 0, 0],
    }


def test_first_prefill_chunk_starts_from_an_empty_ring(builder):
    md = builder.build(
        _common([128], [128], [[5, 6, 0]]),
        torch.arange(128),
        batch_pad=1,
        is_prefill=True,
    )
    assert _conv_rows(md) == {
        "conv_load_blocks": [0],
        "conv_store_blocks": [5],
        "conv_start_positions": [0],
        "conv_query_lens": [128],
        "has_initial_state": [0],
    }


def test_prefill_chunk_entering_a_block_loads_the_previous_one(builder):
    md = builder.build(
        _common([128], [2 * BLOCK + 64], [[5, 6, 7]]),
        torch.arange(2 * BLOCK - 64, 2 * BLOCK + 64),
        batch_pad=1,
        is_prefill=True,
    )
    rows = _conv_rows(md)
    assert rows["conv_load_blocks"] == [6]
    assert rows["conv_store_blocks"] == [7]
    assert rows["has_initial_state"] == [1]
