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

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from tests.vllm.v1.attention.utils import make_builder, make_common_attn_metadata
from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.glm5_next import (
    RBLNGlm5MetadataBuilder,
    RBLNGlm5StateMetadata,
)

# The KDA state group's load / store blocks: the column of each request's block
# table a step reads its state from and the one it writes it to.

MAX_LEN = 32


@pytest.fixture(scope="module")
def cfg():
    return make_vllm_config(max_model_len=MAX_LEN, max_num_batched_tokens=8)


def _builder(cfg, block_size):
    spec = MambaSpec(block_size=block_size, shapes=((4, 512),), dtypes=(torch.uint8,))
    return make_builder(cfg, spec=spec, builder_cls=RBLNGlm5MetadataBuilder)


def _build(builder, *, query_lens, seq_lens, block_table, is_prefill, **kwargs):
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + q)
    md = builder.build(
        make_common_attn_metadata(
            num_reqs=len(seq_lens),
            query_start_loc=torch.tensor(starts),
            seq_lens=torch.tensor(seq_lens),
            block_table_tensor=torch.tensor(block_table),
        ),
        torch.arange(MAX_LEN),
        is_prefill=is_prefill,
        **kwargs,
    )
    assert isinstance(md, RBLNGlm5StateMetadata)
    return md.load_blocks.view(-1).tolist(), md.store_blocks.view(-1).tolist()


def test_whole_sequence_block_reads_and_writes_in_place(cfg):
    # block_size = max_model_len: one block a request, the one updated in place.
    blocks = _build(
        _builder(cfg, MAX_LEN),
        query_lens=[1, 1],
        seq_lens=[9, 20],
        block_table=[[5], [3]],
        is_prefill=False,
        batch_pad=3,
    )
    assert blocks == ([5, 3, 0], [5, 3, 0])


@pytest.mark.parametrize(
    "computed, query_len, expected",
    [
        (0, 4, (10, 10)),  # fresh: nothing read, written at its last token
        (4, 4, (10, 11)),  # tokens 4 .. 7: read behind block 0, written in block 1
        (8, 4, (11, 12)),  # tokens 8 .. 11
        (6, 4, (11, 12)),  # tokens 6 .. 9, crossing into block 2
    ],
)
def test_prefill_reads_last_computed_block_writes_last_token_block(
    cfg, computed, query_len, expected
):
    blocks = _build(
        _builder(cfg, 4),
        query_lens=[query_len],
        seq_lens=[computed + query_len],
        block_table=[[10, 11, 12, 13]],
        is_prefill=True,
        batch_pad=1,
    )
    assert blocks == ([expected[0]], [expected[1]])


def test_decode_moves_on_at_a_block_boundary(cfg):
    # Token 7 stays in block 1; token 8 starts block 2 from block 1's state.
    blocks = _build(
        _builder(cfg, 4),
        query_lens=[1, 1],
        seq_lens=[8, 9],
        block_table=[[10, 11, 12, 13], [20, 21, 22, 23]],
        is_prefill=False,
        batch_pad=2,
    )
    assert blocks == ([11, 21], [11, 22])


def test_verify_reads_where_the_previous_verify_wrote(cfg):
    # The previous step's token 0 sat at position 7 (block 1); its 2 accepted drafts
    # (positions 8, 9) are replayed from block 1's draft buffer. This step's token 0
    # is position 10: written to block 2.
    blocks = _build(
        _builder(cfg, 4),
        query_lens=[2],
        seq_lens=[12],
        block_table=[[10, 11, 12, 13]],
        is_prefill=False,
        batch_pad=1,
        num_accepted=torch.tensor([2], dtype=torch.int32),
    )
    assert blocks == ([11], [12])
