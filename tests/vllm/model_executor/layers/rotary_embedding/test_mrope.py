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

"""The RBLN M-RoPE forward takes [batch, seq] or [3, batch, seq] positions and
must match upstream's forward_native, which takes [N] or [3, N]."""

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from tests.vllm.vllm_config import make_vllm_config

# Qwen3.5 shape: head 256 with partial_rotary_factor 0.25, sections summing to 32.
HEAD_SIZE = 256
ROTARY_DIM = 64
SECTION = [11, 11, 10]
BATCH, SEQ, HEADS, KV_HEADS = 2, 5, 4, 2


@pytest.fixture(scope="module")
def vllm_config():
    return make_vllm_config()


def _rope(
    vllm_config, *, interleaved: bool, is_neox_style: bool = True
) -> MRotaryEmbedding:
    # A CustomOp reads the compilation config while it is constructed.
    with set_current_vllm_config(vllm_config):
        return MRotaryEmbedding(
            HEAD_SIZE,
            ROTARY_DIM,
            max_position_embeddings=64,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=torch.float32,
            mrope_section=SECTION,
            mrope_interleaved=interleaved,
        )


def _qk():
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(BATCH * SEQ, HEADS * HEAD_SIZE, generator=generator)
    k = torch.randn(BATCH * SEQ, KV_HEADS * HEAD_SIZE, generator=generator)
    return q, k


def _mm_positions() -> torch.Tensor:
    # Distinct T/H/W positions so a wrong axis choice shows up.
    text = torch.arange(BATCH * SEQ).view(BATCH, SEQ)
    return torch.stack([text, text * 2 + 1, text * 3 + 2])


class TestForwardOotMatchesNative:
    @pytest.mark.parametrize("interleaved", [True, False])
    def test_text_positions(self, vllm_config, interleaved):
        rope = _rope(vllm_config, interleaved=interleaved)
        q, k = _qk()
        positions = torch.arange(BATCH * SEQ).view(BATCH, SEQ)
        qo, ko = rope.forward_oot(positions, q.clone(), k.clone())
        qn, kn = rope.forward_native(positions.flatten(), q.clone(), k.clone())
        assert torch.allclose(qo, qn, atol=1e-5)
        assert torch.allclose(ko, kn, atol=1e-5)

    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("is_neox_style", [True, False])
    def test_multimodal_positions(self, vllm_config, interleaved, is_neox_style):
        rope = _rope(vllm_config, interleaved=interleaved, is_neox_style=is_neox_style)
        q, k = _qk()
        positions = _mm_positions()
        qo, ko = rope.forward_oot(positions, q.clone(), k.clone())
        qn, kn = rope.forward_native(positions.flatten(1), q.clone(), k.clone())
        assert torch.allclose(qo, qn, atol=1e-5)
        assert torch.allclose(ko, kn, atol=1e-5)

    def test_batched_query_layout(self, vllm_config):
        # The attention layer hands [batch * seq, heads * head]; a caller may also
        # keep [batch, seq, heads * head]. Both come back in their own layout.
        rope = _rope(vllm_config, interleaved=True)
        q, k = _qk()
        positions = _mm_positions()
        qo_flat, _ = rope.forward_oot(positions, q.clone(), k.clone())
        qo_batched, _ = rope.forward_oot(
            positions, q.view(BATCH, SEQ, -1).clone(), k.view(BATCH, SEQ, -1).clone()
        )
        assert qo_batched.shape == (BATCH, SEQ, HEADS * HEAD_SIZE)
        assert torch.allclose(qo_batched.reshape_as(qo_flat), qo_flat)

    def test_offsets_are_added_to_positions(self, vllm_config):
        rope = _rope(vllm_config, interleaved=True)
        q, k = _qk()
        positions = _mm_positions()
        offsets = torch.full_like(positions, 3)
        with_offset, _ = rope.forward_oot(positions, q.clone(), k.clone(), offsets)
        shifted, _ = rope.forward_oot(positions + 3, q.clone(), k.clone())
        assert torch.allclose(with_offset, shifted)


class TestAxisTable:
    def test_interleaved_table_matches_upstream_selection(self, vllm_config):
        rope = _rope(vllm_config, interleaved=True)
        half = ROTARY_DIM // 2
        channel = torch.arange(half)
        expected = torch.zeros(half, dtype=torch.long)
        expected[(channel % 3 == 1) & (channel < SECTION[1] * 3)] = 1
        expected[(channel % 3 == 2) & (channel < SECTION[2] * 3)] = 2
        assert torch.equal(rope.mrope_axis_table[:half], expected)
        # neox caches repeat the half, so the table repeats with it
        assert torch.equal(rope.mrope_axis_table[half:], expected)

    def test_chunked_table_follows_the_sections(self, vllm_config):
        rope = _rope(vllm_config, interleaved=False)
        expected = torch.repeat_interleave(torch.arange(3), torch.tensor(SECTION))
        assert torch.equal(rope.mrope_axis_table[: ROTARY_DIM // 2], expected)
