# Copyright 2025 Rebellions Inc. All rights reserved.
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

"""Regression test for the VLLM_RBLN_USE_FLASHINFER tensor shape contract.

Asserts that ``_forward_flashinfer`` reshapes its inputs to the exact
shapes that the flashinfer_rbln TileLang kernels expect. The kernel
ABIs are:

    Decode  q/o:  [BATCH, Q_HEADS, 1, HEAD_DIM]
    Prefill q/o:  [BATCH * MAX_Q_PER_REQ, Q_HEADS, HEAD_DIM]
    k/v cache:    [NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM]

Source of truth: tilelang_rbln/examples/flash_decode_paged{,_packed}.py,
flash_attn_varlen.py.

Pre-fix, the runtime fed:
    Decode q:    [b, num_heads, head_size]                  (3D, missing seq=1)
    Prefill q:   [b, q_len, num_heads, head_size]           (4D, batch+seq unpacked)
    k/v cache:   [num_blocks, num_kv_heads, 1, block_size, D] (5D, axes swapped)

This test would catch a regression to that state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionImpl


_BSIZE = 2
_NUM_KV_HEADS = 8
_NUM_GROUPS = 1  # num_queries_per_kv
_NUM_HEADS = _NUM_KV_HEADS * _NUM_GROUPS
_HEAD_DIM = 128
_NUM_BLOCKS = 16
_BLOCK_SIZE = 32
_DTYPE = torch.float16


def _make_kv_cache() -> torch.Tensor:
    """vllm-rbln canonical KV cache layout."""
    return torch.empty(
        (2, _NUM_BLOCKS, _NUM_KV_HEADS, 1, _BLOCK_SIZE, _HEAD_DIM),
        dtype=_DTYPE,
    )


def _make_self_stub() -> SimpleNamespace:
    """Minimal stand-in for the bound method: only num_heads, head_size needed."""
    return SimpleNamespace(num_heads=_NUM_HEADS, head_size=_HEAD_DIM)


def _make_attn_metadata(is_prefill: bool) -> MagicMock:
    md = MagicMock()
    md.is_prefill = is_prefill
    md._flashinfer_metadata = MagicMock(name="fi_metadata")
    return md


def test_decode_shapes_match_kernel_abi():
    """Decode path reshapes q/k/v/out to the kernel ABI."""
    q_len = 1
    query = torch.empty(
        (_BSIZE, _NUM_KV_HEADS, _NUM_GROUPS, q_len, _HEAD_DIM), dtype=_DTYPE
    )
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    kv_cache = _make_kv_cache()
    attn_metadata = _make_attn_metadata(is_prefill=False)

    seen = {}

    def fake_run(fi_md, q, k, v, out):
        seen["q"] = q.shape
        seen["k"] = k.shape
        seen["v"] = v.shape
        seen["out"] = out.shape

    with patch("flashinfer_rbln.run", new=fake_run):
        out = RBLNFlashAttentionImpl._forward_flashinfer(
            _make_self_stub(),
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            b_size=_BSIZE,
            q_len=q_len,
        )

    # Decode kernel ABI: q/out [BATCH, Q_HEADS, 1, HEAD_DIM]
    assert seen["q"] == torch.Size((_BSIZE, _NUM_HEADS, 1, _HEAD_DIM)), seen["q"]
    assert seen["out"] == seen["q"], seen["out"]
    # K/V cache: [NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM]
    expected_kv = torch.Size((_NUM_BLOCKS, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM))
    assert seen["k"] == expected_kv, seen["k"]
    assert seen["v"] == expected_kv, seen["v"]
    # Returned tensor reshape: [b, q_len, num_heads * head_size]
    assert out.shape == torch.Size((_BSIZE, q_len, _NUM_HEADS * _HEAD_DIM))


def test_prefill_shapes_match_kernel_abi():
    """Prefill path packs batch+seq into a single leading dim and matches kernel ABI."""
    q_len = 4
    query = torch.empty(
        (_BSIZE, _NUM_KV_HEADS, _NUM_GROUPS, q_len, _HEAD_DIM), dtype=_DTYPE
    )
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    kv_cache = _make_kv_cache()
    attn_metadata = _make_attn_metadata(is_prefill=True)

    seen = {}

    def fake_run(fi_md, q, k, v, out):
        seen["q"] = q.shape
        seen["k"] = k.shape
        seen["v"] = v.shape
        seen["out"] = out.shape

    with patch("flashinfer_rbln.run", new=fake_run):
        out = RBLNFlashAttentionImpl._forward_flashinfer(
            _make_self_stub(),
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            b_size=_BSIZE,
            q_len=q_len,
        )

    # Prefill kernel ABI: q/out [BATCH * MAX_Q_PER_REQ, Q_HEADS, HEAD_DIM]
    assert seen["q"] == torch.Size((_BSIZE * q_len, _NUM_HEADS, _HEAD_DIM)), seen["q"]
    assert seen["out"] == seen["q"], seen["out"]
    expected_kv = torch.Size((_NUM_BLOCKS, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM))
    assert seen["k"] == expected_kv, seen["k"]
    assert seen["v"] == expected_kv, seen["v"]
    assert out.shape == torch.Size((_BSIZE, q_len, _NUM_HEADS * _HEAD_DIM))


@pytest.mark.parametrize("bad_block_size", [16, 24, 48])
def test_metadata_builder_rejects_block_size_below_array_n(bad_block_size):
    """_build_flashinfer_metadata fails fast when block_size is not a
    multiple of array_n (32 on REBEL).  Pre-fix, this would surface as a
    cryptic ValueError from deep inside flashinfer_rbln._planner."""
    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionMetadataBuilder,
    )

    builder = MagicMock(spec=RBLNFlashAttentionMetadataBuilder)
    builder.block_size = bad_block_size
    builder.model_config = MagicMock()
    builder.model_config.max_model_len = 4096

    with pytest.raises(ValueError, match=r"--block-size to be a multiple of array_n"):
        RBLNFlashAttentionMetadataBuilder._build_flashinfer_metadata(
            builder,
            attn_metadata=MagicMock(),
            num_reqs=1,
            num_actual_tokens=1,
            seq_lens_cpu=torch.zeros(1, dtype=torch.int32),
            raw_block_tables=torch.zeros((1, 4), dtype=torch.int32),
            batch_pad=1,
            is_prefill=False,
        )
