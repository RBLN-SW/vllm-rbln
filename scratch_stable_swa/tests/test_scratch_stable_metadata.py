"""Unit tests for `scratch_stable_index_tensors` (Phase 2 of the scratch+stable SWA
prefix-caching plan, new_code_plan/scratch_stable_prefix_cache_design.MD).

Pure tensor-math; no device needed. Run:
    python tests/torch_compile/unit/v1/core/test_scratch_stable_metadata.py
or under pytest.
"""
import torch

from vllm_rbln.v1.attention.backends.flash_attention import (
    scratch_stable_index_tensors,
)

W = 1024  # window == block size (monkey-patched in the real harness)


def _call(n, ql, stable_bt, scratch_ids):
    return scratch_stable_index_tensors(
        torch.tensor(n, dtype=torch.int32),
        torch.tensor(ql, dtype=torch.int32),
        torch.tensor(stable_bt, dtype=torch.int32),
        torch.tensor(scratch_ids, dtype=torch.int32),
        W,
    )


def test_decode_indices_single_request():
    # one request, stable blocks [10, 11, 12, 13], scratch block 5.
    stable_bt = [[10, 11, 12, 13]]
    # (n, expected cache_seq_len, cache_offset, local_seq_idx, stable_block)
    cases = [
        (0, 0, 1, 0, 10),       # empty: window fill 0, write slot 0 of block 0
        (100, 100, 101, 100, 10),
        (1023, 1023, 1024, 1023, 10),  # last slot of block 0
        (1024, 1024, 1025, 0, 11),     # window full (clamp), roll to block 1, slot 0
        (1500, 1024, 1025, 476, 11),   # 1500%1024=476, 1500//1024=1
        (2048, 1024, 1025, 0, 12),     # block 2
    ]
    for n, e_csl, e_cof, e_lsi, e_sb in cases:
        csl, cof, scr, stb, lsi, send = _call([n], [1], stable_bt, [5])
        assert csl.item() == e_csl, (n, csl.item(), e_csl)
        assert cof.item() == e_cof, (n, cof.item(), e_cof)
        assert lsi.item() == e_lsi, (n, lsi.item(), e_lsi)
        assert stb.item() == e_sb, (n, stb.item(), e_sb)
        assert scr.item() == 5
        # stable_end = clamp(local_seq_idx + query_len, 1, W); never 0 (avoids the
        # degenerate window_slice that would zero the chunk), never > W (no roll).
        assert send.item() == min(max(e_lsi + 1, 1), W), (n, send.item())
        # all outputs int32, shape [NB,1]
        for t in (csl, cof, scr, stb, lsi, send):
            assert t.dtype == torch.int32 and tuple(t.shape) == (1, 1)


def test_prefill_chunk_offset_and_local_idx():
    # prefill chunk of L=128 at n=64 -> cache_offset = min(n,W)+L; local = n%W.
    stable_bt = [[20, 21]]
    csl, cof, scr, stb, lsi, send = _call([64], [128], stable_bt, [7])
    assert csl.item() == 64
    assert cof.item() == 64 + 128       # cache_seq_len + query_len
    assert lsi.item() == 64             # chunk starts at slot 64 of stable block 0
    assert stb.item() == 20
    assert scr.item() == 7
    assert send.item() == 64 + 128      # stable_end = local_seq_idx + query_len


def test_batched_requests():
    # NB=2, different positions; stable tables differ per request.
    n = [500, 1100]
    ql = [1, 1]
    stable_bt = [[30, 31, 32], [40, 41, 42]]
    scratch_ids = [8, 9]
    csl, cof, scr, stb, lsi, send = _call(n, ql, stable_bt, scratch_ids)
    assert csl.flatten().tolist() == [500, 1024]
    assert cof.flatten().tolist() == [501, 1025]
    assert lsi.flatten().tolist() == [500, 1100 % W]  # [500, 76]
    assert stb.flatten().tolist() == [30, 41]          # block 0 / block 1
    assert scr.flatten().tolist() == [8, 9]
    assert send.flatten().tolist() == [501, 77]        # local_seq_idx + query_len


TESTS = [
    test_decode_indices_single_request,
    test_prefill_chunk_offset_and_local_idx,
    test_batched_requests,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS {t.__name__}")
    print("all scratch_stable_index_tensors tests passed")
