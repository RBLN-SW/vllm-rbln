import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .conftest import KV_CACHE_SHAPE, KV_SHAPE, QUERY_SHAPE


def _meta(shape, dtype=torch.float16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _build_attention_meta():
    return (
        _meta(QUERY_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_CACHE_SHAPE),
        _meta((1, 1, 1, QUERY_SHAPE[-2], 5)),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
    )


def _build_causal_attention_meta():
    return (
        _meta(QUERY_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_CACHE_SHAPE),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
    )


def _build_flash_attention_meta():
    return (
        _meta(QUERY_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_CACHE_SHAPE),
        _meta((1, 1, 1, QUERY_SHAPE[-2], 5)),
        _meta((), dtype=torch.float16),
        _meta((2, 1), dtype=torch.int16),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
    )


def _build_flash_causal_attention_meta():
    return (
        _meta(QUERY_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_CACHE_SHAPE),
        _meta((), dtype=torch.float16),
        _meta((2, 2), dtype=torch.int16),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
    )


def _build_sliding_window_attention_meta():
    return (
        _meta(QUERY_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_SHAPE),
        _meta(KV_CACHE_SHAPE),
        _meta((2, 1), dtype=torch.int16),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
        _meta((2, 1), dtype=torch.int16),
        _meta((), dtype=torch.float16),
    )


FAKE_OP_SPECS = [
    ("attention_naive_prefill", _build_attention_meta),
    ("attention_naive_decode", _build_attention_meta),
    ("causal_attention_naive_prefill", _build_causal_attention_meta),
    ("causal_attention_naive_decode", _build_causal_attention_meta),
    ("flash_attention_naive_prefill", _build_flash_attention_meta),
    ("flash_attention_naive_decode", _build_flash_attention_meta),
    ("flash_causal_attention_naive_prefill", _build_flash_causal_attention_meta),
    ("flash_causal_attention_naive_decode", _build_flash_causal_attention_meta),
    ("sliding_window_attention_naive_prefill", _build_sliding_window_attention_meta),
    ("sliding_window_attention_naive_decode", _build_sliding_window_attention_meta),
]


@pytest.mark.parametrize(
    ("op_name", "build_meta"),
    FAKE_OP_SPECS,
    ids=[s[0] for s in FAKE_OP_SPECS],
)
def test_fake_op_returns_correct_shape_and_dtype(op_name, build_meta):
    """All fake ops must return a tensor matching the query's shape and dtype."""
    args = build_meta()
    with FakeTensorMode(allow_non_fake_inputs=True):
        op = getattr(torch.ops.rbln_triton_ops, op_name)
        result = op(*args)
    assert result.shape == QUERY_SHAPE
    assert result.dtype == args[0].dtype
