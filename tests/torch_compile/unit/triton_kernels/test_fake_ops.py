import pytest
import torch

import vllm_rbln


QUERY_SHAPE = (2, 1, 4, 3, 32)
KV_SHAPE = (2, 1, 1, 3, 32)
KV_CACHE_SHAPE = (2, 2, 1, 1, 4, 32)


def _meta_tensor(shape, *, dtype=torch.float16):
    return torch.empty(shape, device="meta", dtype=dtype)


def _meta_scalar(*, dtype=torch.float32):
    return torch.empty((), device="meta", dtype=dtype)


def _build_attention_args():
    return (
        _meta_tensor(QUERY_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_CACHE_SHAPE),
        _meta_tensor((1, 1, 1, QUERY_SHAPE[-2], 4), dtype=torch.float32),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
    )


def _build_causal_attention_args():
    return (
        _meta_tensor(QUERY_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_CACHE_SHAPE),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
    )


def _build_flash_attention_args():
    return (
        _meta_tensor(QUERY_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_CACHE_SHAPE),
        _meta_tensor((1, 1, 1, QUERY_SHAPE[-2], 4), dtype=torch.float32),
        _meta_scalar(),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
    )


def _build_flash_causal_attention_args():
    return (
        _meta_tensor(QUERY_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_CACHE_SHAPE),
        _meta_scalar(),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
    )


def _build_sliding_window_attention_args():
    return (
        _meta_tensor(QUERY_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_SHAPE),
        _meta_tensor(KV_CACHE_SHAPE),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
        _meta_tensor((2, 1), dtype=torch.int16),
        _meta_scalar(),
    )


TRITON_FAKE_OP_SPECS = [
    ("attention_naive_prefill", _build_attention_args),
    ("attention_naive_decode", _build_attention_args),
    ("causal_attention_naive_prefill", _build_causal_attention_args),
    ("causal_attention_naive_decode", _build_causal_attention_args),
    ("flash_attention_naive_prefill", _build_flash_attention_args),
    ("flash_attention_naive_decode", _build_flash_attention_args),
    ("flash_causal_attention_naive_prefill", _build_flash_causal_attention_args),
    ("flash_causal_attention_naive_decode", _build_flash_causal_attention_args),
    ("sliding_window_attention_naive_prefill", _build_sliding_window_attention_args),
    ("sliding_window_attention_naive_decode", _build_sliding_window_attention_args),
]


@pytest.fixture(autouse=True)
def register_triton_ops(monkeypatch):
    monkeypatch.setattr(vllm_rbln.envs, "VLLM_RBLN_USE_VLLM_MODEL", True, raising=False)
    vllm_rbln.register_ops()


@pytest.mark.parametrize(("op_name", "build_args"), TRITON_FAKE_OP_SPECS)
def test_triton_kernel_fake_op_given_meta_inputs_returns_query_shaped_meta_tensor(
    op_name: str, build_args
):
    args = build_args()

    output = getattr(torch.ops.rbln_triton_ops, op_name)(*args)

    assert output.device.type == "meta"
    assert output.dtype == args[0].dtype
    assert output.shape == args[0].shape
