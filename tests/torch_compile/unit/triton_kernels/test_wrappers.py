import importlib

import pytest
import torch

import vllm_rbln


QUERY_SHAPE = (2, 1, 4, 3, 32)
KV_SHAPE = (2, 1, 1, 3, 32)
KV_CACHE_SHAPE = (2, 2, 1, 1, 4, 32)


def _build_attention_args():
    return (
        torch.ones(QUERY_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.zeros(KV_CACHE_SHAPE, dtype=torch.float16),
        torch.ones((1, 1, 1, QUERY_SHAPE[-2], 5), dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.5, dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.0, dtype=torch.float16),
    )


def _build_causal_attention_args():
    return (
        torch.ones(QUERY_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.zeros(KV_CACHE_SHAPE, dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.5, dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.0, dtype=torch.float16),
    )


def _build_flash_attention_args():
    return (
        torch.ones(QUERY_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.zeros(KV_CACHE_SHAPE, dtype=torch.float16),
        torch.ones((1, 1, 1, QUERY_SHAPE[-2], 5), dtype=torch.float16),
        torch.tensor(0.5, dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.0, dtype=torch.float16),
    )


def _build_flash_causal_attention_args():
    return (
        torch.ones(QUERY_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.zeros(KV_CACHE_SHAPE, dtype=torch.float16),
        torch.tensor(0.5, dtype=torch.float16),
        torch.zeros((2, 2), dtype=torch.int16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.0, dtype=torch.float16),
    )


def _build_sliding_window_attention_args():
    return (
        torch.ones(QUERY_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.ones(KV_SHAPE, dtype=torch.float16),
        torch.zeros(KV_CACHE_SHAPE, dtype=torch.float16),
        torch.ones((2, 1), dtype=torch.int16),
        torch.full((2, 1), 3, dtype=torch.int16),
        torch.tensor(0.5, dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.int16),
        torch.tensor(0.0, dtype=torch.float16),
    )


WRAPPER_SPECS = [
    (
        "vllm_rbln.triton_kernels.attention",
        "attention_naive_prefill",
        _build_attention_args,
        5,
        (1, 4, 32, 3, 4, 2, 2),
    ),
    (
        "vllm_rbln.triton_kernels.causal_attention",
        "causal_attention_naive_decode",
        _build_causal_attention_args,
        4,
        (1, 4, 32, 3, 4, 2, 2),
    ),
    (
        "vllm_rbln.triton_kernels.flash_attention",
        "flash_attention_naive_prefill",
        _build_flash_attention_args,
        5,
        (1, 4, 32, 3, 2, 4, 5, 2, 2),
    ),
    (
        "vllm_rbln.triton_kernels.flash_causal_attention",
        "flash_causal_attention_naive_decode",
        _build_flash_causal_attention_args,
        4,
        (1, 4, 32, 3, 2, 4, 8, 2, 2),
    ),
    (
        "vllm_rbln.triton_kernels.sliding_window_attention",
        "sliding_window_attention_naive_prefill",
        _build_sliding_window_attention_args,
        4,
        (1, 4, 32, 3, 4, 2, 2, 2),
    ),
]


@pytest.fixture(autouse=True)
def register_triton_ops(monkeypatch):
    monkeypatch.setattr(vllm_rbln.envs, "VLLM_RBLN_USE_VLLM_MODEL", True, raising=False)
    vllm_rbln.register_ops()


@pytest.mark.parametrize(
    ("module_name", "op_name", "build_args", "output_idx", "expected_tail"),
    WRAPPER_SPECS,
)
def test_triton_wrapper_given_real_inputs_calls_warmup_with_expected_derived_params(
    monkeypatch,
    module_name: str,
    op_name: str,
    build_args,
    output_idx: int,
    expected_tail: tuple[int, ...],
):
    module = importlib.import_module(module_name)
    captured = {}

    monkeypatch.setattr(
        module.rblib, "align_tensor_last_dim_to_64", lambda tensor: tensor
    )

    def fake_warmup(func, *args):
        captured["func"] = func
        captured["args"] = args

    monkeypatch.setattr(module, "warmup", fake_warmup)

    args = build_args()
    output = getattr(torch.ops.rbln_triton_ops, op_name)(*args)

    assert captured["func"].__name__ == op_name
    assert tuple(captured["args"][-len(expected_tail) :]) == expected_tail
    assert captured["args"][0].dtype == torch.float32
    assert captured["args"][1].dtype == torch.float32
    assert captured["args"][2].dtype == torch.float32
    assert captured["args"][output_idx].dtype == torch.float32
    assert output.dtype == args[0].dtype
    assert output.shape == args[0].shape
