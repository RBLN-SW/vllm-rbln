import torch
import pytest

import vllm_rbln


TRITON_KERNEL_OPS = [
    "attention_naive_prefill",
    "attention_naive_decode",
    "causal_attention_naive_prefill",
    "causal_attention_naive_decode",
    "flash_attention_naive_prefill",
    "flash_attention_naive_decode",
    "flash_causal_attention_naive_prefill",
    "flash_causal_attention_naive_decode",
    "sliding_window_attention_naive_prefill",
    "sliding_window_attention_naive_decode",
]


@pytest.fixture(autouse=True)
def register_triton_ops(monkeypatch):
    monkeypatch.setattr(vllm_rbln.envs, "VLLM_RBLN_USE_VLLM_MODEL", True, raising=False)
    vllm_rbln.register_ops()


@pytest.mark.parametrize("op_name", TRITON_KERNEL_OPS)
def test_triton_kernel_ops_are_registered(op_name: str):
    op = getattr(torch.ops.rbln_triton_ops, op_name, None)

    assert op is not None, f"Expected {op_name} to be registered"
    assert hasattr(op, "default"), f"Expected {op_name} to expose a default overload"
