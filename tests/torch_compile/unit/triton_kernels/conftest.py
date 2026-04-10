import pytest

import vllm_rbln

QUERY_SHAPE = (2, 1, 4, 3, 32)
KV_SHAPE = (2, 1, 1, 3, 32)
KV_CACHE_SHAPE = (2, 2, 1, 1, 4, 32)

ALL_OPS = [
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
