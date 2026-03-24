from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import set_current_vllm_config


@pytest.fixture
def backend_module():
    from vllm_rbln.v1.attention.backends import flash_attention

    return flash_attention


@pytest.fixture
def attention_impl_factory(vllm_config, backend_module):
    def _factory(**overrides):
        kwargs = {
            "num_heads": 4,
            "head_size": 32,
            "scale": 0.5,
            "num_kv_heads": 1,
            "alibi_slopes": None,
            "sliding_window": None,
            "kv_cache_dtype": "auto",
        }
        kwargs.update(overrides)

        with set_current_vllm_config(vllm_config):
            impl = backend_module.RBLNFlashAttentionImpl(**kwargs)

        impl.enforce_eager = True
        return impl

    return _factory


def _make_inputs(q_len: int = 1, *, batch_size: int = 1, block_size: int = 4):
    query = torch.arange(batch_size * q_len * 4 * 32, dtype=torch.float32).reshape(
        batch_size, q_len, 4 * 32
    )
    key = torch.arange(batch_size * q_len * 32, dtype=torch.float32).reshape(
        batch_size, q_len, 32
    )
    value = (torch.arange(batch_size * q_len * 32, dtype=torch.float32) + 1).reshape(
        batch_size, q_len, 32
    )
    kv_cache = torch.zeros((2, 2, 1, 1, block_size, 32), dtype=torch.float32)
    return query, key, value, kv_cache


def _make_metadata(*, is_prefill: bool, q_len: int = 1):
    return SimpleNamespace(
        is_prefill=is_prefill,
        attn_masks=torch.ones((1, 1, 1, q_len, 4), dtype=torch.float32),
        seq_lens=torch.ones((1, 1), dtype=torch.int16),
        block_tables=torch.zeros((1, 1), dtype=torch.int16),
        cache_seq_lens=torch.ones((1, 1), dtype=torch.int16),
        cache_offsets=torch.full((1, 1), q_len, dtype=torch.int16),
        local_block_tables=torch.zeros((1, 1), dtype=torch.int16),
        swa_attn_masks=torch.ones((1, 1, 1, 4), dtype=torch.float32),
    )


def _configure_runtime(
    monkeypatch, backend_module, *, compile_model: bool, use_custom_kernel: bool
):
    monkeypatch.setattr(
        backend_module.envs,
        "VLLM_RBLN_COMPILE_MODEL",
        compile_model,
        raising=False,
    )
    monkeypatch.setattr(
        backend_module.envs,
        "VLLM_RBLN_USE_CUSTOM_KERNEL",
        use_custom_kernel,
        raising=False,
    )


def _patch_namespace_op(monkeypatch, namespace, op_name: str, stub: Mock):
    monkeypatch.setattr(namespace, op_name, stub, raising=False)


def test_forward_given_compiled_triton_sliding_window_decode_routes_to_triton_namespace(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "sliding_window_attention_naive_decode",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "sliding_window_attention_naive_decode",
        not_selected,
    )

    attention_impl = attention_impl_factory(sliding_window=4)
    metadata = _make_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_custom_flash_prefill_routes_to_custom_namespace(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_attention_naive_prefill",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "flash_attention_naive_prefill",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_metadata(is_prefill=True, q_len=2)
    query, key, value, kv_cache = _make_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
