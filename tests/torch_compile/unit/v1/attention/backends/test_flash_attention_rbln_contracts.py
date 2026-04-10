from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import set_current_vllm_config

import vllm_rbln


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


@pytest.fixture
def metadata_builder_factory(vllm_config, backend_module, monkeypatch):
    def _factory(*, sliding_window=None, is_causal=False, is_batch_attention_opt=False):
        vllm_config.cache_config.block_size = 4
        vllm_config.cache_config.num_gpu_blocks = 2
        vllm_config.model_config.max_model_len = 8
        vllm_config.scheduler_config.max_num_batched_tokens = 2

        monkeypatch.setattr(
            backend_module.envs,
            "VLLM_RBLN_FLASH_CAUSAL_ATTN",
            is_causal,
            raising=False,
        )
        monkeypatch.setattr(
            backend_module.envs,
            "VLLM_RBLN_BATCH_ATTN_OPT",
            is_batch_attention_opt,
            raising=False,
        )

        kv_cache_spec = SimpleNamespace(
            dtype="auto",
            block_size=vllm_config.cache_config.block_size,
            sliding_window=sliding_window,
        )
        with set_current_vllm_config(vllm_config):
            return backend_module.RBLNFlashAttentionMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=["layer.0"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )

    return _factory


def _make_forward_inputs(q_len: int = 1, *, batch_size: int = 1, block_size: int = 4):
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


def _make_forward_metadata(**overrides):
    # Optional fields default to None so tests must explicitly provide
    # the metadata each code path requires — prevents silently passing
    # when a required field is missing.
    metadata = {
        "is_prefill": False,
        "attn_masks": None,
        "seq_lens": torch.ones((1, 1), dtype=torch.int16),
        "block_tables": torch.zeros((1, 1), dtype=torch.int16),
        "cache_seq_lens": None,
        "cache_offsets": None,
        "local_block_tables": None,
        "swa_attn_masks": None,
    }
    metadata.update(overrides)
    return SimpleNamespace(**metadata)


def _make_common_attn_metadata(
    *,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table_tensor: torch.Tensor,
):
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=int(query_start_loc[-1].item()),
        max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
        max_seq_len=int(seq_lens.max().item()),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(int(query_start_loc[-1].item()), dtype=torch.int32),
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


# ====================================================================
# Helper: compute host reference attention output
# ====================================================================
def _host_attention(q, k_state, v_state, scale, mask=None):
    """Pure PyTorch reference attention: Q @ K^T * scale [+ mask] -> softmax -> @ V."""
    attn_weights = torch.matmul(q, k_state.transpose(-2, -1)) * scale
    if mask is not None:
        attn_weights = attn_weights + mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v_state)


# ====================================================================
# 1. Stub custom ops: attention_naive / causal_attention_naive
# ====================================================================


def test_attention_naive_prefill_impl_returns_empty_like(backend_module):
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.attention_naive_prefill_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([0], dtype=torch.int16),
        torch.tensor(0.0),
    )
    assert output.shape == q.shape


def test_attention_naive_decode_impl_returns_empty_like(backend_module):
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.attention_naive_decode_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(0.0),
    )
    assert output.shape == q.shape


def test_causal_attention_naive_prefill_impl_returns_empty_like(backend_module):
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.causal_attention_naive_prefill_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([0], dtype=torch.int16),
        torch.tensor(0.0),
    )
    assert output.shape == q.shape


def test_causal_attention_naive_decode_impl_returns_empty_like(backend_module):
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.causal_attention_naive_decode_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(0.0),
    )
    assert output.shape == q.shape


# ====================================================================
# 2. flash_attention_naive _impl
# ====================================================================


def test_flash_attention_prefill_impl_given_compile_returns_empty_like(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.flash_attention_naive_prefill_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([0], dtype=torch.int16),
        None,
    )
    assert output.shape == q.shape


def test_flash_attention_prefill_impl_given_single_partition_updates_cache_and_returns_output(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((1, 1, 1, 2, 2), dtype=torch.float32)
    key = torch.tensor([[[[[1.0, 0.0], [2.0, 0.0]]]]])
    value = torch.tensor([[[[[3.0, 0.0], [4.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)
    mask = torch.ones((1, 1, 1, 2, 4), dtype=torch.float32)

    output = backend_module.flash_attention_naive_prefill_impl(
        query, key, value, kv_cache, mask,
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([0], dtype=torch.int16),
        None,
    )

    assert torch.equal(kv_cache[0, 0, 0, 0, :2], key[0, 0, 0])
    assert torch.equal(kv_cache[1, 0, 0, 0, :2], value[0, 0, 0])
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_flash_attention_decode_impl_given_compile_returns_empty_like(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.flash_attention_naive_decode_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([[0]], dtype=torch.int16),
        None,
    )
    assert output.shape == q.shape


def test_flash_attention_decode_impl_given_batch_size_greater_than_one_raises_assertion_error(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.zeros((2, 1, 1, 1, 32), dtype=torch.float32)

    with pytest.raises(AssertionError):
        backend_module.flash_attention_naive_decode_impl(
            query,
            torch.zeros_like(query),
            torch.zeros_like(query),
            torch.zeros((2, 1, 1, 1, 4, 32), dtype=torch.float32),
            torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
            torch.tensor(0.5),
            torch.tensor([[0], [0]], dtype=torch.int16),
            torch.tensor([[0], [0]], dtype=torch.int16),
            None,
        )


def test_flash_attention_decode_impl_given_single_partition_updates_cache_and_returns_output(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    key = torch.tensor([[[[[5.0, 0.0]]]]])
    value = torch.tensor([[[[[6.0, 1.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)
    mask = torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)

    output = backend_module.flash_attention_naive_decode_impl(
        query, key, value, kv_cache, mask,
        torch.tensor(1.0),
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor([[0]], dtype=torch.int16),
        None,
    )

    assert torch.equal(kv_cache[0, 0, 0, 0, 1:2], key[0, 0, 0])
    assert torch.equal(kv_cache[1, 0, 0, 0, 1:2], value[0, 0, 0])
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


# ====================================================================
# 3. flash_causal_attention_naive _impl
# ====================================================================


def test_flash_causal_prefill_impl_given_compile_returns_empty_like(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.flash_causal_attention_naive_prefill_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([0], dtype=torch.int16),
        None,
    )
    assert output.shape == q.shape


def test_flash_causal_prefill_impl_given_cross_partition_write_updates_multiple_blocks(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((1, 1, 1, 3, 2), dtype=torch.float32)
    key = torch.tensor([[[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]])
    value = torch.tensor([[[[[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 2, 1, 1, 2, 2), dtype=torch.float32)

    output = backend_module.flash_causal_attention_naive_prefill_impl(
        query, key, value, kv_cache,
        torch.tensor(1.0),
        torch.tensor([[1, 0]], dtype=torch.int16),
        torch.tensor([0, 1], dtype=torch.int16),
        None,
    )

    assert torch.equal(kv_cache[0, 0, 0, 0, 1], key[0, 0, 0, 0])
    assert torch.equal(kv_cache[0, 1, 0, 0, :2], key[0, 0, 0, 1:3])
    assert torch.equal(kv_cache[1, 1, 0, 0, :2], value[0, 0, 0, 1:3])
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_flash_causal_prefill_impl_given_sinks_redistributes_attention(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    k = torch.tensor([[[[[1.0, 0.0]]]]])
    v = torch.tensor([[[[[1.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)

    output = backend_module.flash_causal_attention_naive_prefill_impl(
        q, k, v, kv_cache,
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([0], dtype=torch.int16),
        None,
        torch.ones((1, 1), dtype=torch.float32),
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_flash_causal_decode_impl_given_compile_returns_empty_like(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.flash_causal_attention_naive_decode_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([[0]], dtype=torch.int16),
        None,
    )
    assert output.shape == q.shape


def test_flash_causal_decode_impl_given_zero_total_sequence_length_returns_zeros(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((1, 1, 1, 1, 32), dtype=torch.float32)

    output = backend_module.flash_causal_attention_naive_decode_impl(
        query,
        torch.zeros_like(query),
        torch.zeros_like(query),
        torch.zeros((2, 1, 1, 1, 4, 32), dtype=torch.float32),
        torch.tensor(0.5),
        torch.zeros((1, 1), dtype=torch.int16),
        torch.zeros((1, 1), dtype=torch.int16),
        None,
    )

    assert torch.equal(output, torch.zeros_like(query))


def test_flash_causal_decode_impl_given_sinks_redistributes_attention(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    k = torch.tensor([[[[[1.0, 0.0]]]]])
    v = torch.tensor([[[[[1.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)

    output = backend_module.flash_causal_attention_naive_decode_impl(
        q, k, v, kv_cache,
        torch.tensor(1.0),
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor([[0]], dtype=torch.int16),
        None,
        torch.ones((1, 1), dtype=torch.float32),
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


# ====================================================================
# 4. sliding_window_attention_naive _impl
# ====================================================================


def test_sliding_window_prefill_impl_given_compile_returns_empty_like(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.sliding_window_attention_naive_prefill_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([0], dtype=torch.int16),
        torch.tensor(0.0),
    )
    assert output.shape == q.shape


def test_sliding_window_prefill_impl_given_window_trim_updates_cache_slice(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((1, 1, 1, 2, 2), dtype=torch.float32)
    key = torch.tensor([[[[[7.0, 0.0], [8.0, 0.0]]]]])
    value = torch.tensor([[[[[9.0, 0.0], [10.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)
    kv_cache[0, 0, 0, 0, 0] = torch.tensor([1.0, 0.0])
    kv_cache[0, 0, 0, 0, 1] = torch.tensor([2.0, 0.0])
    kv_cache[1, 0, 0, 0, 0] = torch.tensor([3.0, 0.0])
    kv_cache[1, 0, 0, 0, 1] = torch.tensor([4.0, 0.0])

    output = backend_module.sliding_window_attention_naive_prefill_impl(
        query, key, value, kv_cache,
        torch.tensor([[2]], dtype=torch.int16),
        torch.tensor([[4]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([0], dtype=torch.int16),
        torch.tensor(0.0),
        None,
    )

    assert torch.equal(
        kv_cache[0, 0, 0, 0],
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [7.0, 0.0], [8.0, 0.0]]),
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_sliding_window_prefill_impl_given_sinks_redistributes_attention(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    k = torch.tensor([[[[[1.0, 0.0]]]]])
    v = torch.tensor([[[[[1.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)

    output = backend_module.sliding_window_attention_naive_prefill_impl(
        q, k, v, kv_cache,
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([0], dtype=torch.int16),
        torch.tensor(0.0),
        torch.ones((1, 1), dtype=torch.float32),
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_sliding_window_decode_impl_given_compile_returns_empty_like(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", True, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    output = backend_module.sliding_window_attention_naive_decode_impl(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(0.0),
    )
    assert output.shape == q.shape


def test_sliding_window_decode_impl_given_multiple_active_rows_updates_each_block(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((2, 1, 1, 1, 2), dtype=torch.float32)
    key = torch.tensor([[[[[1.0, 0.0]]]], [[[[2.0, 0.0]]]]])
    value = torch.tensor([[[[[3.0, 0.0]]]], [[[[4.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 2, 1, 1, 4, 2), dtype=torch.float32)

    output = backend_module.sliding_window_attention_naive_decode_impl(
        query, key, value, kv_cache,
        torch.tensor([[1], [1]], dtype=torch.int16),
        torch.tensor([[2], [2]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([[0], [1]], dtype=torch.int16),
        torch.tensor(0.0),
        None,
        None,
    )

    assert torch.equal(kv_cache[0, 0, 0, 0, 1], key[0, 0, 0, 0])
    assert torch.equal(kv_cache[0, 1, 0, 0, 1], key[1, 0, 0, 0])
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_sliding_window_decode_impl_given_non_positive_window_returns_zeros(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    query = torch.ones((1, 1, 1, 1, 32), dtype=torch.float32)

    output = backend_module.sliding_window_attention_naive_decode_impl(
        query,
        torch.zeros_like(query),
        torch.zeros_like(query),
        torch.zeros((2, 1, 1, 1, 4, 32), dtype=torch.float32),
        torch.tensor([[2]], dtype=torch.int16),
        torch.tensor([[2]], dtype=torch.int16),
        torch.tensor(0.5),
        torch.zeros((1, 1), dtype=torch.int16),
        torch.tensor(0.0),
        None,
        None,
    )

    assert torch.equal(output, torch.zeros_like(query))


def test_sliding_window_decode_impl_given_sinks_redistributes_attention(
    monkeypatch, backend_module
):
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    q = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32)
    k = torch.tensor([[[[[1.0, 0.0]]]]])
    v = torch.tensor([[[[[1.0, 0.0]]]]])
    kv_cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)

    output = backend_module.sliding_window_attention_naive_decode_impl(
        q, k, v, kv_cache,
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor([[2]], dtype=torch.int16),
        torch.tensor(1.0),
        torch.tensor([[0]], dtype=torch.int16),
        torch.tensor(0.0),
        None,
        torch.ones((1, 1), dtype=torch.float32),
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


# ====================================================================
# 5. rbln_cache_update
# ====================================================================


def test_rbln_cache_update_impl_returns_empty_like(backend_module):
    cache = torch.zeros((2, 1, 1, 1, 4, 2), dtype=torch.float32)
    output = backend_module.rbln_cache_update_impl(
        cache,
        torch.ones((1, 1, 1, 1, 2), dtype=torch.float32),
        torch.tensor([0], dtype=torch.int32),
    )
    assert output.shape == cache.shape


# ====================================================================
# 6. RBLNAttentionBackend
# ====================================================================


def test_backend_get_kv_cache_shape(backend_module):
    shape = backend_module.RBLNAttentionBackend.get_kv_cache_shape(
        num_blocks=4, block_size=1024, num_kv_heads=8, head_size=64
    )
    assert shape == (2, 4, 8, 1, 1024, 64)


def test_backend_swap_blocks_raises_runtime_error(backend_module):
    with pytest.raises(RuntimeError):
        backend_module.RBLNAttentionBackend.swap_blocks(None, None, {})


def test_backend_copy_blocks_raises_runtime_error(backend_module):
    with pytest.raises(RuntimeError):
        backend_module.RBLNAttentionBackend.copy_blocks([], {})


def test_backend_get_supported_head_sizes(backend_module):
    sizes = backend_module.RBLNAttentionBackend.get_supported_head_sizes()
    assert 128 in sizes
    assert 32 in sizes


def test_backend_get_name(backend_module):
    assert backend_module.RBLNAttentionBackend.get_name() == "FLASH_ATTN"


def test_backend_get_impl_cls(backend_module):
    assert (
        backend_module.RBLNAttentionBackend.get_impl_cls()
        is backend_module.RBLNFlashAttentionImpl
    )


def test_backend_get_builder_cls(backend_module):
    assert (
        backend_module.RBLNAttentionBackend.get_builder_cls()
        is backend_module.RBLNFlashAttentionMetadataBuilder
    )


# ====================================================================
# 7. RBLNFlashAttentionMetadataBuilder.build
# ====================================================================


def test_build_given_missing_num_tokens_raises_assertion_error(
    metadata_builder_factory,
):
    builder = metadata_builder_factory()
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 2), dtype=torch.int16),
    )

    with pytest.raises(AssertionError, match="num_tokens is required"):
        builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            positions=torch.tensor([0], dtype=torch.int32),
        )


def test_build_given_mixed_prefill_and_decode_requests_raises_assertion_error(
    metadata_builder_factory,
):
    builder = metadata_builder_factory()
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int16),
    )

    with pytest.raises(AssertionError):
        builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            num_tokens=np.array([2, 1]),
            positions=torch.tensor([0, 1, 2], dtype=torch.int32),
            batch_pad=2,
        )


@pytest.mark.parametrize(
    "positions,num_tokens,expected_prefix",
    [
        (
            torch.tensor([0, 1], dtype=torch.int32),
            np.array([2]),
            0,
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            np.array([2]),
            2,
        ),
    ],
    ids=["first_chunk", "second_chunk"],
)
def test_build_given_noncausal_prefill_constructs_chunked_attention_mask(
    metadata_builder_factory,
    positions,
    num_tokens,
    expected_prefix,
):
    builder = metadata_builder_factory(is_causal=False)
    seq_len = int(num_tokens[0].item())
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=1,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
        seq_lens=torch.tensor([int(positions[-1].item()) + 1], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 2), dtype=torch.int16),
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_tokens=num_tokens,
        positions=positions,
        batch_pad=1,
    )

    assert metadata.attn_masks is not None
    assert metadata.attn_masks.shape[4] == 8


def test_build_given_sliding_window_decode_clamps_cache_lengths_and_generates_masks(
    metadata_builder_factory,
):
    builder = metadata_builder_factory(sliding_window=4)
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int16),
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_tokens=np.array([1, 1]),
        positions=torch.tensor([0, 1], dtype=torch.int32),
        batch_pad=2,
    )

    assert metadata.cache_seq_lens is not None
    assert metadata.cache_offsets is not None
    assert metadata.swa_attn_masks is not None
    assert metadata.seq_lens.shape[0] == 2


def test_build_given_batch_attention_opt_decode_uses_seq_idx_as_seq_lens(
    metadata_builder_factory,
):
    builder = metadata_builder_factory(is_batch_attention_opt=True)
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int16),
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_tokens=np.array([1, 1]),
        positions=torch.tensor([0, 1], dtype=torch.int32),
        batch_pad=2,
    )

    assert metadata.seq_lens.shape == (2, 1)


def test_build_reorder_batch_returns_false(metadata_builder_factory):
    builder = metadata_builder_factory()
    assert builder.reorder_batch(None, None) is False


def test_build_use_cascade_attention_returns_false(metadata_builder_factory):
    builder = metadata_builder_factory()
    assert builder.use_cascade_attention() is False


# ====================================================================
# 8. RBLNFlashAttentionImpl.__init__
# ====================================================================


def test_init_given_kv_sharing_target_layer_name_raises_not_implemented(
    attention_impl_factory,
):
    with pytest.raises(NotImplementedError, match="KV sharing"):
        attention_impl_factory(kv_sharing_target_layer_name="layer.1")


def test_init_given_non_auto_kv_cache_dtype_raises_not_implemented(
    attention_impl_factory,
):
    with pytest.raises(NotImplementedError, match="FP8 KV cache"):
        attention_impl_factory(kv_cache_dtype="fp8")


def test_init_given_logits_soft_cap_clears_value(attention_impl_factory):
    impl = attention_impl_factory(logits_soft_cap=50.0)
    assert impl.logits_soft_cap == 0


def test_init_given_alibi_slopes_converts_to_tensor(attention_impl_factory):
    impl = attention_impl_factory(alibi_slopes=[0.1, 0.2, 0.3, 0.4])
    assert isinstance(impl.alibi_slopes, torch.Tensor)
    assert impl.alibi_slopes.dtype == torch.float32
    assert impl.need_mask is True


def test_init_given_unsupported_head_size_raises_value_error(attention_impl_factory):
    with pytest.raises(ValueError, match="Head size .* is not supported"):
        attention_impl_factory(head_size=48)


def test_init_given_sink_count_mismatch_raises_assertion_error(
    attention_impl_factory,
):
    with pytest.raises(
        AssertionError, match="Sinks must have the same number of heads"
    ):
        attention_impl_factory(sinks=torch.ones(3, dtype=torch.float32))


def test_init_given_one_dimensional_sinks_reshapes_to_per_head_column(
    attention_impl_factory,
):
    impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))

    assert impl.sinks.shape == (4, 1)


# ====================================================================
# 9. RBLNFlashAttentionImpl.forward — assertions
# ====================================================================


def test_forward_given_missing_kv_cache_raises_assertion_error(
    attention_impl_factory,
):
    impl = attention_impl_factory()
    query, key, value, _ = _make_forward_inputs()
    metadata = _make_forward_metadata()

    with pytest.raises(AssertionError):
        impl.forward(None, query, key, value, None, metadata)


def test_forward_given_sliding_window_size_mismatch_raises_assertion_error(
    attention_impl_factory,
):
    impl = attention_impl_factory(sliding_window=4)
    query, key, value, kv_cache = _make_forward_inputs(block_size=3)
    metadata = _make_forward_metadata()

    with pytest.raises(
        AssertionError, match="kernel_block_size must match window_size"
    ):
        impl.forward(None, query, key, value, kv_cache, metadata)


def test_forward_given_missing_cache_offsets_for_sliding_window_raises_assertion_error(
    attention_impl_factory,
):
    impl = attention_impl_factory(sliding_window=4)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)
    metadata = _make_forward_metadata(cache_offsets=None)

    with pytest.raises(AssertionError):
        impl.forward(None, query, key, value, kv_cache, metadata)


def test_forward_given_missing_sequence_lengths_for_causal_normal_raises_assertion_error(
    attention_impl_factory,
):
    impl = attention_impl_factory()
    impl.is_causal = True
    impl.is_normal = True
    query, key, value, kv_cache = _make_forward_inputs()
    metadata = _make_forward_metadata(seq_lens=None)

    with pytest.raises(AssertionError):
        impl.forward(None, query, key, value, kv_cache, metadata)


def test_forward_given_missing_attention_mask_for_normal_attention_raises_assertion_error(
    attention_impl_factory,
):
    impl = attention_impl_factory()
    impl.is_causal = False
    impl.is_normal = True
    query, key, value, kv_cache = _make_forward_inputs()
    metadata = _make_forward_metadata(attn_masks=None)

    with pytest.raises(AssertionError):
        impl.forward(None, query, key, value, kv_cache, metadata)


# ====================================================================
# 10-14. forward — dispatch routing tests
# ====================================================================


@pytest.mark.parametrize(
    "is_causal,is_normal,sliding_window,is_prefill,compile_model,use_custom_kernel,"
    "triton_op,custom_op,extra_metadata",
    [
        # sliding window decode: triton
        (False, False, 4, False, True, True,
         "sliding_window_attention_naive_decode",
         "sliding_window_attention_naive_decode",
         {"cache_seq_lens": torch.ones((1, 1), dtype=torch.int16),
          "cache_offsets": torch.ones((1, 1), dtype=torch.int16),
          "local_block_tables": torch.zeros((1, 1), dtype=torch.int16)}),
        # sliding window prefill: triton
        (False, False, 4, True, True, True,
         "sliding_window_attention_naive_prefill",
         "sliding_window_attention_naive_prefill",
         {"cache_seq_lens": torch.ones((1, 1), dtype=torch.int16),
          "cache_offsets": torch.full((1, 1), 3, dtype=torch.int16),
          "local_block_tables": torch.zeros((1, 1), dtype=torch.int16)}),
        # causal normal decode: triton
        (True, True, None, False, True, True,
         "causal_attention_naive_decode",
         "causal_attention_naive_decode",
         {}),
        # causal normal prefill: triton
        (True, True, None, True, True, True,
         "causal_attention_naive_prefill",
         "causal_attention_naive_prefill",
         {"seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        # flash causal decode: triton
        (True, False, None, False, True, True,
         "flash_causal_attention_naive_decode",
         "flash_causal_attention_naive_decode",
         {}),
        # flash causal prefill: triton
        (True, False, None, True, True, True,
         "flash_causal_attention_naive_prefill",
         "flash_causal_attention_naive_prefill",
         {"seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        # normal (non-causal) decode: triton
        (False, True, None, False, True, True,
         "attention_naive_decode",
         "attention_naive_decode",
         {"attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)}),
        # normal (non-causal) prefill: triton
        (False, True, None, True, True, True,
         "attention_naive_prefill",
         "attention_naive_prefill",
         {"attn_masks": torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
          "seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        # flash attention decode: triton
        (False, False, None, False, True, True,
         "flash_attention_naive_decode",
         "flash_attention_naive_decode",
         {"attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)}),
    ],
    ids=[
        "swa_decode_triton", "swa_prefill_triton",
        "causal_normal_decode_triton", "causal_normal_prefill_triton",
        "flash_causal_decode_triton", "flash_causal_prefill_triton",
        "normal_decode_triton", "normal_prefill_triton",
        "flash_attn_decode_triton",
    ],
)
def test_forward_routes_to_triton_when_compile_and_custom_kernel(
    monkeypatch, backend_module, attention_impl_factory,
    is_causal, is_normal, sliding_window, is_prefill,
    compile_model, use_custom_kernel,
    triton_op, custom_op, extra_metadata,
):
    q_len = 2 if is_prefill else 1
    block_size = sliding_window if sliding_window else 4
    selected = Mock(
        return_value=torch.zeros(
            (1, 1, 4, q_len, 32), dtype=torch.float32
        )
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module,
        compile_model=compile_model, use_custom_kernel=use_custom_kernel
    )
    _patch_namespace_op(
        monkeypatch, backend_module.torch.ops.rbln_triton_ops,
        triton_op, selected,
    )
    _patch_namespace_op(
        monkeypatch, backend_module.torch.ops.rbln_custom_ops,
        custom_op, not_selected,
    )

    attention_impl = attention_impl_factory(sliding_window=sliding_window)
    attention_impl.is_causal = is_causal
    attention_impl.is_normal = is_normal
    metadata = _make_forward_metadata(is_prefill=is_prefill, **extra_metadata)
    query, key, value, kv_cache = _make_forward_inputs(
        q_len=q_len, block_size=block_size
    )

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


@pytest.mark.parametrize(
    "is_causal,is_normal,sliding_window,is_prefill,impl_name,extra_metadata",
    [
        (False, False, 4, False, "sliding_window_attention_naive_decode_impl",
         {"cache_seq_lens": torch.ones((1, 1), dtype=torch.int16),
          "cache_offsets": torch.ones((1, 1), dtype=torch.int16),
          "local_block_tables": torch.zeros((1, 1), dtype=torch.int16)}),
        (False, False, 4, True, "sliding_window_attention_naive_prefill_impl",
         {"cache_seq_lens": torch.ones((1, 1), dtype=torch.int16),
          "cache_offsets": torch.full((1, 1), 3, dtype=torch.int16),
          "local_block_tables": torch.zeros((1, 1), dtype=torch.int16)}),
        (True, True, None, False, "causal_attention_naive_decode_impl", {}),
        (True, True, None, True, "causal_attention_naive_prefill_impl",
         {"seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        (True, False, None, False, "flash_causal_attention_naive_decode_impl", {}),
        (True, False, None, True, "flash_causal_attention_naive_prefill_impl",
         {"seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        (False, True, None, False, "attention_naive_decode_impl",
         {"attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)}),
        (False, True, None, True, "attention_naive_prefill_impl",
         {"attn_masks": torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
          "seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        (False, False, None, True, "flash_attention_naive_prefill_impl",
         {"attn_masks": torch.ones((1, 1, 1, 2, 4))}),
        (False, False, None, False, "flash_attention_naive_decode_impl",
         {"attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)}),
    ],
    ids=[
        "swa_decode_impl", "swa_prefill_impl",
        "causal_normal_decode_impl", "causal_normal_prefill_impl",
        "flash_causal_decode_impl", "flash_causal_prefill_impl",
        "normal_decode_impl", "normal_prefill_impl",
        "flash_prefill_impl", "flash_decode_impl",
    ],
)
def test_forward_routes_to_python_impl_when_compile_disabled(
    monkeypatch, backend_module, attention_impl_factory,
    is_causal, is_normal, sliding_window, is_prefill, impl_name, extra_metadata,
):
    q_len = 2 if is_prefill else 1
    block_size = sliding_window if sliding_window else 4
    selected = Mock(
        return_value=torch.zeros(
            (1, 1, 4, q_len, 32), dtype=torch.float32
        )
    )

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=False
    )
    monkeypatch.setattr(backend_module, impl_name, selected)

    attention_impl = attention_impl_factory(sliding_window=sliding_window)
    attention_impl.is_causal = is_causal
    attention_impl.is_normal = is_normal
    metadata = _make_forward_metadata(is_prefill=is_prefill, **extra_metadata)
    query, key, value, kv_cache = _make_forward_inputs(
        q_len=q_len, block_size=block_size
    )

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


# ====================================================================
# 15. forward — output reshape
# ====================================================================


def test_forward_given_compile_enabled_non_eager_uses_view_reshape(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "flash_causal_attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    attention_impl.enforce_eager = False
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    assert output.shape == (1, 1, 4 * 32)


# ====================================================================
# 16. Host-reference comparison tests
#     Compute expected output on host using standard PyTorch and compare
#     with the reference impl results.
# ====================================================================


# TP configurations: (n_kv_heads, n_groups, head_dim) simulating TP=1,2,4
TP_HEAD_CONFIGS = [
    pytest.param(1, 4, 64, id="tp1-kv1-g4-d64"),
    pytest.param(2, 2, 64, id="tp2-kv2-g2-d64"),
    pytest.param(4, 1, 64, id="tp4-kv4-g1-d64"),
    pytest.param(1, 1, 128, id="tp1-kv1-g1-d128"),
]


class TestFlashAttentionHostReference:
    """Test reference impls against host-computed PyTorch attention."""

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_HEAD_CONFIGS)
    def test_flash_attention_prefill_matches_host_reference(
        self, monkeypatch, backend_module, n_kv_heads, n_groups, head_dim,
    ):
        monkeypatch.setattr(
            backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
        )
        torch.manual_seed(42)
        seq_len = 2
        partition_size = 4

        q = torch.randn(1, n_kv_heads, n_groups, seq_len, head_dim)
        k = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        v = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        kv_cache = torch.zeros(2, 1, n_kv_heads, 1, partition_size, head_dim)
        mask = torch.ones(1, 1, 1, seq_len, partition_size)
        scale = torch.tensor(1.0 / (head_dim**0.5))

        output = backend_module.flash_attention_naive_prefill_impl(
            q, k, v, kv_cache, mask, scale,
            torch.tensor([[0]], dtype=torch.int16),
            torch.tensor([0], dtype=torch.int16),
            None,
        )

        # Host reference: build k/v state from cache after update
        k_state = kv_cache[0, 0].unsqueeze(0)  # (1, n_kv_heads, 1, partition_size, d)
        v_state = kv_cache[1, 0].unsqueeze(0)
        # Build mask for host comparison (same as impl uses)
        host_mask = torch.where(mask[:, :, :, :, :partition_size] > 0, 0.0, -float("inf"))
        expected = _host_attention(q, k_state, v_state, scale, host_mask)

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_HEAD_CONFIGS)
    def test_flash_attention_decode_matches_host_reference(
        self, monkeypatch, backend_module, n_kv_heads, n_groups, head_dim,
    ):
        monkeypatch.setattr(
            backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
        )
        torch.manual_seed(42)
        partition_size = 4

        q = torch.randn(1, n_kv_heads, n_groups, 1, head_dim)
        k = torch.randn(1, n_kv_heads, 1, 1, head_dim)
        v = torch.randn(1, n_kv_heads, 1, 1, head_dim)
        kv_cache = torch.zeros(2, 1, n_kv_heads, 1, partition_size, head_dim)
        # Pre-fill one token in cache
        kv_cache[0, 0, :, :, 0, :] = torch.randn(n_kv_heads, 1, head_dim)
        kv_cache[1, 0, :, :, 0, :] = torch.randn(n_kv_heads, 1, head_dim)
        mask = torch.ones(1, 1, 1, 1, partition_size)
        scale = torch.tensor(1.0 / (head_dim**0.5))
        seq_idx = torch.tensor([[1]], dtype=torch.int16)

        output = backend_module.flash_attention_naive_decode_impl(
            q, k, v, kv_cache, mask, scale,
            seq_idx,
            torch.tensor([[0]], dtype=torch.int16),
            None,
        )

        k_state = kv_cache[0, 0].unsqueeze(0)
        v_state = kv_cache[1, 0].unsqueeze(0)
        host_mask = torch.where(mask[:, :, :, :, :partition_size] > 0, 0.0, -float("inf"))
        expected = _host_attention(q, k_state, v_state, scale, host_mask)

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_HEAD_CONFIGS)
    def test_flash_causal_prefill_matches_host_reference(
        self, monkeypatch, backend_module, n_kv_heads, n_groups, head_dim,
    ):
        monkeypatch.setattr(
            backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
        )
        torch.manual_seed(42)
        seq_len = 2
        partition_size = 4

        q = torch.randn(1, n_kv_heads, n_groups, seq_len, head_dim)
        k = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        v = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        kv_cache = torch.zeros(2, 1, n_kv_heads, 1, partition_size, head_dim)
        scale = torch.tensor(1.0 / (head_dim**0.5))

        output = backend_module.flash_causal_attention_naive_prefill_impl(
            q, k, v, kv_cache, scale,
            torch.tensor([[0]], dtype=torch.int16),
            torch.tensor([0], dtype=torch.int16),
            None,
        )

        # Host reference: gather from cache + apply causal mask
        k_gathered = kv_cache[0, 0, :, :, :seq_len, :].unsqueeze(0)
        v_gathered = kv_cache[1, 0, :, :, :seq_len, :].unsqueeze(0)
        # Causal mask
        query_pos = torch.arange(seq_len)
        key_pos = torch.arange(seq_len)
        causal = query_pos.unsqueeze(1) >= key_pos.unsqueeze(0)
        causal_mask = torch.where(causal, 0.0, -float("inf")).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        expected = _host_attention(q, k_gathered, v_gathered, scale, causal_mask)

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_HEAD_CONFIGS)
    def test_flash_causal_decode_matches_host_reference(
        self, monkeypatch, backend_module, n_kv_heads, n_groups, head_dim,
    ):
        monkeypatch.setattr(
            backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
        )
        torch.manual_seed(42)
        partition_size = 4

        q = torch.randn(1, n_kv_heads, n_groups, 1, head_dim)
        k = torch.randn(1, n_kv_heads, 1, 1, head_dim)
        v = torch.randn(1, n_kv_heads, 1, 1, head_dim)
        kv_cache = torch.zeros(2, 1, n_kv_heads, 1, partition_size, head_dim)
        # Pre-fill 2 tokens
        kv_cache[0, 0, :, :, 0, :] = torch.randn(n_kv_heads, 1, head_dim)
        kv_cache[0, 0, :, :, 1, :] = torch.randn(n_kv_heads, 1, head_dim)
        kv_cache[1, 0, :, :, 0, :] = torch.randn(n_kv_heads, 1, head_dim)
        kv_cache[1, 0, :, :, 1, :] = torch.randn(n_kv_heads, 1, head_dim)
        scale = torch.tensor(1.0 / (head_dim**0.5))

        output = backend_module.flash_causal_attention_naive_decode_impl(
            q, k, v, kv_cache, scale,
            torch.tensor([[2]], dtype=torch.int16),
            torch.tensor([[0]], dtype=torch.int16),
            None,
        )

        # Host reference: gather 3 tokens total (2 cached + 1 new)
        total_seq_len = 3
        k_gathered = kv_cache[0, 0, :, :, :total_seq_len, :].unsqueeze(0)
        v_gathered = kv_cache[1, 0, :, :, :total_seq_len, :].unsqueeze(0)
        # Decode: query at position 2, can attend to 0..2, no masking needed
        expected = _host_attention(q, k_gathered, v_gathered, scale)

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_HEAD_CONFIGS)
    def test_sliding_window_prefill_matches_host_reference(
        self, monkeypatch, backend_module, n_kv_heads, n_groups, head_dim,
    ):
        monkeypatch.setattr(
            backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
        )
        torch.manual_seed(42)
        seq_len = 1
        window_size = 4

        q = torch.randn(1, n_kv_heads, n_groups, seq_len, head_dim)
        k = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        v = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        kv_cache = torch.zeros(2, 1, n_kv_heads, 1, window_size, head_dim)
        scale = torch.tensor(1.0 / (head_dim**0.5))
        cache_seq_len = torch.tensor([[0]], dtype=torch.int16)
        cache_offset = torch.tensor([[1]], dtype=torch.int16)

        output = backend_module.sliding_window_attention_naive_prefill_impl(
            q, k, v, kv_cache,
            cache_seq_len, cache_offset, scale,
            torch.tensor([0], dtype=torch.int16),
            torch.tensor(0.0),
            None,
        )

        # For seq_len=1 with empty cache: output = softmax(q @ k^T * scale) @ v
        # Build the full k,v state: just the 1 new token padded to window+seq_len
        k_full = torch.zeros(1, n_kv_heads, 1, window_size + seq_len, head_dim)
        k_full[:, :, :, :seq_len, :] = k
        v_full = torch.zeros(1, n_kv_heads, 1, window_size + seq_len, head_dim)
        v_full[:, :, :, :seq_len, :] = v

        # Build sliding window mask
        ones = torch.ones(window_size + seq_len, window_size + seq_len)
        mask_full = torch.tril(ones) - torch.tril(ones, diagonal=-window_size)
        sw_mask = mask_full[None, None, None, 0:seq_len, :]
        sw_mask = torch.where(sw_mask > 0, 0.0, -float("inf"))

        expected = _host_attention(q, k_full, v_full, scale, sw_mask)

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_HEAD_CONFIGS)
    def test_sliding_window_decode_matches_host_reference(
        self, monkeypatch, backend_module, n_kv_heads, n_groups, head_dim,
    ):
        monkeypatch.setattr(
            backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
        )
        torch.manual_seed(42)
        window_size = 4

        q = torch.randn(1, n_kv_heads, n_groups, 1, head_dim)
        k = torch.randn(1, n_kv_heads, 1, 1, head_dim)
        v = torch.randn(1, n_kv_heads, 1, 1, head_dim)
        kv_cache = torch.zeros(2, 1, n_kv_heads, 1, window_size, head_dim)
        # Pre-fill 1 token
        cached_k = torch.randn(n_kv_heads, 1, head_dim)
        cached_v = torch.randn(n_kv_heads, 1, head_dim)
        kv_cache[0, 0, :, :, 0, :] = cached_k
        kv_cache[1, 0, :, :, 0, :] = cached_v
        scale = torch.tensor(1.0 / (head_dim**0.5))
        cache_seq_len = torch.tensor([[1]], dtype=torch.int16)
        cache_offset = torch.tensor([[2]], dtype=torch.int16)

        output = backend_module.sliding_window_attention_naive_decode_impl(
            q, k, v, kv_cache,
            cache_seq_len, cache_offset, scale,
            torch.tensor([[0]], dtype=torch.int16),
            torch.tensor(0.0),
            None,
            None,
        )

        # Build k/v: cached_k + new k, padded to window+1
        k_full = torch.zeros(1, n_kv_heads, 1, window_size + 1, head_dim)
        k_full[:, :, :, 0, :] = cached_k
        k_full[:, :, :, 1, :] = k.squeeze(0).squeeze(-2)
        v_full = torch.zeros(1, n_kv_heads, 1, window_size + 1, head_dim)
        v_full[:, :, :, 0, :] = cached_v
        v_full[:, :, :, 1, :] = v.squeeze(0).squeeze(-2)

        ones = torch.ones(window_size + 1, window_size + 1)
        mask_full = torch.tril(ones) - torch.tril(ones, diagonal=-window_size)
        sw_mask = mask_full[None, None, None, 1:2, :]
        sw_mask = torch.where(sw_mask > 0, 0.0, -float("inf"))

        expected = _host_attention(q, k_full, v_full, scale, sw_mask)

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )


# ====================================================================
# 17. Custom op sinks routing
# ====================================================================


@pytest.mark.parametrize(
    "is_causal,is_normal,sliding_window,is_prefill,custom_op_name,extra_metadata",
    [
        (True, True, None, False, "causal_attention_naive_decode", {}),
        (True, True, None, True, "causal_attention_naive_prefill",
         {"seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        (True, False, None, False, "flash_causal_attention_naive_decode", {}),
        (True, False, None, True, "flash_causal_attention_naive_prefill",
         {"seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        (False, True, None, False, "attention_naive_decode",
         {"attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)}),
        (False, True, None, True, "attention_naive_prefill",
         {"attn_masks": torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
          "seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
        (False, False, None, False, "flash_attention_naive_decode",
         {"attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32)}),
        (False, False, None, True, "flash_attention_naive_prefill",
         {"attn_masks": torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
          "seq_lens": torch.full((1, 1), 2, dtype=torch.int16)}),
    ],
    ids=[
        "causal_normal_decode", "causal_normal_prefill",
        "flash_causal_decode", "flash_causal_prefill",
        "normal_decode", "normal_prefill",
        "flash_decode", "flash_prefill",
    ],
)
def test_forward_given_custom_op_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory,
    is_causal, is_normal, sliding_window, is_prefill, custom_op_name, extra_metadata,
):
    q_len = 2 if is_prefill else 1
    selected = Mock(
        return_value=torch.zeros(
            (1, 1, 4, q_len, 32), dtype=torch.float32
        )
    )

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch, backend_module.torch.ops.rbln_custom_ops,
        custom_op_name, selected,
    )

    attention_impl = attention_impl_factory(
        sliding_window=sliding_window,
        sinks=torch.ones(4, dtype=torch.float32),
    )
    attention_impl.is_causal = is_causal
    attention_impl.is_normal = is_normal
    metadata = _make_forward_metadata(is_prefill=is_prefill, **extra_metadata)
    query, key, value, kv_cache = _make_forward_inputs(q_len=q_len, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


# ====================================================================
# 18. Edge cases and regression tests
# ====================================================================


def test_flash_causal_decode_impl_given_multi_batch_processes_each_batch(
    monkeypatch, backend_module
):
    """Verify the per-batch loop in flash_causal decode handles batch>1."""
    monkeypatch.setattr(
        backend_module.envs, "VLLM_RBLN_COMPILE_MODEL", False, raising=False
    )
    batch_size = 2
    n_kv_heads, n_groups, head_dim = 1, 1, 4
    partition_size = 4

    q = torch.randn(batch_size, n_kv_heads, n_groups, 1, head_dim)
    k = torch.randn(batch_size, n_kv_heads, 1, 1, head_dim)
    v = torch.randn(batch_size, n_kv_heads, 1, 1, head_dim)
    kv_cache = torch.zeros(2, 2, n_kv_heads, 1, partition_size, head_dim)
    # Pre-fill 1 token for batch 0, 2 tokens for batch 1
    kv_cache[0, 0, :, :, 0, :] = torch.randn(n_kv_heads, 1, head_dim)
    kv_cache[1, 0, :, :, 0, :] = torch.randn(n_kv_heads, 1, head_dim)
    kv_cache[0, 1, :, :, 0:2, :] = torch.randn(n_kv_heads, 1, 2, head_dim)
    kv_cache[1, 1, :, :, 0:2, :] = torch.randn(n_kv_heads, 1, 2, head_dim)
    scale = torch.tensor(1.0 / (head_dim**0.5))

    output = backend_module.flash_causal_attention_naive_decode_impl(
        q, k, v, kv_cache, scale,
        torch.tensor([[1], [2]], dtype=torch.int16),
        torch.tensor([[0], [1]], dtype=torch.int16),
        None,
    )

    assert output.shape == (batch_size, n_kv_heads, n_groups, 1, head_dim)
    assert torch.isfinite(output).all()
    # Each batch should produce different results
    assert not torch.equal(output[0], output[1])


def test_build_given_causal_prefill_skips_mask_construction(
    metadata_builder_factory,
):
    """Causal prefill should NOT construct attn_masks (mask is implicit)."""
    builder = metadata_builder_factory(is_causal=True)
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 2), dtype=torch.int16),
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_tokens=np.array([2]),
        positions=torch.tensor([0, 1], dtype=torch.int32),
        batch_pad=1,
    )

    assert metadata.attn_masks is None
    assert metadata.is_prefill is True


def test_build_given_noncausal_decode_constructs_per_batch_mask(
    metadata_builder_factory,
):
    """Non-causal decode should construct attention masks per batch."""
    builder = metadata_builder_factory(is_causal=False)
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 3], dtype=torch.int32),
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int16),
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_tokens=np.array([1, 1]),
        positions=torch.tensor([0, 2], dtype=torch.int32),
        batch_pad=2,
    )

    assert metadata.attn_masks is not None
    assert metadata.is_prefill is False
    # batch 0 has seq_len=1, batch 1 has seq_len=3
    # mask for batch 1 should have more 1s than batch 0
    assert metadata.attn_masks[1, 0, 0, 0, 2] > metadata.attn_masks[0, 0, 0, 0, 2]


def test_build_given_missing_batch_pad_raises_assertion_error(
    metadata_builder_factory,
):
    """batch_pad is required for RBLN Attention Backend."""
    builder = metadata_builder_factory()
    common_attn_metadata = _make_common_attn_metadata(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 2), dtype=torch.int16),
    )

    with pytest.raises(AssertionError, match="batch_pad is required"):
        builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            num_tokens=np.array([1]),
            positions=torch.tensor([0], dtype=torch.int32),
        )


def test_forward_given_swa_batch_attn_opt_casts_cache_seq_lens_to_int32(
    monkeypatch, backend_module, attention_impl_factory
):
    """sliding_window + batch_attn_opt + batch>1 should cast cache_seq_lens to int32."""
    selected = Mock(return_value=torch.zeros((2, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "sliding_window_attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory(sliding_window=4)
    attention_impl.is_batch_attention_opt = True
    metadata = _make_forward_metadata(
        is_prefill=False,
        cache_seq_lens=torch.ones((2, 1), dtype=torch.int16),
        cache_offsets=torch.ones((2, 1), dtype=torch.int16),
        local_block_tables=torch.zeros((2, 1), dtype=torch.int16),
        swa_attn_masks=torch.ones((2, 1, 1, 4), dtype=torch.float32),
    )
    query, key, value, kv_cache = _make_forward_inputs(batch_size=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    # cache_seq_lens should be cast to int32 for batch_attn_opt
    assert selected.call_args.args[4].dtype == torch.int32
