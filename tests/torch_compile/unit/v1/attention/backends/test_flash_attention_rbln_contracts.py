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
    metadata = {
        "is_prefill": False,
        "attn_masks": torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
        "seq_lens": torch.ones((1, 1), dtype=torch.int16),
        "block_tables": torch.zeros((1, 1), dtype=torch.int16),
        "cache_seq_lens": torch.ones((1, 1), dtype=torch.int16),
        "cache_offsets": torch.ones((1, 1), dtype=torch.int16),
        "local_block_tables": torch.zeros((1, 1), dtype=torch.int16),
        "swa_attn_masks": torch.ones((1, 1, 1, 4), dtype=torch.float32),
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
# 1. Stub custom ops: attention_naive / causal_attention_naive
#    (source lines ~44-173)
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
# 2. flash_attention_naive _impl  (source lines ~176-318)
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
# 3. flash_causal_attention_naive _impl  (source lines ~321-673)
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
# 4. sliding_window_attention_naive _impl  (source lines ~676-917)
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
# 5. rbln_cache_update  (source lines ~920-934)
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
# 6. RBLNAttentionBackend  (source lines ~937-989)
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


# ====================================================================
# 7. RBLNFlashAttentionMetadataBuilder.build  (source lines ~1073-1236)
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


# ====================================================================
# 8. RBLNFlashAttentionImpl.__init__  (source lines ~1239-1318)
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
# 9. RBLNFlashAttentionImpl.forward — assertions  (source lines ~1402-1408)
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
# 10. forward — sliding_window routing  (source lines ~1404-1472)
# ====================================================================


def test_forward_given_compiled_triton_sliding_window_decode_routes_to_triton(
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
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_triton_sliding_window_prefill_routes_to_triton(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "sliding_window_attention_naive_prefill",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "sliding_window_attention_naive_prefill",
        not_selected,
    )

    attention_impl = attention_impl_factory(sliding_window=4)
    metadata = _make_forward_metadata(
        is_prefill=True,
        cache_seq_lens=torch.ones((1, 1), dtype=torch.int16),
        cache_offsets=torch.full((1, 1), 3, dtype=torch.int16),
        local_block_tables=torch.zeros((1, 1), dtype=torch.int16),
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_custom_sliding_window_batch_decode_forwards_int32_masks_and_normalized_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((2, 1, 4, 1, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "sliding_window_attention_naive_decode",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "sliding_window_attention_naive_decode",
        not_selected,
    )

    attention_impl = attention_impl_factory(
        sliding_window=4,
        sinks=torch.ones(4, dtype=torch.float32),
    )
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
    not_selected.assert_not_called()
    assert torch.equal(
        selected.call_args.args[4], metadata.cache_seq_lens.to(torch.int32)
    )
    assert selected.call_args.args[9] is metadata.swa_attn_masks
    assert torch.equal(selected.call_args.args[10], attention_impl.sinks)


def test_forward_given_custom_sliding_window_nonbatch_decode_appends_none_mask_and_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "sliding_window_attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory(
        sliding_window=4, sinks=torch.ones(4, dtype=torch.float32)
    )
    attention_impl.is_batch_attention_opt = False
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    args = selected.call_args.args
    assert args[-2] is None
    assert torch.equal(args[-1], attention_impl.sinks)


def test_forward_given_compiled_custom_sliding_window_prefill_routes_with_sinks(
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
        "sliding_window_attention_naive_prefill",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "sliding_window_attention_naive_prefill",
        not_selected,
    )

    attention_impl = attention_impl_factory(
        sliding_window=4,
        sinks=torch.ones(4, dtype=torch.float32),
    )
    metadata = _make_forward_metadata(
        is_prefill=True,
        cache_seq_lens=torch.ones((1, 1), dtype=torch.int16),
        cache_offsets=torch.full((1, 1), 3, dtype=torch.int16),
        local_block_tables=torch.zeros((1, 1), dtype=torch.int16),
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    assert torch.equal(selected.call_args.args[4], metadata.cache_seq_lens)
    assert selected.call_args.args[-1] is attention_impl.sinks


def test_forward_given_compile_disabled_sliding_window_decode_routes_to_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=False
    )
    monkeypatch.setattr(
        backend_module, "sliding_window_attention_naive_decode_impl", selected
    )

    attention_impl = attention_impl_factory(sliding_window=4)
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


def test_forward_given_compile_disabled_sliding_window_prefill_routes_to_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=False
    )
    monkeypatch.setattr(
        backend_module, "sliding_window_attention_naive_prefill_impl", selected
    )

    attention_impl = attention_impl_factory(sliding_window=4)
    metadata = _make_forward_metadata(
        is_prefill=True,
        cache_seq_lens=torch.ones((1, 1), dtype=torch.int16),
        cache_offsets=torch.full((1, 1), 3, dtype=torch.int16),
        local_block_tables=torch.zeros((1, 1), dtype=torch.int16),
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


# ====================================================================
# 11. forward — causal + normal routing  (source lines ~1474-1526)
# ====================================================================


def test_forward_given_compiled_triton_causal_normal_decode_routes_to_triton(
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
        "causal_attention_naive_decode",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "causal_attention_naive_decode",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_triton_causal_normal_prefill_routes_to_triton(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "causal_attention_naive_prefill",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "causal_attention_naive_prefill",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(
        is_prefill=True, seq_lens=torch.full((1, 1), 2, dtype=torch.int16)
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_custom_causal_normal_decode_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "causal_attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compiled_custom_causal_normal_prefill_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "causal_attention_naive_prefill",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(
        is_prefill=True, seq_lens=torch.full((1, 1), 2, dtype=torch.int16)
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compile_model_disabled_causal_normal_decode_routes_to_python_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=True
    )
    monkeypatch.setattr(backend_module, "causal_attention_naive_decode_impl", selected)

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


# ====================================================================
# 12. forward — flash causal routing  (source lines ~1527-1586)
# ====================================================================


def test_forward_given_compiled_triton_flash_causal_decode_routes_to_triton_namespace(
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
        "flash_causal_attention_naive_decode",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_causal_attention_naive_decode",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_triton_flash_causal_prefill_routes_to_triton(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "flash_causal_attention_naive_prefill",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_causal_attention_naive_prefill",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=True, seq_lens=torch.full((1, 1), 2, dtype=torch.int16)
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_custom_flash_causal_decode_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_causal_attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compiled_custom_flash_causal_prefill_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_causal_attention_naive_prefill",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=True, seq_lens=torch.full((1, 1), 2, dtype=torch.int16)
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compile_disabled_flash_causal_decode_routes_to_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=False
    )
    monkeypatch.setattr(
        backend_module, "flash_causal_attention_naive_decode_impl", selected
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


def test_forward_given_compile_disabled_flash_causal_prefill_routes_to_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=False
    )
    monkeypatch.setattr(
        backend_module, "flash_causal_attention_naive_prefill_impl", selected
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=True, seq_lens=torch.full((1, 1), 2, dtype=torch.int16)
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


# ====================================================================
# 13. forward — normal (non-causal) routing  (source lines ~1587-1642)
# ====================================================================


def test_forward_given_compiled_triton_normal_decode_routes_to_triton(
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
        "attention_naive_decode",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "attention_naive_decode",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(
        is_prefill=False,
        attn_masks=torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
    )
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_triton_normal_prefill_routes_to_triton_namespace(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        "attention_naive_prefill",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "attention_naive_prefill",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(
        is_prefill=True,
        attn_masks=torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
        seq_lens=torch.full((1, 1), 2, dtype=torch.int16),
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_custom_normal_decode_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(
        is_prefill=False,
        attn_masks=torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
    )
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compiled_custom_normal_prefill_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "attention_naive_prefill",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(
        is_prefill=True,
        attn_masks=torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
        seq_lens=torch.full((1, 1), 2, dtype=torch.int16),
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compile_model_disabled_normal_decode_routes_to_python_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=True
    )
    monkeypatch.setattr(backend_module, "attention_naive_decode_impl", selected)

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_forward_metadata(is_prefill=False)
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


# ====================================================================
# 14. forward — flash attention (non-causal, non-normal) routing
#     (source lines ~1643-1696)
# ====================================================================


def test_forward_given_compile_model_disabled_flash_prefill_routes_to_python_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=True
    )
    monkeypatch.setattr(backend_module, "flash_attention_naive_prefill_impl", selected)

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=True, attn_masks=torch.ones((1, 1, 1, 2, 4))
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


def test_forward_given_compile_disabled_flash_attention_decode_routes_to_impl(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=False
    )
    monkeypatch.setattr(
        backend_module, "flash_attention_naive_decode_impl", selected
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=False,
        attn_masks=torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
    )
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()


def test_forward_given_compiled_triton_flash_attention_decode_routes_to_triton(
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
        "flash_attention_naive_decode",
        selected,
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_attention_naive_decode",
        not_selected,
    )

    attention_impl = attention_impl_factory()
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=False,
        attn_masks=torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
    )
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()


def test_forward_given_compiled_custom_flash_attention_decode_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 1, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_attention_naive_decode",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=False,
        attn_masks=torch.ones((1, 1, 1, 1, 4), dtype=torch.float32),
    )
    query, key, value, kv_cache = _make_forward_inputs(block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


def test_forward_given_compiled_custom_flash_attention_prefill_passes_sinks(
    monkeypatch, backend_module, attention_impl_factory
):
    selected = Mock(return_value=torch.zeros((1, 1, 4, 2, 32), dtype=torch.float32))

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_namespace_op(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        "flash_attention_naive_prefill",
        selected,
    )

    attention_impl = attention_impl_factory(sinks=torch.ones(4, dtype=torch.float32))
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_forward_metadata(
        is_prefill=True,
        attn_masks=torch.ones((1, 1, 1, 2, 4), dtype=torch.float32),
        seq_lens=torch.full((1, 1), 2, dtype=torch.int16),
    )
    query, key, value, kv_cache = _make_forward_inputs(q_len=2, block_size=4)

    attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    assert torch.equal(selected.call_args.args[-1], attention_impl.sinks)


# ====================================================================
# 15. forward — output reshape  (source lines ~1698-1715)
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
