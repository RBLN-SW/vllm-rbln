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
def attention_impl(vllm_config, backend_module):
    with set_current_vllm_config(vllm_config):
        impl = backend_module.RBLNFlashAttentionImpl(
            num_heads=4,
            head_size=32,
            scale=0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

    impl.enforce_eager = True
    return impl


def _make_inputs(q_len: int = 1, block_size: int = 4, batch_size: int = 1):
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


def _make_metadata(*, is_prefill: bool, q_len: int = 1, batch_size: int = 1):
    return SimpleNamespace(
        is_prefill=is_prefill,
        attn_masks=torch.ones((1, 1, 1, q_len, 4), dtype=torch.float32),
        seq_lens=torch.ones((batch_size, 1), dtype=torch.int16),
        block_tables=torch.zeros((batch_size, 1), dtype=torch.int16),
        cache_seq_lens=torch.ones((batch_size, 1), dtype=torch.int16),
        cache_offsets=torch.full((batch_size, 1), q_len, dtype=torch.int16),
        local_block_tables=torch.zeros((batch_size, 1), dtype=torch.int16),
        swa_attn_masks=torch.ones((1, 1, 1, 4), dtype=torch.float32),
    )


def _make_op_stub(name: str, *, q_len: int, expected_shape: tuple[int, ...]):
    def _stub(*args):
        assert args[0].shape == expected_shape, (
            f"Expected reshaped query for {name}, got {args[0].shape}"
        )
        assert args[1].shape == expected_shape[:2] + (1, q_len, 32)
        assert args[2].shape == expected_shape[:2] + (1, q_len, 32)
        return torch.full(expected_shape, 7.0, dtype=torch.float32)

    return Mock(side_effect=_stub)


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


def _patch_op_pair(
    monkeypatch, namespace, *, selected_name: str, selected: Mock, sibling_name: str
):
    monkeypatch.setattr(namespace, selected_name, selected, raising=False)
    monkeypatch.setattr(namespace, sibling_name, Mock(), raising=False)


def _assert_common_dispatch_args(call_args, *, kv_cache, arg_count: int):
    args = call_args.args
    assert len(args) == arg_count
    assert args[3] is kv_cache


def test_forward_uses_triton_op_for_sliding_window_decode(
    monkeypatch, backend_module, attention_impl
):
    q_len = 1
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "sliding_window_attention_naive_decode",
        q_len=q_len,
        expected_shape=expected_shape,
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        selected_name="sliding_window_attention_naive_decode",
        selected=selected,
        sibling_name="sliding_window_attention_naive_prefill",
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        selected_name="sliding_window_attention_naive_decode",
        selected=not_selected,
        sibling_name="sliding_window_attention_naive_prefill",
    )

    attention_impl.sliding_window = 4
    metadata = _make_metadata(is_prefill=False, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=9)
    assert selected.call_args.args[4] is metadata.cache_seq_lens
    assert selected.call_args.args[5] is metadata.cache_offsets
    assert selected.call_args.args[7] is metadata.local_block_tables
    assert selected.call_args.args[8] is attention_impl.scale
    assert output.shape == (1, q_len, 128)


def test_forward_uses_custom_op_for_causal_normal_decode(
    monkeypatch, backend_module, attention_impl
):
    q_len = 1
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "causal_attention_naive_decode", q_len=q_len, expected_shape=expected_shape
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        selected_name="causal_attention_naive_decode",
        selected=not_selected,
        sibling_name="causal_attention_naive_prefill",
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        selected_name="causal_attention_naive_decode",
        selected=selected,
        sibling_name="causal_attention_naive_prefill",
    )

    attention_impl.sliding_window = None
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_metadata(is_prefill=False, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=9)
    assert torch.equal(selected.call_args.args[4], metadata.seq_lens.to(torch.int16))
    assert torch.equal(
        selected.call_args.args[6], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[7] is attention_impl.scale
    assert selected.call_args.args[8] is None
    assert output.shape == (1, q_len, 128)


def test_forward_uses_triton_op_for_flash_causal_prefill(
    monkeypatch, backend_module, attention_impl
):
    q_len = 2
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "flash_causal_attention_naive_prefill",
        q_len=q_len,
        expected_shape=expected_shape,
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        selected_name="flash_causal_attention_naive_prefill",
        selected=selected,
        sibling_name="flash_causal_attention_naive_decode",
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        selected_name="flash_causal_attention_naive_prefill",
        selected=not_selected,
        sibling_name="flash_causal_attention_naive_decode",
    )

    attention_impl.sliding_window = None
    attention_impl.is_causal = True
    attention_impl.is_normal = False
    metadata = _make_metadata(is_prefill=True, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=8)
    assert selected.call_args.args[4] is attention_impl.scale
    assert torch.equal(selected.call_args.args[5], metadata.seq_lens.to(torch.int16))
    assert torch.equal(
        selected.call_args.args[6], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[7] is attention_impl.scale
    assert output.shape == (1, q_len, 128)


def test_forward_uses_triton_op_for_normal_attention_decode(
    monkeypatch, backend_module, attention_impl
):
    q_len = 1
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "attention_naive_decode", q_len=q_len, expected_shape=expected_shape
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=True
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        selected_name="attention_naive_decode",
        selected=selected,
        sibling_name="attention_naive_prefill",
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        selected_name="attention_naive_decode",
        selected=not_selected,
        sibling_name="attention_naive_prefill",
    )

    attention_impl.sliding_window = None
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_metadata(is_prefill=False, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=9)
    assert selected.call_args.args[4] is metadata.attn_masks
    assert torch.equal(selected.call_args.args[5], metadata.seq_lens.to(torch.int16))
    assert torch.equal(
        selected.call_args.args[7], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[8] is attention_impl.scale
    assert output.shape == (1, q_len, 128)


def test_forward_uses_custom_op_for_flash_attention_prefill(
    monkeypatch, backend_module, attention_impl
):
    q_len = 2
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "flash_attention_naive_prefill", q_len=q_len, expected_shape=expected_shape
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        selected_name="flash_attention_naive_prefill",
        selected=not_selected,
        sibling_name="flash_attention_naive_decode",
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        selected_name="flash_attention_naive_prefill",
        selected=selected,
        sibling_name="flash_attention_naive_decode",
    )

    attention_impl.sliding_window = None
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_metadata(is_prefill=True, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=10)
    assert selected.call_args.args[4] is metadata.attn_masks
    assert selected.call_args.args[5] is attention_impl.scale
    assert torch.equal(selected.call_args.args[6], metadata.seq_lens.to(torch.int16))
    assert torch.equal(
        selected.call_args.args[7], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[8] is attention_impl.scale
    assert selected.call_args.args[9] is None
    assert output.shape == (1, q_len, 128)


def test_forward_uses_python_impl_when_compile_model_disabled(
    monkeypatch, backend_module, attention_impl
):
    q_len = 2
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "flash_attention_naive_prefill_impl",
        q_len=q_len,
        expected_shape=expected_shape,
    )

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=True
    )
    monkeypatch.setattr(
        backend_module,
        "flash_attention_naive_prefill_impl",
        selected,
    )

    attention_impl.sliding_window = None
    attention_impl.is_causal = False
    attention_impl.is_normal = False
    metadata = _make_metadata(is_prefill=True, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=9)
    assert selected.call_args.args[4] is metadata.attn_masks
    assert selected.call_args.args[5] is attention_impl.scale
    assert torch.equal(selected.call_args.args[6], metadata.seq_lens.to(torch.int16))
    assert torch.equal(
        selected.call_args.args[7], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[8] is attention_impl.scale
    assert output.shape == (1, q_len, 128)


def test_forward_uses_python_impl_for_causal_normal_when_compile_model_disabled(
    monkeypatch, backend_module, attention_impl
):
    q_len = 1
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "causal_attention_naive_decode_impl",
        q_len=q_len,
        expected_shape=expected_shape,
    )

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=True
    )
    monkeypatch.setattr(backend_module, "causal_attention_naive_decode_impl", selected)

    attention_impl.sliding_window = None
    attention_impl.is_causal = True
    attention_impl.is_normal = True
    metadata = _make_metadata(is_prefill=False, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=8)
    assert torch.equal(selected.call_args.args[4], metadata.seq_lens.to(torch.int16))
    assert selected.call_args.args[5] is attention_impl.scale
    assert torch.equal(
        selected.call_args.args[6], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[7] is attention_impl.scale
    assert output.shape == (1, q_len, 128)


def test_forward_uses_python_impl_for_normal_attention_when_compile_model_disabled(
    monkeypatch, backend_module, attention_impl
):
    q_len = 1
    expected_shape = (1, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "attention_naive_decode_impl",
        q_len=q_len,
        expected_shape=expected_shape,
    )

    _configure_runtime(
        monkeypatch, backend_module, compile_model=False, use_custom_kernel=True
    )
    monkeypatch.setattr(backend_module, "attention_naive_decode_impl", selected)

    attention_impl.sliding_window = None
    attention_impl.is_causal = False
    attention_impl.is_normal = True
    metadata = _make_metadata(is_prefill=False, q_len=q_len)
    query, key, value, kv_cache = _make_inputs(q_len=q_len, block_size=4)

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=9)
    assert selected.call_args.args[4] is metadata.attn_masks
    assert torch.equal(selected.call_args.args[5], metadata.seq_lens.to(torch.int16))
    assert selected.call_args.args[6] is attention_impl.scale
    assert torch.equal(
        selected.call_args.args[7], metadata.block_tables.to(torch.int16)
    )
    assert selected.call_args.args[8] is attention_impl.scale
    assert output.shape == (1, q_len, 128)


def test_forward_appends_swa_mask_and_sinks_for_custom_sliding_window_decode(
    monkeypatch, backend_module, attention_impl
):
    q_len = 1
    batch_size = 2
    expected_shape = (batch_size, 1, 4, q_len, 32)
    selected = _make_op_stub(
        "sliding_window_attention_naive_decode",
        q_len=q_len,
        expected_shape=expected_shape,
    )
    not_selected = Mock()

    _configure_runtime(
        monkeypatch, backend_module, compile_model=True, use_custom_kernel=False
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_triton_ops,
        selected_name="sliding_window_attention_naive_decode",
        selected=not_selected,
        sibling_name="sliding_window_attention_naive_prefill",
    )
    _patch_op_pair(
        monkeypatch,
        backend_module.torch.ops.rbln_custom_ops,
        selected_name="sliding_window_attention_naive_decode",
        selected=selected,
        sibling_name="sliding_window_attention_naive_prefill",
    )

    attention_impl.sliding_window = 4
    attention_impl.is_batch_attention_opt = True
    attention_impl.sinks = torch.ones((1,), dtype=torch.float32)
    metadata = _make_metadata(is_prefill=False, q_len=q_len, batch_size=batch_size)
    query, key, value, kv_cache = _make_inputs(
        q_len=q_len, block_size=4, batch_size=batch_size
    )

    output = attention_impl.forward(None, query, key, value, kv_cache, metadata)

    selected.assert_called_once()
    not_selected.assert_not_called()
    _assert_common_dispatch_args(selected.call_args, kv_cache=kv_cache, arg_count=11)
    assert torch.equal(
        selected.call_args.args[4], metadata.cache_seq_lens.to(torch.int32)
    )
    assert selected.call_args.args[5] is metadata.cache_offsets
    assert selected.call_args.args[7] is metadata.local_block_tables
    assert selected.call_args.args[8] is attention_impl.scale
    assert selected.call_args.args[9] is metadata.swa_attn_masks
    assert selected.call_args.args[10] is attention_impl.sinks
    assert output.shape == (batch_size, q_len, 128)
