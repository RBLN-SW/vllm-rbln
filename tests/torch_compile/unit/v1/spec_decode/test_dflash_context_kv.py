import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
)

import vllm_rbln.patches.qwen3_dflash as qwen3_dflash_patch
import vllm_rbln.v1.spec_decode.dflash as dflash_mod
from vllm_rbln.patches.qwen3_dflash import (
    _apply_rope,
    _ContextKVPrecompute,
    _ContextKVRun,
    _plan_context_kv_runs,
    _rms_norm,
    _RuntimeInputBindingCache,
    _StableRuntimeGraph,
    patched_scheduler_init,
)
from vllm_rbln.v1.attention.ops.triton_flash_attention_naive import (
    flash_attention_naive_decode,
    flash_attention_naive_prefill,
)
from vllm_rbln.v1.spec_decode.dflash import (
    RBLNDFlashProposer,
    _BoundedHiddenStateCombiner,
    _check_draft_rope_style,
    _dflash_page_crossing_mask,
    _dflash_target_rope_is_neox_style,
    _DFlashSplitForwardGraph,
    _empty_drafts_for_page_crossing,
    _get_dflash_forward_split,
    _validate_dflash_geometry,
)


def test_hidden_state_combiner_reuses_two_stable_input_profiles() -> None:
    calls: list[tuple[torch.Size, int]] = []

    def combine(inputs: torch.Tensor) -> torch.Tensor:
        calls.append((inputs.shape, inputs.data_ptr()))
        return inputs[:, :3] * 2

    helper = _BoundedHiddenStateCombiner(combine)
    outputs = [
        helper(torch.full((run_len, 6), float(run_len)))
        for run_len in (3, 8, 111, 506, 3, 111)
    ]

    assert [shape for shape, _ in calls] == [
        torch.Size([8, 6]),
        torch.Size([8, 6]),
        torch.Size([512, 6]),
        torch.Size([512, 6]),
        torch.Size([8, 6]),
        torch.Size([512, 6]),
    ]
    assert len({pointer for _, pointer in calls[:2] + calls[4:5]}) == 1
    assert len({pointer for _, pointer in calls[2:4] + calls[5:]}) == 1
    assert calls[0][1] != calls[2][1]
    for run_len, output in zip((3, 8, 111, 506, 3, 111), outputs):
        assert output.shape == (run_len, 3)
        torch.testing.assert_close(
            output, torch.full((run_len, 3), float(run_len * 2))
        )


@pytest.mark.parametrize(
    ("run_len", "bucket_len"), [(1, 8), (8, 8), (9, 512), (512, 512)]
)
def test_hidden_state_combiner_bucket_boundaries(
    run_len: int, bucket_len: int
) -> None:
    observed_shapes: list[torch.Size] = []

    def combine(inputs: torch.Tensor) -> torch.Tensor:
        observed_shapes.append(inputs.shape)
        return inputs

    _BoundedHiddenStateCombiner(combine)(torch.empty(run_len, 4))

    assert observed_shapes == [torch.Size([bucket_len, 4])]


@pytest.mark.parametrize("run_len", [0, 513])
def test_hidden_state_combiner_rejects_unverified_lengths(run_len: int) -> None:
    helper = _BoundedHiddenStateCombiner(lambda inputs: inputs)

    with pytest.raises(ValueError, match="combine run length"):
        helper(torch.empty(run_len, 4))


@pytest.mark.parametrize("is_neox", [True, False])
def test_apply_rope_broadcasts_over_kv_heads(is_neox: bool) -> None:
    key = torch.randn(3, 8, 128)
    cos = torch.randn(3, 128)
    sin = torch.randn(3, 128)

    actual = _apply_rope(key, cos, sin, is_neox=is_neox)
    expected = _apply_rope(
        key, cos.unsqueeze(1), sin.unsqueeze(1), is_neox=is_neox
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("is_neox", "rotate_fn"),
    [(True, rotate_neox), (False, rotate_gptj)],
)
def test_apply_rope_selects_rotation_by_style(is_neox: bool, rotate_fn) -> None:
    torch.manual_seed(3)
    key = torch.randn(2, 1, 8)
    cos = torch.randn(2, 8)
    sin = torch.randn(2, 8)

    actual = _apply_rope(key, cos, sin, is_neox=is_neox)
    expected = key * cos.unsqueeze(1) + rotate_fn(key) * sin.unsqueeze(1)
    other = key * cos.unsqueeze(1) + (
        rotate_gptj(key) if is_neox else rotate_neox(key)
    ) * sin.unsqueeze(1)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, other)


def test_projection_runtime_buffers_use_two_stable_buckets() -> None:
    model = SimpleNamespace(
        _fused_kv_weight=torch.empty(16, 32),
        _head_dim=4,
    )
    helper = _ContextKVPrecompute(model)

    decode_inputs = helper._get_run_inputs(3, torch.bfloat16, torch.device("cpu"))
    decode_max_inputs = helper._get_run_inputs(
        8, torch.bfloat16, torch.device("cpu")
    )
    tail_inputs = helper._get_run_inputs(111, torch.bfloat16, torch.device("cpu"))
    prefill_max_inputs = helper._get_run_inputs(
        506, torch.bfloat16, torch.device("cpu")
    )

    assert tuple(t.data_ptr() for t in decode_inputs) == tuple(
        t.data_ptr() for t in decode_max_inputs
    )
    assert tuple(t.data_ptr() for t in tail_inputs) == tuple(
        t.data_ptr() for t in prefill_max_inputs
    )
    assert decode_inputs[0].shape[0] == 8
    assert tail_inputs[0].shape[0] == 506
    assert decode_inputs[0].data_ptr() != tail_inputs[0].data_ptr()


@pytest.mark.parametrize(
    ("run_len", "bucket_len"), [(1, 8), (8, 8), (9, 506), (506, 506)]
)
def test_projection_runtime_bucket_boundaries(
    run_len: int, bucket_len: int
) -> None:
    model = SimpleNamespace(
        _fused_kv_weight=torch.empty(16, 32),
        _head_dim=4,
    )
    helper = _ContextKVPrecompute(model)

    inputs = helper._get_run_inputs(run_len, torch.bfloat16, torch.device("cpu"))

    assert inputs[0].shape[0] == bucket_len


@pytest.mark.parametrize("run_len", [0, 507])
def test_projection_runtime_rejects_lengths_outside_verified_buckets(
    run_len: int,
) -> None:
    model = SimpleNamespace(
        _fused_kv_weight=torch.empty(16, 32),
        _head_dim=4,
    )
    helper = _ContextKVPrecompute(model)

    with pytest.raises(ValueError, match="projection run length"):
        helper._get_run_inputs(run_len, torch.bfloat16, torch.device("cpu"))


def test_context_kv_compiles_projection_only_runtime_per_layer_without_disk_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        _num_attn_layers=2,
        _num_kv_heads=2,
        _head_dim=4,
        _hidden_norm_weight=torch.empty(8),
        _fused_kv_weight=torch.empty(2 * 2 * 2 * 4, 8),
        _fused_kv_bias=None,
        _k_norm_weights=[torch.empty(4), torch.empty(4)],
        _rms_norm_eps=1e-6,
        _rope_is_neox=True,
    )
    helper = _ContextKVPrecompute(model)
    compile_calls: list[dict] = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(qwen3_dflash_patch.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    monkeypatch.setattr(qwen3_dflash_patch, "rbln_compile", fake_compile)

    layer0_decode = helper._get_graph(0, 1)
    layer0_decode_again = helper._get_graph(0, 1)
    layer0_decode_max = helper._get_graph(0, 8)
    layer0_tail = helper._get_graph(0, 111)
    layer0_prefill = helper._get_graph(0, 506)
    layer1_decode = helper._get_graph(1, 1)
    layer1_prefill = helper._get_graph(1, 506)

    assert layer0_decode is layer0_decode_again
    assert layer0_decode is layer0_decode_max
    assert layer0_tail is layer0_prefill
    assert layer0_decode is not layer0_prefill
    assert layer0_decode is not layer1_decode
    assert layer0_prefill is not layer1_prefill
    assert isinstance(layer0_decode, _StableRuntimeGraph)
    assert isinstance(layer0_prefill, _StableRuntimeGraph)
    assert len(compile_calls) == 4
    assert all(call["use_cache"] is False for call in compile_calls)


def test_context_kv_gives_bounded_profiles_unique_dynamo_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers = 5
    model = SimpleNamespace(
        _num_attn_layers=num_layers,
        _num_kv_heads=2,
        _head_dim=4,
        _hidden_norm_weight=torch.empty(8),
        _fused_kv_weight=torch.empty(num_layers * 2 * 2 * 4, 8),
        _fused_kv_bias=None,
        _k_norm_weights=[torch.empty(4) for _ in range(num_layers)],
        _rms_norm_eps=1e-6,
        _rope_is_neox=True,
    )
    helper = _ContextKVPrecompute(model)
    code_objects: list[object] = []

    def fake_compile(fn, **kwargs):
        code_objects.append(fn.__code__)
        return fn

    monkeypatch.setattr(qwen3_dflash_patch.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    monkeypatch.setattr(qwen3_dflash_patch, "rbln_compile", fake_compile)

    for layer_idx in range(num_layers):
        for run_len in (1, 8, 111, 506):
            helper._get_graph(layer_idx, run_len)

    assert len(code_objects) == 2 * num_layers
    assert len({id(code) for code in code_objects}) == 2 * num_layers
    assert helper._graphs.keys() == helper._runtime_holders.keys()


def test_dflash_scheduler_disables_only_eagle_cache_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class SpecConfig:
        method = "dflash"
        num_speculative_tokens = 3

        def use_dflash(self) -> bool:
            return self.method == "dflash"

        def use_eagle(self) -> bool:
            return self.method in ("eagle", "eagle3", "mtp", "dflash")

        def uses_draft_model(self) -> bool:
            return self.method == "draft_model"

    config = SimpleNamespace(speculative_config=SpecConfig())

    def original_init(_self, vllm_config, marker=None):
        spec = vllm_config.speculative_config
        observed.update(
            use_eagle=spec.use_eagle(),
            uses_draft_model=spec.uses_draft_model(),
            marker=marker,
        )

    monkeypatch.setattr(qwen3_dflash_patch, "_ORIGINAL_SCHEDULER_INIT", original_init)

    scheduler = SimpleNamespace()
    patched_scheduler_init(scheduler, config, marker="called")

    assert observed == {
        "use_eagle": False,
        "uses_draft_model": True,
        "marker": "called",
    }
    assert config.speculative_config.method == "dflash"
    assert scheduler.use_eagle is True
    assert scheduler.num_lookahead_tokens == 4


def test_non_dflash_scheduler_keeps_eagle_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    spec = SimpleNamespace(method="eagle3", use_dflash=lambda: False)
    config = SimpleNamespace(speculative_config=spec)

    def original_init(_self, vllm_config):
        observed["method"] = vllm_config.speculative_config.method

    monkeypatch.setattr(qwen3_dflash_patch, "_ORIGINAL_SCHEDULER_INIT", original_init)

    patched_scheduler_init(object(), config)

    assert observed == {"method": "eagle3"}
    assert spec.method == "eagle3"


def test_dflash_maps_checkpoint_swa_types_to_global_layer_names() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=2048,
        )
    )
    names = [f"model.layers.{index}.self_attn.attn" for index in range(62, 67)]
    proposer.draft_attn_groups = [SimpleNamespace(layer_names=list(reversed(names)))]

    RBLNDFlashProposer._configure_dflash_attention_layers(proposer)

    assert proposer._dflash_sliding_layer_names == set(names[:4])
    assert proposer._dflash_sliding_window == 2048


def test_dflash_forward_split_follows_the_swa_full_boundary() -> None:
    assert (
        _get_dflash_forward_split(
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ]
        )
        == 4
    )
    assert _get_dflash_forward_split(["full_attention"] * 5) is None
    assert _get_dflash_forward_split(["sliding_attention"] * 5) is None

    with pytest.raises(ValueError, match="contiguous"):
        _get_dflash_forward_split(
            ["sliding_attention", "full_attention", "sliding_attention"]
        )


def test_dflash_page_crossing_guard_covers_only_unrepresentable_offsets() -> None:
    mask = _dflash_page_crossing_mask(
        torch.tensor([1016, *range(1017, 1024), 1024, 1135]),
        partition_size=1024,
        query_len=8,
    )

    assert mask.tolist() == [False, *([True] * 7), False, False]


def test_dflash_page_crossing_guard_skips_whole_batch_before_kv_insert() -> None:
    assert _empty_drafts_for_page_crossing(torch.tensor([True, False])) == [[], []]
    assert _empty_drafts_for_page_crossing(torch.tensor([True, True])) == [[], []]
    assert _empty_drafts_for_page_crossing(torch.tensor([False, False])) is None


def test_dflash_intermediate_prefill_keeps_only_context_kv_work() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.runner = SimpleNamespace(is_intermediate_chunked_prefill=True)
    proposer.num_speculative_tokens = 7
    proposer.device = torch.device("cpu")

    drafts = RBLNDFlashProposer._intermediate_prefill_drafts(proposer, 2)

    assert drafts is not None
    assert drafts.shape == (2, 7)
    assert drafts.dtype == torch.int64
    assert torch.count_nonzero(drafts) == 0

    proposer.runner.is_intermediate_chunked_prefill = False
    assert RBLNDFlashProposer._intermediate_prefill_drafts(proposer, 2) is None


def test_dflash_split_forward_graph_preserves_hidden_residual_abi() -> None:
    calls: list[tuple] = []

    def sliding(input_ids, positions):
        calls.append(("sliding", input_ids, positions))
        return input_ids + 10, input_ids + 20

    def full(hidden_states, residual, positions, sample_indices):
        calls.append(("full", hidden_states, residual, positions, sample_indices))
        return hidden_states + residual, sample_indices + 1

    graph = _DFlashSplitForwardGraph(sliding, full)

    assert graph(2, 3, 4) == (34, 5)
    assert calls == [
        ("sliding", 2, 3),
        ("full", 12, 22, 3, 4),
    ]


def test_dflash_causal_swa_mask_tracks_each_query_position() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 3
    metadata = SimpleNamespace(attn_masks=torch.zeros(1, 1, 1, 1, 12))
    cad = SimpleNamespace(_seq_lens_cpu=torch.tensor([5], dtype=torch.int32))

    RBLNDFlashProposer._rebuild_block_draft_mask(
        proposer,
        metadata,
        cad,
        num_reqs=1,
        num_reqs_padded=1,
        sliding_window=4,
    )

    expected = torch.zeros(4, 12)
    expected[0, 2:6] = 1
    expected[1, 3:7] = 1
    expected[2, 4:8] = 1
    expected[3, 5:9] = 1
    torch.testing.assert_close(metadata.attn_masks[0, 0, 0], expected)


def test_dflash_preserves_absolute_position_for_compiler_partition_abi() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 7
    proposer._dflash_sliding_layer_names = set()
    proposer._dflash_sliding_window = None
    metadata = SimpleNamespace(
        attn_masks=torch.zeros(1, 1, 1, 1, 4096),
        block_tables=torch.arange(4).view(1, -1),
        seq_lens=torch.tensor([[1135]], dtype=torch.int32),
    )
    cad = SimpleNamespace(
        _seq_lens_cpu=torch.tensor([1135], dtype=torch.int32),
        block_table_tensor=metadata.block_tables,
    )
    group = SimpleNamespace(
        layer_names=["layer.full"],
        kv_cache_spec=SimpleNamespace(block_size=1024),
    )

    per_layer = RBLNDFlashProposer._specialize_layer_attn_metadata(
        proposer,
        group,
        metadata,
        cad,
        num_reqs=1,
        num_reqs_padded=1,
    )

    # The compiler converter is the single owner of absolute-to-partition
    # expansion. Passing [1024, 111, ...] here would make it expand a second
    # time and erase the 111-token tail partition.
    assert per_layer["layer.full"].seq_lens.shape == (1, 1)
    assert per_layer["layer.full"].seq_lens.tolist() == [[1135]]


def test_dflash_keeps_full_metadata_and_specializes_only_swa_layers() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 3
    proposer._dflash_sliding_layer_names = {"layer.sw"}
    proposer._dflash_sliding_window = 4
    block_tables = torch.arange(3).view(1, -1)
    metadata = SimpleNamespace(
        attn_masks=torch.zeros(1, 1, 1, 1, 12),
        block_tables=block_tables,
        seq_lens=torch.tensor([[5]], dtype=torch.int32),
    )
    cad = SimpleNamespace(
        _seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        block_table_tensor=block_tables,
    )
    group = SimpleNamespace(
        layer_names=["layer.sw", "layer.full"],
        kv_cache_spec=SimpleNamespace(block_size=4),
    )

    per_layer = RBLNDFlashProposer._specialize_layer_attn_metadata(
        proposer,
        group,
        metadata,
        cad,
        num_reqs=1,
        num_reqs_padded=1,
    )

    assert per_layer["layer.full"] is metadata
    assert per_layer["layer.sw"] is not metadata
    assert torch.equal(metadata.attn_masks[0, 0, 0, 0], metadata.attn_masks[0, 0, 0, 3])
    assert not torch.equal(
        per_layer["layer.sw"].attn_masks[0, 0, 0, 0],
        per_layer["layer.sw"].attn_masks[0, 0, 0, 3],
    )


def test_dflash_swa_metadata_rebases_to_bounded_physical_partitions() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 7
    proposer._dflash_sliding_layer_names = {"layer.sw"}
    proposer._dflash_sliding_window = 2048
    metadata = SimpleNamespace(
        attn_masks=torch.zeros(1, 1, 1, 1, 49152),
        block_tables=torch.arange(100, 148).view(1, -1),
        seq_lens=torch.tensor([[3497]], dtype=torch.int32),
    )
    cad = SimpleNamespace(
        _seq_lens_cpu=torch.tensor([3497], dtype=torch.int32),
        block_table_tensor=metadata.block_tables,
    )
    group = SimpleNamespace(
        layer_names=["layer.sw", "layer.full"],
        kv_cache_spec=SimpleNamespace(block_size=1024),
    )

    per_layer = RBLNDFlashProposer._specialize_layer_attn_metadata(
        proposer,
        group,
        metadata,
        cad,
        num_reqs=1,
        num_reqs_padded=1,
    )

    sliding = per_layer["layer.sw"]
    # The eight draft queries cover at most four aligned 1024-token cache
    # partitions. Absolute position 3497 starts the local view at partition 1.
    assert sliding.block_tables.shape == (1, 48)
    assert sliding.block_tables[0, :4].tolist() == [101, 102, 103, 104]
    assert torch.count_nonzero(sliding.block_tables[0, 4:]) == 0
    assert sliding.seq_lens.shape == (1, 1)
    assert sliding.seq_lens.tolist() == [[2473]]
    assert sliding.attn_masks.shape == (1, 1, 1, 8, 49152)
    assert torch.count_nonzero(sliding.attn_masks[0, 0, 0, 0, :426]) == 0
    assert torch.all(sliding.attn_masks[0, 0, 0, 0, 426:2474] == 1)
    assert torch.count_nonzero(sliding.attn_masks[0, 0, 0, 0, 2474:]) == 0
    # The full-attention layer must retain its original 48-partition geometry.
    assert per_layer["layer.full"] is metadata
    assert metadata.block_tables.shape == (1, 48)
    assert metadata.seq_lens.shape == (1, 1)
    assert metadata.seq_lens.tolist() == [[3497]]
    assert metadata.attn_masks.shape[-1] == 49152


@pytest.mark.parametrize(
    "kernel", [flash_attention_naive_prefill, flash_attention_naive_decode]
)
def test_noncausal_flash_kernel_derives_each_partition_offset(kernel) -> None:
    source = inspect.getsource(kernel.fn)

    # The compiler expands the raw absolute position into one bounded length
    # per partition. Preserve that maximum on the dynamic index so load/store
    # shape inference cannot grow past the physical cache partition.
    assert "to_dynamic_index(SP_block_ptr, P)" in source


@pytest.mark.parametrize(
    ("is_neox", "rotate_fn"),
    [(True, rotate_neox), (False, rotate_gptj)],
)
def test_context_kv_projection_graph_does_not_take_cache_input(
    is_neox: bool, rotate_fn
) -> None:
    torch.manual_seed(11)
    model = SimpleNamespace(
        _num_attn_layers=1,
        _num_kv_heads=2,
        _head_dim=4,
        _hidden_norm_weight=torch.randn(8),
        _fused_kv_weight=torch.randn(2 * 2 * 4, 8),
        _fused_kv_bias=torch.randn(2 * 2 * 4),
        _k_norm_weights=[torch.randn(4)],
        _rms_norm_eps=1e-6,
        _rope_is_neox=is_neox,
    )
    helper = _ContextKVPrecompute(model)
    states = torch.randn(3, 8)
    cos = torch.randn(3, 4)
    sin = torch.randn(3, 4)

    key, value = helper._graph_fn(layer_idx=0, run_len=3)(states, cos, sin)
    normed = _rms_norm(states, model._hidden_norm_weight, model._rms_norm_eps)
    fused = torch.nn.functional.linear(
        normed, model._fused_kv_weight, model._fused_kv_bias
    ).view(3, 2, 2, 4)
    # The rotation reference is built directly from vLLM's rotate functions so
    # the golden stays independent of _apply_rope's own style dispatch.
    normed_key = _rms_norm(fused[:, 0], model._k_norm_weights[0], model._rms_norm_eps)
    expected_key = (
        normed_key * cos.unsqueeze(1) + rotate_fn(normed_key) * sin.unsqueeze(1)
    ).permute(1, 0, 2)
    expected_value = fused[:, 1].permute(1, 0, 2)

    assert key.shape == (2, 3, 4)
    assert value.shape == (2, 3, 4)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


def test_graph_fn_requires_resolved_rope_style() -> None:
    model = SimpleNamespace(
        _num_attn_layers=1,
        _num_kv_heads=2,
        _head_dim=4,
        _hidden_norm_weight=torch.empty(8),
        _fused_kv_weight=torch.empty(2 * 2 * 4, 8),
        _fused_kv_bias=None,
        _k_norm_weights=[torch.empty(4)],
        _rms_norm_eps=1e-6,
    )
    helper = _ContextKVPrecompute(model)

    with pytest.raises(RuntimeError, match="RoPE rotation style"):
        helper._graph_fn(layer_idx=0, run_len=3)


def test_context_kv_store_uses_one_batched_copy_and_preserves_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, num_heads, head_dim = 2, 2, 4
    caches = [
        torch.full((2, 4, num_heads, 1, 16, head_dim), 7.5) for _ in range(num_layers)
    ]
    model = SimpleNamespace(
        _fused_kv_weight=torch.empty(16, 8),
        _num_kv_heads=num_heads,
        _head_dim=head_dim,
        _attn_layers=[SimpleNamespace(kv_cache=cache) for cache in caches],
    )
    helper = _ContextKVPrecompute(model)
    run = _ContextKVRun(3, 5, 2, 7, 0)
    projected = [
        (torch.randn(num_heads, 5, head_dim), torch.randn(num_heads, 5, head_dim))
        for _ in range(num_layers)
    ]
    real_foreach_copy = torch._foreach_copy_
    calls: list[tuple[int, int]] = []

    def record_foreach_copy(destinations, sources):
        calls.append((len(destinations), len(sources)))
        return real_foreach_copy(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", record_foreach_copy)
    helper._store_projected_kv((0, 1), run, projected)

    assert calls == [(4, 4)]
    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[0, 2, :, 0, 7:12, :], projected[layer_idx][0])
        torch.testing.assert_close(cache[1, 2, :, 0, 7:12, :], projected[layer_idx][1])
        assert torch.count_nonzero(cache[:, 2, :, 0, :7, :] != 7.5) == 0
        assert torch.count_nonzero(cache[:, 2, :, 0, 12:, :] != 7.5) == 0


def test_context_kv_run_uses_bucket_shape_and_stores_only_real_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_heads, head_dim, hidden_size, run_len = 2, 4, 8, 3
    rotary = SimpleNamespace(
        cos_cache=torch.randn(16, head_dim),
        sin_cache=torch.randn(16, head_dim),
    )
    cache = torch.full((2, 2, num_heads, 1, 16, head_dim), 7.5)
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))],
        _fused_kv_weight=torch.empty(2 * num_heads * head_dim, hidden_size),
        _fused_kv_bias=None,
        _num_kv_heads=num_heads,
        _head_dim=head_dim,
        _attn_layers=[SimpleNamespace(kv_cache=cache)],
    )
    helper = _ContextKVPrecompute(model)
    helper._group_runs = {0: (_ContextKVRun(0, run_len, 1, 5, 0),)}
    helper._layer_group = {0: 0}
    graph_input_shapes: list[tuple[torch.Size, ...]] = []

    def projection_graph(*inputs: torch.Tensor):
        graph_input_shapes.append(tuple(tensor.shape for tensor in inputs))
        bucket_len = inputs[0].shape[0]
        output = torch.arange(
            num_heads * bucket_len * head_dim, dtype=torch.float32
        ).view(
            num_heads, bucket_len, head_dim
        )
        return output, output + 1

    monkeypatch.setattr(helper, "_get_graph", lambda _layer, _length: projection_graph)

    helper.run(
        torch.randn(run_len, hidden_size),
        torch.arange(run_len),
        torch.tensor(1),
    )

    assert graph_input_shapes == [
        (
            torch.Size([8, hidden_size]),
            torch.Size([8, head_dim]),
            torch.Size([8, head_dim]),
        )
    ]
    expected = torch.arange(num_heads * 8 * head_dim, dtype=torch.float32).view(
        num_heads, 8, head_dim
    )[:, :run_len]
    torch.testing.assert_close(cache[0, 1, :, 0, 5:8, :], expected)
    torch.testing.assert_close(cache[1, 1, :, 0, 5:8, :], expected + 1)
    assert torch.count_nonzero(cache[:, 1, :, 0, :5, :] != 7.5) == 0
    assert torch.count_nonzero(cache[:, 1, :, 0, 8:, :] != 7.5) == 0


def test_runtime_input_binding_cache_reuses_seen_layer_bindings() -> None:
    bindings = _RuntimeInputBindingCache(dynamic_input_indices=(0, 3))
    layer0 = (10, 11, 12, 13)
    layer1 = (10, 21, 22, 13)

    bindings.observe(layer0)
    assert bindings.plan(layer1).can_reuse is False

    bindings.observe(layer1)
    plan = bindings.plan(layer0)

    assert plan.can_reuse is True
    assert plan.prepare_indices == (0, 3)
    assert plan.patch_indices == (1, 2)


def test_runtime_input_binding_cache_falls_back_for_unseen_address() -> None:
    bindings = _RuntimeInputBindingCache(dynamic_input_indices=(0,))
    bindings.observe((100, 200, 300))

    plan = bindings.plan((100, 201, 300))

    assert plan.can_reuse is False
    assert plan.prepare_indices == ()
    assert plan.patch_indices == ()


def test_runtime_input_binding_cache_executes_minimal_prepare_and_patch() -> None:
    events: list[tuple] = []

    class Handle:
        def begin_io_patch_batch(self) -> None:
            events.append(("begin",))

        def prepare_inputs(self, device_inputs, cpu_inputs) -> None:
            events.append(("prepare", device_inputs, cpu_inputs))

        def update_input_addr(self, index, addresses) -> None:
            events.append(("patch", index, addresses))

        def prepare_outputs(self, device_outputs, cpu_outputs) -> None:
            events.append(("prepare_outputs", device_outputs, cpu_outputs))

        def end_io_patch_batch(self) -> None:
            events.append(("end",))

        def run(self) -> None:
            events.append(("run",))

    runtime = SimpleNamespace(
        _runtime_handle=Handle(),
        _capture_reports_if_needed=lambda: events.append(("reports",)),
    )
    bindings = _RuntimeInputBindingCache(dynamic_input_indices=(0, 3))
    bindings.observe((10, 11, 12, 13))
    bindings.observe((10, 21, 22, 13))
    output = torch.empty(1)

    reused = bindings.execute(
        runtime,
        (10, 11, 12, 13),
        get_device_addrs=lambda pointer: [pointer + 1000],
        outputs=[output],
    )

    assert reused is True
    assert events == [
        ("begin",),
        ("prepare", {0: 10, 3: 13}, {}),
        ("patch", 1, [1011]),
        ("patch", 2, [1012]),
        ("prepare_outputs", {}, {0: output.data_ptr()}),
        ("end",),
        ("run",),
        ("reports",),
    ]


def test_runtime_input_binding_cache_falls_back_for_non_sequence_outputs() -> None:
    bindings = _RuntimeInputBindingCache(dynamic_input_indices=())
    bindings.observe((10,))

    assert (
        bindings.execute(
            SimpleNamespace(),
            (10,),
            get_device_addrs=lambda pointer: [pointer],
            outputs=torch.empty(()),
        )
        is False
    )


def test_runtime_input_binding_cache_retries_patch_after_one_time_prepare() -> None:
    events: list[tuple] = []
    incompatible_patches_remaining = 1

    class Handle:
        def begin_io_patch_batch(self) -> None:
            events.append(("begin",))

        def prepare_inputs(self, device_inputs, cpu_inputs) -> None:
            events.append(("prepare", device_inputs, cpu_inputs))

        def update_input_addr(self, index, addresses) -> None:
            nonlocal incompatible_patches_remaining
            events.append(("patch", index, addresses))
            if index == 1 and incompatible_patches_remaining:
                incompatible_patches_remaining -= 1
                raise RuntimeError(
                    "INIT_INTERNAL (addrs.size() <= rbln_slot.device_allocs().size())"
                )

        def prepare_outputs(self, device_outputs, cpu_outputs) -> None:
            events.append(("prepare_outputs", device_outputs, cpu_outputs))

        def end_io_patch_batch(self) -> None:
            events.append(("end",))

        def run(self) -> None:
            events.append(("run",))

    runtime = SimpleNamespace(
        _runtime_handle=Handle(),
        _capture_reports_if_needed=lambda: events.append(("reports",)),
    )
    bindings = _RuntimeInputBindingCache(dynamic_input_indices=(0, 3))
    layer0 = (10, 11, 12, 13)
    layer1 = (10, 21, 22, 13)
    bindings.observe(layer0)
    bindings.observe(layer1)
    output = torch.empty(1)

    assert bindings.execute(
        runtime,
        layer0,
        get_device_addrs=lambda pointer: [pointer + 1000],
        outputs=[output],
    )
    assert ("prepare", {1: 11}, {}) in events

    events.clear()
    assert bindings.execute(
        runtime,
        layer1,
        get_device_addrs=lambda pointer: [pointer + 1000],
        outputs=[output],
    )
    assert events == [
        ("begin",),
        ("prepare", {0: 10, 3: 13}, {}),
        ("patch", 1, [1021]),
        ("patch", 2, [1022]),
        ("prepare_outputs", {}, {0: output.data_ptr()}),
        ("end",),
        ("run",),
        ("reports",),
    ]


def test_stable_runtime_graph_warms_each_layer_then_reuses_bindings() -> None:
    events: list[tuple] = []

    class Handle:
        def begin_io_patch_batch(self) -> None:
            events.append(("begin",))

        def prepare_inputs(self, device_inputs, cpu_inputs) -> None:
            events.append(("prepare", device_inputs, cpu_inputs))

        def update_input_addr(self, index, addresses) -> None:
            events.append(("patch", index, addresses))

        def prepare_outputs(self, device_outputs, cpu_outputs) -> None:
            events.append(("prepare_outputs", device_outputs, cpu_outputs))

        def end_io_patch_batch(self) -> None:
            events.append(("end",))

        def run(self) -> None:
            events.append(("run",))

    original_inputs: list[tuple] = []
    normal_outputs: list[torch.Tensor] = []

    def original_run(*inputs, out, **kwargs):
        original_inputs.append(inputs)
        output = torch.tensor(len(normal_outputs))
        normal_outputs.append(output)
        return [output]

    runtime = SimpleNamespace(
        _num_inputs=3,
        _input_name_to_index={"args_0": 0, "args_1": 1, "args_2": 2},
        _runtime_utils=SimpleNamespace(
            prepare_inputs=lambda *args, **kwargs: list(args)
        ),
        _runtime_handle=Handle(),
        _capture_reports_if_needed=lambda: events.append(("reports",)),
        run=original_run,
    )

    def compiled(*args):
        # The exported runtime order is intentionally unrelated to the Python
        # function order. The optimization must trust the actual
        # DynamoRuntime.run boundary, not args_N.
        return runtime.run(args[15], args[0], args[2], out=None)[0]

    graph = _StableRuntimeGraph(
        compiled,
        [runtime],
        get_device_addrs=lambda pointer: [pointer + 1000],
        tensor_is_supported=lambda tensor: True,
        dynamic_arg_indices=(0,),
    )
    common = torch.empty(1)
    layer0_weight, layer1_weight = torch.empty(2), torch.empty(3)
    layer0_cache, layer1_cache = torch.empty(4), torch.empty(5)

    def graph_args(weight, cache):
        values = [torch.empty(1) for _ in range(16)]
        values[4] = None
        values[5] = None
        values[0] = common
        values[2] = weight
        values[15] = cache
        return tuple(values)

    first = graph(*graph_args(layer0_weight, layer0_cache))
    second = graph(*graph_args(layer1_weight, layer1_cache))
    third = graph(*graph_args(layer0_weight, layer0_cache))
    reused = graph(*graph_args(layer1_weight, layer1_cache))

    assert first is normal_outputs[0]
    assert second is normal_outputs[1]
    assert third is normal_outputs[2]
    assert reused is normal_outputs[2]
    assert len(normal_outputs) == 3
    assert len(original_inputs) == 3
    assert events[0] == ("begin",)
    assert events[1][0] == "prepare"
    assert events[1][1] == {1: common.data_ptr()}
    assert [event[1] for event in events if event[0] == "patch"] == [0, 2]
    assert events[-4][0] == "prepare_outputs"
    assert events[-2:] == [("run",), ("reports",)]


def _coordinates(
    *, start_block: int, offset: int, length: int, partition_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = torch.arange(offset, offset + length)
    blocks = start_block + absolute // partition_size
    offsets = absolute % partition_size
    return blocks, offsets


def _covered_token_indices(runs: tuple[_ContextKVRun, ...]) -> list[int]:
    return [
        token_idx
        for run in runs
        for token_idx in range(run.token_start, run.token_start + run.token_count)
    ]


@pytest.mark.parametrize("length", [8, 64, 506])
def test_plan_context_kv_runs_keeps_within_block_inputs_maximal(length: int) -> None:
    blocks, offsets = _coordinates(start_block=3, offset=16, length=length)

    runs = _plan_context_kv_runs(
        blocks,
        offsets,
        request_ids=torch.zeros(length, dtype=torch.int64),
        group_id=7,
    )

    assert runs == (_ContextKVRun(0, length, 3, 16, 7),)
    assert _covered_token_indices(runs) == list(range(length))


@pytest.mark.parametrize(
    ("length", "expected_counts"),
    [(507, (506, 1)), (512, (506, 6))],
)
def test_plan_context_kv_runs_caps_verified_projection_profile(
    length: int, expected_counts: tuple[int, ...]
) -> None:
    blocks, offsets = _coordinates(start_block=3, offset=16, length=length)

    runs = _plan_context_kv_runs(
        blocks,
        offsets,
        request_ids=torch.zeros(length, dtype=torch.int64),
        group_id=7,
    )

    assert tuple(run.token_count for run in runs) == expected_counts
    assert tuple(run.block_offset for run in runs) == (16, 522)
    assert _covered_token_indices(runs) == list(range(length))


@pytest.mark.parametrize(
    ("offset", "length", "expected_counts"),
    [(1020, 8, (4, 4)), (1000, 64, (24, 40))],
)
def test_plan_context_kv_runs_splits_1024_partition_straddles(
    offset: int, length: int, expected_counts: tuple[int, int]
) -> None:
    blocks, offsets = _coordinates(start_block=11, offset=offset, length=length)

    runs = _plan_context_kv_runs(
        blocks,
        offsets,
        request_ids=torch.zeros(length, dtype=torch.int64),
        group_id=2,
    )

    assert tuple(run.token_count for run in runs) == expected_counts
    assert tuple(run.physical_block_id for run in runs) == (11, 12)
    assert tuple(run.block_offset for run in runs) == (offset, 0)
    assert _covered_token_indices(runs) == list(range(length))


def test_plan_context_kv_runs_splits_request_and_physical_block_boundaries() -> None:
    # Request 0 straddles blocks 5/6. Request 1 deliberately resumes block 6 at
    # the next offset: the request boundary must still start a new run.
    blocks = torch.tensor([5, 5, 6, 6, 6, 6, 9, 9], dtype=torch.int64)
    offsets = torch.tensor([1022, 1023, 0, 1, 2, 3, 17, 18], dtype=torch.int64)
    request_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)

    runs = _plan_context_kv_runs(blocks, offsets, request_ids, group_id=4)

    assert runs == (
        _ContextKVRun(0, 2, 5, 1022, 4),
        _ContextKVRun(2, 2, 6, 0, 4),
        _ContextKVRun(4, 2, 6, 2, 4),
        _ContextKVRun(6, 2, 9, 17, 4),
    )
    assert _covered_token_indices(runs) == list(range(8))


def test_scheduler_batch_resolves_request_and_physical_block_runs() -> None:
    proposer = SimpleNamespace(
        runner=SimpleNamespace(cache_config=SimpleNamespace(block_size=1024)),
        _dflash_num_context=21,
        _context_positions_cpu_buffer=torch.cat(
            (torch.arange(1020, 1028), torch.arange(31, 44))
        ),
    )
    cad = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 8, 21], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([1028, 44], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3, 4, 5], [8, 9, 10]], dtype=torch.int32),
    )
    group = SimpleNamespace(layer_names=[f"layer.{i}" for i in range(5)])
    metadata = SimpleNamespace(local_block_tables=None, cache_seq_lens=None)

    group_slots, layer_group, request_ids = RBLNDFlashProposer._resolve_group_slots(
        proposer, [(group, metadata)], cad
    )
    helper = _ContextKVPrecompute(
        SimpleNamespace(_fused_kv_weight=torch.empty(16, 32), _head_dim=4)
    )
    helper.set_group_slots(group_slots, layer_group, request_ids)

    assert request_ids.tolist() == [0] * 8 + [1] * 13
    assert helper._group_runs[0] == (
        _ContextKVRun(0, 4, 3, 1020, 0),
        _ContextKVRun(4, 4, 4, 0, 0),
        _ContextKVRun(8, 13, 8, 31, 0),
    )
    assert layer_group == {layer_idx: 0 for layer_idx in range(5)}


def test_rejected_draft_padding_keeps_actual_context_publication_positions() -> None:
    proposer = SimpleNamespace(
        runner=SimpleNamespace(cache_config=SimpleNamespace(block_size=1024)),
        _dflash_num_context=8,
        _context_positions_cpu_buffer=torch.tensor(
            [78, 79, 80, 81, 82, 83, 84, 85], dtype=torch.int64
        ),
    )
    cad = SimpleNamespace(
        num_reqs=1,
        query_start_loc_cpu=torch.tensor([0, 8], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([80], dtype=torch.int32),
        block_table_tensor=torch.tensor([[1, 2, 3]], dtype=torch.int32),
    )
    group = SimpleNamespace(layer_names=[f"layer.{i}" for i in range(5)])
    metadata = SimpleNamespace(local_block_tables=None, cache_seq_lens=None)

    group_slots, layer_group, request_ids = RBLNDFlashProposer._resolve_group_slots(
        proposer, [(group, metadata)], cad
    )
    helper = _ContextKVPrecompute(
        SimpleNamespace(_fused_kv_weight=torch.empty(16, 32), _head_dim=4)
    )
    helper.set_group_slots(group_slots, layer_group, request_ids)

    assert helper._group_runs[0] == (_ContextKVRun(0, 8, 1, 78, 0),)


@pytest.mark.parametrize(
    ("seq_len", "chunk_len", "expected"),
    [
        (
            512,
            512,
            (
                _ContextKVRun(0, 506, 3, 0, 0),
                _ContextKVRun(506, 6, 3, 506, 0),
            ),
        ),
        (
            1024,
            512,
            (
                _ContextKVRun(0, 506, 3, 512, 0),
                _ContextKVRun(506, 6, 3, 1018, 0),
            ),
        ),
        (1135, 111, (_ContextKVRun(0, 111, 4, 0, 0),)),
    ],
)
def test_scheduler_repeated_prefill_chunks_keep_absolute_cache_offsets(
    seq_len: int,
    chunk_len: int,
    expected: tuple[_ContextKVRun, ...],
) -> None:
    proposer = SimpleNamespace(
        runner=SimpleNamespace(cache_config=SimpleNamespace(block_size=1024)),
        _dflash_num_context=chunk_len,
        _context_positions_cpu_buffer=torch.arange(seq_len - chunk_len, seq_len),
    )
    cad = SimpleNamespace(
        num_reqs=1,
        query_start_loc_cpu=torch.tensor([0, chunk_len], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3, 4, 5]], dtype=torch.int32),
    )
    group = SimpleNamespace(layer_names=[f"layer.{i}" for i in range(5)])
    metadata = SimpleNamespace(local_block_tables=None, cache_seq_lens=None)

    group_slots, layer_group, request_ids = RBLNDFlashProposer._resolve_group_slots(
        proposer, [(group, metadata)], cad
    )
    helper = _ContextKVPrecompute(
        SimpleNamespace(_fused_kv_weight=torch.empty(16, 32), _head_dim=4)
    )
    helper.set_group_slots(group_slots, layer_group, request_ids)

    assert helper._group_runs[0] == expected


def test_run_planner_drives_cpu_kv_scatter_bit_exactly() -> None:
    torch.manual_seed(41)
    num_tokens, num_layers = 10, 2
    num_kv_heads, head_dim = 2, 4
    key = torch.randn(num_layers, num_tokens, num_kv_heads, head_dim)
    value = torch.randn(num_layers, num_tokens, num_kv_heads, head_dim)

    blocks = torch.tensor([2, 2, 3, 3, 3, 8, 8, 8, 8, 8])
    offsets = torch.tensor([1022, 1023, 0, 1, 2, 10, 11, 12, 13, 14])
    request_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    runs = _plan_context_kv_runs(blocks, offsets, request_ids, group_id=0)

    for layer_idx in range(num_layers):
        cache_from_runs = torch.full((2, 10, num_kv_heads, 1, 1024, head_dim), 7.5)
        cache_tokenwise = cache_from_runs.clone()

        for run in runs:
            token_slice = slice(run.token_start, run.token_start + run.token_count)
            cache_slice = slice(run.block_offset, run.block_offset + run.token_count)
            cache_from_runs[0, run.physical_block_id, :, 0, cache_slice, :] = key[
                layer_idx, token_slice
            ].permute(1, 0, 2)
            cache_from_runs[1, run.physical_block_id, :, 0, cache_slice, :] = value[
                layer_idx, token_slice
            ].permute(1, 0, 2)

        for token_idx in range(num_tokens):
            block, offset = int(blocks[token_idx]), int(offsets[token_idx])
            cache_tokenwise[0, block, :, 0, offset, :] = key[layer_idx, token_idx]
            cache_tokenwise[1, block, :, 0, offset, :] = value[layer_idx, token_idx]

        assert torch.equal(cache_from_runs, cache_tokenwise)


@pytest.mark.parametrize(
    ("blocks", "offsets", "request_ids"),
    [
        (torch.tensor([1]), torch.tensor([1024]), torch.tensor([0])),
        (torch.tensor([-1]), torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1, 1]), torch.tensor([0]), torch.tensor([0, 0])),
    ],
)
def test_plan_context_kv_runs_rejects_invalid_coordinates(
    blocks: torch.Tensor, offsets: torch.Tensor, request_ids: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        _plan_context_kv_runs(blocks, offsets, request_ids, group_id=0)


# ----------------------------------------------------------------------
# Proposer wiring: propose(), set_inputs_first_pass, rejection rewind
# ----------------------------------------------------------------------


def test_draft_ids_truncates_padded_rows_for_ids_and_logits() -> None:
    proposer = object.__new__(RBLNDFlashProposer)

    padded_ids = torch.arange(56, dtype=torch.int64)
    assert proposer._draft_ids(padded_ids, 35).tolist() == list(range(35))

    logits = torch.zeros(56, 16)
    logits[torch.arange(56), torch.arange(56) % 16] = 1.0
    ids = proposer._draft_ids(logits, 35)
    assert ids.shape == (35,)
    assert ids.tolist() == [row % 16 for row in range(35)]


def _make_first_pass_proposer(num_speculative_tokens: int = 7) -> RBLNDFlashProposer:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.device = torch.device("cpu")
    proposer.parallel_drafting_token_id = 99
    proposer.input_ids = torch.zeros(256, dtype=torch.int32)
    proposer.positions = torch.zeros(256, dtype=torch.int64)
    proposer._context_positions_buffer = torch.zeros(256, dtype=torch.int64)
    proposer._context_positions_cpu_buffer = torch.zeros(256, dtype=torch.int64)
    proposer._dflash_num_context = 0
    proposer._dflash_hidden_states = None
    return proposer


def test_set_inputs_first_pass_flattens_rank3_prefill_hidden() -> None:
    proposer = _make_first_pass_proposer()
    cad = SimpleNamespace(num_reqs=1, seq_lens=torch.tensor([10], dtype=torch.int32))
    # The runner's first-step (prefill) branch hands over 3-D
    # [1, tokens, fused_hidden] with more rows than the sliced token ids.
    hidden = torch.randn(1, 12, 6)

    num_tokens, token_indices = proposer.set_inputs_first_pass(
        target_token_ids=torch.zeros(10, dtype=torch.int32),
        next_token_ids=torch.tensor([7], dtype=torch.int32),
        target_positions=torch.arange(10, dtype=torch.int64),
        target_hidden_states=hidden,
        token_indices_to_sample=None,
        cad=cad,
    )

    assert num_tokens == 8
    assert proposer._dflash_num_context == 10
    assert proposer._dflash_hidden_states.shape == (10, 6)
    torch.testing.assert_close(
        proposer._dflash_hidden_states, hidden.reshape(-1, 6)[:10]
    )
    assert proposer._context_positions_cpu_buffer[:10].tolist() == list(range(10))
    assert proposer.input_ids[0].item() == 7
    assert proposer.input_ids[1:8].tolist() == [99] * 7
    assert proposer.positions[:8].tolist() == list(range(10, 18))
    assert token_indices.dtype == torch.int32
    assert token_indices.tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_set_inputs_first_pass_slices_unsliced_decode_hidden() -> None:
    proposer = _make_first_pass_proposer()
    cad = SimpleNamespace(
        num_reqs=2, seq_lens=torch.tensor([20, 30], dtype=torch.int32)
    )
    # The runner's spec-decode branch flattens but does NOT slice to the total
    # token count, so the hidden rows exceed the 16 context tokens.
    hidden = torch.randn(20, 6)

    num_tokens, token_indices = proposer.set_inputs_first_pass(
        target_token_ids=torch.zeros(16, dtype=torch.int32),
        next_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        target_positions=torch.arange(16, dtype=torch.int64),
        target_hidden_states=hidden,
        token_indices_to_sample=None,
        cad=cad,
    )

    assert num_tokens == 16
    assert proposer._dflash_hidden_states.shape == (16, 6)
    torch.testing.assert_close(proposer._dflash_hidden_states, hidden[:16])
    assert proposer.input_ids[:16].view(2, 8)[:, 0].tolist() == [3, 4]
    assert proposer.positions[:8].tolist() == list(range(20, 28))
    assert proposer.positions[8:16].tolist() == list(range(30, 38))
    assert token_indices.tolist() == list(range(1, 8)) + list(range(9, 16))


def test_dflash_preprocess_skips_discarded_hidden_states_copy() -> None:
    """The decode-path override must pad ids/positions like the base class but
    never touch self.hidden_states, whose padded copy propose() discards."""
    proposer = _make_first_pass_proposer()
    proposer.input_ids[:16] = torch.arange(16, dtype=torch.int32)
    proposer.positions[:16] = torch.arange(100, 116, dtype=torch.int64)
    # Any access raises: the override must not read the unused buffer.
    proposer.hidden_states = object()

    input_ids, positions, hidden, token_indices = proposer._preprocess(
        2, 4, 16, torch.arange(3, dtype=torch.int32), False
    )

    assert hidden is None
    assert input_ids.shape == (4, 8)
    assert input_ids[:2].reshape(-1).tolist() == list(range(16))
    assert input_ids[2:].abs().sum().item() == 0
    assert positions.shape == (4, 8)
    assert positions[:2].reshape(-1).tolist() == list(range(100, 116))
    # dynamo non-view contract, same as the base pad path.
    assert input_ids._base is None
    assert positions._base is None
    assert token_indices.tolist() == [0, 1, 2, 0]


def test_rewind_rejected_tokens_subtracts_once_for_aliased_views() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 7
    base = torch.tensor([100, 200], dtype=torch.int32)
    # rbln_model_runner slices ONE cpu tensor into both fields.
    cad = SimpleNamespace(seq_lens=base[:2], _seq_lens_cpu=base[:2])

    proposer._rewind_rejected_tokens(cad, torch.tensor([3, 5]))

    assert base.tolist() == [97, 195]


def test_rewind_rejected_tokens_corrects_separate_host_shadow() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 7
    cad = SimpleNamespace(
        seq_lens=torch.tensor([100, 200], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([100, 200], dtype=torch.int32),
    )

    proposer._rewind_rejected_tokens(cad, torch.tensor([3, 5]))

    assert cad.seq_lens.tolist() == [97, 195]
    assert cad._seq_lens_cpu.tolist() == [97, 195]


def test_rewind_rejected_tokens_noops_without_rejects_or_speculation() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 7
    cad = SimpleNamespace(
        seq_lens=torch.tensor([100], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([100], dtype=torch.int32),
    )
    proposer._rewind_rejected_tokens(cad, None)
    assert cad.seq_lens.tolist() == [100]

    proposer.num_speculative_tokens = 1
    proposer._rewind_rejected_tokens(cad, torch.tensor([3]))
    assert cad.seq_lens.tolist() == [100]


class _FakeMetadataBuilder:
    def __init__(self) -> None:
        self.build_calls: list[tuple] = []

    def build(self, *, common_attn_metadata, positions, is_prefill, batch_pad):
        self.build_calls.append((positions, is_prefill, batch_pad))
        return SimpleNamespace(name="fake-metadata")


class _FakeContextKVHelper:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_compile_context(self, compile_context) -> None:
        self.calls.append("compile_context")

    def set_group_slots(self, *args, **kwargs) -> None:
        self.calls.append("group_slots")


def _make_propose_proposer(
    num_reqs: int, num_reqs_padded: int, num_speculative_tokens: int = 7
):
    proposer = _make_first_pass_proposer(num_speculative_tokens)
    proposer.input_ids = torch.zeros(4096, dtype=torch.int32)
    proposer.positions = torch.zeros(4096, dtype=torch.int64)
    proposer._context_positions_buffer = torch.zeros(4096, dtype=torch.int64)
    proposer._context_positions_cpu_buffer = torch.zeros(4096, dtype=torch.int64)
    proposer._dflash_sliding_layer_names = set()
    proposer._dflash_sliding_window = None
    proposer._hidden_state_combiner = lambda hidden: hidden
    proposer.vllm_config = SimpleNamespace()
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=num_reqs),
        cache_config=SimpleNamespace(block_size=1024),
        kv_caches=[],
        kv_cache_bases=None,
        kv_cache_view_infos=None,
        compile_context=None,
        is_intermediate_chunked_prefill=False,
    )
    proposer.model = SimpleNamespace(
        model=SimpleNamespace(),
        precompute_and_store_context_kv=lambda *args, **kwargs: None,
    )
    builder = _FakeMetadataBuilder()
    proposer.draft_attn_groups = [
        SimpleNamespace(
            layer_names=["model.layers.0.self_attn.attn"],
            get_metadata_builder=lambda builder=builder: builder,
        )
    ]
    proposer._specialize_layer_attn_metadata = (
        lambda attn_group, attn_metadata, cad, num_reqs, num_reqs_padded: {}
    )
    proposer._resolve_group_slots = lambda per_group_metadata, cad: (
        {0: (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))},
        {},
        torch.zeros(0, dtype=torch.int64),
    )
    proposer._determine_draft_batch_padding = (
        lambda num_reqs_arg, num_tokens, is_prefill: (num_reqs_padded, None, None)
    )
    # Sentinel: the DFlash _preprocess override must never touch the base
    # class's hidden-states buffer. Any access raises immediately.
    proposer.hidden_states = object()
    return proposer


def _make_propose_cad(num_reqs: int, ctx_per_req: int, seq_len: int):
    counts = torch.full((num_reqs,), ctx_per_req, dtype=torch.int32)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(counts, dim=0)
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens,
        block_table_tensor=torch.zeros(num_reqs, 4, dtype=torch.int32),
    )


def test_propose_truncates_padded_draft_output_to_real_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5-request batch padded to the 8-request bucket must return [5, 7].

    The compiled graph samples every PADDED mask slot
    (num_reqs_padded * num_spec rows); the caller reshapes to the real batch.
    This is the seam that shipped broken: _draft_ids returned all padded rows
    and the final view raised for every non-bucket-exact request count.
    """
    num_reqs, num_reqs_padded, num_spec = 5, 8, 7
    ctx_per_req = 8
    proposer = _make_propose_proposer(num_reqs, num_reqs_padded, num_spec)
    cad = _make_propose_cad(num_reqs, ctx_per_req, seq_len=100)

    fake_helper = _FakeContextKVHelper()
    monkeypatch.setattr(dflash_mod, "attach_kv_cache_bindings", lambda *a, **k: None)
    monkeypatch.setattr(dflash_mod, "get_or_create_context_kv", lambda m: fake_helper)
    monkeypatch.setattr(
        dflash_mod, "build_kv_cache_forward_context_kwargs", lambda bases: {}
    )
    monkeypatch.setattr(
        dflash_mod, "set_forward_context", lambda *a, **k: contextlib.nullcontext()
    )

    seen: dict[str, int] = {}

    def fake_model_executable(*, input_ids, positions, token_indices_to_sample):
        seen["sampled_rows"] = int(token_indices_to_sample.shape[0])
        return None, torch.zeros(
            token_indices_to_sample.shape[0], dtype=torch.int64
        )

    proposer.model_executable = fake_model_executable

    num_context = num_reqs * ctx_per_req
    result = proposer.propose(
        target_token_ids=torch.zeros(num_context, dtype=torch.int32),
        target_positions=torch.arange(num_context, dtype=torch.int64),
        target_hidden_states=torch.randn(num_context, 16),
        next_token_ids=torch.arange(num_reqs, dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=cad,
    )

    # The padded sampling contract feeds the graph, the real batch comes back.
    assert seen["sampled_rows"] == num_reqs_padded * num_spec
    assert fake_helper.calls == ["compile_context", "group_slots"]
    assert isinstance(result, torch.Tensor)
    assert result.shape == (num_reqs, num_spec)


# ----------------------------------------------------------------------
# Configuration geometry and RoPE style plumbing
# ----------------------------------------------------------------------


def test_dflash_geometry_accepts_supported_configuration() -> None:
    _validate_dflash_geometry(
        max_num_tokens=512, max_batch_size=64, num_speculative_tokens=7
    )


def test_dflash_geometry_rejects_query_buffers_beyond_token_budget() -> None:
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _validate_dflash_geometry(
            max_num_tokens=511, max_batch_size=64, num_speculative_tokens=7
        )


def test_dflash_geometry_rejects_speculation_past_decode_profile() -> None:
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        _validate_dflash_geometry(
            max_num_tokens=8192, max_batch_size=1, num_speculative_tokens=8
        )


def test_dflash_geometry_rejects_budget_beyond_combiner_prefill_profile() -> None:
    # A 1024-token budget would feed the first long prefill chunk into the
    # fixed 512-row combiner profile at serve time; fail at init instead.
    with pytest.raises(ValueError, match="prefill profile"):
        _validate_dflash_geometry(
            max_num_tokens=1024, max_batch_size=1, num_speculative_tokens=7
        )


class _RotaryModule(nn.Module):
    def __init__(self, style: bool) -> None:
        super().__init__()
        self.is_neox_style = style


class _ModelWithRotary(nn.Module):
    def __init__(self, style: bool) -> None:
        super().__init__()
        self.rotary = _RotaryModule(style)


class _WrappedTarget(nn.Module):
    def __init__(self, style: bool) -> None:
        super().__init__()
        self._language_model = _ModelWithRotary(style)

    def get_language_model(self) -> nn.Module:
        return self._language_model


def test_target_rope_style_reads_first_rotary_module() -> None:
    assert _dflash_target_rope_is_neox_style(_ModelWithRotary(True)) is True
    assert _dflash_target_rope_is_neox_style(_ModelWithRotary(False)) is False
    assert _dflash_target_rope_is_neox_style(_WrappedTarget(False)) is False
    assert _dflash_target_rope_is_neox_style(nn.Linear(2, 2)) is None


def test_check_draft_rope_style_raises_only_on_mismatch() -> None:
    _check_draft_rope_style(_ModelWithRotary(True), True)
    _check_draft_rope_style(_ModelWithRotary(False), False)
    # A rope-free drafter has nothing to contradict.
    _check_draft_rope_style(nn.Linear(2, 2), True)

    with pytest.raises(ValueError, match="RoPE style"):
        _check_draft_rope_style(_ModelWithRotary(True), False)


# ----------------------------------------------------------------------
# Residual guard coverage: combiner dtype keys, config validation branches,
# causal-mismatch guard, SWA local-view bounds guard
# ----------------------------------------------------------------------


def test_hidden_state_combiner_keys_buffers_by_dtype() -> None:
    seen: list[tuple[torch.dtype, int]] = []

    def combine(inputs: torch.Tensor) -> torch.Tensor:
        seen.append((inputs.dtype, inputs.data_ptr()))
        return inputs

    helper = _BoundedHiddenStateCombiner(combine)
    out_fp32 = helper(torch.ones(3, 4, dtype=torch.float32))
    out_bf16 = helper(torch.ones(3, 4, dtype=torch.bfloat16))
    out_fp32_again = helper(torch.ones(3, 4, dtype=torch.float32))

    assert [dtype for dtype, _ in seen] == [
        torch.float32,
        torch.bfloat16,
        torch.float32,
    ]
    # Distinct staging buffers per dtype; same-dtype calls reuse one buffer.
    assert seen[0][1] != seen[1][1]
    assert seen[0][1] == seen[2][1]
    assert out_fp32.dtype == torch.float32
    assert out_bf16.dtype == torch.bfloat16
    assert out_fp32_again.dtype == torch.float32


def _make_configure_proposer(layer_types, sliding_window):
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.draft_attn_groups = [
        SimpleNamespace(
            layer_names=[
                "model.layers.0.self_attn.attn",
                "model.layers.1.self_attn.attn",
            ]
        )
    ]
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            layer_types=layer_types, sliding_window=sliding_window
        )
    )
    proposer._dflash_sliding_layer_names = set()
    proposer._dflash_sliding_window = None
    return proposer


def test_configure_dflash_layers_rejects_layer_type_length_mismatch() -> None:
    proposer = _make_configure_proposer(["full_attention"], sliding_window=None)
    with pytest.raises(ValueError, match="length"):
        proposer._configure_dflash_attention_layers()


def test_configure_dflash_layers_rejects_unknown_layer_type() -> None:
    proposer = _make_configure_proposer(
        ["full_attention", "weird_attention"], sliding_window=None
    )
    with pytest.raises(ValueError, match="Invalid DFlash layer type"):
        proposer._configure_dflash_attention_layers()


def test_configure_dflash_layers_requires_window_for_sliding_layers() -> None:
    proposer = _make_configure_proposer(
        ["sliding_attention", "full_attention"], sliding_window=None
    )
    with pytest.raises(ValueError, match="sliding_window"):
        proposer._configure_dflash_attention_layers()


def test_configure_dflash_layers_handles_absent_layer_types() -> None:
    proposer = _make_configure_proposer(None, sliding_window=None)
    proposer._configure_dflash_attention_layers()
    assert proposer._dflash_sliding_layer_names == set()
    assert proposer._dflash_sliding_window is None


def test_causal_guard_rejects_causal_full_attention_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dflash_mod.RBLNEagleProposer,
        "build_per_group_and_layer_attn_metadata",
        lambda self, cad, draft_index=0: (
            [],
            {"layer.full": SimpleNamespace(causal=True)},
        ),
    )
    proposer = object.__new__(RBLNDFlashProposer)
    proposer._dflash_sliding_layer_names = set()

    with pytest.raises(RuntimeError, match="causal=True"):
        proposer.build_per_group_and_layer_attn_metadata(None)


def test_causal_guard_allows_causal_sliding_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_layer = {"layer.sw": SimpleNamespace(causal=True)}
    monkeypatch.setattr(
        dflash_mod.RBLNEagleProposer,
        "build_per_group_and_layer_attn_metadata",
        lambda self, cad, draft_index=0: ([], per_layer),
    )
    proposer = object.__new__(RBLNDFlashProposer)
    proposer._dflash_sliding_layer_names = {"layer.sw"}

    _, result = proposer.build_per_group_and_layer_attn_metadata(None)

    assert result is per_layer


def test_swa_localization_rejects_undersized_block_table() -> None:
    proposer = object.__new__(RBLNDFlashProposer)
    proposer.num_speculative_tokens = 7
    proposer._dflash_sliding_layer_names = {"layer.sw"}
    proposer._dflash_sliding_window = 2048
    # Sequence 3497 with a 2048 window needs partitions 0..3; a 3-wide block
    # table cannot represent the local view and must fail loudly.
    metadata = SimpleNamespace(
        attn_masks=torch.zeros(1, 1, 1, 1, 49152),
        block_tables=torch.arange(100, 103).view(1, -1),
        seq_lens=torch.tensor([[3497]], dtype=torch.int32),
    )
    cad = SimpleNamespace(
        _seq_lens_cpu=torch.tensor([3497], dtype=torch.int32),
        block_table_tensor=metadata.block_tables,
    )
    group = SimpleNamespace(
        layer_names=["layer.sw"],
        kv_cache_spec=SimpleNamespace(block_size=1024),
    )

    with pytest.raises(ValueError, match="block table"):
        RBLNDFlashProposer._specialize_layer_attn_metadata(
            proposer,
            group,
            metadata,
            cad,
            num_reqs=1,
            num_reqs_padded=1,
        )
