from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.patches.qwen3_dflash as qwen3_dflash_patch
from vllm_rbln.patches.qwen3_dflash import (
    _apply_rope,
    _ContextKVPrecompute,
    _ContextKVRun,
    _DFlashForwardGraph,
    _install_stable_runtime_inputs,
    _plan_context_kv_runs,
    _rms_norm,
    _RuntimeInputBindingCache,
    _StableRuntimeGraph,
)
from vllm_rbln.v1.spec_decode.dflash import RBLNDFlashProposer


def test_apply_rope_broadcasts_over_kv_heads() -> None:
    key = torch.randn(3, 8, 128)
    cos = torch.randn(3, 128)
    sin = torch.randn(3, 128)

    actual = _apply_rope(key, cos, sin)
    expected = _apply_rope(key, cos.unsqueeze(1), sin.unsqueeze(1))

    torch.testing.assert_close(actual, expected)


def test_exact_length_runtime_buffers_keep_stable_addresses() -> None:
    model = SimpleNamespace(
        _fused_kv_weight=torch.empty(16, 32),
        _head_dim=4,
    )
    helper = _ContextKVPrecompute(model)

    first_inputs = helper._get_run_inputs(8, torch.bfloat16, torch.device("cpu"))
    second_inputs = helper._get_run_inputs(8, torch.bfloat16, torch.device("cpu"))

    assert tuple(t.data_ptr() for t in first_inputs) == tuple(
        t.data_ptr() for t in second_inputs
    )


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
    )
    helper = _ContextKVPrecompute(model)
    compile_calls: list[dict] = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(qwen3_dflash_patch.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    monkeypatch.setattr(qwen3_dflash_patch, "rbln_compile", fake_compile)

    layer0 = helper._get_graph(0, 8)
    layer1 = helper._get_graph(1, 8)

    assert layer0 is not layer1
    assert len(compile_calls) == 2
    assert all(call["use_cache"] is False for call in compile_calls)


def test_context_kv_projection_graph_does_not_take_cache_input() -> None:
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
    expected_key = _apply_rope(
        _rms_norm(fused[:, 0], model._k_norm_weights[0], model._rms_norm_eps),
        cos,
        sin,
    ).permute(1, 0, 2)
    expected_value = fused[:, 1].permute(1, 0, 2)

    assert key.shape == (2, 3, 4)
    assert value.shape == (2, 3, 4)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


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

    reused = bindings.execute(
        runtime,
        (10, 11, 12, 13),
        get_device_addrs=lambda pointer: [pointer + 1000],
    )

    assert reused is True
    assert events == [
        ("begin",),
        ("prepare", {0: 10, 3: 13}, {}),
        ("patch", 1, [1011]),
        ("patch", 2, [1012]),
        ("end",),
        ("run",),
        ("reports",),
    ]


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

    assert bindings.execute(
        runtime,
        layer0,
        get_device_addrs=lambda pointer: [pointer + 1000],
    )
    assert ("prepare", {1: 11}, {}) in events

    events.clear()
    assert bindings.execute(
        runtime,
        layer1,
        get_device_addrs=lambda pointer: [pointer + 1000],
    )
    assert events == [
        ("begin",),
        ("prepare", {0: 10, 3: 13}, {}),
        ("patch", 1, [1021]),
        ("patch", 2, [1022]),
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
    assert events[-2:] == [("run",), ("reports",)]


def test_stable_dynamo_runtime_prepares_metadata_but_reuses_cache_binding() -> None:
    events: list[tuple] = []

    class Handle:
        def begin_io_patch_batch(self) -> None:
            events.append(("begin",))

        def prepare_inputs(self, device_inputs, cpu_inputs) -> None:
            events.append(("prepare", device_inputs, cpu_inputs))

        def update_input_addr(self, index, addresses) -> None:
            events.append(("patch", index, addresses))

        def end_io_patch_batch(self) -> None:
            events.append(("end",))

        def run(self) -> None:
            events.append(("device_run",))

    original_calls: list[tuple] = []

    def original_run(*inputs, out, **kwargs):
        original_calls.append(inputs)
        return [torch.tensor(len(original_calls))]

    runtime = SimpleNamespace(
        _num_inputs=3,
        _input_name_to_index={
            "l_input_ids_": 0,
            "l_kv_caches_0_": 1,
            "l_positions_": 2,
        },
        _runtime_utils=SimpleNamespace(
            prepare_inputs=lambda *args, **kwargs: list(args)
        ),
        _runtime_handle=Handle(),
        _capture_reports_if_needed=lambda: events.append(("reports",)),
        run=original_run,
    )
    assert _install_stable_runtime_inputs(
        runtime,
        get_device_addrs=lambda pointer: [pointer + 1000],
        tensor_is_supported=lambda tensor: True,
    )
    input_ids, cache, positions = torch.empty(1), torch.empty(2), torch.empty(3)

    first = runtime.run(input_ids, cache, positions, out=None)
    reused = runtime.run(input_ids, cache, positions, out=None)

    assert first is reused
    assert len(original_calls) == 1
    prepare = next(event for event in events if event[0] == "prepare")
    assert prepare[1] == {0: input_ids.data_ptr(), 2: positions.data_ptr()}
    assert all(event[0] != "patch" for event in events)
    assert events[-2:] == [("device_run",), ("reports",)]


def test_stable_dynamo_runtime_treats_unknown_metadata_as_dynamic() -> None:
    prepared: list[dict[int, int]] = []

    class Handle:
        def begin_io_patch_batch(self) -> None:
            pass

        def prepare_inputs(self, device_inputs, cpu_inputs) -> None:
            prepared.append(device_inputs)

        def end_io_patch_batch(self) -> None:
            pass

        def run(self) -> None:
            pass

    def original_run(*inputs, out, **kwargs):
        return [torch.tensor(1)]

    runtime = SimpleNamespace(
        _input_name_to_index={
            "l_input_ids_": 0,
            "l_kv_caches_0_": 1,
            "l_slot_mapping_": 2,
        },
        _runtime_utils=SimpleNamespace(
            prepare_inputs=lambda *args, **kwargs: list(args)
        ),
        _runtime_handle=Handle(),
        run=original_run,
    )
    assert _install_stable_runtime_inputs(
        runtime,
        get_device_addrs=lambda pointer: [pointer + 1000],
        tensor_is_supported=lambda tensor: True,
    )
    input_ids, cache, slot_mapping = torch.empty(1), torch.empty(2), torch.empty(3)

    runtime.run(input_ids, cache, slot_mapping, out=None)
    runtime.run(input_ids, cache, slot_mapping, out=None)

    assert prepared == [{0: input_ids.data_ptr(), 2: slot_mapping.data_ptr()}]


def test_stable_dynamo_runtime_detects_generic_rank6_kv_cache() -> None:
    prepared: list[dict[int, int]] = []

    class Handle:
        def begin_io_patch_batch(self) -> None:
            pass

        def prepare_inputs(self, device_inputs, cpu_inputs) -> None:
            prepared.append(device_inputs)

        def end_io_patch_batch(self) -> None:
            pass

        def run(self) -> None:
            pass

    def original_run(*inputs, out, **kwargs):
        return [torch.tensor(1)]

    runtime = SimpleNamespace(
        _input_name_to_index={"args_0": 0, "args_1": 1, "args_2": 2},
        _input_profile=[
            SimpleNamespace(shape=(1, 8)),
            SimpleNamespace(shape=(2, 62, 8, 1, 1024, 128)),
            SimpleNamespace(shape=(1,)),
        ],
        _runtime_utils=SimpleNamespace(
            prepare_inputs=lambda *args, **kwargs: list(args)
        ),
        _runtime_handle=Handle(),
        run=original_run,
    )
    assert _install_stable_runtime_inputs(
        runtime,
        get_device_addrs=lambda pointer: [pointer + 1000],
        tensor_is_supported=lambda tensor: True,
    )
    input_ids, cache, slots = torch.empty(1), torch.empty(2), torch.empty(3)

    runtime.run(input_ids, cache, slots, out=None)
    runtime.run(input_ids, cache, slots, out=None)

    assert prepared == [{0: input_ids.data_ptr(), 2: slots.data_ptr()}]


def test_dflash_forward_graph_installs_runtime_reuse_after_compile() -> None:
    def make_runtime():
        return SimpleNamespace(
            _input_name_to_index={"l_kv_caches_0_": 0},
            _runtime_utils=SimpleNamespace(),
            _runtime_handle=SimpleNamespace(),
            run=lambda *args, **kwargs: None,
        )

    runtimes = [make_runtime(), make_runtime()]
    runtime_holder: list = []

    def compiled(value):
        runtime_holder.extend(runtimes)
        return value + 1

    graph = _DFlashForwardGraph(compiled, runtime_holder)

    assert graph(4) == 5
    assert all(runtime._rbln_dflash_stable_inputs is not None for runtime in runtimes)


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


@pytest.mark.parametrize("length", [8, 64, 512])
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
