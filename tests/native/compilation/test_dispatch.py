# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dispatcher's contract: first call per key goes through the compiled
# callable and records the graph, and later calls with that key run the recorded
# bytecode directly. The real thing needs a compiled model; here the Dynamo
# pieces are stubbed.

import json
import types
from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

import vllm_rbln.compilation.dispatch as dispatch
from vllm_rbln.compilation.dispatch import Dispatcher
from vllm_rbln.v1.worker.input_stager import StagedModelInputs


@pytest.fixture
def neutral_forward_context(monkeypatch):
    # The key's forward-context half is exercised separately; keep it constant
    # so the argument half is the only thing varying.
    monkeypatch.setattr(dispatch, "_forward_context_key", lambda: (False, None, False))


def make_target():
    calls: list[tuple] = []

    def target(x, y=None):
        calls.append(("target", x.shape if x is not None else None))
        return "eager"

    return target, calls


def recorder(calls, result):
    def compiled(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    return compiled


def install_entries(monkeypatch, codes):
    """Stub Dynamo's cache so the head of the list is `codes[-1]`."""
    entries = [
        SimpleNamespace(code=c, compile_id=f"0/{i}")
        for i, c in enumerate(reversed(codes))
    ]
    monkeypatch.setattr(
        torch._dynamo.eval_frame, "_debug_get_cache_entry_list", lambda code: entries
    )


class TestDispatch:
    @pytest.fixture(autouse=True)
    def _neutral(self, neutral_forward_context):
        return None

    def test_first_call_goes_through_the_compiled_callable(self, monkeypatch):
        target, _ = make_target()
        seen: list = []
        d = Dispatcher(target, recorder(seen, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        assert d(torch.zeros(1, 4)) == "compiled"
        assert len(seen) == 1

    def test_second_call_with_same_key_runs_the_recorded_code(self, monkeypatch):
        target, calls = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x)
        assert d(x) == "eager"  # the recorded code object is target's own here
        assert len(compiled_calls) == 1
        assert len(calls) == 1

    def test_new_shape_is_compiled_not_rejected(self, monkeypatch):
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        d(torch.zeros(1, 4))
        d(torch.zeros(1, 8))  # a bucket warm-up never covered
        assert len(compiled_calls) == 2

    def test_optional_argument_is_part_of_the_key(self, monkeypatch):
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x, y=None)
        d(x, y=torch.zeros(1))
        assert len(compiled_calls) == 2

    def test_forward_context_is_part_of_the_key(self, monkeypatch):
        # Identical inputs in a different phase must not share a graph.
        target, _ = make_target()
        compiled_calls: list = []
        phase = [True]
        monkeypatch.setattr(
            dispatch, "_forward_context_key", lambda: (phase[0], None, False)
        )
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x)
        phase[0] = False
        d(x)
        assert len(compiled_calls) == 2

    def test_no_cache_entry_keeps_using_the_compiled_callable(self, monkeypatch):
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [])
        x = torch.zeros(1, 4)
        d(x)
        d(x)
        assert len(compiled_calls) == 2

    def test_rejects_a_target_that_is_neither_function_nor_method(self):
        with pytest.raises(TypeError, match="function or a bound method"):
            Dispatcher(torch.nn.Linear(2, 2), lambda *a, **k: None)

    def test_scalar_arguments_are_keyed_by_value(self, monkeypatch):
        # Dynamo guards on a float argument, so two values need two keys; a key
        # built from the type name would serve the first graph for both.
        def target(x, gain=2.0):
            return "eager"

        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x, gain=2.0)
        d(x, gain=5.0)
        assert len(compiled_calls) == 2

    def test_tensors_of_one_shape_on_two_devices_are_two_keys(self, monkeypatch):
        # token_type_ids is built without a device, the staged buffers take the
        # stager's, so a call's tensors are not all on one device.
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        d(torch.zeros(1, 4))
        d(torch.zeros(1, 4, device="meta"))
        assert len(compiled_calls) == 2

    def test_tensors_of_one_shape_with_two_strides_are_two_keys(self, monkeypatch):
        # Same shape and dtype either way; the strides are (4, 1) and (1, 2).
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        d(torch.zeros(4, 4)[:, :2])
        d(torch.zeros(2, 4).t())
        assert len(compiled_calls) == 2

    def test_a_bool_and_the_int_it_equals_are_two_keys(self, monkeypatch):
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x, y=True)
        d(x, y=1)
        assert len(compiled_calls) == 2

    def test_intermediate_tensors_are_keyed_like_the_tensors_they_hold(
        self, monkeypatch
    ):
        # The one slot InputStager passes through: a warm-up run puts a view here,
        # a real step the pipeline group's receive buffer.
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x, y=IntermediateTensors({"hidden_states": torch.zeros(4, 4)[:, :2]}))
        d(x, y=IntermediateTensors({"hidden_states": torch.zeros(2, 4).t()}))
        assert len(compiled_calls) == 2

    def test_rejects_a_mapping_that_is_not_intermediate_tensors(self, monkeypatch):
        target, _ = make_target()
        d = Dispatcher(target, recorder([], "compiled"))
        install_entries(monkeypatch, [target.__code__])
        with pytest.raises(TypeError, match="cannot key a dict argument"):
            d(torch.zeros(1, 4), y={"hidden_states": torch.zeros(1, 4)})

    def test_rejects_an_argument_it_cannot_key(self, monkeypatch):
        target, _ = make_target()
        d = Dispatcher(target, recorder([], "compiled"))
        install_entries(monkeypatch, [target.__code__])
        with pytest.raises(TypeError, match="cannot key a slice argument"):
            d(slice(0, 2))

    def test_dispatches_a_bound_method_without_touching_its_class(self, monkeypatch):
        class Model:
            def __init__(self, bias):
                self.bias = bias

            def forward(self, x):
                return ("eager", self.bias)

        first, second = Model(1), Model(2)
        original_code = Model.forward.__code__
        d = Dispatcher(second.forward, recorder([], "compiled"))
        install_entries(monkeypatch, [original_code])
        x = torch.zeros(1, 4)
        d(x)
        assert d(x) == ("eager", 2)  # bound to `second`, not the class
        assert Model.forward.__code__ is original_code
        assert first.forward(x) == ("eager", 1)

    def test_the_dispatched_region_lands_in_a_profile(self, monkeypatch, tmp_path):
        # Bypassing the frame eval bypasses Dynamo's own annotation, so a dispatched
        # call is a blank in the trace unless the dispatcher names it -- and it must
        # do so without VLLM_CUSTOM_SCOPES_FOR_PROFILING.
        target, _ = make_target()
        d = Dispatcher(target, recorder([], "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            d(x)
        trace = tmp_path / "trace.json"
        prof.export_chrome_trace(str(trace))
        names = {e["name"] for e in json.loads(trace.read_text())["traceEvents"]}
        assert "Dispatched Region: 0/0" in names

    def test_clone_differs_from_the_target_only_in_its_code(self, monkeypatch):
        # The FunctionType constructor takes five of the function's slots; the
        # slots are enumerated off the type so a new one fails here rather than
        # being dropped silently by _clone.
        def target(x, y=None, *, scale=3.0):
            """docstring."""
            return "eager"

        setattr(target, "tagged", "attr")  # noqa: B010
        d = Dispatcher(target, recorder([], "compiled"))
        install_entries(monkeypatch, [target.__code__])
        d(torch.zeros(1, 4))
        clone, _annotation = next(iter(d._graphs.values()))

        slots = [
            name
            for name, member in vars(types.FunctionType).items()
            if isinstance(
                member, (types.GetSetDescriptorType, types.MemberDescriptorType)
            )
        ]
        assert "__kwdefaults__" in slots  # the one that changes results
        missing = object()
        assert [
            name
            for name in slots
            if name not in ("__code__", "__builtins__")
            and getattr(clone, name, missing) != getattr(target, name, missing)
        ] == []


# The situations one runner process stages, as (kwargs, forward-context key).
# Faithful to what reaches a dispatched model: input_stager.StagedModelInputs
# .as_kwargs() names the arguments, rbln_model_runner._prepare_model_inputs
# shapes them (int32 ids, int64 positions, [num_reqs, query_len]), and
# dispatch._forward_context_key reads (is_prefill, dp pad width, kv connector).
#
# Only variation a single process can produce is listed. The batch buckets and
# the DP pad width change from step to step; the query length grows with
# speculative tokens; intermediate tensors arrive on a non-first pipeline rank.
# has_kv_transfer_group() is fixed for the process's lifetime, so it cannot
# collide with itself and is not a case here.
def _staged(num_reqs, query_len, *, token_indices=0, intermediate=False):
    shape = (num_reqs, query_len)
    tensors = None
    if intermediate:
        tensors = IntermediateTensors(
            {"hidden_states": torch.zeros(num_reqs, query_len, 8)}
        )
    return {
        "input_ids": torch.zeros(shape, dtype=torch.int32),
        "positions": torch.zeros(shape, dtype=torch.int64),
        "intermediate_tensors": tensors,
        "inputs_embeds": None,
        "token_indices": (
            torch.zeros(token_indices, dtype=torch.int32) if token_indices else None
        ),
    }


PREFILL = (True, None, False)
DECODE = (False, None, False)

SITUATIONS: dict[str, tuple[dict, tuple]] = {
    "prefill": (_staged(1, 512, token_indices=1), PREFILL),
    "decode-b1": (_staged(1, 1), DECODE),
    "decode-b2": (_staged(2, 1), DECODE),
    "decode-b4": (_staged(4, 1), DECODE),
    "decode-b8": (_staged(8, 1), DECODE),
    # A drafted step carries 1 + num_spec_tokens columns.
    "decode-b4-spec4": (_staged(4, 5), DECODE),
    # Same staged tensors as decode-b4; only the DP-agreed pad width differs,
    # and the graph reads it as dp_metadata.max_pads_across_dp.shape[0].
    "decode-b4-dp8": (_staged(4, 1), (False, 8, False)),
    "decode-b4-dp16": (_staged(4, 1), (False, 16, False)),
    # Same again, with a pipeline rank's inbound activations.
    "decode-b4-pp": (_staged(4, 1, intermediate=True), DECODE),
}


def key_of(monkeypatch, kwargs, context_key):
    """The one key a fresh Dispatcher registers for a single call."""

    def model_wrapper(
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        token_indices=None,
    ):
        return "eager"

    monkeypatch.setattr(dispatch, "_forward_context_key", lambda: context_key)
    d = Dispatcher(model_wrapper, recorder([], "compiled"))
    install_entries(monkeypatch, [model_wrapper.__code__])
    d(**kwargs)
    (key,) = d._graphs
    return key


class TestKeyDiscrimination:
    """The key must separate every situation that needs its own graph.

    Key completeness is the one precondition Dispatcher cannot check: a key
    coarser than Dynamo's guards serves one graph for two situations, forever
    and unreported, and no count of compiled graphs reveals it because the
    second graph is never requested. What this class can do is pin the
    situations down as an explicit list and require the key to tell them apart
    -- so a change that collapses two of them fails here instead of in a model.
    """

    def test_distinct_situations_never_share_a_key(self, monkeypatch):
        keys = {
            name: key_of(monkeypatch, kwargs, context_key)
            for name, (kwargs, context_key) in SITUATIONS.items()
        }
        by_key: dict[tuple, list[str]] = {}
        for name, key in keys.items():
            by_key.setdefault(key, []).append(name)
        collided = sorted(names for names in by_key.values() if len(names) > 1)
        assert not collided, f"situations sharing one graph: {collided}"

    def test_the_same_situation_reuses_its_key(self, monkeypatch):
        # The other direction: a key finer than the guards would recompile every
        # step. Fresh tensors of the same shape and dtype must land on one key.
        for name, (kwargs, context_key) in SITUATIONS.items():
            first = key_of(monkeypatch, kwargs, context_key)
            again = key_of(monkeypatch, {**kwargs}, context_key)
            assert first == again, f"{name} keyed two identical calls apart"
            rebuilt, _ = SITUATIONS[name]
            assert first == key_of(monkeypatch, rebuilt, context_key), (
                f"{name} keys on tensor identity, not shape/dtype"
            )

    def test_the_situations_cover_every_dispatched_argument(self):
        # Drift alarm: a new argument on the model call is a new axis the key may
        # have to carry, and a table that does not mention it would keep passing.
        staged = StagedModelInputs(
            input_ids=torch.zeros(1, 1, dtype=torch.int32),
            positions=torch.zeros(1, 1, dtype=torch.int64),
            intermediate_tensors=None,
            inputs_embeds=None,
            token_indices=None,
        )
        for kwargs, _context_key in SITUATIONS.values():
            assert set(kwargs) == set(staged.as_kwargs())


@pytest.mark.use_device
class TestDynamoContract:
    """Drift alarm against Dynamo. Everything above stubs the cache lookup, so
    what _register reads out of it and what _clone does with the bytecode is
    checked here against the real thing -- on the eager backend, since none of
    it depends on how the graph was lowered."""

    def test_the_clone_runs_the_graph_dynamo_recorded(self):
        compiles: list = []

        def backend(gm, example_inputs):
            compiles.append(gm)
            return gm.forward

        scale = 2.0  # a freevar, so the clone needs the target's __closure__

        def target(x, y=None):
            out = x * scale
            return out if y is None else out + y

        d = Dispatcher(target, torch.compile(target, backend=backend, fullgraph=True))
        x = torch.ones(2, 4)
        assert torch.equal(d(x), target(x))
        assert torch.equal(d(x), target(x))  # the clone, not the compiled callable
        assert len(compiles) == 1
        clone, _annotation = next(iter(d._graphs.values()))
        assert clone.__code__ is not target.__code__

    def test_the_cache_list_head_is_the_graph_the_last_call_used(self):
        # _register takes entries[0], which is only the graph the call it follows
        # used while Dynamo keeps the list MRU-ordered.
        def target(x):
            return x + 1

        compiled = torch.compile(target, backend="eager", dynamic=False, fullgraph=True)
        entries_of = torch._dynamo.eval_frame._debug_get_cache_entry_list
        compiled(torch.zeros(4))
        four = entries_of(target.__code__)[0]
        assert isinstance(four.code, types.CodeType)  # _clone's input
        assert four.compile_id is not None  # the profiler annotation
        compiled(torch.zeros(8))
        assert entries_of(target.__code__)[0] is not four
        compiled(torch.zeros(4))
        assert entries_of(target.__code__)[0] is four


class TestOutsideForwardContext:
    def test_dispatches_a_target_called_without_a_forward_context(self, monkeypatch):
        # The samplers are compiled the same way but run after execute_model has
        # left the forward context, so reading it unconditionally would raise.
        target, calls = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [target.__code__])
        x = torch.zeros(1, 4)
        d(x)
        assert d(x) == "eager"
        assert len(compiled_calls) == 1
        assert len(calls) == 1


class TestForwardContextKey:
    def test_reads_prefill_max_pad_and_connector(self, monkeypatch):
        ctx = SimpleNamespace(
            attn_metadata={"layer": SimpleNamespace(is_prefill=True)},
            dp_metadata=SimpleNamespace(max_pads_across_dp=torch.zeros(7)),
        )
        monkeypatch.setattr(dispatch, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(dispatch, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(dispatch, "has_kv_transfer_group", lambda: True)
        assert dispatch._forward_context_key() == (True, 7, True)

    def test_tolerates_a_context_without_dp_or_attention(self, monkeypatch):
        ctx = SimpleNamespace(attn_metadata={}, dp_metadata=None)
        monkeypatch.setattr(dispatch, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(dispatch, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(dispatch, "has_kv_transfer_group", lambda: False)
        assert dispatch._forward_context_key() == (None, None, False)
