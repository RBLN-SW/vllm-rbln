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
# callable and records the graph, later calls with that key run the recorded
# bytecode directly, and a key that maps to two graphs is an error. The real
# thing needs a compiled model; here the Dynamo pieces are stubbed.

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.compilation.dispatch as dispatch
from vllm_rbln.compilation.dispatch import Dispatcher


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
    entries = [SimpleNamespace(code=c) for c in reversed(codes)]
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

    def test_two_graphs_for_one_key_raises(self, monkeypatch):
        # A key coarser than Dynamo's guards shows up as one key registering two
        # code objects; serving either would be wrong, so it has to raise.
        target, _ = make_target()

        def other_graph(x, y=None):  # a distinct code object, unlike make_target()
            return "other"

        d = Dispatcher(target, lambda *a, **k: "compiled")
        install_entries(monkeypatch, [target.__code__])
        d._register(("key",))
        install_entries(monkeypatch, [other_graph.__code__])
        with pytest.raises(RuntimeError, match="two different graphs"):
            d._register(("key",))

    def test_no_cache_entry_keeps_using_the_compiled_callable(self, monkeypatch):
        target, _ = make_target()
        compiled_calls: list = []
        d = Dispatcher(target, recorder(compiled_calls, "compiled"))
        install_entries(monkeypatch, [])
        x = torch.zeros(1, 4)
        d(x)
        d(x)
        assert len(compiled_calls) == 2

    def test_rejects_a_target_without_code(self):
        with pytest.raises(TypeError, match="plain function"):
            Dispatcher(torch.nn.Linear(2, 2), lambda *a, **k: None)


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
