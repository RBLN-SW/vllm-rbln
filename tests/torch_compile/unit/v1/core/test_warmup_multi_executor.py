# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warm-up multi-executor compile failure (fsw-inference#325).

Symptom
-------
With a v1 KV connector active and the device-tensor path on
(``VLLM_RBLN_USE_DEVICE_TENSOR=1``, which ``RblnNixlConnector`` forces),
model warm-up can abort with::

    RuntimeError: All executors must have the same name if name is not specified
    torch._dynamo.exc.BackendCompilerFailed: backend='logged_rbln_backend' raised

Mechanism
---------
The assertion lives in rebel's ``CompiledModelHolder._select_executor_idx``.
``rbln_backend`` calls ``create_runtime(name=None)``, which only works when a
compiled model's executors all share one name (it treats them as input-size
buckets of one graph). Executor name == the per-graph hash. So the abort means
one compiled model held executors built from *different* graphs (distinct
names) — i.e. the compiler split one attention forward into multiple
distinct-named executors. Whether that split happens depends on the
rebel-compiler / driver version and the model: it reproduces on the cluster
image (rebel dev198, MiniMax-M2.5) but not on rebel dev182/dev214 + Qwen3,
where every warm-up compile yields exactly one executor.

Fix (this branch)
-----------------
Refined fix in ``vllm_rbln/model_executor/layers/attention/attention.py``: only
the two connector hook calls (``wait_for_layer_load`` / ``save_kv_layer``) are
routed through ``@torch._dynamo.disable`` helpers, so dynamo graph-breaks just
at those calls while the attention op stays in the traced graph (still compiled
onto the NPU). The hooks are only reached on the runtime path (connector has
bound metadata), so warm-up (dummy metadata None) and normal serving incur no
graph break. This is lighter than the upstream "fix B" that disables the whole
wrapper (which graph-breaks every attention layer).

Test groups
-----------
1. The rebel assertion surface (mock compiled model) — the failure contract.
2. dynamo behaviour of the *upstream* wrapper in a simplified setting.
3. The refined fix — only the hook helpers are dynamo-disabled; attention
   bindings stay traceable; the runtime path still fires wait/save.
"""

from __future__ import annotations

import pytest

# ===========================================================================
# Group 1 — the rebel assertion surface (mock compiled model, no NPU)
# ===========================================================================

rebel = pytest.importorskip("rebel")
from rebel.compiled_model_holder import CompiledModelHolder  # noqa: E402

_MULTI_EXECUTOR_MSG = "All executors must have the same name if name is not specified"


class _FakeExecutor:
    def __init__(self, context_id: str):
        self._context_id = context_id

    def get_context_id(self) -> str:
        return self._context_id


class _FakeCompiledModel:
    """Stands in for ``PyRblnCompiledModel`` with just the surface
    ``_select_executor_idx`` touches: executor names + per-idx context ids."""

    def __init__(self, names: list[str], context_ids: list[str] | None = None):
        self._names = names
        ctx = context_ids if context_ids is not None else ["ctx0"] * len(names)
        self._executors = [_FakeExecutor(c) for c in ctx]
        self.meta = {}

    def get_executor_names(self) -> list[str]:
        return self._names

    def get_executor_count(self) -> int:
        return len(self._names)

    def get_executor_by_idx(self, idx: int) -> _FakeExecutor:
        return self._executors[idx]


def _holder(names: list[str], context_ids: list[str] | None = None) -> CompiledModelHolder:
    return CompiledModelHolder(_FakeCompiledModel(names, context_ids))


class TestRebelMultiExecutorContract:
    """``_select_executor_idx(name=None, ...)`` is what ``create_runtime``
    calls during warm-up. These pin exactly when it aborts."""

    def test_mixed_executor_names_reproduce_the_warmup_abort(self):
        # Two graphs (distinct hashes) folded into one compiled model: the
        # exact shape of the fsw-inference#325 failure.
        holder = _holder(["72db1d", "ed57dc"])

        with pytest.raises(RuntimeError, match=_MULTI_EXECUTOR_MSG):
            holder._select_executor_idx(None, None)

    def test_default_named_executors_are_safe(self):
        # Multiple executors that all share a name are just input-size buckets
        # of one graph; the nameless select succeeds.
        holder = _holder(["default", "default"], context_ids=["c", "c"])

        context_id, executor_idx = holder._select_executor_idx(None, None)

        assert context_id == "c"
        assert executor_idx == [0, 1]

    def test_single_executor_is_safe(self):
        holder = _holder(["72db1d"], context_ids=["c"])

        context_id, executor_idx = holder._select_executor_idx(None, None)

        assert context_id == "c"
        assert executor_idx == [0]

    def test_explicit_name_disambiguates_mixed_model(self):
        # The escape hatch: with a name supplied the assertion no longer
        # applies — a fix that threads a stable name through would also work.
        holder = _holder(["72db1d", "ed57dc"], context_ids=["c0", "c1"])

        context_id, executor_idx = holder._select_executor_idx("ed57dc", None)

        assert context_id == "c1"
        assert executor_idx == [1]


# ===========================================================================
# Group 2 — dynamo behaviour of the upstream warm-up wrapper (CPU, no NPU)
# ===========================================================================
#
# In a simplified single-attention setting the upstream wrapper is
# constant-folded by dynamo (no graph break). On the real model/compiler it
# nonetheless feeds the split, which is why fix B (Group 3) excludes it from
# the graph entirely rather than relying on this folding.

import torch  # noqa: E402
import vllm.model_executor.layers.attention.attention as vllm_attn_mod  # noqa: E402
import vllm.model_executor.layers.attention.kv_transfer_utils as kvt  # noqa: E402


class _GraphCounter:
    """A torch.compile backend invoked once per FX graph dynamo emits; >1
    call means a graph break occurred."""

    def __init__(self):
        self.node_counts: list[int] = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.node_counts.append(sum(1 for _ in gm.graph.nodes))
        return gm.forward


class _Conn:
    def __init__(self, has_meta: bool):
        self._has_meta = has_meta
        self.waited = 0
        self.saved = 0

    def has_connector_metadata(self) -> bool:
        return self._has_meta

    def wait_for_layer_load(self, layer_name):
        self.waited += 1

    def save_kv_layer(self, layer_name, kv_cache, attn_metadata):
        self.saved += 1


def _attn_fn(query, key, value, layer_name, kv_cache):
    kv_cache.add_(key.sum())
    return query + value


@pytest.fixture
def patch_kv_transfer(monkeypatch):
    """Install a fake KV-transfer group + attention context, return a factory
    that builds the monkeypatched wrapper for a (has_group, has_meta) case."""

    def _factory(has_group: bool, has_meta: bool):
        conn = _Conn(has_meta)
        attn_metadata = object()
        state = {"kv_cache": None}

        def get_attention_context(layer_name):
            return attn_metadata, None, state["kv_cache"], None

        monkeypatch.setattr(
            vllm_attn_mod, "get_attention_context", get_attention_context
        )
        monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: has_group)
        monkeypatch.setattr(kvt, "is_v1_kv_transfer_group", lambda: True)
        monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: conn)

        def fn(query, key, value, layer_name, kv_cache):
            return _attn_fn(query, key, value, layer_name, kv_cache)

        return kvt.maybe_transfer_kv_layer(fn), conn, state

    return _factory


def _compile_and_run(wrapped, state):
    torch._dynamo.reset()
    backend = _GraphCounter()
    q, k, v = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)
    state["kv_cache"] = torch.zeros(2, 4)
    compiled = torch.compile(wrapped, backend=backend, fullgraph=True)
    compiled(q, k, v, "model.layers.0.attn", state["kv_cache"])
    return backend


class TestWarmupWrapperDynamo:
    """In the simplified setting the upstream wrapper folds to one clean
    graph during warm-up (metadata None)."""

    def _baseline_nodes(self) -> int:
        torch._dynamo.reset()
        backend = _GraphCounter()
        q, k, v = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)
        kv = torch.zeros(2, 4)
        compiled = torch.compile(_attn_fn, backend=backend, fullgraph=True)
        compiled(q, k, v, "model.layers.0.attn", kv)
        assert len(backend.node_counts) == 1
        return backend.node_counts[0]

    def test_warmup_path_is_single_clean_graph(self, patch_kv_transfer):
        wrapped, conn, state = patch_kv_transfer(has_group=True, has_meta=False)

        backend = _compile_and_run(wrapped, state)

        assert len(backend.node_counts) == 1
        assert backend.node_counts[0] == self._baseline_nodes()
        assert conn.waited == 0 and conn.saved == 0  # warm-up short-circuit

    def test_no_group_is_noop_single_graph(self, patch_kv_transfer):
        wrapped, conn, state = patch_kv_transfer(has_group=False, has_meta=False)

        backend = _compile_and_run(wrapped, state)

        assert len(backend.node_counts) == 1
        assert backend.node_counts[0] == self._baseline_nodes()
        assert conn.waited == 0 and conn.saved == 0

    def test_runtime_path_invokes_wait_and_save(self, patch_kv_transfer):
        """Runtime path (metadata present): wait/save run eagerly — fix B must
        preserve this, only excluding the wrapper from the *traced* graph."""
        wrapped, conn, state = patch_kv_transfer(has_group=True, has_meta=True)

        torch._dynamo.reset()
        state["kv_cache"] = torch.zeros(2, 4)
        wrapped(
            torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4),
            "model.layers.0.attn", state["kv_cache"],
        )

        assert conn.waited == 1
        assert conn.saved == 1


# ===========================================================================
# Group 3 — fix B regression guard
# ===========================================================================


class TestRefinedHookDisable:
    """fsw-inference#325 refined fix: only the connector wait/save hooks are
    ``torch._dynamo.disable``-wrapped; the attention wrapper/op stay traceable
    so the attention is still compiled onto the NPU, and the runtime transfer
    still fires."""

    @staticmethod
    def _mod():
        import vllm_rbln.model_executor.layers.attention.attention as a
        return a

    def test_hook_helpers_are_dynamo_disabled(self):
        a = self._mod()
        assert getattr(a._kv_transfer_wait, "_torchdynamo_disable", False) is True
        assert getattr(a._kv_transfer_save, "_torchdynamo_disable", False) is True

    def test_attention_bindings_stay_traceable(self):
        # The whole wrapper must NOT be disabled — only the hooks — so the
        # attention op stays in the compiled graph (no per-layer graph break).
        self._mod()  # apply monkeypatch
        import vllm.model_executor.layers.attention.attention as vllm_attn

        for name in ("unified_attention", "unified_attention_with_output"):
            fn = getattr(vllm_attn, name)
            assert getattr(fn, "_torchdynamo_disable", False) is False, (
                f"{name} must stay traceable (attention compiled on NPU)"
            )

    def test_runtime_path_still_invokes_wait_and_save(self, monkeypatch):
        # path C (connector has bound metadata): the disabled helpers must still
        # execute the real transfer eagerly.
        a = self._mod()
        conn = _Conn(has_meta=True)
        attn_metadata = object()
        state = {"kv_cache": torch.zeros(2, 4)}

        def get_ctx(layer_name):
            return attn_metadata, None, state["kv_cache"], None

        monkeypatch.setattr(a, "get_attention_context", get_ctx)
        monkeypatch.setattr(a, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(a, "is_v1_kv_transfer_group", lambda: True)
        monkeypatch.setattr(a, "get_kv_transfer_group", lambda: conn)

        def fn(query, key, value, layer_name, kv_cache):
            return query + value

        wrapped = a._rbln_maybe_transfer_kv_layer(fn)
        wrapped(
            torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4),
            "model.layers.0.attn", state["kv_cache"],
        )

        assert conn.waited == 1
        assert conn.saved == 1
