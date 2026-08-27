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

# RBLN connectors carry hooks upstream does not know about: finalizing KV-cache
# registration after warm-up, and flushing a KV load held until a submission is in
# flight. Under MultiConnector those hooks live on nested children, so the helpers
# flatten the tree and drive each on the children that support it.

import types

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    SupportsDeferredLoad,
    SupportsKVCacheRegistrationFinalize,
    finalize_kv_cache_registrations,
    flush_deferred_loads,
    iter_kv_connectors,
)


class _Finalizable:
    """A connector that implements the deferred-registration hook (NIXL-like)."""

    def __init__(self) -> None:
        self.calls = 0

    def finalize_kv_cache_registration(self) -> None:
        self.calls += 1


class _Plain:
    """A connector without the hook (LMCache-like). Records nothing -- it must
    never have finalize_kv_cache_registration called on it (it has none)."""


class _FakeMultiConnector(MultiConnector):
    """A real MultiConnector with children injected directly, bypassing the
    heavy __init__ (config + sub-connector construction). Staying a genuine
    MultiConnector keeps the isinstance() checks in the helpers honest."""

    def __init__(self, connectors) -> None:
        self._connectors = list(connectors)


class TestSupportsKVCacheRegistrationFinalizeProtocol:
    def test_hook_bearing_instance_matches(self):
        assert isinstance(_Finalizable(), SupportsKVCacheRegistrationFinalize)

    def test_hookless_instance_does_not_match(self):
        assert not isinstance(_Plain(), SupportsKVCacheRegistrationFinalize)

    def test_multiconnector_does_not_match(self):
        # The wrapper lacks the hook -- exactly why it must be expanded into its
        # children rather than finalized directly.
        assert not isinstance(
            _FakeMultiConnector([]), SupportsKVCacheRegistrationFinalize
        )


class TestIterKvConnectors:
    def test_leaf_yields_itself_lazily(self):
        leaf = _Plain()
        result = iter_kv_connectors(leaf)
        assert isinstance(result, types.GeneratorType)  # lazy, not a materialized list
        assert list(result) == [leaf]

    def test_multiconnector_yields_children_in_order(self):
        a, b = _Plain(), _Finalizable()
        assert list(iter_kv_connectors(_FakeMultiConnector([a, b]))) == [a, b]

    def test_nested_multiconnector_is_flattened_in_order(self):
        a, b, c = _Plain(), _Finalizable(), _Plain()
        nested = _FakeMultiConnector([a, _FakeMultiConnector([b, c])])
        assert list(iter_kv_connectors(nested)) == [a, b, c]

    def test_empty_multiconnector_yields_nothing(self):
        assert list(iter_kv_connectors(_FakeMultiConnector([]))) == []


class TestFinalizeKvCacheRegistrations:
    def test_supporting_leaf_is_finalized(self):
        conn = _Finalizable()
        finalize_kv_cache_registrations(conn)
        assert conn.calls == 1

    def test_hookless_leaf_is_skipped_without_error(self):
        finalize_kv_cache_registrations(_Plain())  # must not raise

    def test_only_supporting_children_are_finalized(self):
        # A hook-bearing child nested next to a hookless one must be finalized;
        # the hookless one is left untouched (it has no hook to call).
        plain, nixl = _Plain(), _Finalizable()
        finalize_kv_cache_registrations(_FakeMultiConnector([plain, nixl]))
        assert nixl.calls == 1

    def test_every_supporting_child_is_finalized_once(self):
        first, second = _Finalizable(), _Finalizable()
        finalize_kv_cache_registrations(_FakeMultiConnector([first, _Plain(), second]))
        assert (first.calls, second.calls) == (1, 1)

    def test_nested_mixed_tree_finalizes_only_supporting_leaves(self):
        # Supporting leaves at both the top and a nested level are each finalized
        # once; the interleaved hookless leaves are skipped.
        top = _Finalizable()
        deep = _Finalizable()
        tree = _FakeMultiConnector(
            [_Plain(), _FakeMultiConnector([deep, _Plain()]), top]
        )
        finalize_kv_cache_registrations(tree)
        assert (deep.calls, top.calls) == (1, 1)


class _Deferring:
    """A connector that holds a KV load back (RBLN pull-NIXL-like)."""

    def __init__(self) -> None:
        self.calls = 0

    def flush_deferred_load(self) -> None:
        self.calls += 1


class TestSupportsDeferredLoadProtocol:
    def test_hook_bearing_instance_matches(self):
        assert isinstance(_Deferring(), SupportsDeferredLoad)

    def test_the_other_hook_does_not_match(self):
        # The two hooks are separate: a connector that only finalizes
        # registration must not be asked to flush a load it never held.
        assert not isinstance(_Finalizable(), SupportsDeferredLoad)

    def test_multiconnector_does_not_match(self):
        assert not isinstance(_FakeMultiConnector([]), SupportsDeferredLoad)


class TestFlushDeferredLoads:
    def test_supporting_leaf_is_flushed(self):
        conn = _Deferring()
        flush_deferred_loads(conn)
        assert conn.calls == 1

    def test_hookless_leaf_is_skipped(self):
        # _Plain has no flush_deferred_load; reaching for one would raise.
        flush_deferred_loads(_Plain())

    def test_nested_multiconnector_children_are_flushed(self):
        # The wrapper has no hook, so a read held by a nested child is issued
        # only if the tree is expanded -- otherwise the child waits forever.
        a, b = _Deferring(), _Deferring()
        flush_deferred_loads(
            _FakeMultiConnector([_Plain(), _FakeMultiConnector([a, b])])
        )
        assert (a.calls, b.calls) == (1, 1)

    def test_mixed_children_flush_only_the_holders(self):
        holder, other = _Deferring(), _Finalizable()
        flush_deferred_loads(_FakeMultiConnector([other, holder]))
        assert holder.calls == 1
        assert other.calls == 0
