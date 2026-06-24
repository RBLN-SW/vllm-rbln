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

"""Unit tests for the (multi-)connector KV-cache registration helpers.

Covers the orchestration that finalizes RBLN's deferred KV-cache registration
across a possibly-nested ``MultiConnector``. The per-connector hook itself is
tested in ``test_pd_disaggregation``; here we verify the traversal and the
runtime-checkable protocol that route the call to the right child connectors.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    SupportsKVCacheRegistrationFinalize,
    finalize_kv_cache_registrations,
    iter_kv_connectors,
)


class _Finalizable:
    """Connector that implements the deferred-registration hook (NIXL-like)."""

    def __init__(self) -> None:
        self.calls = 0

    def finalize_kv_cache_registration(self) -> None:
        self.calls += 1


class _Plain:
    """Connector without the hook (e.g. an LMCache-style connector)."""


class _FakeMultiConnector(MultiConnector):
    """A MultiConnector whose children are injected directly.

    Bypasses the heavy ``MultiConnector.__init__`` (config / sub-connector
    construction) while staying a real ``MultiConnector`` instance so the
    ``isinstance`` checks in the helpers behave as in production.
    """

    def __init__(self, connectors) -> None:  # noqa: D107
        self._connectors = list(connectors)


# ---------------------------------------------------------------------------
# SupportsKVCacheRegistrationFinalize protocol
# ---------------------------------------------------------------------------
class TestSupportsKVCacheRegistrationFinalizeProtocol:
    def test_instance_with_hook_matches(self):
        assert isinstance(_Finalizable(), SupportsKVCacheRegistrationFinalize)

    def test_instance_without_hook_does_not_match(self):
        assert not isinstance(_Plain(), SupportsKVCacheRegistrationFinalize)

    def test_multiconnector_does_not_match(self):
        # MultiConnector does not implement the RBLN-specific hook, which is
        # exactly why the wrapper must be expanded into its children.
        assert not isinstance(
            _FakeMultiConnector([]), SupportsKVCacheRegistrationFinalize
        )


# ---------------------------------------------------------------------------
# iter_kv_connectors
# ---------------------------------------------------------------------------
class TestIterKvConnectors:
    def test_single_leaf_yields_itself(self):
        leaf = _Plain()
        assert list(iter_kv_connectors(leaf)) == [leaf]

    def test_multiconnector_yields_children(self):
        a, b = _Plain(), _Finalizable()
        multi = _FakeMultiConnector([a, b])
        assert list(iter_kv_connectors(multi)) == [a, b]

    def test_nested_multiconnector_is_flattened(self):
        a, b, c = _Plain(), _Finalizable(), _Plain()
        nested = _FakeMultiConnector([a, _FakeMultiConnector([b, c])])
        assert list(iter_kv_connectors(nested)) == [a, b, c]

    def test_empty_multiconnector_yields_nothing(self):
        assert list(iter_kv_connectors(_FakeMultiConnector([]))) == []


# ---------------------------------------------------------------------------
# finalize_kv_cache_registrations
# ---------------------------------------------------------------------------
class TestFinalizeKvCacheRegistrations:
    def test_single_connector_with_hook_is_finalized(self):
        conn = _Finalizable()
        finalize_kv_cache_registrations(conn)
        assert conn.calls == 1

    def test_single_connector_without_hook_is_skipped(self):
        # Must not raise for connectors that do not implement the hook.
        finalize_kv_cache_registrations(_Plain())

    def test_multiconnector_finalizes_only_supporting_children(self):
        # Regression: a hook-bearing child (NIXL) nested inside MultiConnector
        # alongside a non-supporting one (LMCache) used to be silently skipped
        # because MultiConnector does not forward the RBLN-specific hook.
        lmcache = _Plain()
        nixl = _Finalizable()
        finalize_kv_cache_registrations(_FakeMultiConnector([lmcache, nixl]))
        assert nixl.calls == 1

    def test_multiconnector_finalizes_every_supporting_child(self):
        first, second = _Finalizable(), _Finalizable()
        finalize_kv_cache_registrations(_FakeMultiConnector([first, _Plain(), second]))
        assert (first.calls, second.calls) == (1, 1)

    def test_nested_multiconnector_reaches_deep_child(self):
        deep = _Finalizable()
        connector = _FakeMultiConnector(
            [_Plain(), _FakeMultiConnector([_Plain(), deep])]
        )
        finalize_kv_cache_registrations(connector)
        assert deep.calls == 1
