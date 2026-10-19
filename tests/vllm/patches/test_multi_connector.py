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

"""MultiConnector forwards its child connectors' worker-side KV events.

Stub children only -- the patches read nothing but ``_connectors``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.patches.multi_connector import (
    _build_init,
    get_kv_connector_kv_cache_events,
)


class _Child:
    def __init__(self, events=None):
        self.events = events
        self.calls = 0

    def get_kv_connector_kv_cache_events(self):
        self.calls += 1
        return self.events


def _multi(*children):
    return SimpleNamespace(_connectors=list(children))


def test_the_patch_is_the_one_installed():
    assert (
        MultiConnector.get_kv_connector_kv_cache_events
        is get_kv_connector_kv_cache_events
    )


def test_the_one_child_with_events_is_forwarded():
    events = object()

    assert get_kv_connector_kv_cache_events(_multi(_Child(), _Child(events))) is events


def test_no_child_with_events_gives_none():
    assert get_kv_connector_kv_cache_events(_multi(_Child(), _Child())) is None


def test_every_child_is_called_even_when_none_reports():
    children = [_Child(), _Child()]

    get_kv_connector_kv_cache_events(_multi(*children))

    assert [child.calls for child in children] == [1, 1]


@pytest.mark.parametrize("reporting_child", [0, 1, 2])
def test_reporting_child_does_not_skip_later_children(reporting_child):
    events = object()
    children = [_Child() for _ in range(3)]
    children[reporting_child].events = events

    assert get_kv_connector_kv_cache_events(_multi(*children)) is events
    assert [child.calls for child in children] == [1, 1, 1]


class _Silent:
    get_kv_connector_kv_cache_events = (
        KVConnectorBase_V1.get_kv_connector_kv_cache_events
    )


def _construct(monkeypatch, *children):
    def upstream_init(self, connectors):
        self._connectors = connectors

    monkeypatch.setattr(MultiConnector, "__init__", upstream_init)
    multi = SimpleNamespace()
    _build_init()(multi, list(children))
    return multi


def test_the_init_guard_is_installed():
    assert MultiConnector.__init__.__wrapped__ is not None


def test_one_reporting_child_constructs(monkeypatch):
    children = [_Silent(), _Child()]

    assert _construct(monkeypatch, *children)._connectors == children


def test_two_reporting_children_are_refused_at_construction(monkeypatch):
    with pytest.raises(NotImplementedError, match="more than one child"):
        _construct(monkeypatch, _Child(), _Silent(), _Child())
