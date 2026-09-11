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

from types import SimpleNamespace

from vllm_rbln.patches import multi_connector_kv_events as mod
from vllm_rbln.patches.multi_connector_kv_events import (
    patched_get_kv_connector_kv_cache_events,
)


class _Events:
    def __init__(self, tag: str) -> None:
        self.tag = tag


class _Connector:
    """A sub-connector that reports ``events`` and counts the calls."""

    def __init__(self, events: _Events | None) -> None:
        self.events = events
        self.calls = 0

    def get_kv_connector_kv_cache_events(self) -> _Events | None:
        self.calls += 1
        return self.events


def _multi(*connectors: _Connector):
    return SimpleNamespace(_connectors=list(connectors))


def test_returns_none_when_no_child_produces_events():
    a, b = _Connector(None), _Connector(None)

    assert patched_get_kv_connector_kv_cache_events(_multi(a, b)) is None
    assert (a.calls, b.calls) == (1, 1)


def test_returns_the_only_child_that_produces_events():
    # The deployed shape: NIXL reports nothing, the offload connector reports.
    events = _Events("offload")
    nixl, offload = _Connector(None), _Connector(events)

    got = patched_get_kv_connector_kv_cache_events(_multi(nixl, offload))

    assert got is events


def test_every_child_is_asked_even_after_one_reports(monkeypatch):
    # Asking all children keeps their internal queues drained; a child that
    # accumulates events must not be skipped just because an earlier one won.
    monkeypatch.setattr(mod, "_warned_multiple", False, raising=False)
    first, second = _Connector(_Events("first")), _Connector(_Events("second"))

    got = patched_get_kv_connector_kv_cache_events(_multi(first, second))

    assert got is first.events
    assert (first.calls, second.calls) == (1, 1)


def test_multiple_producers_warn_once(monkeypatch, caplog):
    monkeypatch.setattr(mod, "_warned_multiple", False, raising=False)
    multi = _multi(_Connector(_Events("a")), _Connector(_Events("b")))

    with caplog.at_level("WARNING", logger=mod.__name__):
        patched_get_kv_connector_kv_cache_events(multi)
        patched_get_kv_connector_kv_cache_events(multi)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
