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

"""Which specs a host's chip lets run. A bug here costs coverage silently -- the
whole-model lane would look green having compiled nothing -- so the skip decision
is pinned per case rather than left to the one chip CI happens to schedule."""

import types

import pytest

from tests.native import conftest
from tests.native.model_specs import (
    ATOM,
    CA25,
    CR03,
    CR13,
    CR23,
    KNOWN_CHIPS,
    REBEL,
    CompileModelSpec,
)


def _item(spec, *, marked=True):
    # test_dp_e2e parametrizes a fixture rather than the test, so the spec
    # arrives under the fixture's name.
    item = types.SimpleNamespace(
        keywords={"model_compile": True} if marked else {},
        callspec=types.SimpleNamespace(params={"dp_lane": spec}),
        marks=[],
    )
    item.add_marker = item.marks.append
    return item


def _skip_reason(item) -> str | None:
    return item.marks[0].kwargs["reason"] if item.marks else None


@pytest.fixture
def on_chip(monkeypatch):
    # The resolver, not its cache: patching the lookup keeps the real one's
    # memoization out of the test's way.
    def use(chip):
        monkeypatch.setattr(conftest, "_host_chip", lambda: chip)

    return use


@pytest.mark.parametrize(
    ("chips", "runs"),
    [
        (KNOWN_CHIPS, True),
        (REBEL, True),
        (frozenset({CR03}), True),
        (ATOM, False),
        (frozenset({CR13, CR23}), False),
    ],
)
def test_host_chip_decides(on_chip, chips, runs):
    on_chip(CR03)
    item = _item(CompileModelSpec("m/x", chips=chips))
    conftest._skip_other_chips([item])
    assert (_skip_reason(item) is None) is runs


def test_skip_reason_names_both_sides(on_chip):
    on_chip(CR03)
    item = _item(CompileModelSpec("m/x", chips=frozenset({CR13, CA25})))
    conftest._skip_other_chips([item])
    assert _skip_reason(item) == "needs RBLN-CA25/RBLN-CR13, host is RBLN-CR03"


def test_only_whole_model_items_are_filtered(on_chip):
    """A spec also feeds unit tests that never touch an NPU."""
    on_chip(CR03)
    item = _item(CompileModelSpec("m/x", chips=ATOM), marked=False)
    conftest._skip_other_chips([item])
    assert _skip_reason(item) is None


def test_unresolved_chip_filters_nothing(on_chip):
    on_chip(None)
    item = _item(CompileModelSpec("m/x", chips=ATOM))
    conftest._skip_other_chips([item])
    assert _skip_reason(item) is None


def test_item_without_a_spec_is_left_alone(on_chip):
    on_chip(CR03)
    item = _item(None)
    item.callspec = None
    conftest._skip_other_chips([item])
    assert _skip_reason(item) is None


@pytest.mark.parametrize(
    ("chips", "message"),
    [
        (frozenset(), "at least one chip"),
        (ATOM & REBEL, "at least one chip"),
        (frozenset({"RBLN-CR99"}), "Unknown chips"),
    ],
)
def test_rejects_unusable_chip_sets(chips, message):
    with pytest.raises(ValueError, match=message):
        CompileModelSpec("m/x", chips=chips)


def test_plain_set_is_frozen():
    spec = CompileModelSpec("m/x", chips={CR13, CR23})
    assert spec.chips == frozenset({CR13, CR23})
    assert isinstance(spec.chips, frozenset)


def test_defaults_to_every_chip():
    assert CompileModelSpec("m/x").chips == KNOWN_CHIPS
