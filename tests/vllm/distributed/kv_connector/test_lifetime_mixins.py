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

"""What one lifetime mixin binds, no sibling may bind or reach.

The worker is assembled from mixins over one shared base. Two of them binding
one name resolves to whichever the MRO reaches first, so a `super()` written to
reach upstream lands on the sibling instead. And a mixin's only base is
`state.py`, so a member parked in one is untyped where a sibling or the base
reads it -- `state.py` is where such a member belongs.

Nothing else catches either. `F811` reads a second binding as the override the
direction subclasses do on purpose, and the untyped read would be a mypy
`attr-defined` if the pre-commit hook's environment had vllm; without it the
upstream base degrades to `Any` and every read of `self` is legal.

The mixins are discovered by their base, not listed, so a fourth is covered the
day it is added. `test_patch_targets.py` guards the other half of the split.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap


def _lifetimes() -> tuple[type, list[type]]:
    """The shared base, and every mixin whose only base is that class."""
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1 import rbln_nixl
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
        RblnNixlWorkerState,
    )

    mixins = {
        obj
        for info in pkgutil.iter_modules(rbln_nixl.__path__)
        for _, obj in inspect.getmembers(
            importlib.import_module(f"{rbln_nixl.__name__}.{info.name}"),
            inspect.isclass,
        )
        if obj.__bases__ == (RblnNixlWorkerState,)
    }
    return RblnNixlWorkerState, sorted(mixins, key=lambda c: c.__name__)


def _own_callables(cls: type) -> set[str]:
    """Names this class itself binds to something callable, not inherited ones.

    Read off `vars`, not off `def` lines, so a decorated method, a `property`, a
    `staticmethod` or a conditional definition counts as what the class binds.
    """
    return {
        name
        for name, value in vars(cls).items()
        if inspect.isfunction(value)
        or isinstance(value, (classmethod, staticmethod, property))
    }


def _collisions(base: type, mixins: list[type]) -> list[str]:
    """Names two lifetimes define, or one defines over the shared base."""
    owners: dict[str, list[str]] = {}
    for cls in mixins:
        for name in _own_callables(cls):
            owners.setdefault(name, []).append(cls.__name__)
    found = [
        f"{name}: {' and '.join(sorted(who))}"
        for name, who in owners.items()
        if len(who) > 1
    ]
    shared = _own_callables(base)
    found += [
        f"{name}: {cls.__name__} over {base.__name__}"
        for cls in mixins
        for name in sorted(_own_callables(cls) & shared)
    ]
    return sorted(found)


def _self_reads(cls: type) -> set[str]:
    """Names this class's own body reaches through `self`.

    Off the source, because the defect is about which class the read sits in and
    an inherited attribute is indistinguishable from an own one at runtime.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _cross_reads(base: type, mixins: list[type]) -> list[str]:
    """A lifetime's own callable that a sibling, or the shared base, reaches.

    The assembled subclasses are not readers here: they inherit every lifetime,
    so the same read is typed there and carries none of this risk.
    """
    owners = {name: cls for cls in mixins for name in _own_callables(cls)}
    return sorted(
        f"{name}: {owners[name].__name__} defines it, {reader.__name__} reads it"
        for reader in [base, *mixins]
        for name in _self_reads(reader)
        if name in owners and owners[name] is not reader
    )


def test_no_lifetime_shadows_a_sibling():
    base, mixins = _lifetimes()
    # A discovery that found one class would pass the assertion below on
    # nothing, which is the state this test exists to tell apart from clean.
    assert len(mixins) >= 3, (
        f"found {len(mixins)} class(es) deriving only from {base.__name__}; "
        "the split produced three, so the discovery is what broke"
    )
    assert not _collisions(base, mixins), (
        "two lifetimes bind the same name, so the one later in the MRO never "
        "runs and a `super()` in the earlier one lands on it instead of on "
        "upstream. Python does not catch it, ruff reads it as a legitimate "
        "override, and each file is correct read on its own. Move the shared "
        "one to state.py, or give them names that say which lifetime owns "
        "them:\n  " + "\n  ".join(_collisions(base, mixins))
    )


def test_no_lifetime_member_is_read_from_a_sibling_or_the_base():
    base, mixins = _lifetimes()
    assert not _cross_reads(base, mixins), (
        "a lifetime's own member is reached from outside that lifetime, where "
        "it is untyped: a mixin's only base is state.py, so the reader's class "
        "does not declare it and the read is not an error. Move it to "
        "state.py, together with any private helper of its own that it "
        "calls:\n  " + "\n  ".join(_cross_reads(base, mixins))
    )


def test_the_check_sees_a_collision():
    # Without this the test above passes on a rule that matches nothing, or on
    # a discovery that returned classes with no methods -- and reads as
    # coverage that exists.
    class Base:
        def shared(self): ...

    class L(Base):
        def both(self): ...

    class R(Base):
        def both(self): ...

        def shared(self): ...

    assert _collisions(Base, [L, R]) == ["both: L and R", "shared: R over Base"]
    assert _collisions(Base, [L]) == []


def test_the_check_sees_what_a_def_line_does_not():
    # Why `vars` and not `def` lines: a decorated `def`, an assignment, and a
    # value that is not callable each bind a name, and a grep over `def` lines
    # sees the first, misses the second, and counts the third.
    class Base:
        pass

    class L(Base):
        @property
        def as_property(self): ...

        @staticmethod
        def as_staticmethod(): ...

    class R(Base):
        as_property = 1  # not callable, so not a collision

        @classmethod
        def as_classmethod(cls): ...

        as_staticmethod = staticmethod(lambda: None)

    assert _collisions(Base, [L, R]) == ["as_staticmethod: L and R"]


def test_the_check_sees_a_read_across_two_lifetimes():
    # Without this the test above passes on a rule that matches nothing. The
    # class names are unique in the file because `inspect.getsource` finds one
    # by searching the source for its name. The reads below are the defect, so
    # mypy reports them here -- which it cannot do in the connector, where the
    # upstream base it needs is absent from this environment and becomes `Any`.
    class CrossBase:
        def reaches_up(self):
            return self.owned_by_one  # type: ignore[attr-defined]

    class CrossOwner(CrossBase):
        def owned_by_one(self): ...

    class CrossSibling(CrossBase):
        def reaches_sideways(self):
            return self.owned_by_one()  # type: ignore[attr-defined]

        def reaches_its_own(self):
            return self.reaches_sideways()

    assert _cross_reads(CrossBase, [CrossOwner, CrossSibling]) == [
        "owned_by_one: CrossOwner defines it, CrossBase reads it",
        "owned_by_one: CrossOwner defines it, CrossSibling reads it",
    ]
    # Nobody owns `owned_by_one` now, and a lifetime reading its own member
    # is what the rule permits -- both must come back clean.
    assert _cross_reads(CrossBase, [CrossSibling]) == []
