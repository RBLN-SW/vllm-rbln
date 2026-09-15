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

"""No two lifetime mixins may bind the same name.

The worker is assembled from mixins over one shared base, so a name two of them
define resolves to whichever the MRO reaches first, and a `super()` in the
earlier one -- written to reach upstream -- lands on the sibling instead, while
the earlier one did not change.

Nothing else catches that. `F811` is a redefinition within one namespace, so two
files binding one name reads as the override the direction subclasses and the
schedulers do on purpose, and each file is correct read on its own.

A name the shared base already binds is flagged too, since `state.py` is where a
name two lifetimes need belongs. The mixins are discovered by their base rather
than listed, so a fourth lifetime is covered the day it is added.
`test_patch_targets.py` guards the other half of what the split created.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil


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
