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

"""No test in this directory may name which module of the connector owns a symbol.

`from x import Y` binds Y in the importing module, so a patch aimed at one
module reaches only the code that module holds. Move that code to a sibling and
the patch still applies, still reports success, and no longer reaches anything.
The test then exercises the real object it meant to substitute, and passes or
fails for reasons unrelated to what it was written for.

Nothing in the language catches that, and neither does reading the diff: the
patch line does not change. So it is asserted here instead, over the source of
every test in this directory.

`utils.py` carries the substitutions to use instead, `patch_in_package` and
`setattr_in_package`, which apply a name to every module of the package that
binds it and refuse a name none of them binds.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# Patching a name ON a module object -- `cm.KVConnectorBase_V1`, `pw.threading`
# -- is not this defect: the attribute travels with the class or the foreign
# module it belongs to. Only the module itself as the target is. Matched on the
# callee as written, so a bare `patch` and `mock.patch` are both seen, and so is
# `patch("<dotted path>")`, the form no import statement reveals.
_PATCHERS = {
    "patch",
    "mock.patch",
    "unittest.mock.patch",
    "patch.object",
    "patch.multiple",
    "mock.patch.object",
    "mock.patch.multiple",
    "monkeypatch.setattr",
    "monkeypatch.delattr",
}

# The two functions that are the sanctioned mechanism; they patch a module on
# purpose, having found it themselves.
_IMPLEMENTATIONS = {"patch_in_package", "setattr_in_package"}


def _package() -> tuple[pathlib.Path, str]:
    """The connector package's directory and its dotted name."""
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1 import rbln_nixl

    return pathlib.Path(rbln_nixl.__file__).parent, rbln_nixl.__name__


def _module_names(tree: ast.Module, package: pathlib.Path, dotted: str) -> set[str]:
    """Local names in this file that refer to a module of the package.

    Both import forms reach one: `import <pkg>.connector as cm` and
    `from <pkg> import metadata as md`. A dotted path is a module of the
    package when the package directory holds the file it names.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head, _, tail = alias.name.rpartition(".")
                if head == dotted and (package / f"{tail}.py").exists():
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module != dotted:
                continue
            for alias in node.names:
                if (package / f"{alias.name}.py").exists():
                    names.add(alias.asname or alias.name)
    return names


def _offences(path: pathlib.Path, package: pathlib.Path, dotted: str) -> list[str]:
    tree = ast.parse(path.read_text())
    modules = _module_names(tree, package, dotted)

    inside: list[str] = []
    found: list[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inside.append(child.name)
                visit(child)
                inside.pop()
                continue
            if (
                isinstance(child, ast.Call)
                and ast.unparse(child.func) in _PATCHERS
                and child.args
                and not (set(inside) & _IMPLEMENTATIONS)
            ):
                first = child.args[0]
                target = ast.unparse(first)
                # Compare the value, not the source: unparsing a constant
                # quotes it, so a prefix test on the source never matches.
                dotted_string = (
                    isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value.startswith(dotted)
                )
                if target in modules or dotted_string:
                    call = ast.unparse(child.func)
                    found.append(f"{path.name}:{child.lineno}  {call}({target}, ...)")
            visit(child)

    visit(tree)
    return found


def test_no_test_pins_which_module_owns_a_symbol():
    package, dotted = _package()
    here = pathlib.Path(__file__).parent
    offences = [
        offence
        for path in sorted(here.glob("*.py"))
        for offence in _offences(path, package, dotted)
    ]
    assert not offences, (
        "these patches name a module of the connector, so they stop reaching "
        "the code if it moves to a sibling module -- silently, since the patch "
        "still succeeds. Use patch_in_package / patched_in_package / "
        "setattr_in_package from utils.py instead:\n  " + "\n  ".join(offences)
    )


def test_a_name_nothing_binds_leaves_no_earlier_patch_applied():
    # Callers pass several names at once, so entering patches as the names are
    # resolved would leave the ones before the failure applied for the rest of
    # the session -- one clear error becoming unrelated failures elsewhere. The
    # good name has to come FIRST: a bad name in front raises before anything is
    # patched, so only this ordering asserts anything.
    from tests.native.distributed.kv_connector.utils import (
        _package_modules,
        patch_in_package,
    )

    holder = next(m for m in _package_modules() if hasattr(m, "rebel"))
    before = holder.rebel
    with pytest.raises(AssertionError, match="nothing binds"):
        patch_in_package(rebel="SUBSTITUTE", no_such_symbol_at_all=1)
    assert holder.rebel is before


def test_the_check_sees_a_module_target(tmp_path):
    # Without this the test above passes on a directory it failed to parse, or
    # on a rule that matches nothing -- and reads as coverage that exists.
    package, dotted = _package()
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl"
        ".base_worker as wm\n"
        "from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl "
        "import metadata as md\n"
        "\n"
        "def f(monkeypatch):\n"
        "    patch.object(wm, 'rebel')\n"
        "    monkeypatch.setattr(md, 'KVSplitAxis', None)\n"
        f"    patch('{dotted}.registration.rebel')\n"
        f"    mock.patch('{dotted}.metadata.KVSplitAxis')\n"
        "    patch.object(wm.SomeClass, 'method')\n"
        "    monkeypatch.setattr(pw.threading, 'Thread', None)\n"
        "    patch('some.other.package.thing')\n"
    )
    offences = _offences(sample, package, dotted)
    # Two module targets and two dotted strings; none of the three that travel
    # with what they are attached to or point outside the package.
    assert len(offences) == 4, offences
    assert "patch.object(wm, ...)" in offences[0]
    assert "monkeypatch.setattr(md, ...)" in offences[1]
    assert "patch(" in offences[2] and "registration.rebel" in offences[2]
    assert "mock.patch(" in offences[3]
