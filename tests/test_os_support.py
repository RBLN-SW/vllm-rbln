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

import os
import subprocess
from pathlib import Path

import pytest
import tomllib
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
REQUIRES_PYTHON = SpecifierSet(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "requires-python"
    ]
)

NO_CP314_NUMPY = pytest.mark.xfail(
    strict=True,
    reason="numpy 2.2.6 ships no cp314 wheel. Drop this once the lock carries one.",
)

# RHEL 9 defaults to 3.9, under our floor, so 3.11 is its lowest usable module.
# uv caps --python-platform at manylinux_2_40; a higher glibc admits strictly more.
OS_MATRIX = [
    ("ubuntu-22.04", "x86_64-manylinux_2_35", "3.10", ()),
    ("ubuntu-24.04", "x86_64-manylinux_2_39", "3.12", ()),
    ("ubuntu-26.04", "x86_64-manylinux_2_40", "3.14", (NO_CP314_NUMPY,)),
    ("rhel-9", "x86_64-manylinux_2_34", "3.11", ()),
    ("rhel-10", "x86_64-manylinux_2_39", "3.12", ()),
]

INSTALLS = [
    pytest.param(platform, python, id=name, marks=marks)
    for name, platform, python, marks in OS_MATRIX
]
PYTHONS = [pytest.param(python, id=name) for name, _, python, _ in OS_MATRIX]


@pytest.mark.parametrize("platform, python", INSTALLS)
@pytest.mark.parametrize("extras", [(), ("kv_connectors",)], ids=["base", "kv"])
def test_locked_stack_installs(platform, python, extras, tmp_path) -> None:
    done = subprocess.run(
        ["uv", "sync", "--locked", "--dry-run", "--no-build", "--no-install-project"]
        + ["--python", python, "--python-platform", platform]
        + [arg for extra in extras for arg in ("--extra", extra)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(tmp_path / "env")},
    )
    assert done.returncode == 0, (
        f"uv.lock has nothing installable on {platform} / CPython {python}"
        f"{', extras ' + ','.join(extras) if extras else ''}:\n{done.stderr.strip()}"
    )


@pytest.mark.parametrize("python", PYTHONS)
def test_matrix_python_is_admitted(python) -> None:
    assert f"{python}.0" in REQUIRES_PYTHON, (
        f"the OS matrix targets CPython {python}, which `requires-python` excludes"
    )
