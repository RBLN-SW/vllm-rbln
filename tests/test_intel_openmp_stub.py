# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify intel-openmp resolves to the vendored empty stub, not Intel's wheel."""

from __future__ import annotations

import importlib.metadata
import re
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stub_wheel(repo_root: Path) -> Path:
    matches = list((repo_root / "vendor").glob("intel_openmp-*+rbln.stub-*.whl"))
    assert len(matches) == 1, matches
    return matches[0]


@pytest.fixture(scope="module")
def stub_version(stub_wheel: Path) -> str:
    match = re.match(r"intel_openmp-(.+?)-py3-none-any\.whl$", stub_wheel.name)
    assert match, stub_wheel.name
    return match.group(1)


def test_stub_wheel_present(stub_wheel: Path) -> None:
    assert stub_wheel.is_file()


def test_stub_wheel_carries_only_dist_info(stub_wheel: Path, stub_version: str) -> None:
    dist_info_prefix = f"intel_openmp-{stub_version}.dist-info/"
    with zipfile.ZipFile(stub_wheel) as z:
        runtime = [n for n in z.namelist() if not n.startswith(dist_info_prefix)]
    assert not runtime, runtime


def test_stub_wheel_declares_expected_version(
    stub_wheel: Path, stub_version: str
) -> None:
    with zipfile.ZipFile(stub_wheel) as z:
        metadata = z.read(f"intel_openmp-{stub_version}.dist-info/METADATA").decode()
    assert f"Version: {stub_version}" in metadata


def test_uv_lock_pins_stub(
    repo_root: Path, stub_wheel: Path, stub_version: str
) -> None:
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "intel-openmp"' in lock
    assert f'version = "{stub_version}"' in lock
    assert f'path = "vendor/{stub_wheel.name}"' in lock


def test_installed_intel_openmp_is_stub(stub_version: str) -> None:
    dist = importlib.metadata.distribution("intel-openmp")
    assert dist.version == stub_version, (
        f"env has intel-openmp {dist.version}; expected {stub_version}. "
        "Run `uv sync --locked` to pick up the vendored stub."
    )
    files = list(dist.files or [])
    binaries = [str(f) for f in files if f.suffix in {".so", ".dll", ".dylib"}]
    assert not binaries, binaries
