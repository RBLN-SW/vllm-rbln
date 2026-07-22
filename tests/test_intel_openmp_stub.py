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
import zipfile
from pathlib import Path

STUB_WHEEL = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "intel_openmp-2024.2.1+rbln.stub-py3-none-any.whl"
)
STUB_VERSION = "2024.2.1+rbln.stub"
DIST_INFO_PREFIX = "intel_openmp-2024.2.1+rbln.stub.dist-info/"


def test_stub_wheel_present() -> None:
    assert STUB_WHEEL.is_file(), STUB_WHEEL


def test_stub_wheel_carries_only_dist_info() -> None:
    with zipfile.ZipFile(STUB_WHEEL) as z:
        names = z.namelist()
    runtime = [n for n in names if not n.startswith(DIST_INFO_PREFIX)]
    assert not runtime, runtime


def test_stub_wheel_declares_expected_version() -> None:
    with zipfile.ZipFile(STUB_WHEEL) as z:
        metadata = z.read(f"{DIST_INFO_PREFIX}METADATA").decode()
    assert f"Version: {STUB_VERSION}" in metadata, metadata


def test_uv_lock_pins_stub() -> None:
    lock = (Path(__file__).resolve().parents[1] / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "intel-openmp"' in lock
    assert f'version = "{STUB_VERSION}"' in lock
    assert 'path = "vendor/intel_openmp-2024.2.1+rbln.stub-py3-none-any.whl"' in lock


def test_installed_intel_openmp_is_stub() -> None:
    dist = importlib.metadata.distribution("intel-openmp")
    assert dist.version == STUB_VERSION, (
        f"env has intel-openmp {dist.version}; expected {STUB_VERSION}. "
        "Run `uv sync --locked` to pick up the vendored stub."
    )
    files = list(dist.files or [])
    binaries = [str(f) for f in files if f.suffix in {".so", ".dll", ".dylib"}]
    assert not binaries, binaries
